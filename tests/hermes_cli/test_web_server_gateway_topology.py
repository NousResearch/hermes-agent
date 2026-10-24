"""Tests for the /api/status profile + gateway topology readout.

Covers the loopback-only ``profiles`` / ``gateway_mode`` / ``gateways`` fields
added to ``/api/status``: profile enumeration, single vs multiplex vs multiple
gateway detection, and per-platform port resolution.
"""

from pathlib import Path

import pytest

from hermes_cli import web_server
import gateway.status as _gw_status
import hermes_cli.web_server_gateway as _web_server_gateway
from hermes_cli.web_server_gateway import _collect_profile_gateway_topology, _profile_platform_ports


# ---------------------------------------------------------------------------
# _profile_platform_ports
# ---------------------------------------------------------------------------

class TestProfilePlatformPorts:
    def test_no_runtime_platforms_returns_empty(self, tmp_path):
        assert _profile_platform_ports(tmp_path, None) == {}
        assert _profile_platform_ports(tmp_path, {"platforms": {}}) == {}


    def test_top_level_platforms_wins_over_gateway_block(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  platforms:\n    webhook:\n      port: 1111\n"
            "platforms:\n  webhook:\n    port: 2222\n",
            encoding="utf-8",
        )
        runtime = {"platforms": {"webhook": {"state": "connected"}}}
        assert _profile_platform_ports(tmp_path, runtime) == {"webhook": 2222}


    def test_dead_platform_states_excluded(self, tmp_path):
        runtime = {
            "platforms": {
                "webhook": {"state": "fatal"},
                "api_server": {"state": "disconnected"},
                "msgraph_webhook": {"state": "connected"},
            }
        }
        assert _profile_platform_ports(tmp_path, runtime) == {"msgraph_webhook": 8646}


# ---------------------------------------------------------------------------
# _collect_profile_gateway_topology
# ---------------------------------------------------------------------------

def _patch_topology(monkeypatch, homes, running, runtimes):
    """Patch the topology collector's collaborators.

    ``homes``: list of (name, Path); ``running``: set of profile names with a
    live gateway; ``runtimes``: {name: runtime dict}.
    """
    import hermes_cli.profiles as profiles_mod
    import gateway.status as status_mod

    monkeypatch.setattr(profiles_mod, "profiles_to_serve", lambda multiplex, **kw: homes)
    monkeypatch.setattr(
        profiles_mod, "_check_gateway_running",
        lambda home: next(n for n, h in homes if h == home) in running,
    )
    by_path = {home / "gateway_state.json": runtimes.get(name) for name, home in homes}
    monkeypatch.setattr(
        status_mod, "read_runtime_status", lambda path=None: by_path.get(path)
    )


class TestCollectProfileGatewayTopology:
    def test_running_standalone_profile_is_in_topology(self, tmp_path, monkeypatch):
        from hermes_cli import profiles

        root = tmp_path / ".hermes"
        solo = root / "profiles" / "solo"
        solo.mkdir(parents=True)
        (solo / "config.yaml").write_text("gateway:\n  standalone: true\n")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(profiles, "_check_gateway_running", lambda home: home == solo)
        topo = _collect_profile_gateway_topology()
        assert "solo" in topo["profiles"]
        assert [g["profile"] for g in topo["gateways"]] == ["solo"]

    def test_no_gateways_running(self, tmp_path, monkeypatch):
        homes = [("default", tmp_path / "d"), ("coder", tmp_path / "c")]
        _patch_topology(monkeypatch, homes, running=set(), runtimes={})
        topo = _collect_profile_gateway_topology()
        assert topo["profiles"] == ["default", "coder"]
        assert topo["gateway_mode"] == "none"
        assert topo["gateways"] == []
        assert topo["profile_platforms"] == {}

    def test_collects_per_profile_platform_maps(self, tmp_path, monkeypatch):
        # Independent per-profile gateways (gateway_mode == "multiple") each
        # write their own gateway_state.json; the collector surfaces every
        # LIVE profile's raw platform map for the /api/status merge (OOF-3).
        # Entries must carry the current live process's writer identity.
        homes = [("default", tmp_path / "d"), ("coder", tmp_path / "c")]
        runtimes = {
            "default": {
                "platforms": {
                    "telegram": {
                        "state": "connected",
                        "writer_pid": 100,
                        "writer_start_time": 111,
                    }
                }
            },
            "coder": {
                "platforms": {
                    "discord": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                        "writer_pid": 200,
                        "writer_start_time": 222,
                    }
                }
            },
        }
        _patch_topology(
            monkeypatch, homes, running={"default", "coder"}, runtimes=runtimes
        )
        identities = {tmp_path / "d": (100, 111), tmp_path / "c": (200, 222)}
        monkeypatch.setattr(
            _web_server_gateway,
            "_profile_gateway_writer_identity",
            lambda home, runtime: identities.get(home),
        )
        topo = _collect_profile_gateway_topology()
        assert topo["gateway_mode"] == "multiple"
        assert topo["profile_platforms"] == {
            "default": {
                "telegram": {
                    "state": "connected",
                    "writer_pid": 100,
                    "writer_start_time": 111,
                }
            },
            "coder": {
                "discord": {
                    "state": "fatal",
                    "error_code": "duplicate_credential",
                    "writer_pid": 200,
                    "writer_start_time": 222,
                }
            },
        }

    def test_stale_platform_entries_are_not_aggregated(self, tmp_path, monkeypatch):
        # Gateway startup preserves plain platform entries across restarts;
        # if the operator removed/disabled the platform and restarted, its
        # old fatal entry must not keep degrading fleet health.  Ownership
        # is exact (pid, start_time) equality with the live process — an
        # entry written by the PREVIOUS process moments before a fast
        # restart, a legacy entry with no writer identity, and a recycled
        # PID with a different start-time fingerprint are all excluded.
        homes = [("default", tmp_path / "d"), ("coder", tmp_path / "c")]
        runtimes = {
            "coder": {
                "platforms": {
                    # Near-boundary: written by the PRIOR process (pid 199)
                    # immediately before a fast restart. A wall-clock
                    # freshness window would admit this; identity must not.
                    "telegram": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                        "updated_at": "2026-08-05T10:00:00.900000+00:00",
                        "writer_pid": 199,
                        "writer_start_time": 110,
                    },
                    # Legacy entry, no writer identity — fail closed.
                    "discord": {"state": "fatal"},
                    # Same PID recycled, different start-time fingerprint —
                    # a different process, not the live writer.
                    "slack": {
                        "state": "fatal",
                        "writer_pid": 200,
                        "writer_start_time": 110,
                    },
                    # Written by the current process — kept.
                    "signal": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                        "writer_pid": 200,
                        "writer_start_time": 222,
                    },
                }
            },
        }
        _patch_topology(
            monkeypatch, homes, running={"coder"}, runtimes=runtimes
        )
        monkeypatch.setattr(
            _web_server_gateway,
            "_profile_gateway_writer_identity",
            lambda home, runtime: (200, 222),
        )
        topo = _collect_profile_gateway_topology()
        assert set(topo["profile_platforms"].get("coder", {})) == {"signal"}

    def test_a_stale_platform_entry_reports_no_port(self, tmp_path, monkeypatch):
        # The live case: a Feishu entry left "connected" by a gateway process from a week earlier
        # (the platform has since been removed) put its port into ``gateways[].ports`` beside the
        # ports the live gateway really binds. Ports follow the same writer-identity ownership as
        # the platform map: a port the current process does not bind is not reported.
        homes = [("default", tmp_path / "d")]
        runtimes = {
            "default": {
                "platforms": {
                    "api_server": {"state": "connected", "writer_pid": 200, "writer_start_time": 222},
                    "feishu": {"state": "connected", "writer_pid": 34105, "writer_start_time": 178945520374},
                }
            },
        }
        _patch_topology(monkeypatch, homes, running={"default"}, runtimes=runtimes)
        monkeypatch.setattr(
            _web_server_gateway,
            "_profile_gateway_writer_identity",
            lambda home, runtime: (200, 222),
        )
        topo = _collect_profile_gateway_topology()
        assert set(topo["gateways"][0]["ports"]) == {"api_server"}

    def test_no_live_process_means_no_aggregation(self, tmp_path, monkeypatch):
        # When the record's PID doesn't validate against a live gateway
        # process, nothing in it is current — the whole map is excluded.
        homes = [("coder", tmp_path / "c")]
        runtimes = {
            "coder": {
                "platforms": {
                    "telegram": {
                        "state": "fatal",
                        "writer_pid": 200,
                        "writer_start_time": 222,
                    }
                }
            },
        }
        _patch_topology(monkeypatch, homes, running={"coder"}, runtimes=runtimes)
        monkeypatch.setattr(
            _web_server_gateway,
            "_profile_gateway_writer_identity",
            lambda home, runtime: None,
        )
        topo = _collect_profile_gateway_topology()
        assert topo["profile_platforms"] == {}



    def test_enumeration_failure_degrades_gracefully(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod

        def _boom(multiplex):
            raise RuntimeError("no profiles root")

        monkeypatch.setattr(profiles_mod, "profiles_to_serve", _boom)
        topo = _collect_profile_gateway_topology()
        assert topo == {
            "profiles": [],
            "gateway_mode": "unknown",
            "gateways": [],
            "profile_platforms": {},
        }


# ---------------------------------------------------------------------------
# /api/status wiring
# ---------------------------------------------------------------------------

class TestStatusEndpointTopology:
    @pytest.fixture(autouse=True)
    def _setup_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_status_includes_full_topology_on_loopback(self, monkeypatch):
        monkeypatch.setattr(
            _web_server_gateway, "_collect_profile_gateway_topology",
            lambda: {
                "profiles": ["default", "coder"],
                "gateway_mode": "single",
                "gateways": [{"profile": "default", "ports": {}}],
            },
        )
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["profiles"] == ["default", "coder"]
        assert data["gateway_mode"] == "single"
        # The per-gateway detail (host ports) is loopback-only recon.
        assert data["gateways"] == [{"profile": "default", "ports": {}}]

    def test_status_preserves_secondary_profile_platform_errors(self, monkeypatch):
        monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            _gw_status,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "telegram": {"state": "connected"},
                    "reviewer:discord": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                        "error_message": "Profiles configure the same credential",
                    },
                },
            },
        )
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        platforms = resp.json()["gateway_platforms"]
        assert platforms["telegram"]["state"] == "connected"
        assert platforms["reviewer:discord"]["error_code"] == "duplicate_credential"

    def test_status_reports_the_default_profiles_flat_adapters_under_multiplex(self, monkeypatch):
        # Under gateway.multiplex_profiles the default's own adapters are the FLAT keys of the
        # multiplexer's record (only secondaries get ``<profile>:`` prefixes). /api/status
        # re-keys the record through profile_platforms_from_multiplexer whenever the
        # multiplexer rung answers, and that re-key used to keep only prefixed keys — so the
        # one profile the record stores un-prefixed reported ``gateway_platforms: {}`` and
        # ``components.platforms.configured: 0`` while its adapters were up and delivering
        # (#123088).
        from gateway.status import GatewayLiveness
        import hermes_cli.web_routers.status as status_router

        record = {
            "gateway_state": "running",
            "served_profiles": ["default", "coder"],
            "platforms": {
                "telegram": {"state": "connected", "writer_pid": 101, "writer_start_time": 5},
                "webhook": {"state": "connected"},
                "coder:discord": {"state": "connected"},
            },
        }
        monkeypatch.setattr(
            status_router, "resolve_gateway_liveness",
            lambda **kwargs: GatewayLiveness(
                running=True, pid=4242, source="multiplexer", runtime=record))
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram", "webhook"},
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        platforms = data["gateway_platforms"]
        assert platforms["telegram"]["state"] == "connected"
        # Writer-identity stamps stay private on the re-keyed map too.
        assert "writer_pid" not in platforms["telegram"]
        assert platforms["webhook"]["state"] == "connected"
        # A secondary's entry never projects onto the default's readout.
        assert "coder:discord" not in platforms
        assert data["components"]["platforms"]["configured"] == 2
        assert data["components"]["platforms"]["connected"] == 2

    def test_profile_scoped_status_folds_default_flat_adapters_on_default_root(self, monkeypatch, tmp_path):
        # ?profile=default resolves the DEFAULT ROOT directory itself — the shipped root is
        # ``~/.hermes``, so the directory's basename (".hermes") is not the profile id. The
        # re-key used to fold on that basename: the multiplexer rung's flat keys never matched
        # (the ``gateway_platforms: {}`` symptom of #123088, one call site over from the fix
        # bare /api/status got), and the basename read as a NAMED profile, so the mirror arm
        # fabricated ``/p/<basename>/...`` ingress URLs for a profile that does not exist.
        from contextlib import nullcontext

        from gateway.status import GatewayLiveness
        import hermes_cli.web_routers.status as status_router
        from hermes_cli import web_server_profiles

        home = tmp_path / ".hermes"  # the shipped default root: basename != profile id
        home.mkdir()
        # The resolver names a default root by IDENTITY, not basename: only the root the
        # process resolves (HERMES_HOME here) folds as "default".
        monkeypatch.setenv("HERMES_HOME", str(home))
        record = {
            "gateway_state": "running",
            "served_profiles": ["default", "coder"],
            "platforms": {
                "telegram": {"state": "connected", "writer_pid": 101, "writer_start_time": 5},
                "webhook": {"state": "connected"},
                "coder:discord": {"state": "connected"},
            },
        }
        monkeypatch.setattr(web_server_profiles, "_resolve_profile_dir", lambda name: home)
        monkeypatch.setattr(web_server_profiles, "_config_profile_scope", lambda profile: nullcontext())
        monkeypatch.setattr(
            status_router, "resolve_gateway_liveness",
            lambda **kwargs: GatewayLiveness(
                running=True, pid=4242, source="multiplexer", runtime=record))
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram", "webhook"},
        )
        monkeypatch.setattr(
            _web_server_gateway, "_collect_profile_gateway_topology",
            lambda: {
                "profiles": ["default", "coder"],
                "gateway_mode": "multiplex",
                "gateways": [],
                "profile_platforms": {},
            },
        )

        resp = self.client.get("/api/status", params={"profile": "default"})

        assert resp.status_code == 200
        data = resp.json()
        platforms = data["gateway_platforms"]
        assert platforms["telegram"]["state"] == "connected"
        assert "writer_pid" not in platforms["telegram"]
        assert platforms["webhook"]["state"] == "connected"
        # The default's own adapters are never presented as shared-listener mirrors.
        assert "mirrored_from" not in platforms["webhook"]
        # A secondary's entry never projects onto the default's readout.
        assert "coder:discord" not in platforms
        assert data["components"]["platforms"]["configured"] == 2
        assert data["components"]["platforms"]["connected"] == 2

    def _scope_to_served_profile(self, monkeypatch, tmp_path, *, own_runtime, lock_pid, mux_pid, mux_runtime):
        from contextlib import nullcontext

        from gateway.status import GatewayLiveness
        import hermes_cli.web_routers.status as status_router
        from hermes_cli import web_server_profiles

        profile = tmp_path / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(web_server_profiles, "_resolve_profile_dir", lambda name: profile)
        monkeypatch.setattr(web_server_profiles, "_config_profile_scope", lambda profile: nullcontext())
        monkeypatch.setattr(status_router, "read_runtime_status", lambda path=None: own_runtime)
        # The PID rung answers first (gateway.lock resolves), so the ladder never attaches a runtime.
        monkeypatch.setattr(
            status_router, "resolve_gateway_liveness",
            lambda **kwargs: GatewayLiveness(running=True, pid=lock_pid, source="pid"),
        )
        def _lookup(home):
            return (mux_pid, mux_runtime) if home == profile else None

        monkeypatch.setattr(_gw_status, "multiplexer_liveness_for_profile", _lookup)
        monkeypatch.setattr(_web_server_gateway, "_load_configured_gateway_platforms", lambda: {"telegram"})

    def test_profile_scoped_status_uses_multiplexer_record_when_lock_names_it(self, monkeypatch, tmp_path):
        # #139080: a profile migrated onto the multiplexer keeps its retired standalone gateway's
        # gateway_state.json (stopped, telegram disconnected) while the shared gateway holds its
        # gateway.lock. Liveness came from that lock, so the platform map must come from the same
        # owner's live record, never the dead one.
        self._scope_to_served_profile(
            monkeypatch, tmp_path,
            own_runtime={
                "gateway_state": "stopped", "updated_at": "2026-10-02T12:33:48Z",
                "platforms": {"telegram": {"state": "disconnected"}},
            },
            lock_pid=4242, mux_pid=4242,
            mux_runtime={
                "gateway_state": "running",
                "served_profiles": ["default", "coder", "other"],
                "platforms": {
                    "telegram": {"state": "disconnected"},
                    "coder:telegram": {"state": "connected"},
                    "other:discord": {"state": "connected"},
                },
            },
        )

        resp = self.client.get("/api/status", params={"profile": "coder"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_state"] == "running"
        # Only coder's adapters: neither the default's flat telegram nor another profile's entry.
        assert set(data["gateway_platforms"]) == {"telegram"}
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["gateway_shared_with"] == ["default", "coder", "other"]
        assert data["components"]["platforms"]["connected"] == 1

    def test_profile_scoped_status_keeps_own_gateway_record_over_multiplexer(self, monkeypatch, tmp_path):
        # A --force-started gateway of the profile's own holds the lock under a different PID: its
        # record is the live one, and the multiplexer's view of the profile must not replace it.
        self._scope_to_served_profile(
            monkeypatch, tmp_path,
            own_runtime={"gateway_state": "running", "platforms": {"telegram": {"state": "connected"}}},
            lock_pid=5151, mux_pid=4242,
            mux_runtime={
                "gateway_state": "running",
                "served_profiles": ["default", "coder"],
                "platforms": {"coder:telegram": {"state": "disconnected"}},
            },
        )

        resp = self.client.get("/api/status", params={"profile": "coder"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["gateway_shared_with"] is None

    def test_unscoped_status_on_a_served_profiles_own_backend_uses_multiplexer_record(self, monkeypatch, tmp_path):
        # `hermes -p coder serve` answers plain /api/status (no ?profile=) for its own home. That
        # home's gateway.lock is the multiplexer's too, so the same live record applies.
        import hermes_cli.web_routers.status as status_router

        self._scope_to_served_profile(
            monkeypatch, tmp_path,
            own_runtime={"gateway_state": "stopped", "platforms": {"telegram": {"state": "disconnected"}}},
            lock_pid=4242, mux_pid=4242,
            mux_runtime={
                "gateway_state": "running",
                "served_profiles": ["default", "coder"],
                "platforms": {"telegram": {"state": "disconnected"},
                              "coder:telegram": {"state": "connected"}},
            },
        )
        monkeypatch.setattr(status_router, "get_process_hermes_home", lambda: tmp_path / "profiles" / "coder")

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_state"] == "running"
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["gateway_shared_with"] == ["default", "coder"]

    def test_profile_scoped_status_keeps_own_record_when_multiplexer_lookup_raises(self, monkeypatch, tmp_path):
        # The multiplexer lookup is optional: a failure there must not 500 the status route or
        # drop the record the ladder already selected.
        self._scope_to_served_profile(
            monkeypatch, tmp_path,
            own_runtime={"gateway_state": "running", "platforms": {"telegram": {"state": "connected"}}},
            lock_pid=4242, mux_pid=4242, mux_runtime={},
        )

        def _boom(home):
            raise OSError("unreadable multiplexer record")

        monkeypatch.setattr(_gw_status, "multiplexer_liveness_for_profile", _boom)

        resp = self.client.get("/api/status", params={"profile": "coder"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_state"] == "running"
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["gateway_shared_with"] is None

    def test_profile_scoped_status_keeps_own_record_when_multiplexer_record_is_missing(self, monkeypatch, tmp_path):
        # A missing/unreadable multiplexer gateway_state.json comes back as (pid, {}): that is no
        # record, so it must not replace the profile's own one with an empty platform map.
        self._scope_to_served_profile(
            monkeypatch, tmp_path,
            own_runtime={"gateway_state": "running", "platforms": {"telegram": {"state": "connected"}}},
            lock_pid=4242, mux_pid=4242, mux_runtime={},
        )

        resp = self.client.get("/api/status", params={"profile": "coder"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["gateway_shared_with"] is None

    def test_remote_health_liveness_never_borrows_the_local_multiplexer_record(self, monkeypatch, tmp_path):
        # A health-probe answer carries the REMOTE container's PID, which can equal an unrelated
        # local host-record PID; only a PID proven on this host may select the local record.
        from gateway.status import GatewayLiveness
        import hermes_cli.web_routers.status as status_router

        mux_runtime = {
            "gateway_state": "running", "served_profiles": ["default", "coder"],
            "platforms": {"coder:telegram": {"state": "disconnected"}},
        }
        self._scope_to_served_profile(
            monkeypatch, tmp_path,
            own_runtime={"gateway_state": "running", "platforms": {"telegram": {"state": "connected"}}},
            lock_pid=4242, mux_pid=4242, mux_runtime=mux_runtime,
        )
        monkeypatch.setattr(
            status_router, "resolve_gateway_liveness",
            lambda **kwargs: GatewayLiveness(running=True, pid=4242, source="health",
                                             health_body={"gateway_state": "running", "pid": 4242}),
        )
        calls = []

        def _lookup(home):
            calls.append(home)
            return 4242, mux_runtime

        monkeypatch.setattr(_gw_status, "multiplexer_liveness_for_profile", _lookup)

        resp = self.client.get("/api/status", params={"profile": "coder"})

        assert resp.status_code == 200
        data = resp.json()
        assert calls == []
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["gateway_shared_with"] is None

    def _status_on_the_host(self, monkeypatch, record):
        # The default profile's own backend on the host: its gateway_state.json is the live
        # multiplexer's record, and the real lookup (host record + default root) resolves to it.
        from gateway import host_topology
        from gateway.status import GatewayLiveness
        import hermes_cli.web_routers.status as status_router
        from hermes_constants import get_process_hermes_home

        monkeypatch.setattr(_gw_status, "read_runtime_status", lambda path=None: record)
        monkeypatch.setattr(_gw_status, "get_runtime_status_running_pid",
                            lambda runtime=None, *, expected_home=None: (runtime or {}).get("pid"))
        monkeypatch.setattr(host_topology, "host_gateway_topology", lambda: host_topology.HostGatewayTopology(
            pid=4242, profiles=("default", "reviewer"), source="host_record"))
        monkeypatch.setattr(status_router, "resolve_gateway_liveness",
                            lambda **kwargs: GatewayLiveness(running=True, pid=4242, source="pid"))
        monkeypatch.setattr(_web_server_gateway, "_load_configured_gateway_platforms", lambda: {"telegram"})
        monkeypatch.setattr(_web_server_gateway, "_collect_profile_gateway_topology", lambda: {
            "profiles": ["default", "reviewer"], "gateway_mode": "multiplex", "gateways": [],
            "profile_platforms": {}})
        # The lookup genuinely succeeds here (no zero-arg PID-probe trap making it raise).
        assert _gw_status.multiplexer_liveness_for_profile(get_process_hermes_home()) == (4242, record)
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        return resp.json()

    def test_status_on_the_host_keeps_other_profiles_platform_errors(self, monkeypatch):
        # The host's own record proves the live gateway, so it is read whole: folding it through
        # the default profile's projection dropped every ``<profile>:<platform>`` entry and the
        # platforms component read ok over another profile's fatal adapter.
        data = self._status_on_the_host(monkeypatch, {
            "pid": 4242, "gateway_state": "running", "served_profiles": ["default", "reviewer"],
            "platforms": {
                "telegram": {"state": "connected"},
                "reviewer:discord": {"state": "fatal", "error_code": "duplicate_credential",
                                     "error_message": "Profiles configure the same credential"},
            },
        })

        assert data["gateway_platforms"]["reviewer:discord"]["error_code"] == "duplicate_credential"
        assert data["gateway_platforms"]["telegram"]["state"] == "connected"
        assert data["components"]["platforms"]["status"] != "ok"

    def test_status_on_the_host_names_the_profiles_it_serves(self, monkeypatch):
        # A restart of the host gateway blips every profile it serves, so its own unscoped
        # readout names them just as a served profile's does.
        data = self._status_on_the_host(monkeypatch, {
            "pid": 4242, "gateway_state": "running", "served_profiles": ["default", "reviewer"],
            "platforms": {"telegram": {"state": "connected"}},
        })

        assert data["gateway_shared_with"] == ["default", "reviewer"]

    def test_status_rejects_malformed_namespaced_platform_key(self, monkeypatch):
        monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            _gw_status,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "reviewer:discord:../../secret": {"state": "fatal"},
                },
            },
        )
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        assert "reviewer:discord:../../secret" not in resp.json()["gateway_platforms"]

    def test_namespaced_key_validation_does_not_fail_open(self, monkeypatch):
        # If loading the configured-platform set throws, plain keys keep the
        # historical pass-through — but colon-containing keys must STILL be
        # validated against the key grammar. A config-load failure must not
        # let malformed keys from a process-local JSON file reach the public
        # endpoint.
        monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            _gw_status,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "telegram": {"state": "connected"},
                    "reviewer:discord": {"state": "fatal"},
                    "reviewer:discord:../../secret": {"state": "fatal"},
                    "REVIEWER:DISCORD": {"state": "fatal"},
                },
            },
        )

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(
            _web_server_gateway, "_load_configured_gateway_platforms", _boom
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        platforms = resp.json()["gateway_platforms"]
        assert "telegram" in platforms
        assert "reviewer:discord" in platforms
        assert "reviewer:discord:../../secret" not in platforms
        assert "REVIEWER:DISCORD" not in platforms

    def test_status_preserves_hyphenated_plugin_platform_key(self, monkeypatch):
        # Plugin platform IDs may contain hyphens (plugins/platforms/<dir>
        # names are lowercased directory names; the Platform enum accepts
        # them). A valid ``reviewer:foo-bar`` fatal entry must survive the
        # public filter.
        monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            _gw_status,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "reviewer:foo-bar": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                    },
                },
            },
        )
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        platforms = resp.json()["gateway_platforms"]
        assert platforms["reviewer:foo-bar"]["error_code"] == "duplicate_credential"

    def test_status_merges_independent_profile_gateway_failures(self, monkeypatch):
        # OOF-3 deployment mode: separate gateway services per profile
        # (gateway_mode == "multiple"). Each profile's failures live in its
        # own gateway_state.json; the machine-level /api/status must fold
        # them in as <profile>:<platform> so NAS health monitoring sees them.
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            _gw_status,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "telegram": {
                        "state": "connected",
                        "writer_pid": 123,
                        "writer_start_time": 456,
                    }
                },
            },
        )
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )
        monkeypatch.setattr(
            profiles_mod, "get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(
            _web_server_gateway, "_collect_profile_gateway_topology",
            lambda: {
                "profiles": ["default", "lead-gen-outreach"],
                "gateway_mode": "multiple",
                "gateways": [
                    {"profile": "default", "ports": {}},
                    {"profile": "lead-gen-outreach", "ports": {}},
                ],
                "profile_platforms": {
                    "default": {"telegram": {"state": "connected"}},
                    "lead-gen-outreach": {
                        "telegram": {
                            "state": "fatal",
                            "error_code": "duplicate_credential",
                            "writer_pid": 789,
                            "writer_start_time": 1011,
                        },
                        "bad key": {"state": "fatal"},
                    },
                },
            },
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        platforms = data["gateway_platforms"]
        # Active profile's own entry untouched, no self-namespacing.
        assert platforms["telegram"]["state"] == "connected"
        assert "default:telegram" not in platforms
        # Independent profile's failure folded in under the namespaced key.
        assert (
            platforms["lead-gen-outreach:telegram"]["error_code"]
            == "duplicate_credential"
        )
        # Keys that don't survive the grammar are dropped, not projected.
        assert not any("bad key" in key for key in platforms)
        # Writer-identity stamps (process recon, same class as the auth-gated
        # gateway_pid) never project onto the endpoint — neither on the active
        # profile's own entries nor on merged cross-profile entries.
        for entry in platforms.values():
            assert "writer_pid" not in entry
            assert "writer_start_time" not in entry
        # The platforms component rollup counts the merged failure.
        assert data["components"]["platforms"]["status"] == "degraded"

    def test_profile_scoped_status_does_not_merge_other_profiles(self, monkeypatch):
        # ?profile=<name> targets one profile's view — merging every other
        # profile's failures into it would misattribute state.
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            _gw_status,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {"telegram": {"state": "connected"}},
            },
        )
        monkeypatch.setattr(
            _web_server_gateway,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )
        monkeypatch.setattr(
            profiles_mod, "get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(
            _web_server_gateway, "_collect_profile_gateway_topology",
            lambda: {
                "profiles": ["default", "coder"],
                "gateway_mode": "multiple",
                "gateways": [],
                "profile_platforms": {
                    "coder": {"telegram": {"state": "fatal"}},
                },
            },
        )

        resp = self.client.get("/api/status", params={"profile": "current"})

        assert resp.status_code == 200
        platforms = resp.json()["gateway_platforms"]
        assert "coder:telegram" not in platforms

    def test_profile_names_and_mode_public_when_auth_gated(self, monkeypatch):
        # Profile NAMES + gateway_mode are low-sensitivity product surface: the
        # Hermes Cloud Portal reads /api/status over the network (a gated bind)
        # to render the profile list, so they must survive the auth gate.
        monkeypatch.setattr(
            _web_server_gateway, "_collect_profile_gateway_topology",
            lambda: {
                "profiles": ["default", "coder"],
                "gateway_mode": "multiplex",
                "gateways": [{"profile": "default", "ports": {"webhook": 8644}}],
            },
        )
        monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
        try:
            resp = self.client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["profiles"] == ["default", "coder"]
            assert data["gateway_mode"] == "multiplex"
            # But the per-gateway detail (host ports = recon) stays gated,
            # alongside hermes_home / gateway_pid.
            assert "gateways" not in data
            assert "hermes_home" not in data
            assert "gateway_pid" not in data
        finally:
            monkeypatch.setattr(
                web_server.app.state, "auth_required", False, raising=False
            )
