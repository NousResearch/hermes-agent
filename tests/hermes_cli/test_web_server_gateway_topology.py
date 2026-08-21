"""Tests for the /api/status profile + gateway topology readout.

Covers the loopback-only ``profiles`` / ``gateway_mode`` / ``gateways`` fields
added to ``/api/status``: profile enumeration, single vs multiplex vs multiple
gateway detection, and per-platform port resolution.
"""

import pytest

from hermes_cli import web_server
from hermes_cli.web_server import (
    _collect_profile_gateway_topology,
    _profile_platform_ports,
)


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

    monkeypatch.setattr(profiles_mod, "profiles_to_serve", lambda multiplex: homes)
    monkeypatch.setattr(
        profiles_mod, "_check_gateway_running",
        lambda home: next(n for n, h in homes if h == home) in running,
    )
    by_path = {home / "gateway_state.json": runtimes.get(name) for name, home in homes}
    monkeypatch.setattr(
        status_mod, "read_runtime_status", lambda path=None: by_path.get(path)
    )


class TestCollectProfileGatewayTopology:
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
            web_server,
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
            web_server,
            "_profile_gateway_writer_identity",
            lambda home, runtime: (200, 222),
        )
        topo = _collect_profile_gateway_topology()
        assert set(topo["profile_platforms"].get("coder", {})) == {"signal"}

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
            web_server,
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
            web_server, "_collect_profile_gateway_topology",
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
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
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
            web_server,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        platforms = resp.json()["gateway_platforms"]
        assert platforms["telegram"]["state"] == "connected"
        assert platforms["reviewer:discord"]["error_code"] == "duplicate_credential"

    def test_status_rejects_malformed_namespaced_platform_key(self, monkeypatch):
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "reviewer:discord:../../secret": {"state": "fatal"},
                },
            },
        )
        monkeypatch.setattr(
            web_server,
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
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
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
            web_server, "_load_configured_gateway_platforms", _boom
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
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
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
            web_server,
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

        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
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
            web_server,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )
        monkeypatch.setattr(
            profiles_mod, "get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(
            web_server, "_collect_profile_gateway_topology",
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

        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {"telegram": {"state": "connected"}},
            },
        )
        monkeypatch.setattr(
            web_server,
            "_load_configured_gateway_platforms",
            lambda: {"telegram"},
        )
        monkeypatch.setattr(
            profiles_mod, "get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(
            web_server, "_collect_profile_gateway_topology",
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
            web_server, "_collect_profile_gateway_topology",
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


# ---------------------------------------------------------------------------
# Public projection allowlist (issue #90700)
# ---------------------------------------------------------------------------

class TestPublicStatusFieldProjection:
    """Regression for issue #90700: /api/status is unauthenticated (NAS
    liveness probing, SPA pre-login bootstrap). Free-text diagnostic
    fields -- error_message on a platform entry, and the top-level
    gateway_exit_reason -- must never reach that public payload; both
    routinely embed endpoint URLs, response bodies, or exception detail
    from adapter failures."""

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

    def test_error_message_never_reaches_the_public_payload(self, monkeypatch):
        """The exact reported leak: a platform entry's free-text
        error_message (here containing an embedded URL, matching the
        issue's own described failure shape) must not survive the
        public projection, on either a running or startup_failed
        gateway."""
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "startup_failed",
                "exit_reason": "startup_failed",
                "platforms": {
                    "telegram": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                        "error_message": (
                            "POST https://api.telegram.org/bot123456:ABCsecret/"
                            "getMe failed: 409 Conflict — terminated by other "
                            "getUpdates request"
                        ),
                        "writer_pid": 123,
                        "writer_start_time": 456,
                    }
                },
            },
        )
        monkeypatch.setattr(
            web_server, "_load_configured_gateway_platforms", lambda: {"telegram"}
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        entry = resp.json()["gateway_platforms"]["telegram"]
        assert entry["state"] == "fatal"
        assert entry["error_code"] == "duplicate_credential"
        assert "error_message" not in entry
        body_text = resp.text
        assert "api.telegram.org" not in body_text
        assert "123456:ABCsecret" not in body_text

    def test_only_the_documented_field_allowlist_survives(self, monkeypatch):
        """Direct allowlist check: every field write_runtime_status()
        actually writes to a platform entry is accounted for -- the
        safe ones pass, error_message and writer identity don't, and an
        unknown future field (simulating an adapter-specific addition
        no reviewer has vetted for the public path) is dropped too."""
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {
                    "telegram": {
                        "state": "connected",
                        "error_code": "some_code",
                        "error_message": "sensitive detail",
                        "needs_attention": True,
                        "retrying_since": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:01Z",
                        "writer_pid": 123,
                        "writer_start_time": 456,
                        "some_future_adapter_field": "unreviewed",
                    }
                },
            },
        )
        monkeypatch.setattr(
            web_server, "_load_configured_gateway_platforms", lambda: {"telegram"}
        )

        resp = self.client.get("/api/status")
        entry = resp.json()["gateway_platforms"]["telegram"]

        assert entry == {
            "state": "connected",
            "error_code": "some_code",
            "needs_attention": True,
            "retrying_since": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:01Z",
        }

    def test_gateway_exit_reason_is_reduced_to_a_coarse_category(self, monkeypatch):
        """gateway_exit_reason must never reach the public payload
        verbatim -- reduced to the same bounded category
        classify_exit_reason() already produces for the authenticated
        monitoring path."""
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "stopped",
                "exit_reason": (
                    "Traceback (most recent call last): ... "
                    "ConnectionError: could not connect to internal-db-host:5432 "
                    "user=admin password=hunter2"
                ),
                "platforms": {},
            },
        )
        monkeypatch.setattr(web_server, "_load_configured_gateway_platforms", lambda: set())

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["gateway_exit_reason"], str)
        assert len(data["gateway_exit_reason"]) < 40, (
            "must be a short, bounded category, not free text"
        )
        raw_leak_markers = ("hunter2", "internal-db-host", "Traceback")
        for marker in raw_leak_markers:
            assert marker not in resp.text

    def test_none_exit_reason_stays_none(self, monkeypatch):
        """Sanity: a genuinely absent exit reason (the common, healthy
        case) must not be coerced into some non-null placeholder."""
        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 123)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda path=None: {
                "gateway_state": "running",
                "platforms": {},
            },
        )
        monkeypatch.setattr(web_server, "_load_configured_gateway_platforms", lambda: set())

        resp = self.client.get("/api/status")
        assert resp.json()["gateway_exit_reason"] is None


# ---------------------------------------------------------------------------
# Value-shape validation on retained fields (review of #90750, P2)
# ---------------------------------------------------------------------------

class TestPublicPlatformEntryValueValidation:
    """Regression for review of #90750: an allowlisted field NAME alone
    was not enough -- a malformed or hijacked VALUE in state, error_code,
    needs_attention, retrying_since, or updated_at previously still
    reached the public response unchanged. _public_platform_entry() must
    validate shape, not just key membership."""

    def test_non_dict_platform_entry_projects_to_empty(self):
        from hermes_cli.web_server import _public_platform_entry

        assert _public_platform_entry("not a dict") == {}
        assert _public_platform_entry(None) == {}
        assert _public_platform_entry(["also", "not", "a", "dict"]) == {}

    def test_unknown_state_value_normalizes_to_unknown(self):
        from hermes_cli.web_server import _public_platform_entry

        entry = _public_platform_entry({
            "state": "https://attacker.example/exfil?data=leaked",
        })
        assert entry["state"] == "unknown"

    def test_known_state_values_pass_through(self):
        from hermes_cli.web_server import _public_platform_entry

        for state in (
            "connected", "connecting", "disabled", "disconnected",
            "fatal", "paused", "retrying", "running",
        ):
            assert _public_platform_entry({"state": state})["state"] == state

    def test_free_text_error_code_is_dropped_not_forwarded(self):
        """The exact residual bypass the review identified: error_code
        is an allowlisted FIELD NAME, but a value that's actually free
        text (a URL, an embedded traceback line, oversized content)
        must not be forwarded just because the key name is safe."""
        from hermes_cli.web_server import _public_platform_entry

        entry = _public_platform_entry({
            "error_code": "see https://internal.example/logs/abc123 for detail",
        })
        assert "error_code" not in entry

    def test_short_snake_case_error_code_passes_through(self):
        from hermes_cli.web_server import _public_platform_entry

        entry = _public_platform_entry({"error_code": "duplicate_credential"})
        assert entry["error_code"] == "duplicate_credential"

    def test_non_bool_needs_attention_is_dropped(self):
        from hermes_cli.web_server import _public_platform_entry

        for bad_value in ("true", 1, "yes", None, ["x"]):
            entry = _public_platform_entry({"needs_attention": bad_value})
            assert "needs_attention" not in entry, f"failed for {bad_value!r}"

    def test_real_bool_needs_attention_passes_through(self):
        from hermes_cli.web_server import _public_platform_entry

        assert _public_platform_entry({"needs_attention": True})["needs_attention"] is True
        assert _public_platform_entry({"needs_attention": False})["needs_attention"] is False

    def test_malformed_timestamp_is_dropped(self):
        from hermes_cli.web_server import _public_platform_entry

        for bad_ts in (
            "not a timestamp",
            "javascript:alert(1)",
            "2026-01-01",  # date only, not the ISO-8601 datetime shape actually written
            12345,
            {"nested": "object"},
        ):
            for key in ("retrying_since", "updated_at"):
                entry = _public_platform_entry({key: bad_ts})
                assert key not in entry, f"failed for {key}={bad_ts!r}"

    def test_valid_iso_timestamp_passes_through(self):
        from hermes_cli.web_server import _public_platform_entry

        entry = _public_platform_entry({
            "retrying_since": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:01.123456+00:00",
        })
        assert entry["retrying_since"] == "2026-01-01T00:00:00Z"
        assert entry["updated_at"] == "2026-01-01T00:00:01.123456+00:00"

    def test_absent_fields_stay_absent_not_defaulted(self):
        """A field the adapter never wrote must not appear with a
        synthetic default -- absence should stay absence."""
        from hermes_cli.web_server import _public_platform_entry

        assert _public_platform_entry({"state": "connected"}) == {"state": "connected"}
