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
        homes = [("default", tmp_path / "d"), ("coder", tmp_path / "c")]
        runtimes = {
            "default": {"platforms": {"telegram": {"state": "connected"}}},
            "coder": {
                "platforms": {
                    "discord": {
                        "state": "fatal",
                        "error_code": "duplicate_credential",
                    }
                }
            },
        }
        _patch_topology(
            monkeypatch, homes, running={"default", "coder"}, runtimes=runtimes
        )
        topo = _collect_profile_gateway_topology()
        assert topo["gateway_mode"] == "multiple"
        assert topo["profile_platforms"] == {
            "default": {"telegram": {"state": "connected"}},
            "coder": {
                "discord": {"state": "fatal", "error_code": "duplicate_credential"}
            },
        }



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
