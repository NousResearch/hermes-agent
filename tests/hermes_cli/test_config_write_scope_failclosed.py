"""G2 guard rail: on a host serving several profiles, a config write with no
explicit profile is REFUSED (409) — it previously defaulted to the launch
profile and wrote the wrong tenant's config (2026-09-25: connecting Bobby's
MCP servers in the GUI wrote them into Nana's config.yaml). Same contract as
``destructive_profile``; 409 so clients distinguish it from route-level 400s.
"""
import json

import pytest
import yaml
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def _multiplex_state_is_per_test():
    """Multi-profile activation is a one-way PROCESS-global flip; contain it."""
    import agent.secret_scope as secret_scope
    from tui_gateway import launch_profile_policy

    was_active = secret_scope.is_multiplex_active()
    snapshot = launch_profile_policy._snapshot
    secret_scope.set_multiplex_active(False)
    launch_profile_policy._snapshot = None
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(was_active)
        launch_profile_policy._snapshot = snapshot


@pytest.fixture
def homes(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated launch home + one named profile, both seeded with real files."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    launch_home = get_hermes_home()
    profiles_root = launch_home / "profiles"
    beta = profiles_root / "worker_beta"
    for home in (launch_home, beta):
        (home / "memories").mkdir(parents=True, exist_ok=True)
        (home / "memories" / "MEMORY.md").write_text(f"memory of {home.name}\n", encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "mcp_servers": {"github": {"command": "npx", "enabled": True}},
            "proxy": {"label": home.name},
        }), encoding="utf-8")
    (beta / ".env").write_text("BETA=1\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"launch": launch_home, "worker_beta": beta}


@pytest.fixture
def multiplexed(homes):
    """Arm the guard the way a real two-profile host does: the BOOT probe."""
    from agent.secret_scope import is_multiplex_active
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting_eagerly

    assert activate_multi_profile_hosting_eagerly() is True, "boot probe did not see two homes"
    assert is_multiplex_active()


class TestConfigWriteScopeFailClosed:
    def test_unnamed_write_refused_while_multiplexing(self, homes, multiplexed):
        from hermes_cli.web_routers._common import config_write_scope

        with pytest.raises(HTTPException) as excinfo:
            with config_write_scope(None):
                pass
        assert excinfo.value.status_code == 409
        assert "profile required" in excinfo.value.detail

    def test_empty_string_profile_refused_while_multiplexing(self, homes, multiplexed):
        from hermes_cli.web_routers._common import config_write_scope

        with pytest.raises(HTTPException) as excinfo:
            with config_write_scope("   "):
                pass
        assert excinfo.value.status_code == 409

    def test_named_profile_write_passes_while_multiplexing(self, homes, multiplexed):
        from hermes_cli.web_routers._common import config_write_scope

        with config_write_scope("worker_beta"):
            pass  # no exception; the lock and scope were entered

    def test_unnamed_write_still_means_launch_profile_when_single_profile(self, homes):
        from hermes_cli.web_routers._common import config_write_scope

        with config_write_scope(None):
            pass  # single-profile host: the destructive_profile precedent holds
