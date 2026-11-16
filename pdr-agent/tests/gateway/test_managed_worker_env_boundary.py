"""Managed worker boundary: a worker FOR another profile never inherits the launch profile's
credentials, and a ledger id never names a path outside worker-outboxes (PR #106742 security lane)."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("multiplex", [True, False])
def test_secondary_profile_worker_gets_no_launch_credentials(tmp_path, monkeypatch, multiplex):
    launch = tmp_path / ".hermes"
    sec = launch / "profiles" / "l106742sec"
    sec.mkdir(parents=True)
    (launch / ".env").write_text("OPENROUTER_API_KEY=LAUNCH-or\nMY_SERVICE_TOKEN=LAUNCH-mysvc\n")
    (sec / ".env").write_text("OPENROUTER_API_KEY=SEC-or\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for key, value in {"OPENROUTER_API_KEY": "LAUNCH-or", "MY_SERVICE_TOKEN": "LAUNCH-mysvc",
                       "NOUS_API_KEY": "LAUNCH-nous", "VAULT_ONLY_TOKEN": "LAUNCH-vault"}.items():
        monkeypatch.setenv(key, value)
    from hermes_cli import env_loader
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "VAULT_ONLY_TOKEN", "bitwarden")
    from agent import secret_scope
    from gateway.session_managed_worker import _worker_env
    secret_scope.set_multiplex_active(multiplex)
    try:
        env = _worker_env(SimpleNamespace(profile_id=str(sec)))
        own = _worker_env(SimpleNamespace(profile_id=str(launch)))
    finally:
        secret_scope.set_multiplex_active(False)
    # The scrub follows the worker's OWNING profile, not the process-wide multiplex flag.
    assert env is not None and env["OPENROUTER_API_KEY"] == "SEC-or"
    assert sorted(k for k, v in env.items() if "LAUNCH" in str(v)) == []
    if not multiplex:
        assert own is None  # single-profile: the launch profile's own worker inherits byte-for-byte


def test_launch_profile_worker_keeps_env_only_tool_credentials_under_multiplex(tmp_path, monkeypatch):
    """Under multiplex the launch profile's own worker resolves what its owner resolves for that
    profile (``launch_secret_scope``): a tool key injected only into the launch environment
    (systemd ``Environment=``, ``op run``) reaches it, while a secondary's worker never sees it."""
    launch = tmp_path / ".hermes"
    sec = launch / "profiles" / "l106742sec"
    sec.mkdir(parents=True)
    (launch / ".env").write_text("OPENROUTER_API_KEY=LAUNCH-or\n")
    (sec / ".env").write_text("OPENROUTER_API_KEY=SEC-or\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "ENVONLY-fc")
    monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "ENVONLY-aux")
    from agent.secret_scope import get_secret, set_multiplex_active
    from gateway.session_managed_worker import _worker_env
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, launch_profile_runtime_scope
    activate_multi_profile_hosting()
    try:
        with launch_profile_runtime_scope(launch):
            assert get_secret("FIRECRAWL_API_KEY") == "ENVONLY-fc"  # what the owner resolves
        own = _worker_env(SimpleNamespace(profile_id=str(launch)))
        other = _worker_env(SimpleNamespace(profile_id=str(sec)))
    finally:
        set_multiplex_active(False)
    assert own["FIRECRAWL_API_KEY"] == "ENVONLY-fc" and own["OPENROUTER_API_KEY"] == "LAUNCH-or"
    assert "AUXILIARY_VISION_API_KEY" not in own  # Hermes-internal secrets never reach a child
    assert "FIRECRAWL_API_KEY" not in other and other["OPENROUTER_API_KEY"] == "SEC-or"
    assert sorted(k for k, v in other.items() if "ENVONLY" in str(v) or "LAUNCH" in str(v)) == []


@pytest.mark.parametrize("bad", ["../../../../escape", "x/../../y", "..\\..\\win", "C:\\evil", ""])
def test_outbox_dir_only_names_paths_inside_worker_outboxes(tmp_path, bad):
    from agent.managed_worker import outbox_dir
    assert outbox_dir(tmp_path, "admission-worker:" + "ab" * 16).parent == tmp_path / "worker-outboxes"
    with pytest.raises(ValueError):
        outbox_dir(tmp_path, "admission-worker:" + bad if bad else bad)
