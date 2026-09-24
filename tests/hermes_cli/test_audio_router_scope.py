"""Scope-isolation regression for the ElevenLabs voices route's env fallback.

``get_elevenlabs_voices`` reads ``ELEVENLABS_API_KEY`` through the secret scope
when ``.env`` has no key. A bound scope whose read fails must leave the route
unavailable -- never borrow the ambient ``os.environ`` key (another profile's
under multiplex). Only the unscoped default-profile path (UnscopedSecretError)
may read the env.
"""

import pytest


@pytest.fixture(autouse=True)
def _restore_process_scope_state():
    """``_config_profile_scope``/``launch_secret_scope`` can freeze the launch-env
    authority (one-way process state); restore it so sibling suites are unaffected."""
    import agent.secret_scope as secret_scope
    from tui_gateway import launch_profile_policy

    was_active = secret_scope.is_multiplex_active()
    authority = launch_profile_policy._authority
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(was_active)
        launch_profile_policy._authority = authority


class _ExplodingScope(dict):
    """A bound secret scope whose resolution fails (resolver/backend error)."""
    def get(self, name, default=None):
        raise RuntimeError("resolver boom")


@pytest.mark.asyncio
async def test_voices_route_scope_failure_never_borrows_env(tmp_path, monkeypatch):
    from agent import secret_scope as ss
    from hermes_cli.web_routers.audio import get_elevenlabs_voices

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-foreign-profile")

    def _urlopen(*a, **k):
        raise AssertionError("network must not be reached without a key")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    was_active = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(_ExplodingScope())
    try:
        result = await get_elevenlabs_voices()
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(was_active)

    assert result == {"available": False, "voices": []}
