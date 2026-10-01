"""OAuth observation retains its supplied profile boundary."""
import pytest
from auth import provider_status
from tests.auth.test_pool_environment import environment, profile_scope, multiplex_scope  # noqa: F401


@pytest.mark.parametrize("observe", [
    lambda e: provider_status.get_codex_auth_status(environment=e),
    lambda e: provider_status.get_xai_oauth_auth_status(environment=e),
    lambda e: provider_status.get_minimax_oauth_auth_status(environment=e),
    lambda e: provider_status.get_plugin_oauth_auth_status("synthetic", environment=e),
])
def test_status_rejects_retained_settings_before_storage(tmp_path, monkeypatch, observe):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    supplied = environment(a)
    monkeypatch.setattr("auth.store._load_auth_store",
                        lambda *args: pytest.fail("foreign authentication store read"))
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            observe(supplied)
    assert not (a / "auth.json").exists()
    assert not (b / "auth.json").exists()


def test_minimax_status_observes_each_profile_and_does_not_rotate(tmp_path):
    from auth.provider_state import save_provider_auth_state
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with profile_scope(a):
        save_provider_auth_state("minimax-oauth", {
            "access_token": "owned-token", "refresh_token": "owned-refresh",
            "region": "cn", "expires_at": "2099-01-01T00:00:00+00:00"})
    before = (a / "auth.json").read_bytes()
    for home, expected in ((a, True), (b, False), (a, True)):
        with profile_scope(home):
            result = provider_status.get_minimax_oauth_auth_status(environment=environment(home))
            assert result["logged_in"] is expected
    assert (a / "auth.json").read_bytes() == before
    assert not (b / "auth.json").exists()
