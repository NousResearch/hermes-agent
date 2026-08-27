from hermes_constants import get_ares_state_root, get_default_hermes_root


def test_ares_state_root_is_installation_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "audit"))
    assert get_ares_state_root() == get_default_hermes_root() / "ares"
