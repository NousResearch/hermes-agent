"""Fresh interactive grants do not select a provider before config commits."""

import argparse
import json

import pytest

from hermes_cli import auth


@pytest.mark.parametrize("active_provider", ["openai-codex", None])
def test_fresh_xai_login_config_failure_preserves_route(tmp_path, monkeypatch, capsys, active_provider):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = {"version": 1, "providers": {}}
    if active_provider:
        store["active_provider"] = active_provider
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(store), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  provider: openai-codex\n  default: old-model\n", encoding="utf-8")
    before = config_path.read_bytes()
    monkeypatch.setattr(auth, "_xai_oauth_device_code_login", lambda **_kw: {
        "tokens": {"access_token": "fresh-access", "refresh_token": "fresh-refresh"},
        "base_url": "https://api.x.ai/v1",
    })

    def disk_full(*_args, **_kwargs):
        raise OSError("config disk full")

    monkeypatch.setattr(auth, "atomic_config_replace", disk_full)
    with pytest.raises(OSError, match="config disk full"):
        auth._login_xai_oauth(argparse.Namespace(no_browser=True), auth.PROVIDER_REGISTRY["xai-oauth"],
                              force_new_login=True)
    after = json.loads(auth_path.read_text(encoding="utf-8"))
    assert after.get("active_provider") == active_provider
    assert after["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "fresh-refresh"
    assert config_path.read_bytes() == before
    assert "Login successful" not in capsys.readouterr().out


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "reuse"])
@pytest.mark.parametrize("update_config", [False, True], ids=["auth-only", "direct"])
def test_login_activation_is_owned_by_config_update(tmp_path, monkeypatch, capsys, reuse, update_config):
    import hermes_yaml as yaml

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1, "active_provider": "openrouter",
        "providers": {"xai-oauth": {"tokens": {
            "access_token": "existing-access", "refresh_token": "existing-refresh",
        }, "last_auth_error": {"code": "superseded"}}},
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  provider: openrouter\n  default: old-model\n", encoding="utf-8")
    before = config_path.read_bytes()
    monkeypatch.setattr("builtins.input", lambda *_a: "y")

    def fresh_grant(**_kw):
        assert not reuse, "reusing credentials must not request a new grant"
        return {"tokens": {"access_token": "fresh-access", "refresh_token": "fresh-refresh"}}

    monkeypatch.setattr(auth, "_xai_oauth_device_code_login", fresh_grant)
    auth._login_xai_oauth(argparse.Namespace(no_browser=True), auth.PROVIDER_REGISTRY["xai-oauth"],
                          force_new_login=not reuse, update_config=update_config)
    after = json.loads(auth_path.read_text(encoding="utf-8"))
    assert after["active_provider"] == ("xai-oauth" if update_config else "openrouter")
    if update_config:
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]["provider"] == "xai-oauth"
    else:
        assert config_path.read_bytes() == before
    if not reuse:
        assert "last_auth_error" not in after["providers"]["xai-oauth"]
    assert "Login successful" in capsys.readouterr().out
