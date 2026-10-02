"""Real auth writes must not migrate another configured identity's pool."""

import json
from types import SimpleNamespace

import pytest
import yaml


@pytest.mark.parametrize("shared_endpoint", [False, True])
@pytest.mark.parametrize("own_first", [False, True])
@pytest.mark.parametrize("requested", ["custom:openrouter", "custom:private-relay", "custom:missing"])
def test_auth_add_migrates_only_the_requested_identity(
    tmp_path, monkeypatch, shared_endpoint, own_first, requested,
):
    from hermes_cli.auth import AuthError
    from hermes_cli.auth_commands import auth_add_command
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    own_url = "http://127.0.0.1:19002/v1"
    other_url = own_url if shared_endpoint else "http://127.0.0.1:19001/v1"
    entries = [
        ("alpha", {"name": "Alpha Relay", "base_url": other_url}),
        ("openrouter", {"name": "Private Relay", "base_url": own_url}),
    ]
    if own_first:
        entries.reverse()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "providers": dict(entries),
    }, sort_keys=False), encoding="utf-8")
    pools = {
        pool_id: [{"id": pool_id, "label": pool_id, "auth_type": "api_key",
                   "priority": 0, "source": "manual", "access_token": token}]
        for pool_id, token in [
            ("custom:alpha-relay", "fake-alpha-sentinel"),
            ("custom:private-relay", "fake-private-old"),
            ("openrouter", "fake-builtin-sentinel"),
        ]
    }
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": pools,
    }), encoding="utf-8")

    auth_add_command(SimpleNamespace(
        provider=requested, auth_type="api_key",
        api_key="fake-private-new", label="private-new",
    ))

    after = json.loads((tmp_path / "auth.json").read_text())["credential_pool"]
    assert after["custom:alpha-relay"] == pools["custom:alpha-relay"]
    assert after["openrouter"] == pools["openrouter"]
    if requested == "custom:missing":
        assert after["custom:private-relay"] == pools["custom:private-relay"]
        assert [row["access_token"] for row in after[requested]] == ["fake-private-new"]
        with pytest.raises(AuthError, match="(?i)(custom|provider|missing)"):
            resolve_runtime_provider(requested=requested)
    else:
        assert "custom:private-relay" not in after
        assert {row["access_token"] for row in after["custom:openrouter"]} == {
            "fake-private-old", "fake-private-new",
        }
        resolved = resolve_runtime_provider(requested=requested)
        assert resolved["base_url"] == own_url
        assert resolved["api_key"] in {"fake-private-old", "fake-private-new"}
