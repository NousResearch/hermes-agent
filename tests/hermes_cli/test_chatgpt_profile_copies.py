"""Rotating ChatGPT grants keep one local owner across profile copies and restores."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hermes_cli import auth, auth_chatgpt


@pytest.fixture
def chatgpt_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-root"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    row = {"id": "personal", "label": "Personal", "auth_type": "oauth", "priority": 0,
           "source": "manual:chatgpt", "access_token": "access-live", "refresh_token": "refresh-live",
           "expires_at_ms": 4102444800000,
           "chatgpt": {"client_id": "issued-one", "subject": "subject-one",
                      "scopes": [auth_chatgpt.DIRECT_SCOPE], "id_token": "identity-live"}}
    state = {"ext_agent_host_id": "host-one", "active_credential_id": row["id"],
             "registrations": [{"id": row["id"], "label": row["label"],
                                "chatgpt": {"client_id": "issued-one", "subject": "subject-one"}}]}
    auth._save_auth_store({"version": 1, "providers": {auth_chatgpt.PROVIDER: state},
                          "credential_pool": {auth_chatgpt.PROVIDER: [row],
                          "openai": [{"id": "static", "auth_type": "api_key", "priority": 0,
                                      "source": "manual", "access_token": "static-key"}]}})
    return home


@pytest.mark.parametrize("copy_mode", ["clone-all", "mirror-credentials"])
def test_profile_copy_requires_its_own_chatgpt_sign_in(chatgpt_home, monkeypatch, copy_mode):
    from agent.credential_pool import load_pool
    from hermes_cli.profiles import create_profile

    before = (chatgpt_home / "auth.json").read_bytes()
    clone = create_profile("copy", clone_all=copy_mode == "clone-all", no_alias=True)
    if copy_mode == "mirror-credentials":
        from tui_gateway import server
        server._mirror_launch_credentials(clone, {"mirror_credentials": True})
    copied = json.loads((clone / "auth.json").read_text())
    assert not copied.get("credential_pool", {}).get(auth_chatgpt.PROVIDER)
    assert auth_chatgpt.PROVIDER not in copied.get("providers", {})
    assert copied["credential_pool"]["openai"][0]["access_token"] == "static-key"
    monkeypatch.setenv("HERMES_HOME", str(clone))
    assert load_pool(auth_chatgpt.PROVIDER).select() is None
    assert (chatgpt_home / "auth.json").read_bytes() == before


@pytest.mark.parametrize("signed_out", [False, True])
@pytest.mark.parametrize("snapshot_providers", ["present", "null"])
def test_snapshot_restore_keeps_current_chatgpt_generation_and_selection(chatgpt_home, signed_out, snapshot_providers):
    from hermes_cli.backup import restore_quick_snapshot

    snapshot = deepcopy(auth._load_auth_store())
    snapshot["credential_pool"][auth_chatgpt.PROVIDER][0].update(
        access_token="access-spent", refresh_token="refresh-spent")
    snapshot["credential_pool"][auth_chatgpt.PROVIDER][0]["chatgpt"]["id_token"] = "identity-spent"
    snapshot["providers"][auth_chatgpt.PROVIDER]["ext_agent_host_id"] = "host-snapshot"
    if snapshot_providers == "null":
        snapshot["providers"] = None
    if signed_out:
        current = auth._load_auth_store()
        row = current["credential_pool"][auth_chatgpt.PROVIDER][0]
        row.update(access_token="", refresh_token=None, expires_at_ms=None)
        row["chatgpt"].pop("id_token")
        current["providers"][auth_chatgpt.PROVIDER].pop("active_credential_id")
        auth._save_auth_store(current)
    live = auth._load_auth_store()
    snap_dir = chatgpt_home / "state-snapshots" / "before-refresh"
    snap_dir.mkdir(parents=True)
    source = snap_dir / "auth.json"
    source.write_text(json.dumps(snapshot))
    (snap_dir / "manifest.json").write_text(json.dumps({"files": {"auth.json": source.stat().st_size}}))

    assert restore_quick_snapshot("before-refresh", hermes_home=chatgpt_home)

    restored = auth._load_auth_store()
    assert restored["credential_pool"][auth_chatgpt.PROVIDER] == live["credential_pool"][auth_chatgpt.PROVIDER]
    assert restored["providers"][auth_chatgpt.PROVIDER] == live["providers"][auth_chatgpt.PROVIDER]
