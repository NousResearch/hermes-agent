"""Remote config: a whole-document save from an older read never undoes a newer write, a Cloud
(Nous credential) boot in a fresh interpreter does not hang, a plane URL is expanded once across
dotenv reloads, a provider switch never half-applies its route, a deleted profile stops being
polled, and a named custom provider added on the plane is seen after a poll."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hermes_cli.config_backend import get_config_backend, read_config_doc, write_config_document

from .conftest import _config_cmd
from .stub_plane import INSTANCE


def test_document_save_sends_exactly_the_edit_made_to_its_own_read(plane):
    from hermes_cli.config import load_config, read_raw_config, save_config
    profile = plane.profile("default")
    profile.update(values={"display": {"personality": "a"}, "agent": {"max_turns": 10}}, version=1)
    backend = get_config_backend()

    def land(path, value):  # another writer's change, picked up by the next poll
        section, key = path
        profile["values"].setdefault(section, {})[key] = value
        profile["version"] += 1
        backend.poll_one(backend._state(plane.home))

    stale = read_raw_config()
    land(("agent", "max_turns"), 99)
    stale["display"]["personality"] = "b"
    save_config(stale)  # a read from before another writer's change keeps that change
    assert profile["values"] == {"display": {"personality": "b"}, "agent": {"max_turns": 99}}

    merged = load_config()  # untagged (a merged copy): matched to the read it came from
    land(("agent", "max_turns"), 7)
    merged["display"]["personality"] = "m"
    save_config(merged)
    assert profile["values"] == {"display": {"personality": "m"}, "agent": {"max_turns": 7}}

    land(("display", "personality"), "c")
    current = read_config_doc(plane.home / "config.yaml")
    current["display"]["personality"] = "b"
    write_config_document(plane.home / "config.yaml", current)  # setting an older value back is an edit too
    assert profile["values"]["display"] == {"personality": "b"}


def test_cloud_boot_in_a_fresh_interpreter_does_not_hang(plane, tmp_path):
    expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
    (plane.home / "auth.json").write_text(json.dumps({"version": 1, "active_provider": "nous", "providers": {
        "nous": {"access_token": "plane-token-1", "expires_at": expires, "refresh_token": "r"}}}))
    plane.profile("default").update(values={"display": {"personality": "remote"}}, version=1)
    child = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR", "SYSTEMROOT")}
    child.update(HERMES_HOME=str(plane.home), HOME=str(tmp_path), HERMES_CONFIG_BACKEND="remote",
                 HERMES_CONFIG_REMOTE_URL=plane.url, HERMES_CONFIG_INSTANCE_ID=INSTANCE,
                 PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    code = ("from hermes_cli.env_loader import load_hermes_dotenv\n"
            "load_hermes_dotenv(load_external_secrets=False)\n"
            "import tui_gateway.server\n"
            "from hermes_cli.config import load_config\n"
            "print('RESULT=' + load_config()['display']['personality'])\n")

    proc = subprocess.run([sys.executable, "-c", code], env=child, capture_output=True, text=True,
                          timeout=90, stdin=subprocess.DEVNULL, cwd=str(tmp_path))

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "RESULT=remote" in proc.stdout + proc.stderr  # tui_gateway.server routes print() to stderr


def test_profile_guest_login_never_borrows_the_root_token(profile_plane, monkeypatch):
    from plugins.config_backends.remote import credentials
    expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
    (profile_plane.home / "auth.json").write_text(json.dumps({"providers": {"nous": {
        "access_token": "root-token", "expires_at": expires}}}))
    guest = profile_plane.home / "profiles" / "guest"
    guest.mkdir(parents=True)
    (guest / "auth.json").write_text(json.dumps({"providers": {"nous": {"guest_id": "g-1"}}}))

    assert credentials._stored_nous_token(guest) == ""  # its own login: exchanged by the resolver
    assert credentials._stored_nous_token(profile_plane.home / "profiles" / "fresh") == "root-token"


def test_reloading_dotenv_expands_each_value_once_across_layers(plane, monkeypatch, tmp_path):
    from hermes_cli.config_backend import bootstrap_deployment
    from hermes_cli.env_loader import load_hermes_dotenv
    project, managed = tmp_path / "project.env", tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_URL", plane.url)
    for name in ("MANAGED_TOKEN_URL", "PLANE_HOST", "OP_SERVICE_ACCOUNT_TOKEN", "OP_ONLY"):  # undone after the test
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    (plane.home / ".env").write_text("HERMES_CONFIG_REMOTE_URL=${HERMES_CONFIG_REMOTE_URL}/\n")
    (plane.home / ".op.env").write_text("OP_SERVICE_ACCOUNT_TOKEN=tok-op\nOP_ONLY=from-op\n")
    project.write_text("PLANE_HOST=https://plane.example\nOP_ONLY=from-project\n")
    (managed / ".env").write_text("MANAGED_TOKEN_URL=${PLANE_HOST}/token\n")

    for _ in range(3):
        bootstrap_deployment(plane.home, project)  # the boot's own pass, as load_hermes_dotenv runs it
        load_hermes_dotenv(hermes_home=plane.home, project_env=project, load_external_secrets=False)

    assert os.environ["HERMES_CONFIG_REMOTE_URL"] == plane.url + "/"  # a self-reference expands once
    assert os.environ["MANAGED_TOKEN_URL"] == "https://plane.example/token"  # an earlier layer is seen
    assert os.environ["OP_ONLY"] == "from-op"  # a gap-filler never takes over an earlier layer's value


def test_provider_switch_with_a_locked_route_changes_nothing(plane, capsys):
    plane.profile("default").update(values={"model": {
        "default": "m", "provider": "anthropic", "base_url": "https://api.anthropic.com"}}, version=1)
    plane.upper_locks = [{"path": "model.base_url", "level": "tenant"}]

    code, err = _config_cmd(capsys, "set", "model.provider", "openrouter")

    assert code == 1 and "model.base_url" in err
    assert plane.patches() == []


def test_deleted_profile_is_no_longer_polled(profile_plane):
    from hermes_constants import mark_named_profile_deleted
    gone = profile_plane.home / "profiles" / "gone"
    gone.mkdir(parents=True)
    backend = get_config_backend()
    backend._state(gone)
    mark_named_profile_deleted(gone)
    before = len(profile_plane.requests)

    backend._poll_due()

    assert backend._key(gone) not in backend._states
    assert not any(r["profile"] == "gone" for r in profile_plane.requests[before:])


def test_named_custom_provider_added_on_the_plane_is_seen_after_a_poll(plane):
    from providers import get_provider_profile
    assert get_provider_profile("my-gw") is None
    plane.profile("default").update(values={"providers": {"my-gw": {
        "base_url": "https://gw.example/v1", "api_key": "${MY_GW_KEY}"}}}, version=1)
    backend = get_config_backend()
    backend.poll_one(backend._state(plane.home))

    assert get_provider_profile("my-gw") is not None
