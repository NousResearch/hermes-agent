"""Dashboard SSH key management — API and on-disk behavior."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import hermes_cli.web_server as web_server
from hermes_cli.ssh_keys import (
    SshKeysError,
    delete_ssh_key,
    generate_ssh_key,
    import_ssh_key,
    list_ssh_keys,
    upsert_ssh_host,
)
from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


@pytest.fixture
def ssh_home(_isolate_hermes_home, monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/ssh-keygen" if name == "ssh-keygen" else None,
        raising=False,
    )
    return home


def test_generate_and_list_ssh_key(ssh_home, monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[0] == "ssh-keygen"

        class Result:
            returncode = 0
            stdout = "256 SHA256:abc123 comment (ED25519)\n"
            stderr = ""

        if "-lf" in cmd:
            return Result()
        if "-y" in cmd:
            result = Result()
            result.stdout = "ssh-ed25519 AAAATEST comment"
            return result

        out_f = None
        for i, arg in enumerate(cmd):
            if arg == "-f" and i + 1 < len(cmd):
                out_f = Path(cmd[i + 1])
                break
        assert out_f is not None
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----\n")
        out_pub = Path(str(out_f) + ".pub")
        out_pub.write_text("ssh-ed25519 AAAATEST comment\n")
        return Result()

    monkeypatch.setattr("hermes_cli.ssh_keys.subprocess.run", fake_run)

    info = generate_ssh_key("id_ed25519", "unit-test")
    assert info["name"] == "id_ed25519"
    assert info["has_private"] is True
    assert info["has_public"] is True
    assert info["public_key"].startswith("ssh-ed25519")
    assert "private" not in str(info).lower() or "has_private" in info

    rows = list_ssh_keys()
    assert len(rows) == 1
    assert rows[0]["name"] == "id_ed25519"
    assert "BEGIN" not in str(rows[0])


def test_import_rejects_invalid_private_key(ssh_home, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.ssh_keys.subprocess.run",
        lambda *a, **k: type("R", (), {"returncode": 1, "stdout": "", "stderr": "bad"})(),
    )
    with pytest.raises(SshKeysError, match="Invalid private key"):
        import_ssh_key("id_ed25519", "not-a-key")


def test_api_list_and_generate_ssh_keys(ssh_home, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.ssh_keys.list_ssh_keys",
        lambda: [{"name": "id_ed25519", "has_private": True, "has_public": True, "public_key": "ssh-ed25519 AAA"}],
    )
    resp = client.get("/api/ssh/keys", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["keys"][0]["name"] == "id_ed25519"
    assert "BEGIN" not in resp.text


def test_api_delete_ssh_key(ssh_home, monkeypatch):
    deleted = []

    def _delete(name):
        deleted.append(name)

    monkeypatch.setattr("hermes_cli.ssh_keys.delete_ssh_key", _delete)
    resp = client.delete("/api/ssh/keys/id_ed25519", headers=HEADERS)
    assert resp.status_code == 200
    assert deleted == ["id_ed25519"]


def test_upsert_host_writes_config(ssh_home, monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 0
            stdout = "256 SHA256:x (ED25519)\n"
            stderr = ""

        if "-lf" in cmd:
            return Result()
        if "-y" in cmd:
            result = Result()
            result.stdout = "ssh-ed25519 AAAATEST comment"
            return result
        out_f = None
        for i, arg in enumerate(cmd):
            if arg == "-f" and i + 1 < len(cmd):
                out_f = Path(cmd[i + 1])
                break
        if out_f is not None:
            out_f.parent.mkdir(parents=True, exist_ok=True)
            out_f.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----\n")
            Path(str(out_f) + ".pub").write_text("ssh-ed25519 AAAATEST comment\n")
        return Result()

    monkeypatch.setattr("hermes_cli.ssh_keys.subprocess.run", fake_run)
    generate_ssh_key("id_ed25519")
    host = upsert_ssh_host(
        alias="prod",
        host_name="10.0.0.1",
        user="dietpi",
        identity_file="id_ed25519",
    )
    assert host["alias"] == "prod"
    config_path = ssh_home / ".ssh" / "config"
    assert config_path.is_file()
    text = config_path.read_text()
    assert "Host prod" in text
    assert "HostName 10.0.0.1" in text
    assert "User dietpi" in text

    delete_ssh_key("id_ed25519")
