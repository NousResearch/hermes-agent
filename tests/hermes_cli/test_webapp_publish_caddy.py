"""Unit tests for the Desktop setup flow's Caddy site management (tmp files)."""
from types import SimpleNamespace

from hermes_cli.webapp_publish import caddy


def test_render_site_block_has_no_placeholders():
    block = caddy.render_site_block("desk.example.com", 9120)
    assert "desk.example.com {" in block
    assert "reverse_proxy 127.0.0.1:9120" in block
    assert "<" not in block and ">" not in block
    assert "tls internal" not in block


def test_render_site_block_internal_tls():
    block = caddy.render_site_block("hermes-desktop.local", 9120, internal_tls=True)
    assert "tls internal" in block


def test_upsert_appends_and_backs_up(tmp_path):
    f = tmp_path / "Caddyfile"
    f.write_text(":80 {\n\trespond \"hi\"\n}\n")
    changed, backup = caddy.upsert_site_block(f, "desk.example.com", 9120)
    assert changed is True
    assert backup is not None and backup.exists()
    text = f.read_text()
    assert ":80 {" in text  # existing block preserved
    assert "desk.example.com {" in text
    assert backup.read_text() == ":80 {\n\trespond \"hi\"\n}\n"


def test_upsert_idempotent(tmp_path):
    f = tmp_path / "Caddyfile"
    f.write_text("")
    caddy.upsert_site_block(f, "desk.example.com", 9120)
    changed, backup = caddy.upsert_site_block(f, "desk.example.com", 9120)
    assert changed is False and backup is None
    assert f.read_text().count("desk.example.com {") == 1


def test_upsert_does_not_match_substring_host(tmp_path):
    f = tmp_path / "Caddyfile"
    f.write_text("other-desk.example.com {\n\treverse_proxy 127.0.0.1:1\n}\n")
    changed, _ = caddy.upsert_site_block(f, "desk.example.com", 9120)
    assert changed is True


def test_validate_uses_caddy_exit_code(tmp_path, monkeypatch):
    f = tmp_path / "Caddyfile"
    f.write_text("")
    monkeypatch.setattr(caddy.shutil, "which", lambda name: "/usr/bin/caddy")
    assert caddy.validate(f, run=lambda cmd, **kw: SimpleNamespace(returncode=0)) is True
    assert caddy.validate(f, run=lambda cmd, **kw: SimpleNamespace(returncode=1)) is False


def test_validate_false_without_caddy(tmp_path, monkeypatch):
    monkeypatch.setattr(caddy.shutil, "which", lambda name: None)
    assert caddy.validate(tmp_path / "Caddyfile") is False


def test_restore_backup_roundtrip(tmp_path):
    f = tmp_path / "Caddyfile"
    f.write_text("original\n")
    _, backup = caddy.upsert_site_block(f, "desk.example.com", 9120)
    assert backup is not None
    caddy.restore_backup(f, backup)
    assert f.read_text() == "original\n"


def test_render_redirect_block():
    block = caddy.render_redirect_block("hermes-desktop.local", "192.168.1.5", 9120)
    assert block.startswith("http://hermes-desktop.local {")
    assert "redir http://192.168.1.5:9120{uri} permanent" in block
    assert "reverse_proxy" not in block


def test_upsert_block_matches_http_prefixed_site(tmp_path):
    f = tmp_path / "Caddyfile"
    f.write_text(caddy.render_redirect_block("hermes-desktop.local", "192.168.1.5", 9120))
    changed, backup = caddy.upsert_block(f, "hermes-desktop.local", "whatever")
    assert changed is False and backup is None


def test_reload_tries_reload_then_restart(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[1])
        return SimpleNamespace(returncode=0 if cmd[1] == "restart" else 1)

    assert caddy.reload_caddy(run=run) is True
    assert calls == ["reload", "restart"]