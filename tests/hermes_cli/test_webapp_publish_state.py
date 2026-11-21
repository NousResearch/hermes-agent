"""Unit tests for the Desktop setup mode-switch state record and cleanup."""
from hermes_cli.webapp_publish import caddy, mdns, state


def test_state_roundtrip(tmp_path):
    state.save_state(tmp_path, {"mode": "local", "mdns_name": "hermes-desktop.local"})
    assert state.load_state(tmp_path)["mode"] == "local"
    state.forget_state(tmp_path)
    assert state.load_state(tmp_path) == {}


def test_load_state_missing_or_corrupt(tmp_path):
    assert state.load_state(tmp_path) == {}
    (tmp_path / state.STATE_REL).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / state.STATE_REL).write_text("{not json")
    assert state.load_state(tmp_path) == {}


def test_cleanup_same_mode_is_noop(tmp_path):
    state.save_state(tmp_path, {"mode": "domain", "caddy_blocks": ["x"]})
    assert state.cleanup_previous_mode(tmp_path, "domain") == []
    assert state.load_state(tmp_path)["mode"] == "domain"  # untouched


def test_cleanup_removes_exact_recorded_block(tmp_path, monkeypatch):
    f = tmp_path / "Caddyfile"
    block = caddy.render_redirect_block("hermes-desktop.local", "192.168.1.5", 9120)
    user_block = "other.example.com {\n\treverse_proxy 127.0.0.1:8080\n}\n"
    f.write_text(user_block + "\n" + block)
    monkeypatch.setattr(caddy, "DEFAULT_CADDYFILE", f)
    monkeypatch.setattr(caddy, "validate", lambda path: True)
    monkeypatch.setattr(caddy, "reload_caddy", lambda: True)
    state.save_state(tmp_path, {"mode": "local", "caddy_blocks": [block]})
    actions = state.cleanup_previous_mode(tmp_path, "domain")
    text = f.read_text()
    assert "hermes-desktop.local" not in text
    assert "other.example.com" in text  # user's block preserved
    assert any("Removed" in a for a in actions)
    assert state.load_state(tmp_path) == {}  # record consumed


def test_cleanup_leaves_edited_block_alone(tmp_path, monkeypatch):
    f = tmp_path / "Caddyfile"
    recorded = caddy.render_redirect_block("hermes-desktop.local", "192.168.1.5", 9120)
    edited = recorded.replace("permanent", "temporary")  # user modified it
    f.write_text(edited)
    monkeypatch.setattr(caddy, "DEFAULT_CADDYFILE", f)
    state.save_state(tmp_path, {"mode": "local", "caddy_blocks": [recorded]})
    state.cleanup_previous_mode(tmp_path, "domain")
    assert f.read_text() == edited  # exact-match only: never guess


def test_cleanup_restores_avahi_with_consent(tmp_path, monkeypatch):
    conf = tmp_path / "avahi-daemon.conf"
    backup = tmp_path / "avahi-daemon.conf.bak-desktop"
    conf.write_text("[server]\nhost-name=hermes-desktop\n")
    backup.write_text("[server]\n#host-name=old\n")
    monkeypatch.setattr(mdns, "DEFAULT_AVAHI_CONF", conf)
    monkeypatch.setattr(mdns, "restart_avahi", lambda: True)
    state.save_state(tmp_path, {"mode": "local", "avahi_backup": str(backup),
                                "mdns_name": "hermes-desktop.local", "caddy_blocks": []})
    actions = state.cleanup_previous_mode(tmp_path, "domain", confirm=lambda q, d: True)
    assert conf.read_text() == backup.read_text()
    assert any("Restored" in a for a in actions)


def test_cleanup_avahi_restore_declined(tmp_path, monkeypatch):
    conf = tmp_path / "avahi-daemon.conf"
    backup = tmp_path / "bak"
    conf.write_text("new")
    backup.write_text("old")
    monkeypatch.setattr(mdns, "DEFAULT_AVAHI_CONF", conf)
    state.save_state(tmp_path, {"mode": "local", "avahi_backup": str(backup), "caddy_blocks": []})
    state.cleanup_previous_mode(tmp_path, "domain", confirm=lambda q, d: False)
    assert conf.read_text() == "new"


def test_cleanup_validation_failure_restores_caddyfile(tmp_path, monkeypatch):
    f = tmp_path / "Caddyfile"
    block = caddy.render_site_block("desk.example.com", 9120)
    original = "keep.example.com {\n\trespond \"ok\"\n}\n\n" + block
    f.write_text(original)
    monkeypatch.setattr(caddy, "DEFAULT_CADDYFILE", f)
    monkeypatch.setattr(caddy, "validate", lambda path: False)
    state.save_state(tmp_path, {"mode": "domain", "caddy_blocks": [block]})
    actions = state.cleanup_previous_mode(tmp_path, "local")
    assert f.read_text() == original  # restored
    assert any("skipped" in a for a in actions)