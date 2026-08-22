"""Tests for the ByteRover memory provider config gates."""

import os
from pathlib import Path

from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


def test_run_brv_child_env_scrubbed_keeps_path_prepend(monkeypatch):
    """The brv CLI child must not inherit gateway credentials, while the
    CLI's own bin dir stays first on PATH."""
    import plugins.memory.byterover as brv_mod

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
    monkeypatch.setenv("EMAIL_PASSWORD", "mail-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        brv_mod, "_resolve_brv_path", lambda: "C:/tmp/fake-brv/brv.cmd"
    )

    captured = {}

    class _Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _Result()

    monkeypatch.setattr(brv_mod.subprocess, "run", _fake_run)

    res = brv_mod._run_brv(["query", "x"])
    assert res["success"] is True

    env = captured["env"]
    assert "GATEWAY_RELAY_SECRET" not in env
    assert "EMAIL_PASSWORD" not in env
    assert "OPENAI_API_KEY" not in env
    # The plugin's own PATH prepend still applies on top of the sanitized env.
    brv_bin_dir = str(Path("C:/tmp/fake-brv/brv.cmd").parent)
    assert env["PATH"].startswith(brv_bin_dir + os.pathsep)


