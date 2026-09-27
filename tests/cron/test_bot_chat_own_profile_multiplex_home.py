"""``deliver: bot-chat`` (the job's OWN profile, no ``-p``) from a multiplexed tick runs the delivery
turn in the ticking profile's home, never the gateway's launch home (#119858).

The multiplexed ticker binds the served profile as a ``HERMES_HOME`` override that never reaches
``os.environ``; the child env must be derived from the override-aware home (``served_profile_child_env``),
never from ``os.environ.copy()``.
"""
from __future__ import annotations

import subprocess

import pytest

import cron.scheduler_delivery as delivery
from agent.secret_scope import set_multiplex_active
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def multiplexer(tmp_path, monkeypatch):
    """A default-hosted multiplexer whose launch ``.env`` is in ``os.environ``, serving alpha and beta."""
    root = tmp_path / "hermes"
    for name in ("alpha", "beta"):
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / ".env").write_text(f"ANTHROPIC_API_KEY=sk-{name}\n{name.upper()}_ONLY=1\n", encoding="utf-8")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-root\nLAUNCH_ONLY_MARKER=1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-root")
    monkeypatch.setenv("LAUNCH_ONLY_MARKER", "1")
    captured: dict = {}

    def _fake_turn(argv, env, report_path, timeout):
        captured.update(env=dict(env), argv=list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(delivery, "_run_bot_chat_turn", _fake_turn)
    set_multiplex_active(True)
    try:
        yield root, captured
    finally:
        set_multiplex_active(False)


def test_an_own_profile_bot_chat_turn_runs_in_the_ticking_profiles_home(multiplexer):
    """A→B→A: each served profile's bare ``bot-chat`` child carries THAT profile's home and
    credentials; the launch profile's marker is absent; no ``-p`` is synthesised."""
    root, captured = multiplexer
    for name in ("alpha", "beta", "alpha"):
        home = root / "profiles" / name
        token = set_hermes_home_override(str(home))
        try:
            assert delivery._deliver_to_bot_chat({"id": f"j-{name}", "name": "nightly"}, "the brief", "") is None
        finally:
            reset_hermes_home_override(token)
        env = captured["env"]
        assert env["HERMES_HOME"] == str(home)
        assert env["ANTHROPIC_API_KEY"] == f"sk-{name}"
        assert "LAUNCH_ONLY_MARKER" not in env, "the launch profile's env reached a served profile's turn"
        assert "-p" not in captured["argv"]


def test_an_explicit_target_profile_still_wins_over_the_ticking_profile(multiplexer):
    """Control: ``bot-chat:beta`` fired from alpha's tick is built for beta, not for alpha."""
    root, captured = multiplexer
    token = set_hermes_home_override(str(root / "profiles" / "alpha"))
    try:
        assert delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "beta") is None
    finally:
        reset_hermes_home_override(token)
    assert captured["env"]["HERMES_HOME"] == str(root / "profiles" / "beta")
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-beta"
    assert "ALPHA_ONLY" not in captured["env"]
