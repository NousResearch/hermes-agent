import json

from agent.cmh_subprocess.envelope import envelope_state_path
from agent.cmh_subprocess.halt_flags import halt_flags_path
from agent.cmh_subprocess.wrappers import (
    prepare_claude_print_invocation,
    prepare_codex_print_invocation,
)


def test_halt_all_prevents_claude_binary_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = halt_flags_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"all": True}), encoding="utf-8")
    calls = []

    def resolver(name):
        calls.append(name)
        return f"/bin/{name}"

    result = prepare_claude_print_invocation("prompt", binary_resolver=resolver)

    assert calls == []
    assert result.ok is False
    assert result.status == "halted"
    assert result.details["active_flag"] == "all"


def test_codex_missing_binary_returns_clean_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = prepare_codex_print_invocation("prompt", binary_resolver=lambda name: None)

    assert result.ok is False
    assert result.status == "missing_binary"
    assert "codex" in result.message.lower()
    assert result.argv == ()


def test_claude_missing_flags_reported_from_help_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = prepare_claude_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/claude",
        help_text="Usage: claude --print",
    )

    assert result.ok is False
    assert result.status == "missing_required_flag"
    assert result.details["missing_required_flags"] == [
        "--max-budget-usd",
        "--output-format",
        "--no-session-persistence",
    ]


def test_claude_prompt_beginning_with_option_prefix_is_unsafe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = prepare_claude_print_invocation(
        "  --dangerous-option prompt",
        binary_resolver=lambda name: "/bin/claude",
    )

    assert result.ok is False
    assert result.status == "unsafe_prompt"
    assert "prompt begins with option prefix" in result.message.lower()
    assert result.argv == ()


def test_claude_malformed_envelope_state_returns_structured_state_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = envelope_state_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")

    result = prepare_claude_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/claude",
    )

    assert result.ok is False
    assert result.status == "state_error"
    assert "state" in result.message.lower() or "envelope" in result.message.lower()
    assert result.details["path"] == str(path)
    assert result.argv == ()


def test_codex_malformed_envelope_state_returns_structured_state_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = envelope_state_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")

    result = prepare_codex_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/codex",
    )

    assert result.ok is False
    assert result.status == "state_error"
    assert "state" in result.message.lower() or "envelope" in result.message.lower()
    assert result.details["path"] == str(path)
    assert result.argv == ()


def test_claude_non_object_envelope_state_returns_structured_state_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = envelope_state_path()
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    result = prepare_claude_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/claude",
    )

    assert result.ok is False
    assert result.status == "state_error"
    assert result.details["path"] == str(path)
    assert result.argv == ()


def test_codex_non_object_envelope_state_returns_structured_state_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = envelope_state_path()
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    result = prepare_codex_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/codex",
    )

    assert result.ok is False
    assert result.status == "state_error"
    assert result.details["path"] == str(path)
    assert result.argv == ()


def test_claude_budget_cap_blocks_non_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = envelope_state_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"anthropic_max": {"envelope_messages_used_5h": 191}}),
        encoding="utf-8",
    )

    result = prepare_claude_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/claude",
        priority=False,
    )

    assert result.ok is False
    assert result.status == "budget_blocked"
    assert result.details["used"] == 191
    assert result.details["cap"] == 191


def test_claude_preflight_assembles_safe_argv_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = prepare_claude_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/claude",
    )

    assert result.ok is True
    assert result.status == "ready"
    assert result.argv == (
        "/bin/claude",
        "--print",
        "--max-budget-usd",
        "0.01",
        "--output-format",
        "text",
        "--no-session-persistence",
        "prompt",
    )


def test_claude_ready_preflight_does_not_create_state_files_when_missing_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = prepare_claude_print_invocation(
        "prompt",
        binary_resolver=lambda name: "/bin/claude",
    )

    assert result.ok is True
    assert not halt_flags_path().exists()
    assert not envelope_state_path().exists()


def test_package_exports_preflight_helpers():
    import agent.cmh_subprocess as cmh_subprocess

    assert callable(cmh_subprocess.prepare_claude_print_invocation)
    assert callable(cmh_subprocess.prepare_codex_print_invocation)
