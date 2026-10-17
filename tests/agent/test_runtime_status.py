import json
import os
from pathlib import Path
from types import SimpleNamespace

import agent.runtime_status as runtime_status
from agent.runtime_status import emit_runtime_status
from hermes_cli._parser import build_top_level_parser


def _agent(*, used=799_123, size=1_000_000, compressions=3, session_id="role-session"):
    return SimpleNamespace(
        runtime_status_file=None,
        session_id=session_id,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=used,
            context_length=size,
            compression_count=compressions,
        ),
    )


def test_runtime_status_is_opt_in(tmp_path: Path):
    agent = _agent()
    target = tmp_path / "runtime-status.json"
    assert emit_runtime_status(agent) is False
    assert not target.exists()


def test_runtime_status_atomically_publishes_per_agent_context_and_compressions(tmp_path: Path):
    target = tmp_path / "runtime-status.json"
    agent = _agent()
    agent.runtime_status_file = str(target)

    assert emit_runtime_status(agent) is True

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": "1.0.0",
        "pid": os.getpid(),
        "session_id": "role-session",
        "context_used": 799_123,
        "context_size": 1_000_000,
        "compression_count": 3,
        "updated_at": payload["updated_at"],
    }
    assert payload["updated_at"].endswith("Z")
    assert os.stat(target).st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".runtime-status.json.*.tmp"))


def test_runtime_status_does_not_require_os_fchmod(monkeypatch, tmp_path: Path):
    target = tmp_path / "runtime-status.json"
    agent = _agent()
    agent.runtime_status_file = str(target)
    monkeypatch.delattr(os, "fchmod")

    assert emit_runtime_status(agent) is True
    assert json.loads(target.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_runtime_status_replace_failure_preserves_existing_file_and_removes_temp(
    monkeypatch, tmp_path: Path
):
    target = tmp_path / "runtime-status.json"
    target.write_text("existing snapshot\n", encoding="utf-8")
    agent = _agent()
    agent.runtime_status_file = str(target)

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(runtime_status.os, "replace", fail_replace)

    assert emit_runtime_status(agent) is False
    assert target.read_text(encoding="utf-8") == "existing snapshot\n"
    assert not list(tmp_path.glob(".runtime-status.json.*.tmp"))


def test_cli_writes_populated_runtime_status_on_initialisation(tmp_path, monkeypatch):
    import cli as cli_mod

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "source": "test",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            self.session_id = kwargs["session_id"]
            self.context_compressor = SimpleNamespace(
                last_prompt_tokens=12_345,
                context_length=272_000,
                compression_count=2,
            )

    monkeypatch.setattr(cli_mod, "AIAgent", _DummyAgent)
    target = tmp_path / "runtime-status.json"
    shell = cli_mod.HermesCLI(
        model="gpt-5",
        compact=True,
        max_turns=1,
        runtime_status_file=str(target),
    )

    assert shell._init_agent() is True
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["session_id"] == shell.session_id
    assert payload["context_used"] == 12_345
    assert payload["context_size"] == 272_000
    assert payload["compression_count"] == 2


def test_runtime_status_clamps_invalid_counters_without_leaking_other_agent_state(tmp_path: Path):
    target = tmp_path / "runtime-status.json"
    agent = _agent(used=2_000_000, size=1_000_000, compressions=-4)
    agent.runtime_status_file = str(target)

    assert emit_runtime_status(agent) is True
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["context_used"] == 1_000_000
    assert payload["context_size"] == 1_000_000
    assert payload["compression_count"] == 0


def test_runtime_status_invalid_home_expansion_is_non_fatal():
    agent = _agent()
    agent.runtime_status_file = "~hermes-agent-user-that-must-not-exist/status.json"

    assert emit_runtime_status(agent) is False


def test_runtime_status_file_is_accepted_before_or_after_chat_subcommand():
    parser, _, chat_parser = build_top_level_parser()
    top = parser.parse_args(["--runtime-status-file", "/tmp/role-status.json"])
    chat = chat_parser.parse_args(["--runtime-status-file", "/tmp/role-status.json"])
    assert top.runtime_status_file == "/tmp/role-status.json"
    assert chat.runtime_status_file == "/tmp/role-status.json"
