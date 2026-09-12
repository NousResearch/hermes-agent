from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_audit(home: str, prompt: str, index: int) -> str:
    import os

    os.environ["HERMES_HOME"] = home
    from hermes_cli.oneshot_audit import OneShotAudit

    audit = OneShotAudit(prompt, "query")
    audit.bind_session(f"session-{index}")
    audit.finish(0)
    return audit.audit_id


def _records(home: Path) -> list[dict]:
    path = home / "logs" / "oneshot-audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_audit_correlates_lifecycle_without_copying_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.oneshot_audit import OneShotAudit

    prompt = "private prompt token: swordfish"
    audit = OneShotAudit.start(prompt, "query-file")
    assert audit is not None
    audit.bind_session("session-123")
    audit.finish(2, "validation_error")

    raw = (tmp_path / "logs" / "oneshot-audit.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert prompt not in raw
    assert [row["event"] for row in rows] == ["started", "finished"]
    assert {row["audit_id"] for row in rows} == {audit.audit_id}
    assert rows[0]["prompt_sha256"] == rows[1]["prompt_sha256"]
    assert rows[0]["prompt_chars"] == len(prompt)
    assert rows[1]["outcome"] == "validation_error"
    assert rows[1]["exit_code"] == 2
    assert rows[1]["session_id"] == "session-123"
    assert "argv" not in rows[0]


def test_concurrent_invocations_append_complete_correlated_records(tmp_path):
    prompts = [f"secret-{index}" for index in range(12)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        audit_ids = list(pool.map(_write_audit, [str(tmp_path)] * len(prompts), prompts, range(len(prompts))))

    rows = _records(tmp_path)
    assert len(rows) == len(prompts) * 2
    assert {row["audit_id"] for row in rows} == set(audit_ids)
    for audit_id in audit_ids:
        lifecycle = [row for row in rows if row["audit_id"] == audit_id]
        assert [row["event"] for row in lifecycle] == ["started", "finished"]
    raw = (tmp_path / "logs" / "oneshot-audit.jsonl").read_text(encoding="utf-8")
    assert not any(prompt in raw for prompt in prompts)


def test_audit_records_normalized_caller_without_raw_author_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(
        "HERMES_TURN_AUTHOR",
        json.dumps({
            "id": "bot:worker",
            "name": "Private Operator Name",
            "is_bot": True,
            "origin": "peer-host",
            "private": "must not be copied",
        }),
    )
    from hermes_cli.oneshot_audit import OneShotAudit

    audit = OneShotAudit("hello", "query", session_source="peer")
    audit.finish()

    raw = (tmp_path / "logs" / "oneshot-audit.jsonl").read_text(encoding="utf-8")
    rows = _records(tmp_path)
    assert rows[0]["caller_id"] == "bot:peer-host/worker"
    assert rows[0]["caller_is_bot"] is True
    assert rows[0]["session_source"] == "peer"
    assert "Private Operator Name" not in raw
    assert "must not be copied" not in raw
    assert "name" not in rows[0]


def test_nonquiet_agent_failure_uses_continuation_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import cli as cli_mod
    from hermes_cli.oneshot_audit import OneShotAudit

    audit = OneShotAudit("hello", "query")
    runner = SimpleNamespace(
        agent=SimpleNamespace(session_id="continuation-session"),
        session_id="compression-parent",
        _oneshot_invocation_audit=audit,
        _claim_active_session=lambda *_args, **_kwargs: True,
        console=SimpleNamespace(print=lambda *_args, **_kwargs: None),
        _show_security_advisories=lambda: None,
        chat=lambda *_args, **_kwargs: None,
        _print_exit_summary=lambda **_kwargs: None,
    )
    monkeypatch.setattr(cli_mod, "_should_seed_interactive", lambda *_args: False)
    monkeypatch.setattr(cli_mod, "_collect_query_images", lambda query, _image: (query, []))
    monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda _images: [])
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)

    cli_mod._run_single_query_mode(runner, "hello", None, False, True)

    finished = _records(tmp_path)[-1]
    assert finished["outcome"] == "agent_error"
    assert finished["exit_code"] == 0
    assert finished["session_id"] == "continuation-session"


@pytest.mark.parametrize("entrypoint", ["startup", "oneshot"])
def test_pre_agent_exit_one_is_validation_error(tmp_path, monkeypatch, entrypoint):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import main as main_mod
    from hermes_cli.oneshot_audit import OneShotAudit

    audit = OneShotAudit("hello", "prompt")
    monkeypatch.setattr(main_mod, "_prepare_oneshot_audit", lambda _args: audit)
    args = SimpleNamespace(oneshot="hello")

    if entrypoint == "startup":
        monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: (_ for _ in ()).throw(SystemExit(1)))
        invoke = main_mod._prepare_agent_startup_audited
    else:
        monkeypatch.setattr(
            main_mod,
            "_confirm_startup_expensive_model_override",
            lambda _args: (_ for _ in ()).throw(SystemExit(1)),
        )
        invoke = main_mod._run_oneshot_from_args

    with pytest.raises(SystemExit) as exc_info:
        invoke(args)

    assert exc_info.value.code == 1
    finished = _records(tmp_path)[-1]
    assert finished["outcome"] == "validation_error"
    assert finished["exit_code"] == 1