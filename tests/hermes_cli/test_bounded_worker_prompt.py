"""Supplied-material worker context uses a verified, pinned provider only."""
import hashlib
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_policy import PolicyViolation
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def board(tmp_path):
    path = tmp_path / "board.db"
    conn = connect(path)
    target = kb.create_task(
        conn, title="Supplied leaf", body="APPROVED MATERIAL", assignee="fixture-worker",
        workspace_kind="scratch", initial_status="blocked",
    )
    unrelated = kb.create_task(
        conn, title="UNRELATED ROLE WORK", assignee="fixture-worker",
        workspace_kind="scratch", initial_status="blocked",
    )
    conn.execute("UPDATE tasks SET status='done', result='PRIVATE RESULT' WHERE id=?", (unrelated,))
    for task_id, summary in ((unrelated, "PRIVATE ROLE SUMMARY"), (target, "PRIVATE PRIOR ATTEMPT")):
        conn.execute(
            "INSERT INTO task_runs(task_id, profile, status, started_at, ended_at, outcome, summary) "
            "VALUES(?, 'fixture-worker', 'done', 1, 2, 'completed', ?)",
            (task_id, summary),
        )
    kb.add_comment(conn, target, author="fixture-worker", body="PRIVATE COMMENT")
    yield conn, path, target
    conn.close()


def _pin_provider(board, hook_source):
    conn, path, _ = board
    provider = path.parent / "execution_policy.py"
    provider.write_text(
        "def validate_receipt(*args, **kwargs):\n    return True\n"
        "def runtime_snapshot(*args, **kwargs):\n    return {}\n" + hook_source,
    )
    marker = path.with_name(path.name + ".policy-required.json")
    conn.execute("CREATE TABLE fixture_policy(value TEXT)")
    conn.execute(
        "CREATE TRIGGER fixture_guard BEFORE DELETE ON fixture_policy "
        "BEGIN SELECT RAISE(ABORT, 'held'); END",
    )
    conn.execute("CREATE TABLE kanban_execution_policy_required(marker TEXT NOT NULL)")
    conn.execute("INSERT INTO kanban_execution_policy_required VALUES(?)", (str(marker),))
    capabilities = {
        row[0]: hashlib.sha256(row[1].encode()).hexdigest()
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name IN "
            "('fixture_policy', 'fixture_guard', 'kanban_execution_policy_required')",
        )
    }
    marker.write_text(json.dumps({
        "version": "1", "database": str(path), "required": True,
        "capabilities": capabilities,
        "protected_tables": ["fixture_policy", "kanban_execution_policy_required"],
        "provider_sha256": hashlib.sha256(provider.read_bytes()).hexdigest(),
    }))
    return provider, marker


def test_pinned_provider_supplies_exact_context_without_implicit_history(board, monkeypatch):
    conn, _, target = board
    exact = "  SUPPLIED MATERIAL ONLY\nStable approved input.\n\n"
    conn.execute("UPDATE tasks SET body=? WHERE id=?", (exact, target))
    _pin_provider(board,
        "def worker_context(conn, task_id):\n"
        "    return conn.execute('SELECT body FROM tasks WHERE id=?', (task_id,)).fetchone()[0]\n",
    )
    first = kb.build_worker_context(conn, target)
    monkeypatch.setattr(kb.time, "time", lambda: 9000000)
    assert kb.build_worker_context(conn, target) == first == exact
    for private in ("PRIVATE ROLE SUMMARY", "UNRELATED ROLE WORK", "PRIVATE PRIOR ATTEMPT", "PRIVATE COMMENT"):
        assert private not in first


@pytest.mark.parametrize("result", ["''", "' \\n\\t'", "False", "42", "{}", "b'bytes'", "'x' * 128001"])
def test_malformed_provider_context_fails_closed(board, result):
    conn, _, target = board
    _pin_provider(board, "def worker_context(conn, task_id):\n    return " + result + "\n")
    with pytest.raises(PolicyViolation, match="worker context"):
        kb.build_worker_context(conn, target)


@pytest.mark.parametrize("hook", [
    "worker_context = 42\n", "worker_context = None\n",
    "def worker_context(conn, task_id):\n    raise RuntimeError('fixture unavailable')\n",
])
def test_unusable_provider_hook_never_falls_back(board, hook):
    conn, _, target = board
    _pin_provider(board, hook)
    with pytest.raises(PolicyViolation, match="worker context"):
        kb.build_worker_context(conn, target)


def test_unguarded_board_keeps_implicit_role_history_attempts_comments(board):
    conn, path, target = board
    # Unpinned files are not plugins and must never be executed.
    (path.parent / "execution_policy.py").write_text("raise AssertionError('untrusted provider')\n")
    context = kb.build_worker_context(conn, target)
    for original in ("APPROVED MATERIAL", "PRIVATE ROLE SUMMARY", "UNRELATED ROLE WORK", "PRIVATE PRIOR ATTEMPT", "PRIVATE COMMENT"):
        assert original in context


@pytest.mark.parametrize("hook", ["", "def worker_context(conn, task_id):\n    return None\n"])
def test_provider_can_leave_other_tasks_on_sdk_context(board, hook):
    conn, _, target = board
    before = kb.build_worker_context(conn, target)
    _pin_provider(board, hook)
    assert kb.build_worker_context(conn, target) == before


@pytest.mark.parametrize("damage", ["provider", "marker", "trigger"])
def test_context_override_still_requires_verified_policy(board, damage):
    conn, _, target = board
    provider, marker = _pin_provider(board, "def worker_context(conn, task_id):\n    return 'approved'\n")
    if damage == "provider":
        provider.write_text("raise AssertionError('tampered provider')\n")
    elif damage == "marker":
        marker.unlink()
    else:
        conn.execute("DROP TRIGGER fixture_guard")
    with pytest.raises(PolicyViolation, match="policy"):
        kb.build_worker_context(conn, target)


def test_provider_context_size_boundary_preserves_exact_bytes(board):
    conn, _, target = board
    _pin_provider(board, "def worker_context(conn, task_id):\n    return 'x' * 128000\n")
    assert kb.build_worker_context(conn, target) == "x" * 128000


@pytest.fixture
def cli_construction(monkeypatch):
    """Run the real setup mixin, replacing model/bootstrap side effects only."""
    import sys
    from types import SimpleNamespace
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from hermes_cli import mcp_startup

    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=construct))
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(
        ChatConsole=lambda: SimpleNamespace(print=lambda *a: None),
        _cprint=lambda *a: None, _prepare_deferred_agent_startup=lambda: None,
        logger=SimpleNamespace(warning=lambda *a: None),
    ))
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **kw: None)
    monkeypatch.setitem(sys.modules, "agent.credits_tracker", SimpleNamespace(
        seed_credits_at_session_start=lambda agent: None,
    ))
    shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    for name in (
        "agent model api_key base_url provider api_mode acp_command acp_args max_turns "
        "enabled_toolsets disabled_toolsets system_prompt prefill_messages reasoning_config service_tier "
        "_providers_only _providers_ignore _providers_order _provider_sort _provider_require_params "
        "_provider_data_collection _openrouter_min_coding_score session_id _fallback_model "
        "checkpoints_enabled checkpoint_max_snapshots checkpoint_max_total_size_mb "
        "checkpoint_max_file_size_mb pass_session_id _pending_title"
    ).split():
        setattr(shell, name, None)
    for name in (
        "finalize_preloaded_skills _install_tool_callbacks _ensure_tirith_security "
        "_current_reasoning_callback _on_thinking _on_tool_progress _on_notice "
        "_on_notice_clear _on_reaction _agent_status_print"
    ).split():
        setattr(shell, name, lambda: None)
    shell._ensure_runtime_credentials = lambda: True
    shell._session_db = object()
    shell._resumed = shell.verbose = shell.ignore_rules = False
    shell._inline_diffs_enabled = shell.streaming_enabled = False
    shell._single_query_mode = True
    shell.config = {"agent": {"require_execution_receipt": True}}
    return shell, calls


def test_receipt_worker_skips_memory_before_constructor_without_skipping_rules(cli_construction, monkeypatch):
    shell, calls = cli_construction
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    shell.system_prompt = "PROJECT / SOUL RULES"
    assert shell._init_agent() is True
    assert calls[0]["skip_memory"] is True
    assert calls[0]["skip_context_files"] is False
    assert calls[0]["ephemeral_system_prompt"] == "PROJECT / SOUL RULES"


@pytest.mark.parametrize("requirement", ["true", 1, None, [], {}])
def test_malformed_receipt_requirement_blocks_before_constructor(cli_construction, monkeypatch, requirement):
    shell, calls = cli_construction
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    shell.config["agent"]["require_execution_receipt"] = requirement
    assert shell._init_agent() is False
    assert calls == []


@pytest.mark.parametrize("mode", ["ordinary", "receipt-disabled", "no-task", "descendant", "non-owner", "ignore-rules"])
def test_memory_isolation_respects_ownership_and_existing_rules_flag(cli_construction, monkeypatch, mode):
    from contextlib import nullcontext
    from agent.delegation_context import non_dispatcher_owned_context
    shell, calls = cli_construction
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    if mode == "ordinary":
        shell.config = {}
    elif mode == "receipt-disabled":
        shell.config["agent"]["require_execution_receipt"] = False
    elif mode == "no-task":
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    elif mode == "descendant":
        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "fixture-board-root")
    elif mode == "ignore-rules":
        shell.ignore_rules = True
    context = non_dispatcher_owned_context() if mode == "non-owner" else nullcontext()
    with context:
        assert shell._init_agent() is True
    assert calls[0]["skip_memory"] is (mode == "ignore-rules")
    assert calls[0]["skip_context_files"] is (mode == "ignore-rules")


@pytest.fixture
def real_memory_phases(tmp_path, monkeypatch):
    """No-model constructor carrier double; memory and prompt phases are real SDK code."""
    from pathlib import Path
    from types import SimpleNamespace
    from agent.agent_init import _init_memory
    from agent.system_prompt import _memory_parts
    from agent.tool_permissions import ToolPermissionPolicy, apply_agent_tool_policy

    home = tmp_path / "fixture-home"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("UNSUPPLIED_FIXTURE_MEMORY")
    (home / "memories" / "USER.md").write_text("UNSUPPLIED_FIXTURE_USER")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reads = []
    original_open = Path.open

    def observed_open(path, mode="r", *args, **kwargs):
        if path.name in ("MEMORY.md", "USER.md") and "r" in mode:
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    provider_calls = []
    # Explicit external-manager double: proves reuse is skipped, never contacts a provider.
    manager = SimpleNamespace(
        get_all_tool_schemas=lambda: provider_calls.append("schemas") or [],
        build_system_prompt=lambda: provider_calls.append("prompt") or "FIXTURE_EXTERNAL_MEMORY",
    )

    def exercise(kwargs, config):
        carrier = SimpleNamespace(
            enabled_toolsets=kwargs["enabled_toolsets"],
            disabled_toolsets=kwargs["disabled_toolsets"],
            tools=[{"function": {"name": name}} for name in ("kanban_complete", "memory", "terminal")],
            _tool_policy=ToolPermissionPolicy.from_config(config),
        )
        apply_agent_tool_policy(carrier)
        _init_memory(carrier, config, skip_memory=kwargs["skip_memory"],
                     platform=kwargs["platform"], memory_manager=manager)
        apply_agent_tool_policy(carrier)
        return carrier, "\n".join(_memory_parts(carrier))

    return exercise, home, reads, provider_calls, manager


@pytest.mark.parametrize("ignore_rules,disabled", [
    (False, None), (False, ["terminal"]), (False, ("terminal",)),
    (False, {"terminal"}), (False, ["memory", "terminal"]), (True, ["terminal"]),
])
def test_owned_receipt_worker_cannot_load_enabled_memory(
    cli_construction, real_memory_phases, monkeypatch, ignore_rules, disabled,
):
    from copy import deepcopy
    shell, calls = cli_construction
    exercise, _, reads, provider_calls, _ = real_memory_phases
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    shell.ignore_rules = ignore_rules
    shell.enabled_toolsets = ["kanban", "memory"]
    shell.disabled_toolsets = disabled
    shell.config.update(memory={"memory_enabled": True, "user_profile_enabled": True})
    shell.config["agent"]["allowed_tools"] = ["kanban_complete"]
    shell.system_prompt = "PROJECT / SOUL RULES"
    before_config, before_disabled = deepcopy(shell.config), deepcopy(disabled)

    assert shell._init_agent() is True
    carrier, memory_prompt = exercise(calls[0], shell.config)
    assert "UNSUPPLIED_FIXTURE_MEMORY" not in memory_prompt
    assert "UNSUPPLIED_FIXTURE_USER" not in memory_prompt
    assert carrier._memory_store is None
    assert carrier._memory_manager is None
    assert reads == []
    assert provider_calls == []
    assert carrier.valid_tool_names == {"kanban_complete"}
    assert calls[0]["enabled_toolsets"] is shell.enabled_toolsets
    assert set(calls[0]["disabled_toolsets"]) == set(disabled or []) | {"memory"}
    assert list(calls[0]["disabled_toolsets"]).count("memory") == 1
    assert shell.disabled_toolsets is disabled
    assert shell.disabled_toolsets == before_disabled
    assert shell.config == before_config
    assert calls[0]["skip_memory"] is True
    assert calls[0]["skip_context_files"] is ignore_rules
    assert calls[0]["ephemeral_system_prompt"] == "PROJECT / SOUL RULES"


@pytest.mark.parametrize("mode", [
    "ordinary", "receipt-disabled", "no-task", "descendant", "non-owner", "unowned-ignore-rules",
])
def test_non_owned_memory_tool_semantics_remain_unchanged(
    cli_construction, real_memory_phases, monkeypatch, mode,
):
    from contextlib import nullcontext
    from copy import deepcopy
    from agent.delegation_context import non_dispatcher_owned_context
    shell, calls = cli_construction
    exercise, home, reads, provider_calls, manager = real_memory_phases
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    shell.enabled_toolsets = ["kanban", "memory"]
    shell.disabled_toolsets = ["terminal"]
    shell.config.update(memory={"memory_enabled": True, "user_profile_enabled": True})
    if mode == "ordinary":
        shell.config["agent"].pop("require_execution_receipt")
    elif mode == "receipt-disabled":
        shell.config["agent"]["require_execution_receipt"] = False
    elif mode in ("no-task", "unowned-ignore-rules"):
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        shell.ignore_rules = mode == "unowned-ignore-rules"
    elif mode == "descendant":
        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "fixture-board-root")
    before = deepcopy(shell.config)
    context = non_dispatcher_owned_context() if mode == "non-owner" else nullcontext()
    with context:
        assert shell._init_agent() is True
    carrier, prompt = exercise(calls[0], shell.config)
    assert "UNSUPPLIED_FIXTURE_MEMORY" in prompt
    assert "UNSUPPLIED_FIXTURE_USER" in prompt
    assert set(reads) == {home / "memories" / name for name in ("MEMORY.md", "USER.md")}
    assert calls[0]["enabled_toolsets"] is shell.enabled_toolsets
    assert calls[0]["disabled_toolsets"] is shell.disabled_toolsets
    assert calls[0]["skip_memory"] is shell.ignore_rules
    assert shell.config == before
    if shell.ignore_rules:
        assert carrier._memory_manager is None
        assert provider_calls == []
    else:
        assert carrier._memory_manager is manager
        assert provider_calls == ["schemas", "prompt"]
        assert "FIXTURE_EXTERNAL_MEMORY" in prompt


@pytest.mark.parametrize("disabled", [None, ["memory"]])
def test_background_skip_memory_keeps_explicit_builtin_tool_semantics(real_memory_phases, disabled):
    exercise, _, reads, provider_calls, _ = real_memory_phases
    carrier, prompt = exercise({
        "enabled_toolsets": ["memory"], "disabled_toolsets": disabled,
        "skip_memory": True, "platform": "cron",
    }, {"memory": {"memory_enabled": True, "user_profile_enabled": True}})
    assert ("UNSUPPLIED_FIXTURE_MEMORY" in prompt) is (disabled is None)
    assert ("UNSUPPLIED_FIXTURE_USER" in prompt) is (disabled is None)
    assert bool(reads) is (disabled is None)
    assert carrier._memory_manager is None
    assert provider_calls == []


@pytest.mark.parametrize("requirement", ["true", 1, None, [], {}])
def test_ignore_rules_cannot_bypass_receipt_requirement_validation(cli_construction, monkeypatch, requirement):
    shell, calls = cli_construction
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    shell.ignore_rules = True
    shell.config["agent"]["require_execution_receipt"] = requirement
    assert shell._init_agent() is False
    assert calls == []


def test_owned_memory_isolation_preserves_real_project_and_soul_prompt_parts(
    cli_construction, real_memory_phases, tmp_path, monkeypatch,
):
    from types import SimpleNamespace
    from agent.system_prompt import _context_files_part, _identity_parts
    shell, calls = cli_construction
    exercise, home, reads, provider_calls, _ = real_memory_phases
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("FIXTURE_PROJECT_RULES")
    (home / "SOUL.md").write_text("FIXTURE_SOUL_RULES")
    monkeypatch.chdir(project)
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    shell.enabled_toolsets = ["kanban", "memory"]
    assert shell._init_agent() is True
    carrier, memory_prompt = exercise(calls[0], shell.config)
    carrier.skip_context_files = calls[0]["skip_context_files"]
    carrier.load_soul_identity = False
    carrier.platform = calls[0]["platform"]
    carrier._session_db = SimpleNamespace(db_path=str(home / "state.db"))
    identity, loaded = _identity_parts(carrier, None)
    context = _context_files_part(carrier, None, loaded)
    prompt = "\n".join([*identity, *context, memory_prompt])
    assert loaded is True
    assert "FIXTURE_SOUL_RULES" in prompt
    assert "FIXTURE_PROJECT_RULES" in prompt
    assert "UNSUPPLIED_FIXTURE_MEMORY" not in prompt
    assert "UNSUPPLIED_FIXTURE_USER" not in prompt
    assert reads == provider_calls == []
