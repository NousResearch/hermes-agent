import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def _load_plugin():
    plugin_path = Path(__file__).resolve().parents[2] / "plugins" / "harness_engineering" / "__init__.py"
    spec = importlib.util.spec_from_file_location("harness_engineering_under_test", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_intake_helper():
    script = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "software-development"
        / "harness-agenting-engineering"
        / "scripts"
        / "harness_intake.py"
    )
    spec = importlib.util.spec_from_file_location("harness_intake_under_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_allows_plain_explanation(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_configured_preflight_mode", lambda: "advisory")

    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="解释一下什么是 MCP"))

    assert result == {"action": "allow"}


def test_preflight_rewrites_engineering_task(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_configured_preflight_mode", lambda: "advisory")

    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="请修复这个 WebUI bug 并加测试"))

    assert result["action"] == "rewrite"
    assert "Harness / Agenting Engineering preflight" in result["text"]
    assert result["text"].endswith("请修复这个 WebUI bug 并加测试")


def test_preflight_strict_uses_intake_notice(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_configured_preflight_mode", lambda: "strict")

    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="重构 gateway 认证流程"))

    assert result["action"] == "rewrite"
    assert "intake required" in result["text"]


def test_preflight_advisory_requires_intake_for_high_risk(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_configured_preflight_mode", lambda: "advisory")

    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="请修改 OAuth token 存储并 force-push"))

    assert result["action"] == "rewrite"
    assert "intake required" in result["text"]


def test_task_classifier_routes_research_without_harness():
    plugin = _load_plugin()

    result = plugin.classify_task("调研并对比三个 MCP server 方案")

    assert result.task_type == "research"
    assert result.harness_required is False
    assert result.route == "research_then_report"


def test_task_classifier_routes_multi_agent_as_intake_required():
    plugin = _load_plugin()

    result = plugin.classify_task("把这个功能拆成 Kanban 多代理并行执行")

    assert result.task_type == "multi_agent_project"
    assert result.harness_required is True
    assert result.route == "intake_required"


def test_task_classifier_routes_small_code_change():
    plugin = _load_plugin()

    result = plugin.classify_task("请做一个 small fix 并加测试")

    assert result.task_type == "small_code_change"
    assert result.harness_required is False
    assert result.route == "bounded_engineering"


def test_classify_cli_emits_json(capsys):
    plugin = _load_plugin()

    try:
        plugin._handle_harness_cli(SimpleNamespace(harness_action="classify", text="重构认证流程", format="json"))
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert '"task_type": "high_risk_change"' in out
    assert '"route": "intake_required"' in out


def test_preflight_uses_config_mode(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_configured_preflight_mode", lambda: "off")

    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="请修复这个 bug"))

    assert result == {"action": "allow"}


def test_preflight_ignores_legacy_env_override(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setenv("HERMES_HARNESS_PREFLIGHT", "strict")
    monkeypatch.setattr(plugin, "_configured_preflight_mode", lambda: "off")

    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="请修改 OAuth token 存储"))

    assert result == {"action": "allow"}


def test_preflight_loads_mode_from_config_yaml(monkeypatch, tmp_path):
    hermes_home = tmp_path / "profile-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "harness_engineering:\n  preflight_mode: strict\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    plugin = _load_plugin()

    assert plugin._configured_preflight_mode() == "strict"
    result = plugin._handle_pre_gateway_dispatch(SimpleNamespace(text="请修复这个 WebUI bug"))

    assert result["action"] == "rewrite"
    assert "intake required" in result["text"]


def test_helper_command_prefers_bundled_script(monkeypatch, tmp_path):
    plugin = _load_plugin()
    bundled = tmp_path / "repo" / "skills" / "software-development" / "harness-agenting-engineering" / "scripts" / "harness_intake.py"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    user_helper = tmp_path / "home" / ".hermes" / "bin" / "hermes-harness"
    user_helper.parent.mkdir(parents=True)
    user_helper.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("PYTHON", "python3.12")
    monkeypatch.setattr(plugin, "_bundled_helper_path", lambda: bundled)
    monkeypatch.setattr(plugin, "_user_helper_path", lambda: user_helper)

    assert plugin._helper_command() == ["python3.12", str(bundled)]


def test_helper_command_falls_back_to_profile_helper(monkeypatch, tmp_path):
    plugin = _load_plugin()
    bundled = tmp_path / "missing" / "harness_intake.py"
    profile_helper = tmp_path / "profiles" / "coder" / "bin" / "hermes-harness"
    profile_helper.parent.mkdir(parents=True)
    profile_helper.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(plugin, "_bundled_helper_path", lambda: bundled)
    monkeypatch.setattr(plugin, "_user_helper_path", lambda: profile_helper)

    assert plugin._helper_command() == [str(profile_helper)]


def test_user_helper_path_uses_active_profile_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "coder"))
    plugin = _load_plugin()

    assert plugin._user_helper_path() == tmp_path / "profiles" / "coder" / "bin" / "hermes-harness"


def test_harness_new_accepts_documented_output_flag(monkeypatch, tmp_path):
    plugin = _load_plugin()
    calls = []

    def fake_run(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(plugin, "_run_helper", fake_run)
    try:
        plugin._handle_harness_cli(
            SimpleNamespace(
                harness_action="new",
                title="Task",
                workspace="/repo",
                mode="Implement changes",
                output=str(tmp_path / "intake.md"),
                print_path=False,
            )
        )
    except SystemExit as exc:
        assert exc.code == 0

    assert calls == [[
        "new",
        "--title",
        "Task",
        "--workspace",
        "/repo",
        "--mode",
        "Implement changes",
        "--output",
        str(tmp_path / "intake.md"),
    ]]


def test_harness_new_parser_rejects_legacy_out_flag(capsys):
    plugin = _load_plugin()
    parser = argparse.ArgumentParser(prog="hermes harness")
    plugin._setup_harness_cli(parser)

    try:
        parser.parse_args(["new", "--out", "/tmp/intake.md"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("legacy --out flag should not parse")

    assert "--out" in capsys.readouterr().err


def test_helper_default_output_dir_honors_hermes_home(monkeypatch, tmp_path):
    helper = _load_intake_helper()
    profile_home = tmp_path / "profiles" / "writer"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert helper._default_out_dir() == profile_home / "harness" / "intake"


def test_register_exposes_harness_command_and_gateway_hook():
    plugin = _load_plugin()
    ctx = MagicMock()

    plugin.register(ctx)

    cli_call = ctx.register_cli_command.call_args
    assert cli_call.args[0] == "harness"
    ctx.register_command.assert_called_once()
    assert ctx.register_command.call_args.args[0] == "intake"
    ctx.register_hook.assert_called_once()
    assert ctx.register_hook.call_args.args[0] == "pre_gateway_dispatch"


def _filled_intake(tmp_path):
    intake = tmp_path / "intake.md"
    intake.write_text(
        """# Harness / Agenting Engineering Task Intake

- Task title: Close lifecycle loop
- Workspace / repo: /repo/ws
- Problem / request: Bridge intake to Kanban and evidence.

## 0. Submission mode
- [x] Implement changes

## 2. Acceptance criteria
1. Kanban card can be created from intake.
2. Evidence report can be generated.

## 3. Risk surface
- [x] Plugins / hooks / tools

## 6. Verification evidence required before done
Run plugin unit tests.
""",
        encoding="utf-8",
    )
    return intake


def test_harness_kanban_create_dry_run_from_intake(tmp_path, capsys):
    plugin = _load_plugin()
    intake = _filled_intake(tmp_path)

    code = plugin._handle_harness_kanban_cli(
        SimpleNamespace(
            harness_kanban_action="create",
            file=str(intake),
            assignee="architect",
            workspace="",
            triage=True,
            dry_run=True,
            json=True,
        )
    )

    assert code == 0
    out = capsys.readouterr().out
    assert '"hermes"' in out
    assert '"kanban"' in out
    assert '"create"' in out
    assert '"Close lifecycle loop"' in out
    assert '"--triage"' in out
    assert "harness-intake:" in out


def test_harness_kanban_decompose_defaults_to_dry_run(capsys):
    plugin = _load_plugin()

    code = plugin._handle_harness_kanban_cli(
        SimpleNamespace(
            harness_kanban_action="decompose",
            task_id="t123",
            worker="loop-worker",
            reviewer="loop-reviewer",
            workspace="worktree:/tmp/wt",
            branch="wt/t123",
            execute=False,
            json=False,
        )
    )

    assert code == 0
    out = capsys.readouterr().out
    assert '"dry_run": true' in out
    assert '"--parent"' in out
    assert '"t123"' in out
    assert '"--initial-status"' in out
    assert '"blocked"' in out


def test_harness_evidence_writes_report(tmp_path, monkeypatch):
    plugin = _load_plugin()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    out = tmp_path / "evidence.md"

    def fake_run(command, cwd=None):
        if command[:2] == ["git", "rev-parse"]:
            return 0, "abc1234", ""
        if command[:2] == ["git", "status"]:
            return 0, "", ""
        if command[:2] == ["git", "diff"]:
            return 0, " plugins/harness_engineering/__init__.py | 10 +", ""
        return 0, "{}", ""

    monkeypatch.setattr(plugin, "_run_capture", fake_run)

    code = plugin._handle_harness_evidence_cli(
        SimpleNamespace(task_id="t123", workspace=str(workspace), output=str(out))
    )

    assert code == 0
    content = out.read_text(encoding="utf-8")
    assert "## Harness Evidence" in content
    assert "abc1234" in content
    assert "Required completion notes" in content


def test_harness_gc_template_documents_non_mutating_boundary(tmp_path):
    plugin = _load_plugin()
    out = tmp_path / "gc.md"

    code = plugin._handle_harness_gc_template_cli(
        SimpleNamespace(output=str(out), board="hermes-engineering-loop")
    )

    assert code == 0
    content = out.read_text(encoding="utf-8")
    assert "Weekly Harness GC" in content
    assert "Do not auto-repair" in content
    assert "hermes-engineering-loop" in content


def test_harness_migration_pack_generates_cross_agent_files(tmp_path, capsys):
    plugin = _load_plugin()

    code = plugin._handle_harness_migration_pack_cli(
        SimpleNamespace(output_dir=str(tmp_path), force=False, json=False)
    )

    assert code == 0
    out = capsys.readouterr().out
    expected = [
        "CODEX.md",
        "CLAUDE.md",
        "OPENCODE.md",
        ".cursor/rules/harness.mdc",
        ".windsurfrules",
        "prompts/task-intake.md",
    ]
    for rel in expected:
        target = tmp_path / rel
        assert target.exists()
        assert str(target.resolve()) in out
    assert "hermes harness classify" in (tmp_path / "CODEX.md").read_text(encoding="utf-8")
    assert "docs/CONTRACTS.md" in (tmp_path / ".cursor/rules/harness.mdc").read_text(encoding="utf-8")


def test_harness_migration_pack_skips_existing_without_force(tmp_path, capsys):
    plugin = _load_plugin()
    target = tmp_path / "CODEX.md"
    target.write_text("custom", encoding="utf-8")

    code = plugin._handle_harness_migration_pack_cli(
        SimpleNamespace(output_dir=str(tmp_path), force=False, json=False)
    )

    assert code == 0
    assert target.read_text(encoding="utf-8") == "custom"
    assert "SKIP existing" in capsys.readouterr().out
