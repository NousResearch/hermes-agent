import importlib.util
import json
import logging
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PATH = (
    REPO_ROOT
    / "profile-distributions"
    / "elixir-analytics"
    / "profile_plugins"
    / "elixir-analytics-runner"
    / "__init__.py"
)


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "test_elixir_analytics_runner_plugin",
        PLUGIN_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Completed:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _Ctx:
    def __init__(self):
        self.tools = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def test_tool_schema_tells_model_to_use_answer_question_first():
    module = _load_plugin_module()
    ctx = _Ctx()

    module.register(ctx)

    schema = ctx.tools[0]["schema"]
    assert "Use mode='answer_question' first" in schema["description"]
    assert "payload.slackText" in schema["description"]
    assert "answer_question" in schema["parameters"]["properties"]["mode"]["enum"]
    assert "source_change_plan" in schema["parameters"]["properties"]["mode"]["enum"]
    assert "self_improvement_check" in schema["parameters"]["properties"]["mode"]["enum"]
    assert "self_improvement_plan" in schema["parameters"]["properties"]["mode"]["enum"]
    assert "query_log" in schema["parameters"]["properties"]
    assert (
        "use answer_question first"
        in schema["parameters"]["properties"]["mode"]["description"]
    )


def test_plan_mode_invokes_question_planner_on_stdin(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "route": "saved_topic"}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {"mode": "plan", "question": "show GTV last 30 days by week"}
    )

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/plan-analytics-question.ts",
    ]
    assert kwargs["input"] == "show GTV last 30 days by week"


def test_ad_hoc_mode_sends_request_json_on_stdin_and_defaults_base_url(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "rows": [{"user": "Ada"}]}))

    monkeypatch.delenv("ANALYTICS_BASE_URL", raising=False)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {
            "mode": "supabase_ad_hoc",
            "request": {"question": "which users spent on Swiggy this week?"},
            "max_rows": 25,
        }
    )

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/run-ad-hoc-query.ts",
        "--max-rows",
        "25",
    ]
    assert json.loads(kwargs["input"]) == {
        "question": "which users spent on Swiggy this week?"
    }
    assert kwargs["env"]["ANALYTICS_BASE_URL"] == "https://analytics.joinelixir.club"


def test_answer_question_mode_invokes_shortcut_runner_on_stdin(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "route": "supabase_ad_hoc"}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {
            "mode": "answer_question",
            "question": "which users spent on Swiggy this week?",
            "max_rows": 25,
        }
    )

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/run-analytics-question.ts",
        "--max-rows",
        "25",
    ]
    assert kwargs["input"] == "which users spent on Swiggy this week?"


def test_runner_logs_safe_route_summary_without_rows_or_sql(monkeypatch, caplog):
    module = _load_plugin_module()
    row = {"email": "ada@example.com", "gross_spend_inr": 420}
    payload = {
        "ok": True,
        "route": "supabase_ad_hoc",
        "shortcut": "swiggy_users_this_week",
        "payload": {
            "ok": True,
            "rowCount": 1,
            "truncated": False,
            "dryRun": False,
            "dashboardUrlPath": "/query?payload=compact",
            "metadata": {
                "resultType": "users",
                "sql": "select * from transactions",
            },
            "rows": [row],
        },
    }

    def fake_run(command, **kwargs):
        return _Completed(json.dumps(payload))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with caplog.at_level(logging.INFO, logger="hermes.elixir_analytics_runner"):
        result = module.run_elixir_analytics_runner(
            {
                "mode": "answer_question",
                "question": "which users spent on Swiggy this week?",
            }
        )

    assert result["ok"] is True
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "mode=answer_question" in log_text
    assert "route=supabase_ad_hoc" in log_text
    assert "shortcut=swiggy_users_this_week" in log_text
    assert "rowCount=1" in log_text
    assert "resultType=users" in log_text
    assert "dashboard=True" in log_text
    assert "ada@example.com" not in log_text
    assert "select * from transactions" not in log_text
    assert "which users spent on Swiggy this week" not in log_text


def test_answer_question_mode_defaults_to_compact_row_cap(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "route": "posthog_ad_hoc"}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {
            "mode": "answer_question",
            "question": "how many app active users this week?",
        }
    )

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/run-analytics-question.ts",
        "--max-rows",
        "100",
    ]
    assert kwargs["input"] == "how many app active users this week?"


def test_question_without_mode_defaults_to_answer_question(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "route": "saved_topic"}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {"question": "show GTV last 30 days by week", "dry_run": True}
    )

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert result["mode"] == "answer_question"
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/run-analytics-question.ts",
        "--max-rows",
        "100",
        "--dry-run",
    ]
    assert kwargs["input"] == "show GTV last 30 days by week"


def test_source_change_plan_mode_invokes_source_change_planner_on_stdin(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "kind": "metric_definition"}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {
            "mode": "source_change_plan",
            "request": "GTV definition is wrong, wallet loads should be included",
        }
    )

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/plan-source-change.ts",
    ]
    assert kwargs["input"] == "GTV definition is wrong, wallet loads should be included"


def test_self_improvement_plan_mode_invokes_self_improvement_planner_with_query_log(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "entriesReviewed": 5}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner({"mode": "self_improvement_plan"})

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/plan-self-improvement.ts",
        "--query-log",
        "QUERY_LOG.md",
    ]
    assert kwargs["input"] is None


def test_self_improvement_check_mode_invokes_cadence_checker_with_query_log(monkeypatch):
    module = _load_plugin_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _Completed(json.dumps({"ok": True, "status": "not_due"}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner({"mode": "self_improvement_check"})

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert command == [
        "node",
        "--import",
        "tsx",
        "scripts/check-self-improvement-cadence.ts",
        "--query-log",
        "QUERY_LOG.md",
    ]
    assert kwargs["input"] is None


def test_runner_failure_returns_structured_error(monkeypatch):
    module = _load_plugin_module()

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=3, output="partial")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_elixir_analytics_runner(
        {"mode": "plan", "question": "show GTV", "timeout_seconds": 3}
    )

    assert result["ok"] is False
    assert result["errorType"] == "timeout"
    assert result["timeoutSeconds"] == 3
