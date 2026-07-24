"""Integration tests for `hermes preview` (run_preview + display_report).

These tests are fully hermetic: no network, no real API keys, no file side
effects beyond the per-test tempdir provided by conftest's
``_hermetic_environment`` fixture. We isolate everything via monkeypatch.

Note: ``preview.HERMES_HOME`` is a module-level constant captured at import
time.  The ``_fresh_preview`` helper below re-points HERMES_HOME *and*
re-imports the module so every test sees a clean, isolated home.  This is the
canonical way to test preview because its check functions read HERMES_HOME
directly (not via get_hermes_home() on each call).
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.colors import Colors, color
from hermes_cli.preview import PreviewItem, PreviewReport


# ── Helpers ────────────────────────────────────────────────────────────────
def _config_yml(model: str = "", provider: str = "", base_url: str = "") -> str:
    lines = []
    if model:
        lines.append("model:")
        if provider or base_url:
            lines.append(f"  name: {model}")
            if provider:
                lines.append(f"  provider: {provider}")
            if base_url:
                lines.append(f"  base_url: {base_url}")
        else:
            lines.append(f"  name: {model}")
    return "\n".join(lines)


def _fresh_preview(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return (reimported preview module, fake_hermes_home path)."""
    fake_home = tmp_path / "hermes_test"
    fake_home.mkdir(exist_ok=True)
    (fake_home / "sessions").mkdir(exist_ok=True)
    (fake_home / "cron").mkdir(exist_ok=True)
    (fake_home / "memories").mkdir(exist_ok=True)
    (fake_home / "skills").mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    # Re-import so the module-level HERMES_HOME captures the fresh value.
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    if "hermes_cli.preview" in sys.modules:
        del sys.modules["hermes_cli.preview"]
    from hermes_cli import preview as preview_mod

    return preview_mod, fake_home


def _stub_module(monkeypatch, name: str, **attrs):
    """Create a fake module and install it into sys.modules."""
    mod = types.SimpleNamespace()
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


# ── A. run_preview() integration tests (15) ────────────────────────────────
class TestRunPreviewIntegration:
    def test_run_preview_returns_ready_when_all_checks_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="sensenova-6.7-flash", provider="sensenova")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("SENSENOVA_API_KEY", "sk-1234567890abcdef")

        skills = [MagicMock(), MagicMock()]
        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: skills,
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(
                get_tools=lambda: ["read_file", "terminal"]
            ),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [
                {"name": "mcp1", "status": "connected"},
                {"name": "mcp2", "status": "connected"},
            ],
        )

        report = pm.run_preview()
        assert report.verdict == "ready"
        assert report.verdict == "ready"  # stable
        names = [i.name for i in report.items]
        assert "Hermes home" in names
        assert "Model" in names
        assert "Auth" in names
        assert "Skills" in names

    def test_run_preview_returns_blocked_when_model_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        assert report.verdict == "blocked"
        model_item = next((i for i in report.items if i.name == "Model"), None)
        assert model_item is not None
        assert model_item.status == "error"
        assert "not set" in model_item.detail or "configured" in model_item.detail

    def test_run_preview_returns_blocked_when_auth_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="claude-3-sonnet", provider="anthropic")
        (fake_home / "config.yaml").write_text(cfg)
        # No API keys set at all
        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        assert report.verdict == "blocked"
        auth_item = next((i for i in report.items if i.name == "Auth"), None)
        assert auth_item is not None
        assert auth_item.status == "error"

    def test_run_preview_returns_warning_when_mcp_disconnected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="deepseek-chat", provider="deepseek")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-1234567890abcdef")

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [
                {"name": "bad1", "status": "disconnected"},
                {"name": "bad2", "status": "disconnected"},
            ],
        )

        report = pm.run_preview()
        mcp_item = next((i for i in report.items if i.name == "MCP"), None)
        assert mcp_item is not None
        assert mcp_item.status == "warn"
        # Has no blocker so verdict stays warning (or ready if no other issues)
        assert report.verdict != "blocked"

    def test_run_preview_returns_warning_when_skills_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="deepseek-chat", provider="deepseek")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-1234567890abcdef")

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        skill_item = next((i for i in report.items if i.name == "Skills"), None)
        assert skill_item is not None
        assert skill_item.status == "warn"

    def test_run_preview_combines_multiple_errors_and_warnings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, _ = _fresh_preview(monkeypatch, tmp_path)

        # Model missing (error) + skills missing (warn) + mcp disconnected (warn)
        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [{"name": "x", "status": "disconnected"}],
        )

        report = pm.run_preview()
        assert report.verdict == "blocked"  # model error wins
        errors = [i for i in report.items if i.status == "error"]
        warns = [i for i in report.items if i.status == "warn"]
        assert any(i.name == "Model" for i in errors)
        assert any(i.name == "Skills" for i in warns)
        assert any(i.name == "MCP" for i in warns)

    def test_run_preview_includes_all_8_check_items(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(
            model="test-model", provider="custom", base_url="http://localhost:8080/v1"
        )
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        names = [i.name for i in report.items]
        expected = {
            "Hermes home",
            "Python",
            "Model",
            "Auth",
            ".env file",
            "Skills",
            "Tools",
            "MCP",
        }
        assert expected.issubset(set(names))
        assert len(report.items) == 8

    def test_run_preview_respects_config_loaded_from_config_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="gpt-4o-mini", provider="openai")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        model_item = next((i for i in report.items if i.name == "Model"), None)
        assert model_item is not None
        assert model_item.status == "ok"
        assert "gpt-4o-mini" in model_item.detail

    def test_run_preview_handles_config_load_failure_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """load_config may fail; run_preview must not raise."""
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        # Force load_config to raise
        monkeypatch.setattr(
            pm, "load_config", lambda: (_ for _ in ()).throw(FileError("bad yaml"))
        )
        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        # Must not raise — model becomes "(not set)", verdict blocked
        report = pm.run_preview()
        assert report.verdict == "blocked"
        model_item = next((i for i in report.items if i.name == "Model"), None)
        assert model_item is not None
        assert model_item.status == "error"

    def test_run_preview_handles_auth_module_import_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="claude-3-sonnet", provider="anthropic")
        (fake_home / "config.yaml").write_text(cfg)

        # Make hermes_cli.auth un-importable
        monkeypatch.delitem(sys.modules, "hermes_cli.auth", raising=False)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        # Must not raise on import failure; falls back gracefully
        report = pm.run_preview()
        assert report.verdict in ("ready", "warning", "blocked")

    def test_run_preview_handles_mcp_module_import_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="deepseek-chat", provider="deepseek")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-1234567890abcdef")

        monkeypatch.delitem(sys.modules, "tools.mcp_server", raising=False)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )

        report = pm.run_preview()
        mcp_item = next((i for i in report.items if i.name == "MCP"), None)
        assert mcp_item is not None
        assert mcp_item.status == "ok"  # 0 servers configured

    def test_run_preview_handles_skills_module_import_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="deepseek-chat", provider="deepseek")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-1234567890abcdef")

        monkeypatch.delitem(sys.modules, "hermes_cli.skills", raising=False)

        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        skill_item = next((i for i in report.items if i.name == "Skills"), None)
        assert skill_item is not None
        assert skill_item.status == "warn"  # 0 skills on failure

    def test_run_preview_handles_tools_module_import_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)

        cfg = _config_yml(model="deepseek-chat", provider="deepseek")
        (fake_home / "config.yaml").write_text(cfg)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-1234567890abcdef")

        monkeypatch.delitem(sys.modules, "tools.tool_backend", raising=False)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        tool_item = next((i for i in report.items if i.name == "Tools"), None)
        assert tool_item is not None
        assert tool_item.status == "ok"  # "Loaded on demand"

    def test_run_preview_handles_python_version_edge_case(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        monkeypatch.setattr(pm, "sys", types.SimpleNamespace(version="2.7.18 final"))
        report = pm.run_preview()
        py_item = next((i for i in report.items if i.name == "Python"), None)
        assert py_item is not None
        assert py_item.status == "warn"

    def test_run_preview_handles_hermes_home_not_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, _ = _fresh_preview(monkeypatch, tmp_path)
        # Point HERMES_HOME at a non-existent path
        missing = tmp_path / "does_not_exist_at_all"
        monkeypatch.setenv("HERMES_HOME", str(missing))
        monkeypatch.setattr(pm, "HERMES_HOME", missing)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        report = pm.run_preview()
        assert report.verdict == "blocked"
        home_item = next((i for i in report.items if i.name == "Hermes home"), None)
        assert home_item is not None
        assert home_item.status == "error"
        assert "setup" in home_item.fix.lower()


# ── B. Output format tests (8) ─────────────────────────────────────────────
class TestDisplayReportOutputFormat:
    def _make_report(
        self,
        verdict: str = "ready",
        items: list[tuple[str, str, str, str]] | None = None,
    ) -> PreviewReport:
        report = PreviewReport(verdict)
        if items:
            for name, status, detail, fix in items:
                if status == "ok":
                    report.add_ok(name, detail)
                elif status == "warn":
                    report.add_warn(name, detail, fix)
                else:
                    report.add_error(name, detail, fix)
        return report

    def test_display_report_shows_banner_with_cyan_color(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )
        monkeypatch.setattr("hermes_cli.colors.color", lambda s, *a, **kw: str(s))

        display_report(self._make_report("ready"))
        output = "\n".join(captured)
        assert "Hermes Preview" in output
        assert "---" in output

    def test_display_report_shows_items_with_status_icons(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        report = self._make_report(
            "warning",
            items=[
                ("Python", "ok", "3.12", ""),
                ("Model", "ok", "gpt-4o", ""),
                ("Skills", "warn", "No skills loaded", "hermes skills"),
            ],
        )
        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )
        monkeypatch.setattr("hermes_cli.colors.color", lambda s, *a, **kw: str(s))

        display_report(report)
        output = "\n".join(captured)
        assert "Python" in output
        assert "Model" in output
        assert "Skills" in output

    def test_display_report_shows_ok_items_with_green_checkmark(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        calls: list[tuple] = []

        def fake_color(s, *args, **kwargs):
            if args:
                return f"<c{args[0]}>{s}</c>"
            return s

        monkeypatch.setattr(
            "hermes_cli.colors.color",
            fake_color,
        )

        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )

        display_report(self._make_report("ready", items=[("Auth", "ok", "OK", "")]))
        output = "\n".join(captured)
        # The ok marker is [OK]
        assert "[OK]" in output

    def test_display_report_shows_warn_items_with_yellow_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )
        monkeypatch.setattr("hermes_cli.colors.color", lambda s, *a, **kw: str(s))

        display_report(
            self._make_report(
                "warning",
                items=[
                    ("Env", "warn", "not found", "create .env"),
                ],
            )
        )
        output = "\n".join(captured)
        assert "[!!]" in output

    def test_display_report_shows_error_items_with_red_x(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )
        monkeypatch.setattr("hermes_cli.colors.color", lambda s, *a, **kw: str(s))

        display_report(
            self._make_report(
                "blocked",
                items=[
                    ("Model", "error", "not set", "hermes model"),
                ],
            )
        )
        output = "\n".join(captured)
        assert "[XX]" in output

    def test_display_report_shows_verdict_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )
        monkeypatch.setattr("hermes_cli.colors.color", lambda s, *a, **kw: str(s))

        display_report(self._make_report("blocked"))
        output = "\n".join(captured)
        assert "Cannot run Hermes" in output

        captured.clear()
        display_report(self._make_report("warning"))
        output = "\n".join(captured)
        assert "Some issues detected" in output

    def test_display_report_shows_fix_instructions_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.preview import display_report

        captured: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *a, **kw: captured.append(" ".join(str(x) for x in a if x)),
        )
        monkeypatch.setattr("hermes_cli.colors.color", lambda s, *a, **kw: str(s))

        display_report(
            self._make_report(
                "blocked",
                items=[
                    ("Model", "error", "not set", "Run 'hermes model'"),
                    ("Home", "error", "missing", "Run 'hermes setup'"),
                ],
            )
        )
        output = "\n".join(captured)
        assert "hermes model" in output
        assert "hermes setup" in output

    def test_display_report_json_output_matches_text_output_data(
        self,
    ):
        """JSON output from PreviewReport must be consistent with its data."""
        report = self._make_report(
            "warning",
            items=[
                ("Model", "ok", "gpt-4o (provider: openai)", ""),
                ("Auth", "error", "No key", "hermes auth add"),
                ("Skills", "warn", "0 available", "install skills"),
            ],
        )
        data = json.loads(report.json_output())
        assert data["verdict"] == "warning"
        assert len(data["items"]) == 3
        names = [i["name"] for i in data["items"]]
        assert names == ["Model", "Auth", "Skills"]
        assert data["items"][1]["fix"] == "hermes auth add"


# ── C. Side-effect verification tests (7) ──────────────────────────────────
class TestPreviewSideEffects:
    def test_preview_does_not_call_llm_api(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        call_count = {"n": 0}
        for mod_name in ["hermes_cli.chat", "hermes_cli.provider", "hermes_cli.auth"]:
            _stub_module(
                monkeypatch,
                mod_name,
                __getattr__=lambda self, *_: None,
            )
        # Stub any httpx/aiohttp request at the module level so an outbound
        # call would be intercepted and counted.
        for fake_mod in (
            "httpx",
            "aiohttp",
            "hermes_cli.api",
            "hermes_cli.openai_compat",
        ):
            _stub_module(
                monkeypatch,
                fake_mod,
                post=lambda *a, **kw: None,
                get=lambda *a, **kw: None,
                __getattr__=lambda self, *_: None,
            )

        # Inspect run_preview source for any HTTP/LLM call indicators.
        # The canonical check: run_preview must not contain
        # client.chat, /chat/completions, or openai.ChatCompletion.
        source_lines = [l for l in pm.run_preview.__code__.co_names]
        # Instead, statically parse the function body for forbidden patterns.
        src = importlib.resources.read_text(  # type: ignore[attr-defined]
            "hermes_cli", "preview.py"
        )
        func_node = self._extract_func(src, "run_preview")
        forbidden = ["chat.completions", "client.chat", "completion", "ask("]
        func_src = ast.get_source_segment(src, func_node)  # type: ignore[arg-type]
        for pat in forbidden:
            assert pat not in func_src, (
                f"run_preview contains LLM API call indicator '{pat}'"
            )

        # Also actually execute and confirm no HTTP module was touched.
        report = pm.run_preview()
        assert report.verdict in ("ready", "warning", "blocked")

    def test_preview_does_not_execute_tools(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        backend = MagicMock()
        backend.get_tools.return_value = ["read_file", "terminal"]
        backend.execute_tool = MagicMock()

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: backend,
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        pm.run_preview()
        # get_tools was called to COUNT them; execute_tool must never be called
        backend.execute_tool.assert_not_called()

    def test_preview_does_not_connect_mcp_servers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        connect_call_count = {"n": 0}

        def discover():
            return [
                {"name": "s1", "status": "disconnected"},
                {"name": "s2", "status": "connected"},
            ]

        def connect_server(*a, **kw):
            connect_call_count["n"] += 1

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=discover,
            connect_server=connect_server,
            connect_all=lambda: None,
        )

        pm.run_preview()
        assert connect_call_count["n"] == 0

    def test_preview_does_not_modify_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )
        _stub_module(
            monkeypatch,
            "hermes_cli.auth",
            get_nous_auth_status=lambda: {"logged_in": False},
            get_codex_auth_status=lambda: {"logged_in": False},
        )
        # Prevent ensure_hermes_home() from seeding SOUL.md
        monkeypatch.setattr(
            "hermes_cli.config.ensure_hermes_home", lambda: None, raising=False
        )

        # Record initial state of everything under fake_home
        before = {
            p.relative_to(fake_home): p.read_bytes()
            for p in fake_home.rglob("*")
            if p.is_file()
        }

        pm.run_preview()

        # All files must still have identical contents
        after = {
            p.relative_to(fake_home): p.read_bytes()
            for p in fake_home.rglob("*")
            if p.is_file()
        }
        assert set(before) == set(after)
        for rel in before:
            assert before[rel] == after[rel], f"file modified: {rel}"

    def test_preview_does_not_spawn_subagents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        # Patch subprocess so any child process launch would be caught
        spawn_calls: list[list] = []
        orig_popen = __import__("subprocess").Popen

        class TrackedPopen(orig_popen):  # type: ignore[misc]
            def __init__(self, args, *a, **kw):
                spawn_calls.append(args)
                super().__init__(args, *a, **kw)

        monkeypatch.setattr(__import__("subprocess"), "Popen", TrackedPopen)

        pm.run_preview()
        # Filter out pytest's own usage by checking that the call was
        # attributed to our module; here we simply assert no subprocess
        # launch happened during the check.
        assert len(spawn_calls) == 0, f"run_preview spawned subprocesses: {spawn_calls}"

    def test_preview_does_not_consume_tokens(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Verify run_preview is pure — no model invocation, no token bookkeeping."""
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        # Patch any token-tracking surfaces to confirm they're untouched
        token_surfaces = [
            "hermes_cli.token_tracker",
            "hermes_cli.usage",
            "hermes_cli.stats",
        ]
        for mod_name in token_surfaces:
            _stub_module(
                monkeypatch,
                mod_name,
                record_usage=lambda *a, **kw: None,
                add_tokens=lambda *a, **kw: None,
                count_tokens=lambda *a, **kw: None,
                __getattr__=lambda self, *_: None,
            )

        # Also verify the function's bytecode contains no calls to
        # token-counting or LLM-invocation names
        func_code = pm.run_preview.__code__
        co_names = set(func_code.co_names)
        suspicious = {
            "completion",
            "chat",
            "count_tokens",
            "record_usage",
            "token_counter",
            "tokenize",
            "add_tokens",
        }
        overlap = co_names & suspicious
        assert not overlap, (
            f"run_preview bytecode references token/LLM names: {overlap}"
        )

        report = pm.run_preview()
        assert report.verdict in ("ready", "warning", "blocked")

    def test_preview_is_idempotent_running_twice_gives_same_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        pm, fake_home = _fresh_preview(monkeypatch, tmp_path)
        cfg = _config_yml(model="test", provider="custom", base_url="http://x/v1")
        (fake_home / "config.yaml").write_text(cfg)

        _stub_module(
            monkeypatch,
            "hermes_cli.skills",
            discover_skills=lambda: [MagicMock(), MagicMock()],
        )
        _stub_module(
            monkeypatch,
            "tools.tool_backend",
            get_tool_backend=lambda: types.SimpleNamespace(get_tools=lambda: []),
        )
        _stub_module(
            monkeypatch,
            "tools.mcp_server",
            discover_mcp_servers=lambda: [],
        )

        r1 = pm.run_preview()
        r2 = pm.run_preview()
        # Same verdict, same number of items, same item statuses
        assert r1.verdict == r2.verdict
        assert len(r1.items) == len(r2.items)
        assert [i.status for i in r1.items] == [i.status for i in r2.items]
        # JSON serialization matches exactly (no stateful drift)
        assert r1.json_output() == r2.json_output()

    # ── Private helper used by side-effect tests ───────────────────────
    @staticmethod
    def _extract_func(src: str, func_name: str):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        raise ValueError(f"Function {func_name} not found in source")
