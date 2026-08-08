"""Seam tests for the cli.py R4 extraction (usage/insights -> CLIUsageMixin).

Verifies the mixin seam contract:
  * identity: ``HermesCLI.<member> is CLIUsageMixin.<member>`` for all 5 members
  * MRO order / no shadowing
  * no module-level back-import of ``cli`` (subprocess, cli blocked)
  * import-order permutation (mixin first, then cli)
  * patch-binding through the seam and the cross-mixin billing contract
  * behavioral pins: /usage dispatch, reset (non-codex + timeout), context
    breakdown shape, insights aggregation, logging-level flip
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import hermes_cli.cli_usage_mixin as cli_usage_mixin
from cli import HermesCLI
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.cli_billing_mixin import CLIBillingMixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_usage_mixin import CLIUsageMixin

MEMBERS = (
    "_handle_usage_command",
    "_usage_reset",
    "_show_context_breakdown",
    "_show_usage",
    "_show_insights",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bare_cli():
    """Bare composite instance (no __init__ side effects), house pattern."""
    return HermesCLI.__new__(HermesCLI)


# ── seam identity / MRO ─────────────────────────────────────────────────────


def test_seam_identity_five_members():
    for name in MEMBERS:
        assert getattr(HermesCLI, name) is getattr(CLIUsageMixin, name), name


def test_seam_mro_order_no_shadowing():
    mro = HermesCLI.__mro__
    assert CLIUsageMixin in mro
    mixins = [m for m in mro if m.__name__.endswith("Mixin")]
    assert mixins == [CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, CLIUsageMixin]


# ── import discipline (subprocess, fresh interpreters) ──────────────────────


def test_seam_no_back_import():
    code = (
        "import sys\n"
        "sys.modules['cli'] = None\n"
        "import hermes_cli.cli_usage_mixin\n"
        "assert sys.modules['cli'] is None\n"
        "print('OK')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_seam_import_order_mixin_first_then_cli():
    code = (
        "import hermes_cli.cli_usage_mixin\n"
        "import cli\n"
        "assert cli.HermesCLI._show_usage is hermes_cli.cli_usage_mixin.CLIUsageMixin._show_usage\n"
        "print('OK')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


# ── patch-binding through the seam ──────────────────────────────────────────


def test_seam_patch_binding_through_mro(monkeypatch):
    cli_obj = _bare_cli()
    fake = MagicMock()
    monkeypatch.setattr(CLIUsageMixin, "_show_usage", fake)
    cli_obj._show_usage()
    fake.assert_called_once_with()


def test_seam_cross_mixin_billing_contract(monkeypatch):
    """_show_usage's self._print_nous_credits_block/_print_usage_cta resolve
    into CLIBillingMixin through the composite — even with self.agent None."""
    cli_obj = _bare_cli()  # self.agent is None -> billing helpers must resolve
    cli_obj.agent = None
    block = MagicMock(return_value=True)
    cta = MagicMock()
    monkeypatch.setattr(CLIBillingMixin, "_print_nous_credits_block", block)
    monkeypatch.setattr(CLIBillingMixin, "_print_usage_cta", cta)
    cli_obj._show_usage()
    block.assert_called_once_with()
    cta.assert_called_once_with()


# ── behavioral pins ─────────────────────────────────────────────────────────


def test_usage_command_dispatch(monkeypatch, capsys):
    cli_obj = _bare_cli()
    calls = []
    monkeypatch.setattr(cli_obj, "_usage_reset", lambda force: calls.append(("reset", force)))
    monkeypatch.setattr(cli_obj, "_show_usage", lambda: calls.append(("show",)))

    cli_obj._handle_usage_command("/usage reset --force")
    assert calls == [("reset", True)]

    cli_obj._handle_usage_command("/usage")
    assert calls == [("reset", True), ("show",)]

    cli_obj._handle_usage_command("/usage bogus")
    assert calls == [("reset", True), ("show",)]
    assert "Unknown /usage subcommand" in capsys.readouterr().out


def test_usage_reset_non_codex_early_return(capsys):
    cli_obj = _bare_cli()
    cli_obj.agent = MagicMock(provider="anthropic")
    cli_obj._usage_reset()
    assert "only available on the openai-codex provider" in capsys.readouterr().out


def test_usage_reset_timeout_path(monkeypatch, capsys):
    cli_obj = _bare_cli()
    cli_obj.agent = MagicMock(provider="openai-codex")

    class _Future:
        def result(self, timeout=None):
            raise cli_usage_mixin.concurrent.futures.TimeoutError()

    class _Pool:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, *args, **kwargs):
            return _Future()

    monkeypatch.setattr(
        cli_usage_mixin.concurrent.futures, "ThreadPoolExecutor", lambda *a, **k: _Pool()
    )
    cli_obj._usage_reset()
    assert "Timed out talking to the Codex backend" in capsys.readouterr().out


def test_context_breakdown_shape(monkeypatch, capsys):
    cli_obj = _bare_cli()
    agent = MagicMock()
    cli_obj.agent = agent
    cli_obj.conversation_history = [{"role": "user"}, {"role": "assistant"}]
    cli_obj.model = "fallback-model"

    payload = {"model": "test-model", "buckets": {"conv": 50}}
    details = {"skills": [], "toolsets": []}
    monkeypatch.setattr(
        "agent.context_breakdown.compute_session_context_breakdown",
        lambda a, history: payload,
    )
    monkeypatch.setattr(
        "agent.context_breakdown.render_context_breakdown_lines",
        lambda p, details=None, grid=True: ["▓▓▓░ 50%", "tail line"],
    )
    monkeypatch.setattr("agent.context_breakdown.compute_context_details", lambda a: details)

    cli_obj._show_context_breakdown("/context all")
    out = capsys.readouterr().out
    assert "Context Usage — test-model" in out
    assert "▓▓▓░ 50%" in out
    assert "tail line" in out

    # Expanded path must also render (details pulled, grid printed)
    cli_obj._show_context_breakdown("/context all")
    assert "Context Usage" in capsys.readouterr().out


def test_context_breakdown_no_agent_early_return(capsys):
    cli_obj = _bare_cli()
    cli_obj.agent = None
    cli_obj._show_context_breakdown("/context")
    assert "No active agent" in capsys.readouterr().out


def test_insights_aggregation(capsys):
    cli_obj = _bare_cli()

    class _Engine:
        calls = []

        def __init__(self, db):
            self.db = db

        def generate(self, *, days=30, source=None):
            self.calls.append((days, source))
            return {"days": days, "source": source}

        def format_terminal(self, report):
            return f"aggregated days={report['days']} source={report['source']}"

    _Engine.calls = []
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db), patch(
        "agent.insights.InsightsEngine", _Engine
    ):
        cli_obj._show_insights("/insights --days 7 --source discord")

    assert _Engine.calls == [(7, "discord")]
    db.close.assert_called_once()
    assert "aggregated days=7 source=discord" in capsys.readouterr().out


def test_insights_invalid_days(capsys):
    cli_obj = _bare_cli()
    cli_obj._show_insights("/insights --days nope")
    assert "Invalid --days value" in capsys.readouterr().out


def test_show_usage_logging_flip(monkeypatch, capsys):
    """The global logging-level flip travels verbatim with _show_usage:
    verbose -> root DEBUG + noisy loggers WARNING; quiet -> INFO."""
    cli_obj = _bare_cli()
    agent = MagicMock()
    agent.session_api_calls = 5
    agent.get_rate_limit_state.return_value = None
    agent.model = "test-model"
    agent.session_input_tokens = 10
    agent.session_output_tokens = 20
    agent.session_reasoning_tokens = 0
    agent.session_prompt_tokens = 100
    agent.session_completion_tokens = 200
    agent.session_total_tokens = 300
    agent.provider = None
    agent.base_url = None
    agent.api_key = None
    agent.context_compressor.last_prompt_tokens = 100
    agent.context_compressor.context_length = 20000
    agent.context_compressor.compression_count = 1
    cli_obj.agent = agent
    cli_obj.conversation_history = []
    cli_obj.session_start = datetime.now()
    cli_obj.verbose = True
    cli_obj._print_nous_credits_block = lambda: False
    cli_obj._print_usage_cta = lambda: None

    monkeypatch.setattr("agent.account_usage.render_account_usage_lines", lambda s: [])
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda *a, **k: None)

    root = logging.getLogger()
    noisy = logging.getLogger("openai")
    saved = (root.level, noisy.level)
    try:
        cli_obj._show_usage()
        assert root.level == logging.DEBUG
        assert noisy.level == logging.WARNING

        cli_obj.verbose = False
        cli_obj._show_usage()
        assert root.level == logging.INFO
    finally:
        root.setLevel(saved[0])
        noisy.setLevel(saved[1])
    capsys.readouterr()
