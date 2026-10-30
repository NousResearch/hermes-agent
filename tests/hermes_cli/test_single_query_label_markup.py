"""Single-query label must render literally, not as Rich markup (#98789).

``-q`` / ``--query-file`` text is user-controlled (the Bot Mode DM transport
feeds arbitrary handoff prose through it, e.g. an issue comment containing
``[/<agent>]``). The single-query branch prints it via ``Console.print``,
which parses the string as Rich markup — an unmatched closing tag such as a
pytest id ``test_case[/fb-images/a/../b.webp?w=336]`` raised
``rich.errors.MarkupError`` before the agent turn even started.

These tests drive ``_run_single_query_mode`` through the same facade-monkeypatch
seams it late-binds from ``cli`` but give the fake CLI a *real*
``rich.console.Console`` so the markup parser actually runs.
"""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import pytest
from rich.console import Console

from agent.i18n import t

import cli as cli_mod
from hermes_cli.cli_single_query import _run_single_query_mode

# The exact reproduction token from the issue: Rich sees ``[/fb-images/...]``
# as an unmatched closing tag.
_MARKUP_BREAKING_QUERY = "test_case[/fb-images/a/../b.webp?w=336]"


@pytest.fixture(autouse=True)
def _isolated_single_query_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "_should_seed_interactive", lambda *a: False)
    monkeypatch.setattr(cli_mod, "_collect_query_images", lambda query, image=None: (query, None))
    monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda images: [])
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda fake_cli: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: SimpleNamespace(_cli_ref=None),
    )
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_startup.record_cli_one_shot_ready",
        lambda: None,
    )
    yield
    # _run_single_query_mode sets this unconditionally on the real environ.
    os.environ.pop("HERMES_SINGLE_QUERY_SESSION", None)


@pytest.fixture
def fake_cli_factory():
    calls = []
    stdout = io.StringIO()

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = Console(
                file=stdout, width=80, no_color=True, highlight=False
            )
            self.session_id = "single-query-session"
            self.agent = SimpleNamespace(
                session_id="single-query-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            self._last_turn_result = {"completed": True}
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    return calls, stdout, FakeCLI


def _run_single_query(monkeypatch, fake_cli_factory, query):
    calls, stdout, fake_cls = fake_cli_factory

    with pytest.raises(SystemExit) as excinfo:
        _run_single_query_mode(fake_cls(), query, None, quiet=False, oneshot=False)

    assert excinfo.value.code == 0
    assert calls == [
        ("claim", "cli", False),
        "advisories",
        ("chat", query, None),
        "summary",
    ]
    return stdout.getvalue()


def test_single_query_label_with_unmatched_closing_tag_reaches_chat(
    monkeypatch, fake_cli_factory
):
    # Before the fix this raises MarkupError inside Console.print, so the
    # agent turn never starts; after it the run completes and the label is
    # rendered literally.
    output = _run_single_query(monkeypatch, fake_cli_factory, _MARKUP_BREAKING_QUERY)

    assert _MARKUP_BREAKING_QUERY in output


def test_single_query_label_plain_text_still_renders(monkeypatch, fake_cli_factory):
    output = _run_single_query(monkeypatch, fake_cli_factory, "plain query")

    # The label prefix comes from the i18n catalog ("Query:" in English);
    # read it through the same translator so the assertion holds whatever
    # catalog the test home resolved.
    label = t("cli.single_query.query_label")
    assert f"{label} plain query" in output
