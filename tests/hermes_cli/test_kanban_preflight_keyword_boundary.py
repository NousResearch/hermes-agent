"""Tests for the kanban pre-flight API-key guard.

The guard used to do substring matching, so innocent words containing
provider substrings (e.g. "fal" inside "Falsifizierbarkeit") blocked
tasks. The fix uses word-boundary regex plus a required API/auth/call
context marker so that public/compound contexts such as "GitHub-Repo"
do not trigger a block either.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_task(title: str, body: str = "") -> kb.Task:
    """Build a minimal Task object for direct preflight testing."""
    return kb.Task(
        id="t_test",
        title=title,
        body=body,
        assignee="operator",
        status="ready",
        priority=0,
        created_by="test",
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path="/tmp",
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )


@pytest.mark.parametrize(
    "title,body",
    [
        ("Falsifizierbarkeit prüfen", "Task beschäftigt sich mit Falsifizierbarkeit"),
        ("Wegfallen von Anforderungen", "Prüfe Wegfallen alter Anforderungen"),
        ("Falsch-Positive vermeiden", "Vermeide Falsch-Positive im Test"),
        ("GitHub-Repo public", "Öffentliches GitHub-Repo ohne Token"),
        ("Anthropic Beispieltext", "anthropic ist ein Beispielunternehmen"),
        ("Claude erklärt", "Claude als Personenname in einer Geschichte"),
        ("OpenAI History", " Geschichte über OpenAI ohne Authentifizierungskontext"),
    ],
)
def test_false_positives_do_not_block(
    kanban_home: Path, title: str, body: str
) -> None:
    with kb.connect_closing() as conn:
        pf = kb.preflight_check_task(conn, _make_task(title, body))
        assert not pf.blocked, f"blocked unexpectedly: {title!r} -> {pf.reason}"
        assert pf.route_to is None


@pytest.mark.parametrize(
    "title,body,env_var",
    [
        ("Anthropic API call", "Mache einen Anthropic API-Call", "ANTHROPIC_API_KEY"),
        ("GitHub Token", "Setze GitHub Token für private Repo", "GITHUB_TOKEN"),
        ("GitHub push", "Push zu GitHub private repo", "GH_TOKEN"),
        ("Fal API key", "Nutze fal API key", "FAL_API_KEY"),
        ("OpenAI client", "Baue OpenAI client", "OPENAI_API_KEY"),
        ("DeepSeek token", "Holen DeepSeek token", "DEEPSEEK_API_KEY"),
    ],
)
def test_real_api_context_blocks_when_key_missing(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str,
    body: str,
    env_var: str,
) -> None:
    for key in (env_var,):
        monkeypatch.delenv(key, raising=False)
    with kb.connect_closing() as conn:
        pf = kb.preflight_check_task(conn, _make_task(title, body))
        assert pf.blocked, f"should block: {title!r}"
        assert pf.kind == "capability"
        assert pf.pattern == "missing-api-keys"
        assert env_var in pf.reason


@pytest.mark.parametrize(
    "title,body,env_var",
    [
        ("Anthropic API", "Call Anthropic API", "ANTHROPIC_API_KEY"),
    ],
)
def test_real_api_context_passes_when_key_present(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str,
    body: str,
    env_var: str,
) -> None:
    monkeypatch.setenv(env_var, "dummy-key")
    with kb.connect_closing() as conn:
        pf = kb.preflight_check_task(conn, _make_task(title, body))
        assert not pf.blocked, f"should pass when {env_var} is set: {pf.reason}"


def test_config_override_can_disable_keyword(kanban_home: Path, monkeypatch) -> None:
    """An empty env-var list in config disables the keyword entirely."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg_path = kanban_home / "config.yaml"
    cfg_path.write_text(
        "kanban:\n  preflight_checks:\n    api_key_keywords:\n      anthropic: []\n"
    )
    with kb.connect_closing() as conn:
        pf = kb.preflight_check_task(
            conn, _make_task("Anthropic API", "Call Anthropic API")
        )
        assert not pf.blocked, "config override should disable anthropic check"
