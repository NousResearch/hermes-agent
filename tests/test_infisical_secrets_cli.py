"""Tests for the Infisical secrets CLI helpers."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from io import StringIO
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import infisical_secrets_cli as cli  # noqa: E402


def test_sync_apply_respects_override_existing_false(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing")
    monkeypatch.delenv("NEW_KEY", raising=False)
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    cli._print_sync_actions(
        console,
        Namespace(apply=True),
        {"override_existing": False},
        {"OPENAI_API_KEY": "fresh", "NEW_KEY": "new"},
        [],
        set(),
    )

    assert os.environ["OPENAI_API_KEY"] == "existing"
    assert os.environ["NEW_KEY"] == "new"
