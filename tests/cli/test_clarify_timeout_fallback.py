"""Regression coverage for the CLI clarify timeout fallback."""

import os
import re
import time
from pathlib import Path
from tempfile import gettempdir

# The hermetic runner clears Windows home variables.  cli.py resolves its
# platform default at import time, before pytest's HERMES_HOME fixture runs.
os.environ.setdefault("LOCALAPPDATA", str(Path(gettempdir()) / "hermes-test-localappdata"))

import cli as cli_module
from hermes_cli import managed_scope


def _timeout_used_by_callback(monkeypatch, config: dict) -> int:
    """Run the callback through its timeout path and return its displayed limit."""
    messages: list[str] = []
    monkeypatch.setattr(cli_module, "CLI_CONFIG", config)
    monkeypatch.setattr(cli_module, "_cprint", messages.append)

    cli = cli_module.HermesCLI.__new__(cli_module.HermesCLI)
    cli._app = None

    calls = 0

    def monotonic() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 3 else 1e9

    monkeypatch.setattr(time, "monotonic", monotonic)
    cli._clarify_callback("Continue?", ["yes", "no"])

    for message in messages:
        match = re.search(r"timed out after (\d+)s", message)
        if match:
            return int(match.group(1))
    raise AssertionError("clarify callback did not report a timeout")


def test_cli_uses_agent_clarify_timeout_without_clarify_override(monkeypatch):
    """A user-set gateway timeout is the CLI fallback when clarify is unset."""
    timeout = _timeout_used_by_callback(
        monkeypatch,
        {
            "clarify": {"timeout": 120, "_clarify_timeout_explicitly_configured": False},
            "agent": {"clarify_timeout": 600},
            "display": {"persist_prompts": False},
        },
    )

    assert timeout == 600


def test_managed_clarify_timeout_at_default_value_beats_agent_fallback(
    monkeypatch, tmp_path
):
    """An administrator pin of 120 remains authoritative over a user fallback."""
    user_home = tmp_path / "user"
    user_home.mkdir()
    (user_home / "config.yaml").write_text("agent:\n  clarify_timeout: 600\n")

    managed_home = tmp_path / "managed"
    managed_home.mkdir()
    (managed_home / "config.yaml").write_text("clarify:\n  timeout: 120\n")

    monkeypatch.setattr(cli_module, "_hermes_home", user_home)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    managed_scope.invalidate_managed_cache()
    try:
        config = cli_module.load_cli_config()

        assert config["clarify"]["_clarify_timeout_explicitly_configured"] is True
        assert _timeout_used_by_callback(monkeypatch, config) == 120
    finally:
        managed_scope.invalidate_managed_cache()
