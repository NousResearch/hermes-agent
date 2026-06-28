"""Tests for ``export `` prefix handling in the hand-rolled .env parsers.

Bash-compatible .env files commonly prefix lines with ``export `` (users
copy-paste from shell profiles, cloud provider docs, tutorials). The three
hand-rolled parsers — ``hermes_cli.config.load_env``,
``hermes_cli.main._has_any_provider_configured``, and
``tools.skills_tool.load_env`` — split on ``line.partition("=")`` and must
strip the ``export `` prefix first, otherwise ``export API_KEY=sk-...`` is
stored under the wrong key ``"export API_KEY"`` and the real key is lost
(setup wizard re-triggers, providers undetected, skill env passthrough drops
the var). See PR #6659.

These assert the behavior contract (prefix stripped → canonical key resolves),
not the literal parser source.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch


def _write_env(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_config_load_env_strips_export_prefix(tmp_path):
    from hermes_cli.config import invalidate_env_cache, load_env

    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        'export OPENAI_API_KEY=sk-export-123\n'
        'export OPENROUTER_API_KEY="sk-or-456"\n'
        'ANTHROPIC_API_KEY=sk-plain-789\n',
    )
    invalidate_env_cache()
    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            env = load_env()
    finally:
        invalidate_env_cache()

    # Canonical keys resolve, export-prefixed wrong keys never appear.
    assert env["OPENAI_API_KEY"] == "sk-export-123"
    assert env["OPENROUTER_API_KEY"] == "sk-or-456"
    assert env["ANTHROPIC_API_KEY"] == "sk-plain-789"
    assert "export OPENAI_API_KEY" not in env


def test_config_load_env_does_not_mangle_non_export(tmp_path):
    """A bare 'export' word without trailing space is not a prefix."""
    from hermes_cli.config import invalidate_env_cache, load_env

    env_path = tmp_path / ".env"
    _write_env(env_path, "PLAIN_KEY=val1\nexportNOSPACE=val2\nexport REAL=val3\n")
    invalidate_env_cache()
    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            env = load_env()
    finally:
        invalidate_env_cache()

    assert env["PLAIN_KEY"] == "val1"
    # No trailing space → NOT an export prefix; the key stays intact.
    assert env["exportNOSPACE"] == "val2"
    assert env["REAL"] == "val3"
    assert "export REAL" not in env


def test_skills_tool_load_env_strips_export_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(
        "export SOME_SKILL_KEY=skillval\nPLAIN=plainval\n", encoding="utf-8"
    )

    # skills_tool.load_env reads get_hermes_home()/.env directly.
    import importlib

    import tools.skills_tool as skills_tool

    importlib.reload(skills_tool)
    with patch.object(skills_tool, "get_hermes_home", return_value=tmp_path):
        env = skills_tool.load_env()

    assert env["SOME_SKILL_KEY"] == "skillval"
    assert env["PLAIN"] == "plainval"
    assert "export SOME_SKILL_KEY" not in env


def test_has_any_provider_configured_with_export_prefix(tmp_path, monkeypatch):
    """An export-prefixed provider key in .env counts as configured.

    Exercises the .env-reading branch of _has_any_provider_configured by
    blanking provider creds from the process environment first, so detection
    depends solely on parsing the file.
    """
    import importlib

    # Blank any provider-shaped creds so os.environ short-circuit can't mask
    # the .env parse path.
    for key in list(__import__("os").environ):
        if key.endswith(("_API_KEY", "_TOKEN")) and key != "BWS_ACCESS_TOKEN":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(
        "export OPENAI_API_KEY=sk-export-only-123\n", encoding="utf-8"
    )

    import hermes_cli.main as hmain

    importlib.reload(hmain)
    # get_env_path() derives from HERMES_HOME (set above) → tmp_path/.env, so
    # no patching is needed. Re-clear os.environ provider keys that
    # load_hermes_dotenv may have populated at import/reload time, forcing the
    # function down its .env-reading branch.
    for key in list(__import__("os").environ):
        if key.endswith(("_API_KEY", "_TOKEN")) and key != "BWS_ACCESS_TOKEN":
            monkeypatch.delenv(key, raising=False)
    assert hmain._has_any_provider_configured() is True
