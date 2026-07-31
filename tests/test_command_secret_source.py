"""E2E tests for the ``command`` secret source.

These exercise the REAL resolution path: real helper shell scripts written
to a temp dir (chmod +x), real ``/bin/sh -c`` subprocesses, and a real temp
HERMES_HOME with a config.yaml routing ``secrets.provider: command`` through
``hermes_cli.env_loader._apply_external_secret_sources``.

Security invariants under test (ported from the desktop TS provider):

* the requested key travels ONLY via the ``HERMES_SECRET_KEY`` env var —
  never interpolated into the shell string (hostile key names are inert);
* cross-key misroute guard: a single env-shaped line for a DIFFERENT key
  never leaks as the wanted key's value;
* base64 '=' padding is not misclassified as a dotenv line;
* hard timeout + degrade-to-empty on every failure mode, never raise;
* failure logging carries structured fields only — never the command
  string or any secret value.

NOTE: tests assert on key NAMES, lengths, and presence — never log secret
values themselves.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources.command import (  # noqa: E402
    apply_command_secrets,
    get_command_secret,
    list_command_secrets,
    parse_secret_output,
    unquote_dotenv_value,
)
from hermes_cli import env_loader  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the command secret provider is POSIX-only"
)


def _write_helper(tmp_path: Path, body: str, name: str = "helper.sh") -> Path:
    """Write a real executable helper script and return its path."""
    script = tmp_path / name
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with a clean source map, applied-home guard, and no
    leftover test keys in os.environ."""
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    for key in ("CMDTEST_API_KEY", "CMDTEST_TOKEN", "CMDTEST_OTHER_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    for key in ("CMDTEST_API_KEY", "CMDTEST_TOKEN", "CMDTEST_OTHER_KEY"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Parsing semantics (pure functions, mirroring the TS parseSecretOutput)
# ---------------------------------------------------------------------------


def test_unquote_strips_one_layer_of_matching_quotes():
    assert unquote_dotenv_value('"abc"') == "abc"
    assert unquote_dotenv_value("'abc'") == "abc"
    assert unquote_dotenv_value('""') == ""
    assert unquote_dotenv_value('"') == '"'  # lone quote left intact
    assert unquote_dotenv_value("  plain  ") == "plain"


def test_parse_base64_padding_not_misclassified_as_dotenv():
    # "dGVzdA==" looks env-shaped (key `dGVzdA`, value `=`) but is a bare
    # base64 secret and must round-trip unchanged.
    assert parse_secret_output("dGVzdA==\n", "CMDTEST_API_KEY") == "dGVzdA=="








# ---------------------------------------------------------------------------
# Real-subprocess resolution
# ---------------------------------------------------------------------------


def test_bare_value_helper_resolves_single_value(tmp_path):
    helper = _write_helper(tmp_path, "printf 'sk-test-bare-12345'")
    value = get_command_secret(command=str(helper), key="CMDTEST_API_KEY")
    assert value == "sk-test-bare-12345"












def test_timeout_kills_hung_helper_and_degrades_to_empty(tmp_path):
    helper = _write_helper(tmp_path, "sleep 30")
    start = time.monotonic()
    value = get_command_secret(
        command=str(helper), key="CMDTEST_API_KEY", timeout_seconds=2.0
    )
    elapsed = time.monotonic() - start
    assert value is None
    assert elapsed < 6.0, f"helper not killed within the bound (took {elapsed:.1f}s)"






def test_failure_logging_never_leaks_command_or_secret(tmp_path, capfd):
    secret_value = "sk-super-secret-value-do-not-log"
    helper = _write_helper(
        tmp_path,
        f"echo '{secret_value}' >&2\nexit 7",
        name="my-distinctive-helper-name.sh",
    )
    value = get_command_secret(command=str(helper), key="CMDTEST_API_KEY")
    assert value is None
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    # Structured fields only — never the command string, the helper's
    # stderr, or any secret value.
    assert "my-distinctive-helper-name" not in combined
    assert secret_value not in combined
    assert "code=7" in combined  # the structured field IS logged






def test_apply_dotenv_blob_sets_environ(tmp_path):
    helper = _write_helper(
        tmp_path,
        "printf 'CMDTEST_API_KEY=sk-applied\\nCMDTEST_TOKEN=tok-applied\\n'",
    )
    result = apply_command_secrets(command=str(helper))
    assert sorted(result.applied) == ["CMDTEST_API_KEY", "CMDTEST_TOKEN"]
    assert result.error is None
    assert os.environ["CMDTEST_API_KEY"] == "sk-applied"
    assert os.environ["CMDTEST_TOKEN"] == "tok-applied"






# ---------------------------------------------------------------------------
# Dispatch E2E through env_loader against a real temp HERMES_HOME
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    from agent.secret_sources import registry
    registry._reset_registry_for_tests()
    yield
    registry._reset_registry_for_tests()


def test_registry_command_source_applies_and_records_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    helper = _write_helper(
        tmp_path, "printf 'CMDTEST_API_KEY=sk-dispatch\\nCMDTEST_TOKEN=tok-dispatch\\n'"
    )
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  command:\n"
        "    enabled: true\n"
        f"    command: {helper}\n",
        encoding="utf-8",
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ.get("CMDTEST_API_KEY") == "sk-dispatch"
    assert env_loader.get_secret_source("CMDTEST_API_KEY") == "command"
    assert env_loader.get_secret_source("CMDTEST_TOKEN") == "command"
    assert (
        env_loader.format_secret_source_suffix("CMDTEST_API_KEY")
        == " (from Command helper)"
    )


def test_registry_status_line_printed_once_per_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    helper = _write_helper(tmp_path, "printf 'CMDTEST_API_KEY=sk-once\\n'")
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n"
        f"    command: {helper}\n",
        encoding="utf-8",
    )

    for _ in range(3):  # idempotency guard: only the first call does work
        env_loader._apply_external_secret_sources(tmp_path)

    err = capsys.readouterr().err
    assert err.count("Command helper: applied 1 secret") == 1




def test_registry_failing_helper_does_not_block_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: exit 9\n",
        encoding="utf-8",
    )
    # Must not raise — config/helper errors never block startup.
    env_loader._apply_external_secret_sources(tmp_path)
    assert env_loader.get_secret_source("CMDTEST_API_KEY") is None




def test_registry_helper_error_prints_remediation(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: ''\n",
        encoding="utf-8",
    )
    env_loader._apply_external_secret_sources(tmp_path)
    err = capsys.readouterr().err
    assert "secrets.command.command" in err
