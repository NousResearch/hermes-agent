"""``HERMES_REDACT_SECRETS`` parsing: unset or blank keeps redaction ON (#17691 secure default)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.redact import redaction_flag_enabled

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("raw", [None, "", "   ", "\t"])
def test_unset_or_blank_keeps_redaction_on(raw):
    assert redaction_flag_enabled(raw) is True


@pytest.mark.parametrize("raw", ["true", " TRUE ", "1", "yes", "on", True])
def test_truthy_values_enable(raw):
    assert redaction_flag_enabled(raw) is True


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", " False ", False])
def test_explicit_opt_out_disables(raw):
    assert redaction_flag_enabled(raw) is False


@pytest.mark.parametrize("raw, expected", [("", "True"), (" true ", "True"), ("false", "False")])
def test_import_snapshot_uses_same_rule(raw, expected):
    env = {**os.environ, "HERMES_REDACT_SECRETS": raw}
    out = subprocess.run(
        [sys.executable, "-c", "import agent.redact as r; print(r._REDACT_ENABLED)"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    assert out.stdout.strip() == expected
