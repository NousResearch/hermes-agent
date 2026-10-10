"""An applied secret must be redactable after its value stops looking like an assignment.

A secret source is the only producer of secrets in the process that the exact-value
scrub registry does not know about. Once its value is split across lines the
shape-based passes lose the `KEY=value` form and the bare body survives, so the apply
funnel registers the value — and each of its substantial lines — for exact-value
redaction.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.redact import (  # noqa: E402
    clear_vault_redaction_values,
    redact_registered_vault_values,
    redact_terminal_output,
)
from agent.secret_sources import registry as reg  # noqa: E402
from agent.secret_sources.base import FetchResult, SecretSource  # noqa: E402

PRIVATE_KEY_HEADER = "-----BEGIN OPENSSH PRIVATE KEY-----"
PRIVATE_KEY_FOOTER = "-----END OPENSSH PRIVATE KEY-----"
BODY_LINE = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtz"


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    reg._reset_registry_for_tests()
    monkeypatch.setattr(reg, "_ensure_builtin_sources", lambda: None)
    clear_vault_redaction_values()
    yield
    reg._reset_registry_for_tests()
    clear_vault_redaction_values()


def _apply(secrets: dict) -> dict:
    class _Src(SecretSource):
        def fetch(self, cfg, home_path):
            res = FetchResult()
            res.secrets = dict(secrets)
            return res

        def override_existing(self, cfg):
            return True

    _Src.name = "applied"
    _Src.label = "Applied"
    _Src.shape = "mapped"
    _Src.scheme = "applied://x"
    _Src.api_version = reg.SECRET_SOURCE_API_VERSION
    assert reg.register_source(_Src())

    env: dict = {}
    reg.apply_all({"applied": {"enabled": True}}, Path("/tmp/applied-home"), environ=env)
    return env


def test_applied_value_is_registered_for_exact_redaction():
    _apply({"DEPLOY_TOKEN": "opaque-value-never-seen-by-shape-passes"})

    assert "opaque-value-never-seen-by-shape-passes" not in redact_registered_vault_values(
        "leaked: opaque-value-never-seen-by-shape-passes"
    )


def test_each_line_of_a_multiline_value_is_registered():
    key = "\n".join([PRIVATE_KEY_HEADER, BODY_LINE, PRIVATE_KEY_FOOTER])
    _apply({"SSH_KEY": key})

    # A line of the body on its own, with no name attached to it — what a pipe between
    # the command and the output leaves behind.
    assert redact_registered_vault_values(f"line: {BODY_LINE}") == "line: «redacted-vault-secret»"


def test_multiline_secret_survives_no_pipeline_and_each_line_oriented_one():
    key = "\n".join([PRIVATE_KEY_HEADER, BODY_LINE, PRIVATE_KEY_FOOTER])
    _apply({"SSH_KEY": key})
    env_text = f"HOME=/tmp\nSSH_KEY={key}\nPATH=/usr/bin\n"
    body = [line for line in key.splitlines() if not line.startswith("-----")]

    pipelines = {
        "env": "cat",
        "env | sort": "sort",
        "env | sort | grep -v -i key": "sort | grep -v -i key",
        "env | cut -d= -f1": "cut -d= -f1",
    }
    for command, pipe in pipelines.items():
        out = subprocess.run(["sh", "-c", pipe], input=env_text,
                             capture_output=True, text=True).stdout
        redacted = redact_terminal_output(out, command=command, force=True)
        leaked = [line for line in body if line in redacted]
        assert not leaked, f"{command} leaked {len(leaked)}/{len(body)} body lines"


def test_short_fragments_are_not_registered_on_their_own():
    # A fragment common enough to occur by chance must not blank out unrelated text:
    # the scrub is an unanchored substring match over every later result.
    key = "\n".join([PRIVATE_KEY_HEADER, "abc", PRIVATE_KEY_FOOTER])
    _apply({"SSH_KEY": key})

    assert redact_registered_vault_values("unrelated text with abc inside") == (
        "unrelated text with abc inside"
    )


def test_a_secret_source_failure_does_not_block_applying():
    class _Src(SecretSource):
        def fetch(self, cfg, home_path):
            raise RuntimeError("source is broken")

    _Src.name = "broken"
    _Src.label = "Broken"
    _Src.shape = "mapped"
    _Src.scheme = "broken://x"
    _Src.api_version = reg.SECRET_SOURCE_API_VERSION
    assert reg.register_source(_Src())

    env: dict = {}
    reg.apply_all({"broken": {"enabled": True}}, Path("/tmp/broken-home"), environ=env)
    assert env == {}
