"""vault.otp_commands: a user-configured helper supplies an emailed/SMS one-time code server-side.

Real helper subprocesses and a real config.yaml under the test HERMES_HOME; only the page is faked.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ORIGIN = "https://login.example.test"
CODE = "482913"


def _helper(tmp_path: Path, body: str) -> str:
    script = tmp_path / "otp-helper"
    script.write_text(f"#!{sys.executable}\nimport os, sys, pathlib\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _configure(origin: str, command: str) -> None:
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(json.dumps({"vault": {"otp_commands": {origin: command}}}))


def _enter_code(page_origin: str = ORIGIN):
    from tools import browser_vault_tool

    controls = [{"index": 0, "type": "text", "name": "c", "label": "Verification code", "autocomplete": ""}]
    seen = {}

    def fake_eval(task_id, expr):
        return {"success": True, "result": json.dumps(controls) if "querySelectorAll" in expr else page_origin + "/verify"}

    def fake_secret(task_id, expr):
        seen["expr"] = expr
        return {"success": True, "result": json.dumps({"filled": 1})}

    with patch("agent.vault_backends.unlock.can_prompt_here", return_value=False), \
         patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
         patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret):
        raw = browser_vault_tool.browser_vault_enter_code(task_id="t")
    return raw, seen.get("expr", "")


def test_headless_session_gets_the_code_from_the_helper_without_the_model_seeing_it(tmp_path):
    marker = tmp_path / "env.json"
    _configure(ORIGIN, _helper(tmp_path, (
        f"pathlib.Path({str(marker)!r}).write_text(__import__('json').dumps("
        "{k: os.environ.get(k) for k in ('HERMES_OTP_ORIGIN', 'HERMES_OTP_SINCE')}))\n"
        f"print(__import__('json').dumps({{'code': {CODE!r}, 'reference': 'msg-123'}}))")))
    raw, expr = _enter_code()
    out = json.loads(raw)
    assert out["success"] is True and out["source"] == "otp_command" and out["reference"] == "msg-123"
    assert CODE not in raw  # result goes to the model
    assert re.search(rf'"value": "{CODE}"', expr)  # the code went to the page
    env = json.loads(marker.read_text())
    assert env["HERMES_OTP_ORIGIN"] == ORIGIN and int(env["HERMES_OTP_SINCE"]) > 0

    from agent.redact import redact_registered_vault_values
    assert CODE not in redact_registered_vault_values(f"page echoed {CODE}")


def test_helper_is_polled_until_the_email_arrives(tmp_path):
    from agent.vault_otp_command import fetch_otp_code

    counter = tmp_path / "calls"
    _configure(ORIGIN, _helper(tmp_path, (
        f"p = pathlib.Path({str(counter)!r}); n = int(p.read_text()) + 1 if p.exists() else 1; p.write_text(str(n))\n"
        f"print({CODE!r} if n >= 3 else '')")))
    fetched = fetch_otp_code(ORIGIN, deadline_s=10, poll_s=0.01)
    assert fetched is not None and fetched.code == CODE and counter.read_text() == "3"


def test_helper_failure_never_echoes_its_output(tmp_path):
    _configure(ORIGIN, _helper(tmp_path, f"print({CODE!r}); print('boom ' + {CODE!r}, file=sys.stderr); sys.exit(3)"))
    raw, expr = _enter_code()
    out = json.loads(raw)
    assert out["error_type"] == "otp_command_failed" and "status 3" in out["error"]
    assert CODE not in raw and expr == ""


@pytest.mark.parametrize("stdout", ["two lines\\n123456", "{not json", "12 34 56 78 90 12 34"])
def test_output_that_is_not_one_code_is_refused(tmp_path, stdout):
    _configure(ORIGIN, _helper(tmp_path, f"print({stdout!r})"))
    raw, expr = _enter_code()
    assert json.loads(raw)["error_type"] == "otp_command_failed" and expr == ""


def test_helper_for_another_origin_is_never_run(tmp_path):
    ran = tmp_path / "ran"
    _configure(ORIGIN, _helper(tmp_path, f"pathlib.Path({str(ran)!r}).write_text('x'); print({CODE!r})"))
    raw, expr = _enter_code(page_origin="https://login.example.test.evil.invalid")
    assert json.loads(raw)["error_type"] == "prompt_unavailable"
    assert not ran.exists() and expr == ""
