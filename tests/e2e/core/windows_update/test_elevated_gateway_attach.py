"""A standard-user client dialing an elevated gateway's control pipe (#140706, #134889 W9).

The runner session is elevated, so this process hosts the real control pipe
(``NativeControlServer``) as an elevated gateway would. A child under the UAC filtered-shape
token (``_tokens.StandardUserProcess``: Administrators deny-only, Medium integrity) then dials it
the way the Desktop and CLI do:

* ``query_runtime_control``: the client half of the pipe identity check must refuse before
  sending anything, with the named ``gateway_elevated`` reason and its remedy, never the
  generic ``cannot verify bootstrap peer process``;
* ``discover_gateway_endpoint`` (what ``hermes gateway ensure`` and the Desktop boot read):
  ``inaccessible`` / ``gateway_elevated`` with the same remedy, without any gateway record.

The gateway's handler must never run: no request crosses the elevation boundary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration, pytest.mark.live_system_guard_bypass]

_REPO = Path(__file__).resolve().parents[4]

_CLIENT = r"""
import json, sys
from pathlib import Path
from gateway.control_socket import windows_pipe_name
from gateway import runtime_bootstrap_windows as native
from hermes_dir_acl import process_is_elevated
from hermes_cli.gateway_runtime import discover_gateway_endpoint
home = Path(sys.argv[1])
out = {"pipe": windows_pipe_name(home), "elevated": process_is_elevated()}
try:
    native.query_runtime_control(home, b'{"protocol": 1, "id": 1, "verb": "identify"}', 10)
    out["outcome"] = "attached"
except PermissionError as exc:
    out.update(outcome="refused", reason_code=getattr(exc, "reason_code", None), message=str(exc))
found = discover_gateway_endpoint(home, timeout=10)
out["discovery"] = [found.state, found.reason_code, found.detail]
print("RESULT " + json.dumps(out))
"""


def test_a_standard_user_client_gets_the_named_refusal_from_an_elevated_gateway(tmp_path):
    from gateway import runtime_bootstrap_windows as native
    from gateway.control_socket import windows_pipe_name
    from hermes_dir_acl import process_is_elevated
    from tests.e2e.core.windows_update._tokens import StandardUserProcess, hand_to_user, probe_token_elevation

    if not process_is_elevated():
        pytest.skip("the gateway side needs an elevated session (the Windows E2E runner's)")
    home, temp = tmp_path / "home", tmp_path / "temp"
    home.mkdir()
    temp.mkdir()
    hand_to_user(tmp_path)  # the user's own home, as on a real machine
    env = {**os.environ, "PYTHONPATH": str(_REPO), "TEMP": str(temp), "TMP": str(temp),
           "HERMES_HOME": str(home), "HERMES_STATE_DB_GUARD_BYPASS": "1"}
    # Harness control: the child really is a standard user (Medium, Administrators deny-only).
    assert probe_token_elevation(tmp_path, env, tmp_path / "token.log") == "elevated=False admin=False"

    served = []

    def handler(raw, subject):
        served.append((raw, subject))
        return b'{"ok": false}\n'
    server = native.NativeControlServer(home, handler)
    server.start()
    try:
        log = tmp_path / "client.log"
        child = StandardUserProcess([sys.executable, "-c", _CLIENT, str(home)], cwd=_REPO, env=env, log=log)
        rc = child.wait(120)
    finally:
        server.close()
    text = log.read_text(encoding="utf-8", errors="replace")
    assert rc == 0, text
    result = json.loads(next(line for line in text.splitlines() if line.startswith("RESULT "))[7:])
    assert result["pipe"] == windows_pipe_name(home), "harness: the child dialed another pipe"
    assert result["elevated"] is False
    remedy = native.refused_owner_remedy(native.GATEWAY_ELEVATED, os.getpid(), native._session_of(os.getpid()))
    assert (result["outcome"], result.get("reason_code"), result.get("message")) == (
        "refused", native.GATEWAY_ELEVATED, remedy), text
    assert result["discovery"] == ["inaccessible", native.GATEWAY_ELEVATED, remedy], text
    assert served == [], "a request from the standard-user client reached the elevated gateway"
