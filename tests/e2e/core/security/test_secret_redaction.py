"""Secrets never reach disk or the next provider request through the classic CLI (``hermes chat -q``).

Contract under test (``security.redact_secrets``, on by default; website/docs/user-guide/configuration.md
§ Security, hermes_logging.py, user-guide/sessions.md § Export Sessions):

* tool output is redacted before it enters the conversation, so it never reaches state.db (any table,
  FTS, WAL), a session export, or the next provider request; a read of a secret-bearing file (``.env``)
  masks credential-shaped assignments whatever the value looks like;
* the model's own answer is redacted at the storage boundary (history / state.db / exports / replay);
* every log file (agent.log, errors.log, gateway.log) goes through ``RedactingFormatter``;
* ``hermes sessions export --redact`` masks message content AND tool-call arguments.

Raw BY DESIGN (not asserted, see NOT COVERED in the suite notes): a tool argument and the user's own prompt
stay as executed in state.db, in the default (non ``--redact``) export and in the provider replay of the
conversation. Every scenario first proves the secret really travelled (the tool echoed it into a workspace
file, curl hit the endpoint with it, the log recorded the turn) before asserting absence.
"""

from __future__ import annotations

import re
import sys

import pytest

from tests.e2e.core.security._helpers import known_param, run_hermes, write_home
from tests.e2e.core.security._redact import (
    CONFIG, SCENARIOS, Ctx, Director, Secrets, World, assert_harness_sane, check, collect, echo_preconditions,
    prompt_for, seed_workspace,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell commands (cat | tee, curl)")

KNOWN: dict[str, str] = {}

_SESSION_RE = re.compile(r"session_id:\s*(\S+)")


@pytest.fixture(scope="module")
def cli_world(tmp_path_factory) -> World:
    root = tmp_path_factory.mktemp("redact-cli")
    home, ws, keys = root / "home", root / "ws", Secrets()
    seed_workspace(ws, keys)
    ctx = Ctx(keys, ws, home / ".hermes" / ".env")
    with FakeLLMServer(Director(ctx), api_key=keys.provider, record_get=True) as llm:
        ctx.port = llm.port
        write_home(home / ".hermes", llm.base_url, api_key=keys.provider, env=keys.env(), config=CONFIG)
        runs: dict[str, str] = {}
        for name, scenario in SCENARIOS.items():
            r = run_hermes(["chat", "-q", prompt_for(name, keys), "-Q"], home, cwd=ws, timeout=150)
            runs[name] = f"rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
            if scenario.followup:
                sid = _SESSION_RE.search(r.stdout + r.stderr)
                assert r.returncode == 0 and sid, f"{name}: first turn failed\n{runs[name]}"
                r2 = run_hermes(["chat", "-q", prompt_for(name, keys, followup=True), "-Q", "--resume", sid.group(1)],
                                home, cwd=ws, timeout=150)
                assert r2.returncode == 0, f"{name}: follow-up turn failed\n{r2.stdout}\n{r2.stderr}"
            elif name != "provider_error_echo":
                assert r.returncode == 0, f"{name}: turn failed\n{runs[name]}"
        sinks = collect(home, list(llm.requests))
        gets = [r for r in llm.requests if r["kind"] == "get"]
    assert_harness_sane(sinks)
    logs = "\n".join(sinks.texts["logs"].values())
    return World(keys, sinks, echo_preconditions(ws, keys, gets, logs), runs)


@pytest.mark.parametrize("scenario", [known_param(n, KNOWN) for n in SCENARIOS])
def test_cli_turn_never_persists_or_replays_a_secret(cli_world: World, scenario: str) -> None:
    check(cli_world, scenario)
