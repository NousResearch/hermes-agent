"""Regression coverage for P0-D: kanban workers spawned via the dispatcher's
`-q` single-query path (see `hermes_cli/kanban_db.py::_default_spawn`) must
never silently exit 0 on a total turn failure.

Background: the dispatcher spawns workers as `hermes ... chat -q "<prompt>"`
(lowercase `-q`, the query-text flag — NOT `-Q`/`--quiet`, the machine-
readable flag). Before this fix, only the `-Q` branch of `cli.main()`
translated a failed turn (`result["failed"]`) into a non-zero exit code
(1, or KANBAN_RATE_LIMIT_EXIT_CODE for a quota/billing wall). The `-q`
branch the dispatcher actually uses had no such logic at all — `cli.chat()`
returns a plain string there, discarding `failed`/`failure_reason` — so a
worker whose API calls failed completely (invalid/unavailable model,
persistent auth error, provider outage, retries+fallback exhausted) fell
through to the interpreter's default exit code 0. The dispatcher's
`detect_crashed_workers()` then saw `clean_exit` on a still-`running` task
and reported `protocol_violation` — a mischaracterization: the model never
got a turn at all, so it never had a chance to call kanban_complete /
kanban_block, let alone decline to.

This observed exactly in a real incident: profile `claude-coder`'s
provider/model config pointed at an unavailable free-tier OpenRouter slug
(`tencent/hy3:free`, HTTP 404 "unavailable for free"), so every one of 3
dispatcher-spawned worker runs failed at the very first API call (0 tool
calls, 1 message, per the worker log) and one of the three was reaped as
`protocol_violation` at rc=0.

The fix: `HermesCLI.chat()` now stashes the structured turn result on
`self._last_chat_result`; `cli.kanban_worker_exit_code()` maps that result
to a process exit code using the exact same rules the `-Q` path already
used; and the `-q` single-query branch in `main()` now calls
`sys.exit(kanban_worker_exit_code(...))` whenever `HERMES_KANBAN_TASK` is
set (dispatcher-spawned worker) — so a total failure becomes a real
non-zero exit, which the dispatcher's reap classifier sees as
`nonzero_exit` (a normal crash, using the default multi-strike failure
limit) instead of `clean_exit` (`protocol_violation`, single-strike).

Human / non-dispatcher `-q` usage (no `HERMES_KANBAN_TASK`) is completely
unaffected — it keeps the pre-existing "always falls through, implicit
exit 0" behavior.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli as cli_mod
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


# ---------------------------------------------------------------------------
# Unit tests: cli.kanban_worker_exit_code()
# ---------------------------------------------------------------------------

def test_exit_code_none_result_is_zero():
    assert cli_mod.kanban_worker_exit_code(None) == 0


def test_exit_code_non_dict_result_is_zero():
    assert cli_mod.kanban_worker_exit_code("not a dict") == 0


def test_exit_code_succeeded_turn_is_zero():
    assert cli_mod.kanban_worker_exit_code({"failed": False, "final_response": "ok"}) == 0


def test_exit_code_missing_failed_key_is_zero():
    # A dict without "failed" at all (e.g. an unexpected/partial shape)
    # must not be treated as a failure — no invented non-zero exit.
    assert cli_mod.kanban_worker_exit_code({"final_response": "ok"}) == 0


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_exit_code_quota_wall_uses_rate_limit_sentinel(reason):
    result = {"failed": True, "failure_reason": reason, "error": "quota exhausted"}
    assert cli_mod.kanban_worker_exit_code(result) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_exit_code_total_api_failure_with_no_failure_reason_is_one():
    """The exact shape observed in the real incident: the turn failed
    (all retries + fallback exhausted on an invalid/unavailable model)
    but `failure_reason` was never a quota/billing classification — it
    must become a plain 1, NOT the rate-limit sentinel and NOT 0."""
    result = {
        "failed": True,
        "final_response": "",
        "error": "HTTP 404: This model is unavailable for free.",
    }
    assert cli_mod.kanban_worker_exit_code(result) == 1


def test_exit_code_other_failure_reason_is_one():
    result = {"failed": True, "failure_reason": "context_overflow", "error": "too big"}
    assert cli_mod.kanban_worker_exit_code(result) == 1


# ---------------------------------------------------------------------------
# Integration-style tests: cli.main()'s `-q` single-query branch, driven the
# same way tests/cli/test_single_query_session_finalize.py drives it — a
# FakeCLI stand-in for HermesCLI so no real model/network call happens.
# ---------------------------------------------------------------------------

class _FakeCLIBase:
    def __init__(self, **_kwargs):
        self.console = SimpleNamespace(print=lambda *a, **k: None)
        self.session_id = "worker-session"
        self.agent = SimpleNamespace(session_id="worker-session", platform="cli")
        self._last_chat_result = None

    def _claim_active_session(self, surface, *, stderr=False):
        return True

    def _show_security_advisories(self):
        pass

    def _print_exit_summary(self):
        pass


@pytest.fixture(autouse=True)
def _stub_finalize(monkeypatch):
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_k: None)


def test_kanban_worker_q_path_exits_zero_on_success(monkeypatch):
    class FakeCLI(_FakeCLIBase):
        def chat(self, query, images=None):
            # Mirrors the real HermesCLI.chat(): stash the structured
            # result, return a plain string.
            self._last_chat_result = {"failed": False, "final_response": "done"}
            return "done"

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="work kanban task t_fake", quiet=False, toolsets="terminal")
    assert exc_info.value.code == 0


def test_kanban_worker_q_path_exits_one_on_total_api_failure(monkeypatch):
    """This is the exact regression case: the dispatcher's `-q` path, a
    worker whose model is entirely unavailable, must exit non-zero — not
    fall through silently to 0."""
    class FakeCLI(_FakeCLIBase):
        def chat(self, query, images=None):
            self._last_chat_result = {
                "failed": True,
                "final_response": "",
                "error": "HTTP 404: This model is unavailable for free.",
            }
            return "Error: HTTP 404: This model is unavailable for free."

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="work kanban task t_fake", quiet=False, toolsets="terminal")
    assert exc_info.value.code == 1


def test_kanban_worker_q_path_exits_rate_limit_sentinel_on_quota_wall(monkeypatch):
    class FakeCLI(_FakeCLIBase):
        def chat(self, query, images=None):
            self._last_chat_result = {
                "failed": True,
                "final_response": "",
                "error": "rate limited",
                "failure_reason": "rate_limit",
            }
            return "Error: rate limited"

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="work kanban task t_fake", quiet=False, toolsets="terminal")
    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE


def test_non_kanban_q_path_falls_through_on_failure_unchanged(monkeypatch):
    """No HERMES_KANBAN_TASK set (plain human/automation `-q` usage, no
    dispatcher involved) — behavior must be completely untouched: no
    sys.exit() call at all, even when the turn failed."""
    class FakeCLI(_FakeCLIBase):
        def chat(self, query, images=None):
            self._last_chat_result = {
                "failed": True,
                "final_response": "",
                "error": "HTTP 404: This model is unavailable for free.",
            }
            return "Error: HTTP 404: This model is unavailable for free."

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    # Must NOT raise SystemExit — falls through to the function's natural
    # return, exactly as before this fix.
    cli_mod.main(query="hello", quiet=False, toolsets="terminal")
