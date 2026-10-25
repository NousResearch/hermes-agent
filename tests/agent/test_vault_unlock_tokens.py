"""Password-manager session tokens: in memory, out of reach of a plain import, gone at the idle TTL,
and handed to exactly the manager invocation that needs them.

This is accidental/casual-access protection, not isolation: code running inside the Hermes process can
still reach process memory. The module-globals walk below is what ``from ... import`` or a state dump
sees; it deliberately does not descend into function closures.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import types
import uuid

import pytest

from agent.vault_backends import unlock
from tests.waiting import wait_until

_CONTAINERS = (dict, list, tuple, set, frozenset)


def _reachable(obj, needle: str, seen: set, *, closures: bool = False, depth: int = 0) -> bool:
    """Is *needle* inside *obj*? With ``closures`` the walk also opens function closure cells, i.e.
    it sees what in-process code can reach on purpose rather than by a plain import."""
    if id(obj) in seen or depth > 8:
        return False
    seen.add(id(obj))
    if isinstance(obj, str):
        return obj == needle
    if isinstance(obj, dict):
        children = [*obj.keys(), *obj.values()]
    elif isinstance(obj, _CONTAINERS):
        children = list(obj)
    elif isinstance(obj, types.FunctionType):
        if not closures:
            return False
        children = [cell.cell_contents for cell in obj.__closure__ or () if cell.cell_contents is not obj]
    elif isinstance(obj, (types.ModuleType, type, types.BuiltinFunctionType)):
        return False
    elif dataclasses.is_dataclass(obj):
        children = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    else:
        return False
    return any(_reachable(child, needle, seen, closures=closures, depth=depth + 1) for child in children)


def _held_by_unlock(token: str, *, closures: bool) -> bool:
    return any(_reachable(value, token, set(), closures=closures) for value in vars(unlock).values())


@pytest.fixture(autouse=True)
def _profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    unlock.lock()
    yield
    unlock.lock()


def test_session_token_is_not_reachable_from_the_unlock_module_globals():
    token = f"dummy-session-{uuid.uuid4().hex}"
    assert unlock.store_session_token("bitwarden", token)
    assert unlock.get_session_token("bitwarden") == token  # the real call path still gets it
    assert not _held_by_unlock(token, closures=False), "a plain import or module-state dump reaches the token"


def test_idle_token_is_dropped_at_the_ttl_without_another_vault_call(monkeypatch):
    monkeypatch.setattr(unlock, "_IDLE_TTL_S", 0.2)
    token = f"dummy-idle-{uuid.uuid4().hex}"
    assert unlock.store_session_token("onepassword", token)
    assert _held_by_unlock(token, closures=True), "the deep walk must see the live store, or it proves nothing"
    wait_until(lambda: not _held_by_unlock(token, closures=True), timeout=5,
               what="an expired token released without waiting for the next vault call")


# A stand-in `bw` that logs the environment of every invocation beside itself.
_FAKE_BW = r'''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bw.log"), "a") as log:
    log.write(json.dumps({"argv": argv, "env": dict(os.environ)}) + "\n")
if argv[:2] == ["unlock", "--raw"]:
    print("dummy-bw-session-7f3a"); sys.exit(0)
if argv[:2] == ["list", "items"]:
    print("[]"); sys.exit(0)
sys.exit(2)
'''


@pytest.mark.platforms("posix")  # the fake manager is a shebang script; the env contract is host-agnostic
def test_session_token_reaches_only_the_manager_call_that_uses_it(tmp_path):
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from tools.environments.local import build_subprocess_env

    exe = tmp_path / "bw"
    exe.write_text(_FAKE_BW, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    backend = BitwardenLoginBackend({"binary_path": str(exe)})

    backend.unlock("dummy master password")
    assert backend.list_items() == []

    token = "dummy-bw-session-7f3a"
    calls = [json.loads(line) for line in (tmp_path / "bw.log").read_text(encoding="utf-8").splitlines()]
    unlock_call, list_call = calls
    assert "BW_SESSION" not in unlock_call["env"], "the unlock call has no session yet"
    assert list_call["env"]["BW_SESSION"] == token
    assert token not in os.environ.values(), "the token never enters Hermes' own environment"
    # Every other child (terminal tool, MCP servers, hooks) starts from that environment; even the
    # unscrubbed factory variant must not carry it.
    assert token not in build_subprocess_env(scrub_secrets=False).values()
    assert token not in build_subprocess_env().values()
