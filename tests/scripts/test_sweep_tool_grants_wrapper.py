"""W1-E regression: sweep-tool-grants.py active wrapper contract.

Proves the restored wrapper:
  - invokes sweep_expired_grants(ttl_hours=24)
  - emits exact JSON {"expired_grants_revoked": n, "ttl_hours": 24} only when n > 0
  - prints nothing when n == 0

Isolation: fakes tools.tool_grants.sweep_expired_grants before the wrapper
is imported so the ``from ... import`` binds the stub.  The real
~/.hermes/grants ledger is never touched.
"""
import importlib.util
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WRAPPER_PATH = os.path.join(REPO_ROOT, "scripts", "sweep-tool-grants.py")
_MOD_NAME = "sweep_tool_grants_wrapper"


def _run_wrapper(monkeypatch, capsys, fake_n, call_log):
    """Import the wrapper fresh with a faked engine and capture stdout."""
    import tools.tool_grants as tg

    def fake_sweep(ttl_hours=None):
        call_log["ttl_hours"] = ttl_hours
        return fake_n

    monkeypatch.setattr(tg, "sweep_expired_grants", fake_sweep)
    # Purge any stale import so ``from ... import`` re-binds to the fake.
    sys.modules.pop(_MOD_NAME, None)
    spec = importlib.util.spec_from_file_location(_MOD_NAME, WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return capsys.readouterr()


def test_wrapper_exists():
    """RED guard: the active wrapper must be present at scripts/sweep-tool-grants.py."""
    assert os.path.isfile(WRAPPER_PATH), (
        "scripts/sweep-tool-grants.py is absent — active wrapper not restored"
    )


def test_wrapper_invokes_sweep_ttl_24_and_emits_json(monkeypatch, capsys):
    call_log = {}
    out = _run_wrapper(monkeypatch, capsys, fake_n=3, call_log=call_log)
    assert call_log["ttl_hours"] == 24, (
        f"wrapper must call sweep_expired_grants(ttl_hours=24), got {call_log.get('ttl_hours')!r}"
    )
    parsed = json.loads(out.out)
    assert parsed == {"expired_grants_revoked": 3, "ttl_hours": 24}


def test_wrapper_silent_when_zero(monkeypatch, capsys):
    call_log = {}
    out = _run_wrapper(monkeypatch, capsys, fake_n=0, call_log=call_log)
    assert call_log["ttl_hours"] == 24
    assert out.out == "", f"expected no output for zero revocations, got {out.out!r}"
