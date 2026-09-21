"""W1-E regression: sweep-skill-grants.py wrapper portability contract.

Proves the wrapper is portable (derives sys.path from __file__, not a
hardcoded absolute path) by importing it from a *relocated copy* with a
fixture ``tools.skill_grants`` module placed alongside the copy. The
real ~/.hermes is never touched.

Behavioural tests:
  - wrapper file present at active path
  - invokes sweep_expired_grants(ttl_hours=24), emits exact JSON for n>0
  - silent (no stdout) for n==0
  - portable: a *copy* of the wrapper at a temp path imports and runs
    against a *fixture* tools/skill_grants module (proving the sys.path
    is derived from __file__, not a hardcoded literal)
"""
import importlib.util
import json
import os
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import subprocess
    _out = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if _out.returncode == 0 and _out.stdout.strip():
        REPO_ROOT = _out.stdout.strip()
except Exception:
    pass
WRAPPER_PATH = os.path.join(REPO_ROOT, "scripts", "sweep-skill-grants.py")
_MOD_NAME = "sweep_skill_grants_wrapper"


def _run_wrapper(monkeypatch, capsys, fake_n, call_log):
    """Import the wrapper fresh with a faked engine and capture stdout."""
    import tools.skill_grants as sg

    def fake_sweep(ttl_hours=None):
        call_log["ttl_hours"] = ttl_hours
        return fake_n

    monkeypatch.setattr(sg, "sweep_expired_grants", fake_sweep)
    sys.modules.pop(_MOD_NAME, None)
    spec = importlib.util.spec_from_file_location(_MOD_NAME, WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return capsys.readouterr()


def test_wrapper_exists():
    assert os.path.isfile(WRAPPER_PATH), (
        "scripts/sweep-skill-grants.py is absent — active wrapper not restored"
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


def test_wrapper_portable_from_relocated_copy(tmp_path, capsys, monkeypatch):
    """The wrapper must resolve its sys.path from __file__ so it runs
    from any location. Prove this behaviourally by copying the wrapper
    to a temp dir that mimics the repo layout (scripts/ + tools/), placing
    a *fixture* ``tools.skill_grants`` module under it, and importing the
    copy. The copy must bind the fixture (not the repo's real module and
    not a hardcoded path).

    The wrapper computes repo_root = dirname(dirname(abspath(__file__))),
    i.e. it expects to live at <root>/scripts/sweep-skill-grants.py and
    inserts <root> on sys.path. So the fixture layout must be
    <fake>/scripts/sweep-skill-grants.py + <fake>/tools/skill_grants.py.
    """
    # Mimic the repo layout: <fake>/scripts/ + <fake>/tools/.
    fake_root = tmp_path / "relocated"
    fake_root.mkdir()
    scripts_dir = fake_root / "scripts"
    scripts_dir.mkdir()
    copy_path = scripts_dir / "sweep-skill-grants.py"
    with open(WRAPPER_PATH) as fh:
        copy_path.write_text(fh.read())

    # Fixture package + module at <fake>/tools/ (the wrapper inserts
    # <fake> on sys.path via dirname(dirname(__file__))).
    pkg = fake_root / "tools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "skill_grants.py").write_text(textwrap.dedent("""\
        # Fixture tools.skill_grants for the portability test.
        # Records its invocation on the module so the test can verify
        # the relocated wrapper bound THIS module, not the repo one.
        _last_call = None

        def sweep_expired_grants(ttl_hours=None):
            global _last_call
            _last_call = {"invoked": True, "ttl_hours": ttl_hours}
            return 2
    """))

    # Isolate sys.path/sys.modules so the fixture tools.skill_grants is
    # the one the wrapper imports. The repo's real tools.skill_grants is
    # already cached in sys.modules from earlier tests; we must drop it
    # and the tools package so the import re-resolves against the
    # relocated dir the wrapper inserts at sys.path[0].
    saved_path = list(sys.path)
    saved_tools = sys.modules.pop("tools", None)
    saved_sg = sys.modules.pop("tools.skill_grants", None)
    monkeypatch.setattr("sys.path", [str(fake_root)] + saved_path)
    bound_fixture = None
    try:
        mod_name = "sweep_skill_grants_wrapper_relocated"
        sys.modules.pop(mod_name, None)
        spec = importlib.util.spec_from_file_location(mod_name, str(copy_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = capsys.readouterr()
        # Capture the fixture module the wrapper actually bound BEFORE
        # we restore the real module cache.
        bound_fixture = sys.modules.get("tools.skill_grants")
    finally:
        # Restore the real sys.path and module cache.
        sys.path[:] = saved_path
        if saved_tools is not None:
            sys.modules["tools"] = saved_tools
        if saved_sg is not None:
            sys.modules["tools.skill_grants"] = saved_sg

    # The relocated wrapper bound the fixture and called it.
    assert bound_fixture is not None, (
        "relocated wrapper did not import a tools.skill_grants module at all"
    )
    # The fixture's __file__ must point at the relocated copy, not the repo.
    assert str(fake_root) in str(getattr(bound_fixture, "__file__", "")), (
        f"relocated wrapper bound the repo module "
        f"({getattr(bound_fixture, '__file__', 'NA')}) instead of the fixture "
        f"under {fake_root} — sys.path is not derived from __file__"
    )
    assert getattr(bound_fixture, "_last_call", None) is not None, (
        "relocated wrapper did not invoke the fixture sweep_expired_grants; "
        "sys.path is not derived from __file__ (hardcoded path regression)"
    )
    assert bound_fixture._last_call["invoked"] is True
    assert bound_fixture._last_call["ttl_hours"] == 24
    parsed = json.loads(out.out)
    assert parsed == {"expired_grants_revoked": 2, "ttl_hours": 24}
