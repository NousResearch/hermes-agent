"""The dependency sync fails after the checkout moved while an idle gateway is paused, then recovers.

The end-to-end check of RE2 migration stage S2d (the launch guard that refuses to run without the
regex engine), part of the upgrade rehearsal (stage S2c). Release N-1 (or ``HERMES_E2E_UPGRADE_BASE``),
installed by its own installer, runs one idle gateway by hand. Upstream publishes HEAD plus a fault
seam in ``pm/client.py``: while ``~/.e2e-rehearsal-break-deps`` exists, every dependency sync of the
new tree raises, the way a sync fails without network. The old updater pauses the gateway, moves the
checkout, its completion child's sync fails, so the update owes ``dependencies``. The paused gateway
then either resumes on the previous dependencies, or (every supported start so far) the updater says it
stays stopped until the next launch (``DEFERRED``) and none may run; a gateway left stopped without that
line fails the cell before the recovery could hide it. The network comes back (the toggle goes) and the
next plain ``hermes update`` must finish the sync and bring the gateway back on the new code without
any other command.

What the resumed gateway does depends on the guard, probed at run time by running the tree's own guard
without the engine (``guard_armed``), one parametrized cell per state (the other one skips):

* ``guard-armed`` (from the first ``hermes_re`` swap, S6a PR 1, on): the new tree on the previous
  dependencies has no google-re2, so the resumed gateway exits 1 with ``hermes: cannot start:`` in its
  log. That line is the only place ``re2`` and a manual step may appear.
* ``guard-off`` (before S2d lands, or while ``REGEX_ENGINE_REQUIRED`` is False): no refusal; the
  resumed gateway runs the new code on the previous dependencies, and no line anywhere names the
  regex engine.

Lazy installs stay on (``HERMES_DISABLE_LAZY_INSTALLS`` is dropped), as on a user's machine: each
launch's own dependency sync is part of both the failure and the recovery.
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade import _logs as L
from tests.e2e.core.upgrade.handoff import _handoff as X
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.skipif(not H.BWRAP_OK, reason="bubblewrap sandbox unavailable"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
    pytest.mark.live_system_guard_bypass,  # one bwrap sandbox per cell; killing it reaps everything
]

BREAK_DEPS = ".e2e-rehearsal-break-deps"
_MARK = "# e2e (test_rehearsal_engine_guard): failing dependency sync toggle"
_SEAM = f"""    {_MARK}
    if __import__("os").path.exists(__import__("os").path.join(__import__("os").path.expanduser("~"), "{BREAK_DEPS}")):
        raise RuntimeError("e2e: dependency sync fails on purpose ({BREAK_DEPS})")
"""
DEPS_OWED = re.compile(r"⚠ Update follow-up 'dependencies' did not finish: .*dependencies not installed yet")
REFUSED = "hermes: cannot start:"
# The updater's deferral (``update_cmd_windows._resume_unless_torn``, every supported start): the paused set
# stays stopped, owed to the next launch, because the dependencies are not current for the new code.
DEFERRED = "Paused gateway(s) left stopped: dependencies are not current for the updated code yet"
REPLY = "engine guard reply"


# Runs the launch guard of the checkout under test (argv[1]) the way ``hermes_bootstrap`` does for a
# gateway launch, with ``re2`` unimportable. Exit status: 20 refused (armed), 10 ran on (off or no guard
# yet); anything else (a traceback, a guard imported from another tree) is a broken premise.
_GUARD_PROBE = """
import importlib.abc, sys
from pathlib import Path

class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] == "re2":
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by the e2e guard probe)", name=name)
        return None

root = Path(sys.argv[1]).resolve()
sys.meta_path.insert(0, NoEngine())
sys.path.insert(0, str(root))
import hermes_cli.venv_sync as vs
if Path(vs.__file__).resolve().parent != root / "hermes_cli":
    sys.exit(f"premise: imported {vs.__file__}, not the checkout under test")
guard = getattr(vs, "guard_regex_engine", None)
if guard is None:
    sys.exit(10)
try:
    guard(root, ["gateway", "run"])
except SystemExit as exc:
    sys.exit(20 if exc.code == 1 else 30)
sys.exit(10)
"""


@functools.cache
def guard_armed() -> bool:
    """Whether the tree under test refuses a gateway launch without the regex engine (S2d adds the guard
    inert, S6a PR 1 arms it): its own guard, run in a child that cannot import ``re2``."""
    cp = subprocess.run([sys.executable, "-c", _GUARD_PROBE, str(H.WORKTREE)], cwd=str(H.WORKTREE),
                        capture_output=True, text=True, encoding="utf-8", timeout=120, check=False)
    assert cp.returncode in (10, 20), f"premise: the launch-guard probe broke\n{H.describe(cp)}"
    if cp.returncode == 20:
        assert REFUSED in cp.stderr, f"premise: the guard exited 1 without its refusal\n{H.describe(cp)}"
    return cp.returncode == 20


def _publish_seamed_head(inst: X.Install) -> str:
    """HEAD plus the toggle in ``pm.client.ensure_tools_for_sync`` (every sync of the new tree runs it)."""
    X.publish_target(inst)
    text = I.git("show", f"{X.refs().head}:pm/client.py", cwd=H.WORKTREE) + "\n"
    m = re.search(r'^def ensure_tools_for_sync\(\) -> None:\n    """.*?"""\n', text, re.DOTALL | re.MULTILINE)
    assert m, "premise: pm/client.py has no ensure_tools_for_sync() with a docstring to inject after"
    text = text[:m.end()] + _SEAM + text[m.end():]
    inst.target = I.publish_commit(inst.origin, inst.root, "release: e2e rehearsal failing dependency sync",
                                   {"pm/client.py": text})
    return inst.target


@pytest.mark.parametrize("armed", [pytest.param(True, id="guard-armed"), pytest.param(False, id="guard-off")])
def test_failed_sync_after_the_move_then_the_next_update_recovers(root, armed):
    if armed != guard_armed():
        pytest.skip(f"the tree under test has the launch guard {'off' if armed else 'armed'} "
                    "(S2d adds it inert; the first hermes_re swap, S6a PR 1, arms it)")
    if not X.refs().base:
        pytest.skip("no release tag reachable (shallow checkout)")
    with FakeLLMServer(default_text=REPLY) as srv, \
            X.cell("n1", root, srv.base_url, drop_env=("HERMES_DISABLE_LAZY_INSTALLS",)) as inst:
        _cell(inst, armed)


def _log_offsets(inst: X.Install) -> dict:
    return L.log_offsets([inst.hermes_home], inst.logs)


def _log_text_since(inst: X.Install, offsets: dict) -> str:
    return L.log_text_since(offsets, [inst.hermes_home], inst.logs)


def _engine_lines(text: str, armed: bool) -> list[str]:
    """Regex-engine lines that are failures: with the guard armed its own refusal is the expected one."""
    return [line for line in I.engine_error_lines(text) if not (armed and REFUSED in line)]


def _cell(inst: X.Install, armed: bool) -> None:
    before = X.start_gateway(inst)
    target = _publish_seamed_head(inst)
    toggle = inst.home / BREAK_DEPS
    toggle.write_text("on\n", encoding="utf-8")
    offsets = _log_offsets(inst)
    up = inst.update()
    out = up.stdout + up.stderr
    # The updater either resumes the paused gateway now or says it leaves it stopped until the next launch.
    deferred = DEFERRED in out
    resumed = None
    if up.returncode == 0 and not deferred and armed:
        resumed = X.wait_for(lambda: REFUSED in _log_text_since(inst, offsets), timeout=300, interval=1.0,
                             what=f"the resumed gateway's {REFUSED!r}")
    elif up.returncode == 0 and not deferred:
        resumed = X.settle_gateway(inst, before["pid"], timeout=300)
    logs = _log_text_since(inst, offsets)
    diag = inst.diagnostics(up) + f"\n--- logs since the update ---\n{logs[-8000:]}"
    assert inst.sha() == target, f"premise: the update did not move the checkout\n{diag}"
    assert DEPS_OWED.search(out), f"premise: the dependency sync did not fail after the move\n{diag}"
    assert up.returncode == 0, f"a dependency sync that failed after the move failed the update\n{diag}"
    bad = _engine_lines(f"{out}\n{logs}", armed)
    assert not bad, "a line names the regex engine or a missing module:\n" + "\n".join(bad[:20]) + f"\n{diag}"
    # Checked before the recovery: the second update restarts the gateway and would hide a first update
    # that left it stopped without saying so.
    if deferred:
        assert X.identify(inst.hermes_home) is None, (
            f"the update says the paused gateway stays stopped, but a gateway runs\n{diag}")
    elif armed:
        assert resumed, f"the resumed gateway did not refuse to start without the regex engine\n{diag}"
        assert X.identify(inst.hermes_home) is None, f"a gateway runs without the regex engine\n{diag}"
    else:
        assert resumed is not None, (f"the update neither resumed the paused gateway on {target[:12]} nor "
                                     f"said it leaves it stopped ({DEFERRED!r})\n{diag}")

    # The network is back: the next plain `hermes update` finishes the sync and the gateway serves the
    # new code again, with no other command.
    toggle.unlink()
    offsets = _log_offsets(inst)
    again = inst.update()
    out = again.stdout + again.stderr
    ident = X.settle_gateway(inst, before["pid"], timeout=300)
    gateways = X.gateway_pids(inst)
    logs = _log_text_since(inst, offsets)
    diag = inst.diagnostics(again) + f"\n--- logs since the second update ---\n{logs[-8000:]}"
    assert again.returncode == 0 and X.TRACEBACK not in out, f"the second update failed\n{diag}"
    bad = I.engine_error_lines(f"{out}\n{logs}")
    assert not bad, "a line names the regex engine or a missing module:\n" + "\n".join(bad[:20]) + f"\n{diag}"
    manual = I.manual_step_lines(out)
    assert not manual, "the second update still asks for a manual step:\n" + "\n".join(manual) + f"\n{diag}"
    assert ident is not None, f"no gateway serves {target[:12]} after the second update\n{diag}"
    assert [p["pid"] for p in gateways] == [ident["pid"]], f"expected exactly one gateway: {gateways}\n{diag}"
    status, body = X.chat(inst.port, "hello after the recovery")
    assert status == 200 and X.reply_text(body) == REPLY, f"no turn after the recovery: {body}\n{diag}"
