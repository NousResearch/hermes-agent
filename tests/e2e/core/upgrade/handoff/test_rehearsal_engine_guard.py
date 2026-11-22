"""The dependency sync fails after the checkout moved while an idle gateway is paused, then recovers.

The end-to-end check of RE2 migration stage S2d (the launch guard that refuses to run without the
regex engine), part of the upgrade rehearsal (stage S2c). Release N-1 (or ``HERMES_E2E_UPGRADE_BASE``),
installed by its own installer, runs one idle gateway by hand. Upstream publishes HEAD plus a fault
seam in ``pm/client.py``: while ``~/.e2e-rehearsal-break-deps`` exists, every dependency sync of the
new tree raises, the way a sync fails without network. The old updater pauses the gateway, moves the
checkout, its completion child's sync fails, so the update owes ``dependencies`` and the OLD process
resumes the paused gateway itself through new-tree modules (``update_cmd.py:833-841@v0.23.0``). The
network comes back (the toggle goes) and the next plain ``hermes update`` must finish the sync and
bring the gateway back on the new code without any other command.

What the resumed gateway does depends on the guard, read at run time from the tree under test
(``guard_armed``), one parametrized cell per state (the other one skips):

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

import re

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
REPLY = "engine guard reply"


def guard_armed() -> bool:
    """Whether the published tree (HEAD, not whatever this process imports) refuses a launch without the
    regex engine (S2d, armed by S6a PR 1)."""
    text = I.git("show", f"{X.refs().head}:hermes_cli/venv_sync.py", cwd=H.WORKTREE)
    return "def guard_regex_engine" in text and "REGEX_ENGINE_REQUIRED = False" not in text


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
    with FakeLLMServer(default_text=REPLY) as srv:
        inst = X.stage("n1", root / "n1")
        inst.env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
        inst.port = X.free_port()
        X.write_config(inst.hermes_home, srv.base_url, port=inst.port)
        X.start_host(inst)
        leaked: list[str] = []
        try:
            _cell(inst, armed)
        finally:
            leaked = inst.close()
        assert not leaked, f"processes outlived the sandbox: {leaked}"


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
    resumed = None
    if armed:
        resumed = X.wait_for(lambda: REFUSED in _log_text_since(inst, offsets), timeout=300, interval=1.0,
                             what=f"the resumed gateway's {REFUSED!r}") if up.returncode == 0 else None
    else:
        resumed = X.settle_gateway(inst, before["pid"], timeout=300)
    logs = _log_text_since(inst, offsets)
    diag = inst.diagnostics(up) + f"\n--- logs since the update ---\n{logs[-8000:]}"
    assert inst.sha() == target, f"premise: the update did not move the checkout\n{diag}"
    assert DEPS_OWED.search(out), f"premise: the dependency sync did not fail after the move\n{diag}"
    assert up.returncode == 0, f"a dependency sync that failed after the move failed the update\n{diag}"
    bad = _engine_lines(f"{out}\n{logs}", armed)
    assert not bad, "a line names the regex engine or a missing module:\n" + "\n".join(bad[:20]) + f"\n{diag}"
    if armed:
        assert resumed, f"the resumed gateway did not refuse to start without the regex engine\n{diag}"
        assert X.identify(inst.hermes_home) is None, f"a gateway runs without the regex engine\n{diag}"

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
