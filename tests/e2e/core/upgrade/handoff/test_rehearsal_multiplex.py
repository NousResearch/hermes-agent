"""An old release whose second profile runs its own gateway, updated to HEAD.

The upgrade rehearsal's multi-profile cell (RE2 migration, stage S2c). Starting from release N-1 (or
``HERMES_E2E_UPGRADE_BASE``), installed by its own installer, the user created a second profile whose
own (standalone) gateway runs by hand and idle. ``hermes update`` must fold it into one multiplexed
default gateway on its own (``gateway_migrate.maybe_auto_migrate_after_update``, run by the NEW tree's
completion child on the NEW dependencies), leave nothing for the user to do, and trip over the regex
engine nowhere.

The fleet the supported releases (v0.21.6 to v0.23.3) let a sandbox build without a service manager:
a NEW per-profile gateway needs ``gateway run --force`` ("does not get a gateway of its own"), and a
default gateway started beside it cannot start (it reserves every profile, ``work`` included);
``gateway.standalone: true`` is no way around it, the migration leaves such a profile alone. So the
secondary runs forced and the default profile has no gateway: the shape those releases describe as
"Another profile's standalone gateway owns this host ... Fold every profile onto one gateway".
"""

from __future__ import annotations

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

WORK = "work"
WORK_KEY = "e2e-rehearsal-work-api-server-key"  # its own key: a shared one blocks the fold (duplicate credential)
REPLY = "multiplex reply"
MIGRATING = "Migrating per-profile gateways onto one multiplexed default gateway"
MIGRATED = "✓ Migrated: the default gateway now serves"


def _start_work_gateway(inst: X.Install, port: int) -> None:
    inst.spawn("gateway-work", "-p", WORK, "gateway", "run", "--force")
    try:
        X.wait_for(lambda: X.http("GET", f"http://127.0.0.1:{port}/health/detailed", key=WORK_KEY, timeout=5)[0] == 200,
                   timeout=240, what=f"the {WORK} gateway's API server")
    except AssertionError as exc:
        raise AssertionError(f"premise: the {WORK} gateway never came up: {exc}\n{inst.diagnostics()}") from None


def test_update_from_old_release_folds_profile_gateways_without_manual_steps(root):
    if not X.refs().base:
        pytest.skip("no release tag reachable (shallow checkout)")
    with FakeLLMServer(default_text=REPLY) as srv:
        inst = X.stage("n1", root / "n1")
        inst.port = X.free_port()
        X.write_config(inst.hermes_home, srv.base_url, port=inst.port)
        X.start_host(inst)
        leaked: list[str] = []
        try:
            _fold_cell(inst, srv.base_url)
        finally:
            leaked = inst.close()
        assert not leaked, f"processes outlived the sandbox: {leaked}"


def _fold_cell(inst: X.Install, provider_url: str) -> None:
    cp = inst.cli("profile", "create", WORK, "--no-alias", timeout=300)
    assert cp.returncode == 0, "premise: profile create failed\n" + H.describe(cp)
    work_home = inst.hermes_home / "profiles" / WORK
    work_port = X.free_port()
    X.write_config(work_home, provider_url, port=work_port)
    (work_home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}\nAPI_SERVER_KEY={WORK_KEY}\n", encoding="utf-8")

    _start_work_gateway(inst, work_port)
    standalone = X.gateway_pids(inst)
    assert len(standalone) == 1, f"premise: only {WORK}'s own gateway runs: {standalone}\n{inst.diagnostics()}"

    X.publish_target(inst)
    offsets = L.log_offsets([inst.hermes_home], inst.logs)
    up = inst.update()
    out = up.stdout + up.stderr
    ident = X.settle_gateway(inst, standalone[0]["pid"], timeout=300)
    gateways = X.gateway_pids(inst)
    work_chat = X.http("POST", f"http://127.0.0.1:{inst.port}/p/{WORK}/v1/chat/completions",
                       {"model": "hermes-agent", "messages": [{"role": "user", "content": "hello work"}]},
                       key=WORK_KEY, timeout=180)
    diag = inst.diagnostics(up) + f"\n--- gateways after ---\n{gateways}\n--- /p/{WORK} chat ---\n{work_chat}"

    assert up.returncode == 0 and X.TRACEBACK not in out, f"the update failed\n{diag}"
    assert inst.sha() == inst.target, f"the update did not move the checkout\n{diag}"
    bad = I.engine_error_lines(f"{out}\n{L.log_text_since(offsets, [inst.hermes_home], inst.logs)}")
    assert not bad, "the update or a log line written since names the regex engine or a missing module:\n" + \
        "\n".join(bad[:20]) + f"\n{diag}"
    assert MIGRATING in out, f"the update never ran the multiplex migration\n{diag}"
    assert MIGRATED in out, f"the migration did not confirm both profiles served\n{diag}"
    manual = I.manual_step_lines(out)
    assert not manual, "the update asks for a manual step:\n" + "\n".join(manual) + f"\n{diag}"
    assert ident is not None, f"no default gateway serves {inst.target[:12]} after the update\n{diag}"
    # The migration starts the gateway under the `stderr_timestamp` wrapper: one process tree whose root is
    # the wrapper and whose child is the gateway that answers `identify`.
    assert len(gateways) == 1, f"expected one multiplexed gateway: {gateways}\n{diag}"
    # gateway_pids drops a gateway whose parent is a gateway: only the old process's death shows that the
    # standalone gateway did not survive under the multiplexed tree.
    assert not inst.host.alive(standalone[0]["pid"]), f"the {WORK} standalone gateway still runs\n{diag}"
    parents = {p["pid"]: p["ppid"] for p in inst.host.procs()}
    assert gateways[0]["pid"] in (ident["pid"], parents.get(ident["pid"])), (
        f"the gateway that answers identify ({ident['pid']}) is not the one running: {gateways}\n{diag}")
    status, body = X.chat(inst.port, "hello default")
    assert status == 200 and X.reply_text(body) == REPLY, f"no turn on the default profile: {body}\n{diag}"
    status, body = work_chat
    assert status == 200 and X.reply_text(body) == REPLY, f"the {WORK} profile is not served by the default gateway\n{diag}"
