"""Upgrading a real pre-#106742 install over its running gateway, elevated and not (#134889 W4).

Failure class: the release that introduced the unified gateway runtime (#106742, eb75627faf5)
changed who owns ``gateway.lock`` and how it is opened. A user upgrading from the release before
it has that OLD gateway running and holding ``gateway.lock`` when ``hermes update`` starts. The
update must stop it, take the checkout to the new release, start the NEW gateway on that same
lock, and leave every file the gateway needs usable by the same user without elevation:

* ``base``: the release before the merge (``d04056d609b``, its first parent), installed with
  ITS OWN ``install.ps1`` from the local origin, exactly what that user ran then.
* ``hermes update`` to this checkout's HEAD (current main plus nothing but tests).

Two journeys, one per file-level fixture, each on its own fresh profile:

* ``elevated``: an administrator console (the runner's session) installs, hosts the old gateway
  and updates. Afterwards the same user WITHOUT elevation (a UAC filtered-shape token, see
  ``_tokens.py``) must be able to read and write ``gateway.lock`` and the gateway's state, and,
  once the elevated gateway is stopped, start their own gateway on that lock, see it in
  ``hermes gateway status`` and get a reply.
* ``standard user``: the whole journey under the filtered token: install, old gateway, update,
  new gateway, status, turn.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import wait_until
from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    Machine,
    fail_with,
    new_machine,
    one_shot_turn,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

# The first parent of the #106742 merge (eb75627faf5): the last main before the unified runtime.
PRE_RUNTIME_RELEASE = os.environ.get("HERMES_E2E_W4_BASE", "d04056d609bb53530843a6c630493243b2ebdc24")
_STATUS_LINE = re.compile(r"^.*[✓✗].*[Gg]ateway.*$", re.MULTILINE)
_RESTART_FAILURE = re.compile(
    r"^.*(recovery failed|restart could not be verified|restart incomplete|not verified alive).*$", re.MULTILINE)


def _status_line(text: str) -> str:
    found = _STATUS_LINE.findall(text)
    return found[0].strip() if found else "<no status line>"


def _alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).is_running()
    except psutil.Error:
        return False


def _gateway_files(m: Machine) -> list[Path]:
    """What a gateway opens for writing at start: its home, lock, pid and state records, logs, the
    per-user host lock dir, and every top-level entry the elevated run left in the home."""
    home = m.hermes_home
    paths = [home, home / "gateway.lock", home / "gateway.pid", home / "gateway_state.json", home / "logs"]
    lock_dir = m.profile / ".local" / "state" / "hermes" / "gateway-locks"
    if lock_dir.is_dir():
        paths += [lock_dir, *sorted(lock_dir.iterdir())]
    skip = {"hermes-agent", "tools", "bin"}  # code and toolchain: read+execute (test_paths_and_acls.py)
    paths += [p for p in sorted(home.iterdir()) if p.name not in skip]
    seen, out = set(), []
    for p in paths:
        key = os.path.normcase(str(p))
        if key not in seen and p.exists():
            seen.add(key)
            out.append(p)
    return out


def _user_write_access(m: Machine) -> dict[Path, str]:
    from tests.e2e.core.windows_update._tokens import FILE_READ_WRITE, standard_user_access

    return standard_user_access(_gateway_files(m), FILE_READ_WRITE)


def _lock_held(m: Machine) -> bool:
    """True while some process holds ``gateway.lock`` (a byte-range lock or a sharing deny)."""
    import msvcrt

    path = m.hermes_home / "gateway.lock"
    try:
        fh = open(path, "r+b")
    except PermissionError:
        return True
    except OSError:
        return False
    with fh:
        fh.seek(1024 * 1024)  # gateway/status.py _WINDOWS_LOCK_OFFSET, the same in both releases
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        fh.seek(1024 * 1024)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    return False


def _machine(tmp_path_factory, srv: FakeLLMServer, label: str) -> Machine:
    from tests.e2e.core.windows_update._tokens import grant_user_profile_acl, hand_to_user

    machine = new_machine(tmp_path_factory.mktemp(label), srv.base_url, label=label, system_git=True,
                          base_ref=PRE_RUNTIME_RELEASE)
    # A real profile's DACL (the user owns their home), and the origin + gitconfig the user can
    # read: harness facts, so a standard-user verdict is about Hermes and nothing else.
    grant_user_profile_acl(machine.profile)
    hand_to_user(machine.root)
    return machine


def _token_control(m: Machine) -> str:
    from tests.e2e.core.windows_update._tokens import harness_control, probe_token_elevation

    harness_control(m.root)
    return probe_token_elevation(m.profile, m.env(as_user=True), m.logs / "00-token-probe.log")


def _wait_new_gateway(m: Machine, old_pid: int | None) -> dict:
    return m.wait_gateway_running(not_pid=old_pid)


def _pid(state) -> int:
    return int(state.get("pid") or 0) if isinstance(state, dict) else 0


def _stopped(pid: int) -> bool:
    try:
        wait_until(lambda: not _alive(pid), 90, f"gateway pid {pid} to exit", interval=0.5)
    except AssertionError:
        return False
    return True


@pytest.fixture(scope="module")
def elevated(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = _machine(tmp_path_factory, srv, "w4e")
        j = Journey(machine)
        try:
            j.step("control", lambda: _token_control(machine))
            install = j.step("install", machine.install)
            if j.ok("install"):
                j.step("install_ok", lambda: j.require("install", install.returncode == 0,
                                                        f"install.ps1 ({PRE_RUNTIME_RELEASE[:11]}) exited "
                                                        f"{install.returncode}", install))
            if j.ok("install_ok"):
                with machine.gateway_phase():
                    j.step("installed_base", machine.installed_head)
                    j.step("spawn_old", machine.spawn_gateway)
                    old = j.step("old_state", machine.wait_gateway_running)
                    j.step("old_lock_held", lambda: _lock_held(machine))
                    machine.advance()  # the new release lands on main: an update is available
                    j.step("update", machine.update)
                    j.step("new_state", lambda: _wait_new_gateway(machine, _pid(old) or None))
                    j.step("status_admin", lambda: machine.hermes("gateway", "status", label="status-admin"))
                    j.step("status_user", lambda: machine.hermes("gateway", "status", label="status-user",
                                                                  as_user=True))
                    j.step("turn_admin", lambda: one_shot_turn(machine, srv, "turn-admin"))
                    j.step("user_access", lambda: _user_write_access(machine))
                    # The elevated gateway goes; the same user, unelevated, takes the lock next.
                    new_pid = _pid(j.results.get("new_state"))
                    j.step("stop_admin", lambda: machine.hermes("gateway", "stop", label="stop-admin"))
                    j.step("admin_stopped", lambda: _stopped(new_pid))
                    j.step("spawn_user", lambda: machine.spawn_gateway(as_user=True))
                    j.step("user_state", lambda: _wait_new_gateway(machine, new_pid or None))
                    j.step("user_status", lambda: machine.hermes("gateway", "status", label="status-user-gw",
                                                                  as_user=True))
                    j.step("user_turn", lambda: one_shot_turn(machine, srv, "turn-user", as_user=True))
                    user_pid = _pid(j.results.get("user_state"))
                    j.step("stop_user", lambda: machine.hermes("gateway", "stop", label="stop-user", as_user=True))
                    j.step("user_stopped", lambda: _stopped(user_pid))
                    machine.kill_owned()
            yield j
        finally:
            machine.teardown()


@pytest.fixture(scope="module")
def standard_user(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = _machine(tmp_path_factory, srv, "w4u")
        j = Journey(machine)
        try:
            j.step("control", lambda: _token_control(machine))
            install = j.step("install", lambda: machine.install(as_user=True))
            if j.ok("install"):
                j.step("install_ok", lambda: j.require("install", install.returncode == 0,
                                                        f"install.ps1 ({PRE_RUNTIME_RELEASE[:11]}) as a standard "
                                                        f"user exited {install.returncode}", install))
            if j.ok("install_ok"):
                with machine.gateway_phase():
                    j.step("installed_base", machine.installed_head)
                    j.step("spawn_old", lambda: machine.spawn_gateway(as_user=True))
                    old = j.step("old_state", machine.wait_gateway_running)
                    j.step("old_lock_held", lambda: _lock_held(machine))
                    machine.advance()
                    j.step("update", lambda: machine.update(as_user=True))
                    j.step("new_state", lambda: _wait_new_gateway(machine, _pid(old) or None))
                    j.step("status_user", lambda: machine.hermes("gateway", "status", label="status-user",
                                                                  as_user=True))
                    j.step("user_turn", lambda: one_shot_turn(machine, srv, "turn-user", as_user=True))
                    new_pid = _pid(j.results.get("new_state"))
                    j.step("stop_user", lambda: machine.hermes("gateway", "stop", label="stop-user", as_user=True))
                    j.step("user_stopped", lambda: _stopped(new_pid))
                    machine.kill_owned()
            yield j
        finally:
            machine.teardown()


# -- shared verdicts ---------------------------------------------------------------------------


def _assert_old_gateway_held_lock(j: Journey) -> None:
    m, state = j.machine, j["old_state"]
    assert j["installed_base"] == m.head, fail_with(
        m, f"the install is not the pre-runtime release: checkout {j['installed_base']}, expected {m.head}")
    assert str(state.get("code_sha") or "")[:7] == m.head[:7] or not state.get("code_sha"), fail_with(
        m, f"the gateway before the update serves {state.get('code_sha')!r}, expected the base {m.head}")
    assert j["old_lock_held"], fail_with(
        m, f"the old gateway (pid {state.get('pid')}) is running but nothing holds gateway.lock")


def _assert_update_replaced_gateway(j: Journey) -> None:
    m, run = j.machine, j["update"]
    failure = _RESTART_FAILURE.search(run.stdout)
    assert run.returncode == 0 and failure is None and m.installed_head() == m.next, fail_with(
        m, f"hermes update over the running pre-runtime gateway failed (rc={run.returncode}, checkout "
           f"{m.installed_head()}, expected {m.next}): "
           f"{failure.group(0).strip() if failure else '<no restart message>'}", run)
    state = j["new_state"]
    sha = str(state.get("code_sha") or "")
    assert len(sha) >= 7 and m.next.startswith(sha), fail_with(
        m, f"after the update the gateway (pid {state.get('pid')}) serves code_sha={sha!r}, expected {m.next}")
    assert _pid(state) != _pid(j["old_state"]), fail_with(m, "the old gateway was never replaced")


def _assert_status_sees(j: Journey, status_key: str, state_key: str, who: str) -> None:
    m, status, state = j.machine, j[status_key], j[state_key]
    line = _status_line(status.stdout)
    assert status.returncode == 0 and line.startswith("✓") and str(_pid(state)) in line, fail_with(
        m, f"`hermes gateway status` ({who}) does not see the running gateway (pid {_pid(state)}): {line!r}",
        status)


def _assert_turn(j: Journey, key: str, who: str) -> None:
    m, turn = j.machine, j[key]
    assert turn.ok, fail_with(
        m, f"a turn ({who}) after the upgrade got no reply (reply printed={turn.reply_id in turn.run.stdout}, "
           f"prompt reached provider={turn.reached_wire})", turn.run)


# -- elevated journey ----------------------------------------------------------------------------


def test_elevated_harness_token_is_standard(elevated: Journey) -> None:
    assert elevated["control"] == "elevated=False admin=False"


def test_elevated_old_gateway_held_lock_before_update(elevated: Journey) -> None:
    _assert_old_gateway_held_lock(elevated)


def test_elevated_update_replaces_old_gateway(elevated: Journey) -> None:
    _assert_update_replaced_gateway(elevated)


def test_elevated_status_sees_new_gateway(elevated: Journey) -> None:
    _assert_status_sees(elevated, "status_admin", "new_state", "administrator console")
    _assert_status_sees(elevated, "status_user", "new_state", "same user, not elevated")


def test_elevated_turn_replies_after_update(elevated: Journey) -> None:
    _assert_turn(elevated, "turn_admin", "administrator console")


def test_elevated_upgrade_leaves_gateway_files_usable_unelevated(elevated: Journey) -> None:
    m, verdicts = elevated.machine, elevated["user_access"]
    elevated["update"]
    assert any(p.name == "gateway.lock" for p in verdicts), fail_with(
        m, f"no gateway.lock after the upgrade under {m.hermes_home}")
    broken = [f"{p}: {why}" for p, why in verdicts.items() if why]
    assert not broken, fail_with(
        m, f"after an elevated upgrade the same user, not elevated, cannot write: {'; '.join(broken)}")


def test_elevated_upgrade_then_user_gateway_takes_lock(elevated: Journey) -> None:
    m = elevated.machine
    stop = elevated["stop_admin"]
    assert stop.returncode == 0 and elevated["admin_stopped"], fail_with(
        m, f"`hermes gateway stop` left the elevated gateway (pid {_pid(elevated['new_state'])}) running", stop)
    elevated["spawn_user"]
    state = elevated["user_state"]
    sha = str(state.get("code_sha") or "")
    assert m.next.startswith(sha) and len(sha) >= 7, fail_with(
        m, f"the unelevated gateway serves {sha!r}, expected {m.next}")
    _assert_status_sees(elevated, "user_status", "user_state", "same user, not elevated")
    _assert_turn(elevated, "user_turn", "same user, not elevated")
    assert elevated["stop_user"].returncode == 0 and elevated["user_stopped"], fail_with(
        m, f"the unelevated `hermes gateway stop` left its gateway (pid {_pid(state)}) running",
        elevated["stop_user"])


# -- standard-user journey -----------------------------------------------------------------------


def test_standard_user_harness_token_is_standard(standard_user: Journey) -> None:
    assert standard_user["control"] == "elevated=False admin=False"


def test_standard_user_old_gateway_held_lock_before_update(standard_user: Journey) -> None:
    _assert_old_gateway_held_lock(standard_user)


def test_standard_user_update_replaces_old_gateway(standard_user: Journey) -> None:
    _assert_update_replaced_gateway(standard_user)


def test_standard_user_status_and_turn_after_update(standard_user: Journey) -> None:
    _assert_status_sees(standard_user, "status_user", "new_state", "standard user")
    _assert_turn(standard_user, "user_turn", "standard user")
    m = standard_user.machine
    assert standard_user["stop_user"].returncode == 0 and standard_user["user_stopped"], fail_with(
        m, f"`hermes gateway stop` left the gateway (pid {_pid(standard_user['new_state'])}) running",
        standard_user["stop_user"])
