"""A BOOTSTRAP-LAUNCHED multiplexer is the ONE gateway the default home asks about.

``hermes update`` / the source installer publish ``<python> -I -c '<bootstrap>' gateway run``
(``hermes_cli._launchers``: ``_launcher_script`` + ``_mint_shell_launcher``), so on every
PM-managed install the live host gateway's own argv is an interpreter running INLINE SOURCE.
Since the #107002 fix (#121635) the canonical identity matchers answer ``None`` for that shape
on purpose -- everything after ``-c`` is data a program may spawn LATER -- and
``_record_matches_live_gateway_pid`` consulted that answer BEFORE the host-multiplexer proof
that answers correctly. The result: a healthy, serving gateway reads as absent from every
default-home consumer (``live_default_gateway_pid`` -> ``recorded_served_profiles`` ->
``hermes gateway migrate --multiplex`` never confirms, and ``gateway.pid``/``gateway.lock`` are
unlinked from under the live process).

The argv under test is REAL: read off ``/proc`` from a process actually spawned with the published
launcher's own bootstrap source -- the premise of the defect, executed rather than asserted from a
string literal. The liveness ladder, the host rendezvous record and the served set are all the real
code, with this process playing the gateway as the converged-host fixture does; only the owner's
``identify`` answer over its control socket (the wire) is faked.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import hermes_constants

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def published_launcher_argv() -> Iterator[str]:
    """A real ``/proc`` command line from a process launched exactly as ``hermes`` publishes it.

    The bootstrap source is the install's own, so the argv is the shipped one and not a
    hand-written lookalike; only the entry point is swapped for a sleep.
    """
    from hermes_cli._launchers import _launcher_script

    script = (
        _launcher_script("hermes", _REPO_ROOT, None) + "\nimport time; time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", script, "gateway", "run"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        live = ""
        while time.monotonic() < deadline and not live:
            try:
                live = (
                    Path(f"/proc/{proc.pid}/cmdline")
                    .read_bytes()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", errors="ignore")
                    .strip()
                )
            except OSError:
                live = ""
            if not live:
                time.sleep(0.1)
        assert live, (
            "the launcher-shaped stand-in never published a readable command line"
        )
        yield live
    finally:
        proc.kill()
        proc.wait(timeout=10)


#: The spawned stand-in carries a real ``gateway run`` argv on purpose -- that argv IS the defect --
#: and is spawned and reaped by ``published_launcher_argv``, so it never outlives the file. The
#: premise is read off ``/proc`` and the shape is the POSIX shell launcher: Windows has the same
#: inline-source defect through its ``.cmd`` launcher, but proving it there belongs to the Windows
#: live suite, not a ``/proc`` read this file would fail on.
pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.platforms("posix"),
]


@pytest.fixture
def host_home(tmp_path, monkeypatch):
    root = tmp_path / "home" / ".hermes"
    for sub in ("profiles/coder", "profiles/ops"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(
        hermes_constants, "_default_hermes_root_memo", None, raising=False
    )
    assert str(hermes_constants.get_default_hermes_root()).startswith(str(tmp_path))
    return root


@pytest.fixture
def bootstrap_host(host_home, published_launcher_argv, monkeypatch):
    """default + coder + ops served by ONE live host gateway (this process) whose live argv is the
    published launcher's."""
    from gateway import host_attach, host_rendezvous as hr
    from gateway import status as status_mod

    served = ["default", "coder", "ops"]
    pid = os.getpid()
    start = status_mod._get_process_start_time(pid)
    # Premise of the defect, asserted on the real argv so this file cannot silently stop exercising it.
    assert status_mod.command_line_runs_inline_source([
        t.strip("\"'").replace("\\", "/")
        for t in shlex.split(published_launcher_argv, posix=False)
    ]), "the published launcher is expected to be an inline-source argv"
    assert (
        status_mod.looks_like_gateway_runtime_command_line(published_launcher_argv)
        is False
    )

    # What a live multiplexer leaves on disk: its own pid file + lock + runtime record in the launch
    # home, and a runtime record stamped into EVERY served profile's home.
    identity = {
        "pid": pid,
        "kind": "hermes-gateway",
        "start_time": start,
        "hermes_home": str(host_home),
    }
    (host_home / "gateway.pid").write_text(json.dumps(identity), encoding="utf-8")
    (host_home / "gateway.lock").write_text(json.dumps(identity), encoding="utf-8")
    (host_home / "gateway_state.json").write_text(
        json.dumps({
            "pid": pid,
            "hermes_home": str(host_home),
            "gateway_state": "running",
            "served_profiles": served,
        }),
        encoding="utf-8",
    )
    for name in served[1:]:
        (host_home / "profiles" / name / "gateway_state.json").write_text(
            json.dumps({
                "pid": pid,
                "hermes_home": str(host_home / "profiles" / name),
                "gateway_state": "running",
            }),
            encoding="utf-8",
        )

    # This process IS the gateway and its live command line is the launcher's.
    monkeypatch.setattr(
        status_mod, "_read_process_cmdline", lambda _pid: published_launcher_argv
    )
    # The WIRE only: the owner's control socket answer. Everything that reads it is real.
    # The signature matches ``identify_gateway`` so the probe's real ``timeout=`` reaches it.
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda home, timeout=None: {
            "pid": pid,
            "hermes_home": str(host_home),
            "served_profiles": served,
        },
    )
    hr.publish_record(hr.ROLE_GATEWAY, profiles=tuple(served), home=str(host_home))
    host_attach.invalidate_host_gateway_cache()
    status_mod._clear_running_pid_cache()

    yield pid, served
    hr.clear_record(hr.ROLE_GATEWAY)
    host_attach.invalidate_host_gateway_cache()
    status_mod._clear_running_pid_cache()


def test_the_live_gateway_reports_its_own_pid_for_the_home_it_serves(
    host_home, bootstrap_host
):
    """The argv check's refusal must not veto the host-multiplexer proof: the issue's table, row one."""
    pid, _ = bootstrap_host

    from hermes_cli.gateway_multiplex_served import live_default_gateway_pid

    assert live_default_gateway_pid() == pid


def test_the_served_roster_is_readable_so_migration_confirmation_can_land(
    host_home, bootstrap_host
):
    """The consumer surface the operator sees: ``served_profiles`` is gated on that PID.

    ``recorded_served_profiles`` short-circuits to ``None`` when the identity check refuses the
    process, which is what makes ``hermes gateway migrate --multiplex`` print "has not confirmed
    serving" for every profile and leave ``gateway_migration.json`` on disk forever.
    """
    _, served = bootstrap_host

    from hermes_cli.gateway_multiplex_served import (
        multiplexer_served_secondaries,
        recorded_served_profiles,
        served_profile_ingress_urls,
    )

    assert recorded_served_profiles(host_home) == served
    assert multiplexer_served_secondaries() == ["coder", "ops"]
    # Ingress URLs are what an operator pastes into a vendor console; they were unreported too.
    assert callable(served_profile_ingress_urls)


def test_a_second_profile_on_the_same_host_reports_that_gateway_as_live(
    host_home, bootstrap_host
):
    """The same false negative on the per-profile rung, which is what the migration plan reads."""
    pid, _ = bootstrap_host

    from gateway import status as status_mod

    for name in ("coder", "ops"):
        assert (
            status_mod.live_gateway_pid_for_home(host_home / "profiles" / name) == pid
        )


def test_the_unscoped_pid_query_keeps_a_live_gateways_identity_files(
    host_home, bootstrap_host
):
    """The damage, not just the reporting gap: the same false negative fed the poison-file path.

    ``get_running_pid(cleanup_stale=True)`` is what ``hermes gateway status`` / ``profile list``
    call; unable to adopt the record it force-unlinks ``gateway.pid`` and ``gateway.lock`` while
    the process still holds the lock on the unlinked inode.
    """
    pid, _ = bootstrap_host
    pid_path, lock_path = host_home / "gateway.pid", host_home / "gateway.lock"

    from gateway import status as status_mod

    assert status_mod.get_running_pid() == pid
    assert pid_path.exists() and lock_path.exists(), (
        "a live gateway's identity files were unlinked"
    )


def test_a_restart_watchers_borrowed_argv_still_never_reads_as_this_homes_gateway(
    host_home, monkeypatch
):
    """The #107002 property must survive the rescue: a rescue, not a weakening.

    The detached restart watcher is spawned as ``python -c <watcher source> <old_pid> <python> -m
    hermes_cli.main gateway run`` -- it carries a real gateway argv in its TAIL. With no live host
    multiplexer behind that PID, the record must still refuse it.
    """
    from gateway import status as status_mod

    watcher_pid = 4242
    watcher_cmd = (
        f"{sys.executable} -c 'import os, subprocess, time; time.sleep(120)' {watcher_pid} "
        f"{sys.executable} -m hermes_cli.main gateway run"
    )
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: 1000)
    monkeypatch.setattr(status_mod, "_read_process_cmdline", lambda pid: watcher_cmd)

    # No host record in this test: the proof has nothing to answer with, so the argv check stands alone.
    assert status_mod._host_gateway_serves_home(watcher_pid, host_home) is False
    assert (
        status_mod._record_matches_live_gateway_pid(
            {
                "kind": "hermes-gateway",
                "argv": [sys.executable, "-c", "src", "gateway", "run"],
            },
            watcher_pid,
            expected_home=host_home,
        )
        is False
    )


def test_a_bootstrap_launched_owner_keeps_its_scoped_token_locks(
    host_home, bootstrap_host, published_launcher_argv, monkeypatch
):
    """The same false negative also took the owner's scoped token locks away from it.

    ``_scoped_lock_record_is_stale`` judged a live bootstrap-launched gateway dead on a READABLE
    command line, so ``acquire_scoped_lock`` treated a token it is actively serving as free and
    let a second process take it over (a duplicate Telegram bot on one token).
    """
    pid, _ = bootstrap_host

    from gateway import status as status_mod

    monkeypatch.setattr(status_mod, "_pid_exists", lambda _p: True)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda _p: 1000)
    monkeypatch.setattr(status_mod, "_process_is_stopped", lambda _p: False)
    record = {
        "pid": pid,
        "kind": "hermes-gateway",
        "start_time": 1000,
        "hermes_home": str(host_home),
        "scope": "platform:telegram",
        "identity_hash": "deadbeef",
    }
    assert status_mod._scoped_lock_record_is_stale(record, pid) is False
    # Same record, a PID that is not the recorded host owner: still stale, lock still reclaimable.
    assert status_mod._scoped_lock_record_is_stale(dict(record, pid=pid + 1), pid + 1) is True


def test_the_rescue_protects_the_owner_and_nobody_else(
    host_home, bootstrap_host, published_launcher_argv, monkeypatch
):
    """Over-blocking would be its own defect: the proof keys on the live owner's identity, not argv.

    A foreign live PID carrying the very same inline-source argv must still lose the lock, or a
    squatter's lock could never be reclaimed. The real host record published by ``bootstrap_host``
    answers; nothing here stubs the proof.
    """
    owner_pid, _ = bootstrap_host
    from gateway import status as status_mod

    monkeypatch.setattr(status_mod, "_pid_exists", lambda _p: True)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda _p: 1000)
    monkeypatch.setattr(status_mod, "_process_is_stopped", lambda _p: False)
    # Same argv for every PID, so only the owner's identity can decide.
    monkeypatch.setattr(status_mod, "_read_process_cmdline", lambda _p: published_launcher_argv)
    record = {"kind": "hermes-gateway", "start_time": 1000, "hermes_home": str(host_home)}

    # The recorded owner, serving this home: its lock is NOT stale.
    assert status_mod._scoped_lock_record_is_stale(dict(record, pid=owner_pid), owner_pid) is False
    # A foreign live process wearing the very same argv still loses its lock.
    foreign = owner_pid + 1
    assert status_mod._scoped_lock_record_is_stale(dict(record, pid=foreign), foreign) is True
    # And the owner judged against a profile it does NOT serve: not its gateway, so stale.
    assert (
        status_mod._scoped_lock_record_is_stale(
            dict(record, pid=owner_pid, hermes_home=str(host_home / "profiles" / "stranger")),
            owner_pid,
        )
        is True
    )


def test_a_second_install_on_the_host_is_not_vouched_for_by_our_owner(
    host_home, bootstrap_host, monkeypatch
):
    """The served set is keyed by PROFILE NAME, so two installs collide on ``default``.

    A home outside any ``<root>/profiles/`` dir normalizes to ``default`` -- the name this owner
    serves. Without a home check, one installation's owner would vouch for the OTHER's gateway
    and hold its token locks unreclaimable. The owner's own root and its served profiles must
    still pass: the fix cannot cost the multiplexer its secondaries.
    """
    owner_pid, _ = bootstrap_host
    from gateway import status as status_mod

    other_install = host_home.parent / "other-installation" / ".hermes"
    other_install.mkdir(parents=True, exist_ok=True)

    assert status_mod._host_gateway_serves_home(owner_pid, host_home) is True
    # A served secondary lives under the owner's own root and must still be served.
    assert status_mod._host_gateway_serves_home(owner_pid, host_home / "profiles" / "coder") is True
    # A second install resolves to the same profile name, and must not be vouched for.
    assert status_mod._host_gateway_serves_home(owner_pid, other_install) is False


def test_replace_can_still_reclaim_a_bootstrap_launched_owners_lock(
    host_home, published_launcher_argv, monkeypatch
):
    """A lock nothing can take over is a wedge, so ``--replace`` must still work on this owner.

    Protecting the owner from a plain second start (above) is only correct while a sanctioned
    takeover remains: ``take_over_scoped_lock_holder`` resolves the owner through
    ``_validated_scoped_lock_gateway_owner``, which had the same argv blindness, and returned
    ``None`` for a bootstrap-launched owner — so its scoped token locks became unreclaimable.

    The owner here is a FOREIGN live pid (the resolver refuses ``os.getpid()`` outright) and the
    host record is published for it, so the proof is the real one.
    """
    import json

    import gateway.host_rendezvous as hr
    from gateway import host_attach
    from gateway import status as status_mod

    owner_pid = os.getppid()  # real, live, and not this process
    start = 1000
    monkeypatch.setattr(status_mod, "_pid_exists", lambda _p: True)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda _p: start)
    monkeypatch.setattr(status_mod, "_process_is_stopped", lambda _p: False)
    monkeypatch.setattr(status_mod, "_read_process_cmdline", lambda _p: published_launcher_argv)

    # Publish the host record for a pid that is not us, through the real record type.
    record = hr.HostRecord(
        role=hr.ROLE_GATEWAY, pid=owner_pid, create_time=hr.process_create_time(owner_pid),
        host="", port=None, protocol_version=hr.HOST_PROTOCOL_VERSION,
        token_fingerprint=hr.token_fingerprint(""), profiles=("default",),
        updated_at="2026-09-25T00:00:00+00:00", home=str(host_home),
    )
    hr.ensure_host_state_dir()
    hr.atomic_json_write(hr.record_path(hr.ROLE_GATEWAY), record.to_json(), mode=0o600)
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda home, timeout=None: {
        "pid": owner_pid, "hermes_home": str(host_home), "served_profiles": ["default"]})
    host_attach.invalidate_host_gateway_cache()
    try:
        assert hr.read_record(hr.ROLE_GATEWAY) is not None, "the record must be a real one"
        # What the real launcher persists: argv[0] is '-c'.
        boot_argv = ["-c", "gateway", "run"]
        (host_home / "gateway.pid").write_text(json.dumps({
            "pid": owner_pid, "kind": "hermes-gateway", "start_time": start,
            "hermes_home": str(host_home), "argv": boot_argv}))
        lock = {
            "pid": owner_pid, "kind": "hermes-gateway", "start_time": start,
            "hermes_home": str(host_home), "argv": boot_argv,
            "scope": "platform:telegram", "identity_hash": "deadbeef",
        }
        assert status_mod._record_looks_like_gateway(lock) is False, "premise: argv alone refuses"
        resolved = status_mod._validated_scoped_lock_gateway_owner(lock)
        assert resolved is not None, "--replace could not reclaim this owner's lock"
        assert resolved[0] == owner_pid
    finally:
        hr.record_path(hr.ROLE_GATEWAY).unlink(missing_ok=True)
        host_attach.invalidate_host_gateway_cache()
