"""C7 two-tenant canary: ONE Desktop backend (``hermes serve``) serving three profiles over /api/ws.

Class: multiplex / tenancy isolation (issue_classes.md C7). Users hit it as "a secondary profile's
chat showed/stored the default profile's model", "a key or setting I saved for profile B landed in
profile A's .env / config.yaml", "the Desktop cron ticker ran B's job with A's env", "creating a
profile restarted my backend" (#85669, #101719, #105396, #107422, #107692, #111151).

Harness: the real app backend, ``hermes serve --port 0`` with ``HERMES_DESKTOP=1`` (which also runs
the in-process cron ticker for every served profile), driven over the same ``/api/ws`` JSON-RPC
socket the Desktop uses: ``session.create {profile}``, ``prompt.submit``, ``config.set``,
``model.save_key``, ``profiles.create``, ``session.resume``. Each profile owns its own loopback
provider and distinct canaries (see ``_helpers.Tenant``); every turn runs ``env | sort; pwd`` in
the terminal tool so the provider receives a snapshot taken inside the tool subprocess.

Phases (same invariant after each): interleaved turns + a cron tick in every profile ->
session-bound and profile-bound settings writes via RPC -> profile create/delete next to the live
backend (PID must not change) -> restart + resume + a second cron tick. Invariant: providers see only
their own tenant's key/model/canaries, tool subprocesses see only their own cwd/env, each write
changes only its own profile's files, no file in any profile home carries another tenant's canary,
and no RPC reply/event for a session carries another tenant's canary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


from . import _helpers as H

NAMES = ("default", "alpha", "beta")


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("c7-serve")
    home = root / "home"
    tenants = H.make_tenants(root, NAMES)
    for t in tenants.values():
        assert t.srv is not None
        t.srv.start()
    home.mkdir(parents=True)
    H.assert_profiles_root_under(root, home)  # HOME-anchored profile root: prove it is sandboxed first
    for t in tenants.values():
        H.write_tenant_home(t)
    backends: list[H.ServeBackend] = []
    try:
        yield root, tenants, backends
    finally:
        for b in backends:
            b.close()
            assert not H.pid_alive(b.proc.pid), f"serve pid {b.proc.pid} survived teardown"
        for t in tenants.values():
            assert t.srv is not None
            t.srv.stop()


def _profile_param(name: str) -> dict[str, str]:
    return {} if name == "default" else {"profile": name}


def _files(tenants: dict[str, H.Tenant]) -> dict[str, bytes]:
    return {f"{t.name}/{n}": (t.home / n).read_bytes() for t in tenants.values() for n in ("config.yaml", ".env")}


def _changed(before: dict[str, bytes], after: dict[str, bytes]) -> set[str]:
    return {k for k in after if before.get(k) != after[k]}


def _expect_only(before: dict[str, bytes], after: dict[str, bytes], expected: set[str], what: str) -> None:
    if (changed := _changed(before, after)) != expected:
        raise H.LaunchProfileBleed(f"{what} changed {sorted(changed)}, expected only {sorted(expected)}")


def _rpc_leaks(b: H.ServeBackend, owners: dict[str, str], tenants: dict[str, H.Tenant]) -> list[str]:
    """Replies/events bound to a session must carry no other tenant's canary."""
    out: list[str] = []
    for m in b.seen:
        sid = (m.get("params") or {}).get("session_id") or (m.get("result") or {}).get("session_id")
        if sid in owners:
            label = f"rpc {m.get('method') or 'reply'}:{(m.get('params') or {}).get('type', '')} for {owners[sid]}"
            out += H.text_leaks(label, json.dumps(m), tenants, owner=owners[sid])
    return out


def _start(root: Path, tenants: dict[str, H.Tenant], backends: list[H.ServeBackend]) -> H.ServeBackend:
    for t in tenants.values():
        H.seed_due_cron_job(t, root / "home")  # due now: the backend's first cron tick fires it
    b = H.ServeBackend(root / "home", root / "serve.log")
    backends.append(b)
    return b


def _await_cron(tenants: dict[str, H.Tenant], fires: int) -> None:
    H.poll(lambda: all(H.cron_requests(t) >= 2 * fires for t in tenants.values()), 150,
           f"cron fire #{fires} in every profile")


def test_desktop_backend_never_crosses_tenants(fleet, request: pytest.FixtureRequest) -> None:
    root, tenants, backends = fleet

    # Phase 1: one backend, three profile sessions, interleaved turns, one cron fire per profile.
    b = _start(root, tenants, backends)
    pid = b.proc.pid
    sids: dict[str, str] = {}
    stored: dict[str, str] = {}
    for name in ("alpha", "default", "beta"):
        res = b.ok("session.create", _profile_param(name))
        sids[name], stored[name] = res["session_id"], res["stored_session_id"]
        if res["info"]["model"] != tenants[name].model:
            raise H.LaunchProfileBleed(f"session.create for {name} reports model {res['info']['model']!r}")
    owners = {sid: name for name, sid in sids.items()}
    for name in ("alpha", "default", "beta", "alpha", "beta"):
        b.turn(sids[name], f"turn for {name}")
    _await_cron(tenants, 1)
    H.check_isolation(tenants, min_snapshots=2, extra=_rpc_leaks(b, owners, tenants))

    # Phase 2: settings writes via RPC touch only the addressed profile.
    for name, t in tenants.items():
        new_cwd = root / f"work2-{name}-{t.tag}"
        new_cwd.mkdir()
        t.extra["workdir2"] = str(new_cwd)  # the live session may keep its cwd; both dirs are t's own
        before = _files(tenants)
        b.ok("config.set", {"session_id": sids[name], "key": "cwd", "value": str(new_cwd)})
        _expect_only(before, _files(tenants), {f"{name}/config.yaml"}, f"cwd write for {name}")
    beta = tenants["beta"]
    beta.extra["custom_prompt"] = f"prompt-canary-beta-{beta.tag}"
    before = _files(tenants)
    b.ok("config.set", {"profile": "beta", "key": "prompt", "value": beta.extra["custom_prompt"]})
    _expect_only(before, _files(tenants), {"beta/config.yaml"}, "profile-bound prompt write")
    alpha = tenants["alpha"]
    alpha.extra["saved_key"] = f"sk-savekey-alpha-{alpha.tag}"
    before = _files(tenants)
    b.ok("model.save_key", {"session_id": sids["alpha"], "slug": "deepseek", "api_key": alpha.extra["saved_key"]})
    _expect_only(before, _files(tenants), {"alpha/.env"}, "session-bound model.save_key write")
    for name in ("default", "beta", "alpha"):
        b.turn(sids[name], f"after settings for {name}")
    H.check_isolation(tenants, min_snapshots=3, extra=_rpc_leaks(b, owners, tenants))

    # Phase 3: profile churn next to the live backend never replaces it.
    b.ok("profiles.create", {"name": "gamma"})
    deleted = H.run_hermes(["profile", "delete", "gamma", "--yes"], root / "home")
    assert deleted.returncode == 0, deleted.stderr[-2000:]
    assert b.proc.pid == pid and b.proc.poll() is None, "profile create/delete replaced the backend"
    b.turn(sids["alpha"], "after profile churn")
    H.check_isolation(tenants, min_snapshots=3, extra=_rpc_leaks(b, owners, tenants))

    # Phase 4: restart; resume each stored session in its own profile; cron ticks again.
    b.close()
    assert not H.pid_alive(pid), "old backend still alive"
    b = _start(root, tenants, backends)
    for name in ("beta", "default", "alpha"):
        res: dict[str, Any] = b.ok("session.resume", {"session_id": stored[name], **_profile_param(name)})
        sids[name] = res["session_id"]
    owners = {sid: name for name, sid in sids.items()}
    for name in ("beta", "default", "alpha"):
        b.turn(sids[name], f"after restart for {name}")
    _await_cron(tenants, 2)
    H.check_isolation(tenants, min_snapshots=5, extra=_rpc_leaks(b, owners, tenants))

    b.close()
    log = (root / "serve.log").read_text(encoding="utf-8", errors="replace")
    H.assert_no_text_leaks("serve output", log, tenants)
