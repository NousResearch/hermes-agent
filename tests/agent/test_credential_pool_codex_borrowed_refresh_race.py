"""Regression: N profiles borrowing ONE root Codex grant must not race its single-use refresh.

Codex (ChatGPT OAuth) refresh tokens are single-use. A profile with no ``openai-codex`` rows of
its own borrows the global root's row (#100339). Before the fix, ``_refresh_entry`` serialised
the rotation under the *profile's* ``auth.lock`` — a different file per profile — so N profiles
hitting expiry together all POSTed the same refresh token: one won, the rest got
``refresh_token_reused``. The in-lock re-sync could not rescue them either: it read the
``providers.openai-codex`` singleton block, which ``manual:device_code`` / borrowed rows never
write (#39236), so a waiter saw no rotation and replayed the spent pair.

Fix: borrowed rows lock the ROOT store, and the Codex waiter re-reads the POOL row.

Real OS processes, real temp root + profiles, real auth.json I/O and real kernel file locks —
the same shape as the live race witnessed on a 10-profile Windows fleet. The token endpoint is
replaced at ``hermes_cli.auth.refresh_codex_oauth_pure`` (looked up by name at call time) with
genuine single-use semantics, enforced across processes with an ``O_EXCL`` claim file.
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest


def _jwt(exp: int) -> str:
    hdr = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps({
        "sub": "user-1", "exp": exp,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"},
    }).encode()).rstrip(b"=").decode()
    return f"{hdr}.{body}."


def _fake_refresh_codex_oauth_pure(access_token, refresh_token, **_):
    """Single-use token endpoint shared across processes via an O_EXCL claim file."""
    import errno
    scratch = Path(os.environ["CODEX_RACE_SCRATCH"])
    try:
        fd = os.open(scratch / f"spent-{refresh_token}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        won = True
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        won = False
    with open(scratch / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "rt": refresh_token, "won": won}) + "\n")
    if not won:
        from hermes_cli.auth_codex import _codex_err
        raise _codex_err(
            "Your refresh token has already been used to generate a new access token.",
            "refresh_token_reused", relogin=True)
    time.sleep(0.3)  # widen the window: every waiter must queue on the lock, not slip past it
    gen = int(refresh_token.rsplit("gen", 1)[1]) + 1
    return {
        "access_token": _jwt(int(time.time()) + 3600),
        "refresh_token": f"RT-gen{gen}",
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _worker(idx: int, scratch: str, repo: str, go: str, out):
    os.environ["CODEX_RACE_SCRATCH"] = scratch
    os.environ["HERMES_HOME"] = str(Path(scratch) / "profiles" / f"p{idx}")
    os.environ.pop("PYTEST_CURRENT_TEST", None)  # child is a plain Hermes process
    os.environ["HOME"] = str(Path(scratch) / "fakehome")
    sys.path.insert(0, repo)
    import hermes_cli.auth as auth_mod
    auth_mod.refresh_codex_oauth_pure = _fake_refresh_codex_oauth_pure
    from agent.credential_pool import load_pool
    while not Path(go).exists():
        time.sleep(0.005)
    pool = load_pool("openai-codex")
    refreshed = pool.try_refresh_current()
    selected = pool.select()
    own_rows = json.loads(
        (Path(os.environ["HERMES_HOME"]) / "auth.json").read_text(encoding="utf-8-sig")
    ).get("credential_pool", {}).get("openai-codex")
    out.put({
        "idx": idx,
        "refreshed_rt": None if refreshed is None else refreshed.refresh_token,
        "selected_rt": None if selected is None else selected.refresh_token,
        "selected_status": None if selected is None else selected.last_status,
        "own_rows": own_rows,
    })


@pytest.fixture
def fleet(tmp_path):
    """Temp root with one expired-but-refreshable borrowed Codex grant + N empty profiles."""
    scratch = tmp_path / "fleet"
    (scratch / "profiles").mkdir(parents=True)
    (scratch / "fakehome").mkdir()
    for marker in ("config.yaml", ".env"):
        (scratch / marker).write_text("", encoding="utf-8")
    root_store = {
        "version": 1, "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "root01", "label": "root-grant", "auth_type": "oauth", "priority": 0,
            "source": "manual:device_code",
            "access_token": _jwt(int(time.time()) - 10),  # expired -> every process must refresh
            "refresh_token": "RT-gen0",
            "last_refresh": "2026-01-01T00:00:00Z",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }]},
    }
    (scratch / "auth.json").write_text(json.dumps(root_store, indent=1), encoding="utf-8")

    def make_profiles(n: int) -> None:
        for i in range(1, n + 1):
            pdir = scratch / "profiles" / f"p{i}"
            pdir.mkdir()
            for marker in ("config.yaml", ".env"):
                (pdir / marker).write_text("", encoding="utf-8")
            (pdir / "auth.json").write_text(
                json.dumps({"version": 1, "providers": {}, "credential_pool": {}}), encoding="utf-8")

    return {"scratch": scratch, "make_profiles": make_profiles}


def _run_race(fleet, n: int):
    repo = str(Path(__file__).resolve().parents[2])
    scratch = fleet["scratch"]
    fleet["make_profiles"](n)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    go = scratch / "GO"
    procs = [ctx.Process(target=_worker, args=(i, str(scratch), repo, str(go), out)) for i in range(1, n + 1)]
    for p in procs:
        p.start()
    time.sleep(3.0)  # let every interpreter finish importing before the barrier drops
    go.write_text("go", encoding="utf-8")
    results = sorted((out.get(timeout=120) for _ in procs), key=lambda r: r["idx"])
    for p in procs:
        p.join(timeout=15)
    ledger_path = scratch / "ledger.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8-sig").splitlines()] if ledger_path.exists() else []
    root_rows = json.loads((scratch / "auth.json").read_text(encoding="utf-8-sig"))["credential_pool"]["openai-codex"]
    return results, ledger, root_rows


@pytest.mark.parametrize("n", [6])
def test_borrowed_codex_grant_refreshes_once_across_racing_profiles(fleet, n):
    results, ledger, root_rows = _run_race(fleet, n)

    won = [l for l in ledger if l["won"]]
    reused = [l for l in ledger if not l["won"]]
    assert len(won) == 1, f"expected exactly one rotation POST, ledger={ledger}"
    assert reused == [], f"a waiter replayed the spent refresh token: {reused}"

    # Root holds the rotated pair and it is healthy.
    assert [(r["id"], r["refresh_token"], r.get("last_status")) for r in root_rows] == [("root01", "RT-gen1", "ok")]

    # Every process ends on the SAME rotated token, none benched, none forked a local copy.
    assert {r["selected_rt"] for r in results} == {"RT-gen1"}, results
    assert all(r["selected_status"] in (None, "ok") for r in results), results
    assert all(not r["own_rows"] for r in results), f"a profile materialised its own copy: {results}"


def test_single_use_refresh_lock_target_is_root_for_borrowed_rows(tmp_path, monkeypatch):
    """Unit-level witness for the lock-target choice, independent of the process race."""
    import hermes_constants
    root = tmp_path / "root"
    prof = root / "profiles" / "p"
    prof.mkdir(parents=True)
    for d in (root, prof):
        for marker in ("config.yaml", ".env"):
            (d / marker).write_text("", encoding="utf-8")
    (root / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "r1", "auth_type": "oauth", "source": "manual:device_code", "priority": 0,
            "access_token": _jwt(int(time.time()) + 3600), "refresh_token": "RT-a",
        }]},
    }), encoding="utf-8")
    (prof / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {}}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(prof))
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    import hermes_cli.auth as auth_mod
    auth_mod._global_auth_store_cache = None
    auth_mod._oauth_heal_clean_marks.clear()

    from agent.credential_pool import load_pool
    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None and entry.id in pool._borrowed_root_ids
    target = pool._single_use_refresh_lock_target(entry)
    assert target is not None and target.resolve() == (root / "auth.json").resolve(), (
        "borrowed Codex row must serialise its rotation under ROOT's auth.lock")

    # An owned row (classic mode / profile with its own rows) keeps the active-store lock.
    monkeypatch.setenv("HERMES_HOME", str(root))
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    auth_mod._global_auth_store_cache = None
    pool_root = load_pool("openai-codex")
    owned = pool_root.select()
    assert owned is not None and pool_root._single_use_refresh_lock_target(owned) is None
