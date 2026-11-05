"""The SQL mapping checker on bodies that reach their transaction other than through a direct wrapper call: a
``partial`` or ``asyncio.to_thread`` of a wrapper, and a callable handed to a parameter its callee calls inside a
wrapper. Each rule passes on matching trees, and a body that runs in another transaction, or with other wrapper
arguments, still fails."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[3] / "persistence" / "sqlite" / "_sqlmap" / "check.py"
_spec = importlib.util.spec_from_file_location("sqlmap_check", CHECK)
check = importlib.util.module_from_spec(_spec)
sys.modules["sqlmap_check"] = check
_spec.loader.exec_module(check)


def run(tmp_path: Path, legacy: str, backend: str, units: list) -> list[str]:
    files = {"m1/hermes_state_x.py": legacy, "branch/persistence/sqlite/x.py": backend,
             "branch/persistence/sqlite/_sqlmap/x.json": json.dumps({"family": "x", "units": units})}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    branch = tmp_path / "branch"
    return check.Checker(tmp_path / "m1", branch, check.load_manifests(branch / check.MANIFEST_DIR)).run().failures


def unit(name: str) -> dict:
    return {"backend": f"persistence/sqlite/x.py::{name}", "legacy": f"hermes_state_x.py::{name}"}


LEGACY_FORWARDED = '''
import asyncio
from functools import partial


def drop(db, key):
    def write(conn):
        conn.execute("DELETE FROM meta WHERE key = ?", (key,))
    yield HANDED


async def forget(db, key):
    await asyncio.to_thread(db._execute_write, lambda conn: conn.execute("DELETE FROM picks WHERE key = ?", (key,)))
'''
BACKEND_FORWARDED = '''
def drop(holder, key):
    def write(conn):
        conn.execute("DELETE FROM meta WHERE key = ?", (key,))
    return holder.WRAPPER(write)


def forget(holder, key):
    return holder.WRAPPER(lambda conn: conn.execute("DELETE FROM picks WHERE key = ?", (key,)))
'''
FORWARDED_UNITS = [unit("drop"), unit("forget")]


def test_a_partial_or_to_thread_of_a_wrapper_runs_its_body_in_that_wrapper(tmp_path):
    legacy = LEGACY_FORWARDED.replace("HANDED", "partial(db._execute_write, write)")
    assert run(tmp_path, legacy, BACKEND_FORWARDED.replace("WRAPPER", "_execute_write"), FORWARDED_UNITS) == []
    tree = check.Tree(tmp_path / "m1", ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
    views = [tree.sites[("hermes_state_x.py", f)][0].view()["transaction"] for f in ("drop", "forget")]
    assert views == ["write", "write"]


def test_a_body_handed_to_another_wrapper_kind_still_fails(tmp_path):
    legacy = LEGACY_FORWARDED.replace("HANDED", "partial(db._execute_write, write)")
    failures = run(tmp_path, legacy, BACKEND_FORWARDED.replace("WRAPPER", "_read_retrying_ioerr"), FORWARDED_UNITS)
    assert sorted(f.split(" transaction:")[0] for f in failures) == ["x.json: persistence/sqlite/x.py::drop",
                                                                     "x.json: persistence/sqlite/x.py::forget"]


def test_the_arguments_a_partial_adds_stay_in_the_label(tmp_path):
    legacy = LEGACY_FORWARDED.replace("HANDED", "partial(db._execute_write, write, patience_s=5.0)")
    (failure,) = run(tmp_path, legacy, BACKEND_FORWARDED.replace("WRAPPER", "_execute_write"), FORWARDED_UNITS)
    assert "x.py::drop transaction:" in failure and "patience_s=5.0" in failure


def test_a_partial_of_a_function_that_is_no_wrapper_runs_no_transaction(tmp_path):
    legacy = LEGACY_FORWARDED.replace("HANDED", "partial(print, write)")
    (failure,) = run(tmp_path, legacy, BACKEND_FORWARDED.replace("WRAPPER", "_execute_write"), FORWARDED_UNITS)
    assert "x.py::drop transaction:" in failure and "legacy ...'none'" in failure


LEGACY_HOOKS = '''
from functools import partial


def settle(db, key, *, _terminal_write=None):
    SETTLE


def close(conn, key):
    conn.execute("INSERT INTO notes(key) VALUES(?)", (key,))


def guard(conn, key, *, who):
    conn.execute("SELECT 1 FROM grants WHERE who = ?", (who,))
'''
# Each caller hands the hook one way, and the function the hook runs: a lambda through a local assigned
# under a condition, a lambda passed directly, a partial through a local.
CALLERS = {
    "finish": ("close", '''
def finish(db, key, notice):
    write = None if notice is None else (lambda conn, k: close(conn, k))
    return settle(db, key, _terminal_write=write)
'''),
    "finish_now": ("close", '''
def finish_now(db, key):
    return settle(db, key, _terminal_write=lambda conn, k: close(conn, k))
'''),
    "finish_guarded": ("guard", '''
def finish_guarded(db, key, who):
    hook = None
    if who:
        hook = partial(guard, who=who)
    return settle(db, key, _terminal_write=hook)
'''),
}
SETTLE_INSIDE = '''def write(conn):
        conn.execute("UPDATE items SET done = 1 WHERE key = ?", (key,))
        if _terminal_write is not None:
            _terminal_write(conn, key)
    return db.WRAPPER(write)'''
SETTLE_OUTSIDE = '''def write(conn):
        conn.execute("UPDATE items SET done = 1 WHERE key = ?", (key,))
    db._execute_write(write)
    if _terminal_write is not None:
        _terminal_write(None, key)'''
BACKEND_HOOKS = '''
def settle(holder, key, *, notice=False, who=None):
    def write(conn):
        conn.execute("UPDATE items SET done = 1 WHERE key = ?", (key,))
        if notice:
            close(conn, key)
        if who:
            guard(conn, key, who=who)
    return holder._execute_write(write)


def close(conn, key):
    conn.execute("INSERT INTO notes(key) VALUES(?)", (key,))


def guard(conn, key, *, who):
    conn.execute("SELECT 1 FROM grants WHERE who = ?", (who,))
'''
HOOK_UNITS = [unit("settle"), unit("close"), unit("guard")]


def failing_units(failures: list[str]) -> list[str]:
    return sorted(f.split(" transaction:")[0].rsplit("::", 1)[1] for f in failures)


def legacy_hooks(settle: str, *callers: str) -> str:
    return LEGACY_HOOKS.replace("SETTLE", settle) + "".join(CALLERS[c][1] for c in callers or CALLERS)


@pytest.mark.parametrize("caller", list(CALLERS))
def test_a_callable_handed_to_a_parameter_the_callee_calls_in_its_wrapper_runs_in_that_wrapper(tmp_path, caller):
    legacy = legacy_hooks(SETTLE_INSIDE.replace("WRAPPER", "_execute_write"), caller)
    (tmp_path / "hermes_state_x.py").write_text(legacy, encoding="utf-8")
    tree = check.Tree(tmp_path, ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
    (site,) = tree.sites[("hermes_state_x.py", CALLERS[caller][0])]
    assert site.view()["transaction"] == "write"


def test_hooked_callables_match_a_backend_that_calls_them_inside_its_write(tmp_path):
    legacy = legacy_hooks(SETTLE_INSIDE.replace("WRAPPER", "_execute_write"))
    assert run(tmp_path, legacy, BACKEND_HOOKS, HOOK_UNITS) == []


def test_a_parameter_the_callee_calls_outside_its_wrapper_hands_on_no_transaction(tmp_path):
    assert failing_units(run(tmp_path, legacy_hooks(SETTLE_OUTSIDE), BACKEND_HOOKS, HOOK_UNITS)) == ["close", "guard"]


def test_a_hook_run_in_another_wrapper_kind_still_fails(tmp_path):
    legacy = legacy_hooks(SETTLE_INSIDE.replace("WRAPPER", "_read_retrying_ioerr"))
    assert failing_units(run(tmp_path, legacy, BACKEND_HOOKS, HOOK_UNITS)) == ["close", "guard", "settle"]


LEGACY_ADMIT = '''
def admit(db, key, *, _authorize_write=None):
    def write(conn):
        if _authorize_write is not None:
            _authorize_write(conn)
        conn.execute("INSERT INTO items(key) VALUES (?)", (key,))
    return db._execute_write(write)


def retire(conn, old):
    conn.execute("UPDATE items SET status = 'retired' WHERE key = ?", (old,))
'''
LEGACY_ADMIT_HOOK_OUTSIDE = LEGACY_ADMIT.replace(
    "        if _authorize_write is not None:\n            _authorize_write(conn)\n", "").replace(
    "    return db._execute_write(write)",
    "    if _authorize_write is not None:\n        _authorize_write(db._conn)\n    return db._execute_write(write)")
LEGACY_OWNER = '''
from hermes_state_x import admit


class Authority:
    def gate(self, authorize=None):
        def guard(conn):
            if authorize is not None:
                authorize(conn)
        return guard

    def submit(self, key, *, _authorize_write=None):
        return self._submit(key, _authorize_write=_authorize_write)

    def _submit(self, key, *, _authorize_write=None):
        def run():
            return admit(self.db, key, _authorize_write=self.gate(_authorize_write))
        return run()
'''
LEGACY_CONTROLS = '''
from functools import partial

from hermes_state_x import retire


async def replace(authority, key, old):
    authority.db._conn.execute("SELECT 1 FROM items")
    return await authority.submit(key, _authorize_write=partial(retire, old=old))
'''
BACKEND_ADMIT = '''
def admit(holder, key, *, old=None):
    def write(conn):
        if old is not None:
            retire(conn, old)
        conn.execute("INSERT INTO items(key) VALUES (?)", (key,))
    return holder._execute_write(write)


def retire(conn, old):
    conn.execute("UPDATE items SET status = 'retired' WHERE key = ?", (old,))
'''
CALLER_FILES = ["gateway/owner.py", "gateway/controls.py"]


def hook_checker(tmp_path: Path, name: str, legacy: str = LEGACY_ADMIT, callers=CALLER_FILES,
                 units=(unit("admit"), unit("retire"))) -> "check.Checker":
    files = {"m1/hermes_state_x.py": legacy, "m1/gateway/owner.py": LEGACY_OWNER, "m1/gateway/controls.py": LEGACY_CONTROLS,
             f"{name}/persistence/sqlite/x.py": BACKEND_ADMIT,
             f"{name}/persistence/sqlite/_sqlmap/x.json": json.dumps({"family": "x", "callers": callers, "units": list(units)})}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    branch = tmp_path / name
    return check.Checker(tmp_path / "m1", branch, check.load_manifests(branch / check.MANIFEST_DIR))


def test_a_hook_passed_on_through_a_guard_factory_in_caller_files_runs_in_the_write_it_reaches(tmp_path):
    checker = hook_checker(tmp_path, "branch")
    assert checker.run().failures == []
    assert {path for path, _ in checker.legacy.sites} == {"hermes_state_x.py"}


def test_a_caller_file_named_as_a_sql_source_is_refused(tmp_path):
    named = [unit("retire"), {"backend": "persistence/sqlite/x.py::admit", "legacy": "gateway/controls.py::replace"}]
    with pytest.raises(check.ManifestError, match="caller files named as SQL sources"):
        hook_checker(tmp_path, "named", units=named)


@pytest.mark.parametrize("legacy, callers", [
    (LEGACY_ADMIT_HOOK_OUTSIDE, CALLER_FILES),
    (LEGACY_ADMIT, CALLER_FILES[:1]),
], ids=["hook_called_outside_the_write", "caller_file_not_read"])
def test_a_hook_the_checker_cannot_follow_into_the_write_still_fails(tmp_path, legacy, callers):
    failures = hook_checker(tmp_path, "planted", legacy=legacy, callers=callers).run().failures
    assert any("x.py::retire transaction:" in f for f in failures)


LEGACY_AUTOMATION = '''
from functools import partial

from hermes_state_x import admit


def _cancel(old, replaced):
    def guard(conn):
        conn.execute("UPDATE items SET status = 'cancelled' WHERE key = ?", (old,))
        replaced[:] = [old]
    return guard


def run(authority, key, old, replaced):
    HANDED
'''
HANDED_TO_ADMIT = "return authority.tracked(partial(admit, authority.db, key, _authorize_write=authority.gate(old and _cancel(old, replaced))))"
LEGACY_MUTATIONS = '''
from functools import partial

from hermes_state_x import admit, retire


def mutate(authority, key, old, cold):
    def authorize_write(conn):
        retire(conn, old)
    return write_mutation(authority, key, authorize_write=authorize_write if cold else None)


def write_mutation(authority, key, *, authorize_write=None):
    COMMIT
    return commit()
'''
COMMIT_WITH_HOOK = "commit = partial(admit, authority.db, key, _authorize_write=authorize_write)"
BACKEND_CANCEL = '''

def _cancel(conn, old):
    conn.execute("UPDATE items SET status = 'cancelled' WHERE key = ?", (old,))
'''


def closure_checker(tmp_path: Path, name: str, handed: str = HANDED_TO_ADMIT, commit: str = COMMIT_WITH_HOOK) -> "check.Checker":
    files = {"m1/hermes_state_x.py": LEGACY_ADMIT, "m1/gateway/owner.py": LEGACY_OWNER,
             "m1/gateway/auto.py": LEGACY_AUTOMATION.replace("HANDED", handed),
             "m1/gateway/mutations.py": LEGACY_MUTATIONS.replace("COMMIT", commit),
             f"{name}/persistence/sqlite/x.py": BACKEND_ADMIT.replace(
                 "            retire(conn, old)\n", "            retire(conn, old)\n            _cancel(conn, old)\n") + BACKEND_CANCEL,
             f"{name}/persistence/sqlite/_sqlmap/x.json": json.dumps({
                 "family": "x", "callers": ["gateway/owner.py", "gateway/mutations.py"],
                 "units": [unit("admit"), unit("retire"),
                           {"backend": "persistence/sqlite/x.py::_cancel", "legacy": "gateway/auto.py::_cancel"}]})}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    branch = tmp_path / name
    return check.Checker(tmp_path / "m1", branch, check.load_manifests(branch / check.MANIFEST_DIR))


def test_hooks_reach_the_write_through_partial_arguments_conditional_arguments_and_returned_closures(tmp_path):
    assert closure_checker(tmp_path, "branch").run().failures == []


@pytest.mark.parametrize("handed, commit, unit_name", [
    ("guard = old and _cancel(old, replaced)\n    return authority.tracked(partial(admit, authority.db, key, _authorize_write=authority.gate()))",
     COMMIT_WITH_HOOK, "_cancel"),
    (HANDED_TO_ADMIT, "authorize_write(authority.db._conn)\n    commit = partial(admit, authority.db, key)", "retire"),
], ids=["closure_never_handed_on", "hook_called_outside_the_write"])
def test_a_closure_or_hook_kept_out_of_the_write_still_fails(tmp_path, handed, commit, unit_name):
    failures = closure_checker(tmp_path, "planted", handed, commit).run().failures
    assert any(f"x.py::{unit_name} transaction:" in f for f in failures)
