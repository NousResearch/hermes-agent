"""Fixtures shared across hermes_cli tests."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )


@pytest.fixture(autouse=True)
def _private_repo_checkout_lock(monkeypatch, tmp_path_factory):
    """``cmd_update`` locks the checkout it runs from: in tests, this repository, whose git
    common dir every parallel test process (and, from a linked worktree, the live install's
    ``hermes update``) shares. Give each test its own lock file for THIS repo; lock tests on
    fixture checkouts are unaffected."""
    from pathlib import Path

    from hermes_cli import update_lock

    repo = Path(update_lock.__file__).resolve().parents[1]
    real = update_lock.checkout_lock_path
    private = tmp_path_factory.mktemp("checkout-lock") / "hermes-update.lock"

    def checkout_lock_path(install_root=None):
        root = Path(install_root) if install_root else repo
        return private if root.resolve() == repo else real(install_root)

    monkeypatch.setattr(update_lock, "checkout_lock_path", checkout_lock_path)


@pytest.fixture(autouse=True)
def _source_channels_resolve_locally(request, monkeypatch):
    """Every unflagged ``hermes update`` resolves its channel through R2; tests must
    not reach the network for that. Default every channel to a ``source-branch``
    record delivering ``origin/<name>`` through the documented seam. Channel tests
    that model records themselves re-patch ``_resolve_channel`` after this runs;
    the marker opts out entirely for tests of the reader's own network path.
    """
    if request.node.get_closest_marker("real_release_channels"):
        return
    from hermes_cli import source_releases
    from hermes_cli.release_channels import ChannelResolution

    def resolve(name, repository):
        record = {"schema": 1, "name": name, "repository": repository, "policy": "source-branch",
                  "state": "active", "identity": None, "nextSequence": 1, "head": None,
                  "delivery": {"kind": "source-branch", "branch": name}}
        return ChannelResolution(record, record, None)

    monkeypatch.setattr(source_releases, "_resolve_channel", resolve)
    # stable reads GitHub releases directly; keep it off the network the same way.
    monkeypatch.setattr(source_releases, "_resolve_stable", lambda repository, *_: source_releases.SourceTarget(
        "stable", "stable", repository, branch="stable"))


@pytest.fixture(autouse=True)
def _no_detached_commit_graph_writer(monkeypatch):
    """The update's maintenance starts a detached ``git commit-graph write`` on the checkout it ran
    from: in a test, this repository or a tmp copy deleted under it. In-process runs never spawn it."""
    from hermes_cli import git_commit_graph

    monkeypatch.setattr(git_commit_graph, "spawn_commit_graph_refresh", lambda root: False)


@pytest.fixture(autouse=True)
def _discharge_host_update_obligation():
    """Start and end every ``hermes_cli`` test with NO host update-restart obligation.

    The record is host-scoped on purpose (one multiplexer per host), so it lives in the
    per-OS-USER host state dir — not in the per-test ``HERMES_HOME``. The root conftest pins
    that dir per test only when the caller supplied no ``HERMES_GATEWAY_LOCK_DIR`` (#118097
    keeps the documented override working), so with one set every test in a file shares it and
    a test that arms the obligation makes the next one read a restart it never owed. Clearing
    the record — rather than re-pinning the dir — leaves that override rule untouched.
    """

    def _clear() -> None:
        try:
            from hermes_cli.update_host_obligation import clear_host_obligation

            clear_host_obligation()
        except Exception:
            # Import/env failure here must never error an unrelated test.
            pass

    _clear()
    yield
    _clear()


@pytest.fixture
def isolated_source_completion(monkeypatch):
    """Unit-test the completion tail in-process; real transport is tested separately."""
    from hermes_cli import update_cmd, update_completion

    monkeypatch.setattr("hermes_cli.source_build.build_update_products", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", lambda *a: None)

    def complete(request):
        update_completion._complete_selected(request)
        return {"exit_code": 0, "receipt": update_completion._read_terminal_receipt(request),
                "windows_resume": request["windows_resume"]}

    monkeypatch.setattr(update_cmd, "run_completion", complete)


@pytest.fixture(autouse=True)
def _reset_prompt_toolkit_output_cache():
    """Clear prompt_toolkit's cached AppSession output around each CLI test.

    See the module docstring for the capsys/prompt_toolkit interaction this
    guards against.
    """

    def _clear() -> None:
        try:
            from prompt_toolkit.application.current import get_app_session

            get_app_session()._output = None
        except Exception:
            # prompt_toolkit not importable / internal shape changed — the
            # tests that rely on this simply keep their prior behavior.
            pass

    _clear()
    yield
    _clear()


_KANBAN_MODULE_ROOTS = ("hermes_cli", "hermes_state", "hermes_constants")


def _is_kanban_family(name: str) -> bool:
    return name.split(".", 1)[0] in _KANBAN_MODULE_ROOTS


def _pin_parent_attribute(module_name: str, module) -> None:
    """Point the parent package's attribute back at ``module`` (or drop a dead one).

    ``sys.modules`` is not the only registry: ``import hermes_cli.gateway_runtime``
    also leaves the submodule on its parent package. A purge that only touches
    ``sys.modules`` leaves that attribute holding the dead copy, so the next
    ``from hermes_cli import gateway_runtime`` hands out the ghost while
    ``from hermes_cli.gateway_runtime import …`` — production's function-local form —
    re-imports the fresh module. Repairing both halves is what makes the identity
    one module again.
    """
    parent_name, _, child = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is None or not hasattr(parent, child):
        return
    if module is None:
        delattr(parent, child)
    else:
        setattr(parent, child, module)


@pytest.fixture(autouse=True)
def _restore_kanban_module_identity():
    """Hand every test the ``hermes_cli*``/``hermes_state*``/``hermes_constants``
    ``sys.modules`` state it would have had running alone.

    Two fixtures purge that whole family from ``sys.modules`` so the next import
    re-reads a fake ``HERMES_HOME`` (``test_kanban_per_profile_cap.py``,
    ``test_kanban_cli_dispatch_passthrough.py``). Followed by a re-import the purge
    is one-way, and that splits the process in two: modules imported during
    collection (the test modules themselves, ``hermes_cli.kanban_db``, …) keep the
    ghost copy a test patched, while a function-local ``import`` in production
    (``hermes_cli/kanban_worker_client.py``) resolves the fresh copy and cannot see
    that patch — so the real ``connect_gateway`` runs. A file that never loads a
    purging fixture is green; the package run, which does, is not. Snapshotting and
    restoring the family around every test keeps a purge a local act, so file order
    no longer decides the result.
    """
    saved = {name: module for name, module in sys.modules.items() if _is_kanban_family(name)}
    yield
    live = [n for n in sys.modules if _is_kanban_family(n)]
    for name in live:
        del sys.modules[name]
    sys.modules.update(saved)
    for name in live:
        _pin_parent_attribute(name, saved.get(name))
    for name, module in saved.items():
        _pin_parent_attribute(name, module)


@pytest.fixture
def probe_root(tmp_path):
    """A fixture checkout the installation launcher can boot from.

    ``runtime_command`` prepends the checkout root and runs ``import hermes_bootstrap``
    before the probe body, exactly as production does. Tests that point the import
    guard at a scratch tree need that module present, or the probe dies before its
    health marker — a developer venv whose editable ``.pth`` shadows the root hides
    the dependency, CI's clean environment does not.
    """
    (tmp_path / "hermes_bootstrap.py").write_text("", encoding="utf-8")
    return tmp_path
