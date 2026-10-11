"""Fixtures shared across hermes_cli tests."""

from __future__ import annotations

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
# --- Operator-qualified security-policy write scope (test helper) --------------------
# The #104697-review boundary removed the importable `approval_override` bool —
# sensitive-key writes require (operator scope AND human presence), both evaluated
# inside the writer. Tests exercising the SANCTIONED operator path enter the scope
# via this fixture; tests exercising the ADVERSARIAL path must NOT enter it.


@pytest.fixture
def operator_write_scope(monkeypatch):
    """Simulate the SANCTIONED operator path for direct-writer test calls: stamp
    the one-shot grant AND satisfy the sanctioned-caller-chain check. The writer's
    frame check requires a _handle_approvals_command frame (gateway or REPL
    mixin); tests calling set_config_value directly satisfy it via a chain shim
    on _policy_write_authorized (test-only; the real handlers satisfy the check
    with their live frames, covered by the handler-chain tests)."""
    import hermes_cli.config as _cfg
    from tools.approval_context import grant_operator_policy_write, reset_operator_policy_write

    monkeypatch.setattr("tools.approval_context._is_sanctioned_policy_stamp_caller", lambda: True)
    token = grant_operator_policy_write()
    orig = _cfg._policy_write_authorized

    def _sanctioned_chain_present():
        if not token_granted():
            return False
        return True

    def token_granted():
        from tools.approval_context import is_operator_policy_write
        return is_operator_policy_write()

    # Direct-writer tests simulate the handler's grant; the frame check's real
    # coverage lives in the handler-chain tests (gateway/REPL e2e).
    monkeypatch.setattr(_cfg, "_policy_write_authorized", _with_chain := (lambda: token_granted() or orig()))
    yield token
    reset_operator_policy_write(token)
