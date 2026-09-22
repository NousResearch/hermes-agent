"""T3/T4 pinned apply, immutable hand-off, and correlated receipt tests.

These tests use local temporary Git repositories only. They never install Hermes,
open SSH, or touch a real user checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_handoff, update_receipt
from hermes_cli.update_target import (
    TargetRequest,
    SourceBinding,
    validate_source_binding,
    TargetAdmissionError,
    apply_pinned_target,
    verify_pinned_post_swap,
)


INSTALL_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _authoritative_install_identity(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "install_id").write_text(INSTALL_ID + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def _commit(cwd: Path, message: str) -> str:
    _git(cwd, "add", ".")
    _git(cwd, "commit", "-m", message)
    return _git(cwd, "rev-parse", "HEAD")


def _remote_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    bare = tmp_path / "origin.git"
    author = tmp_path / "author"
    checkout = tmp_path / "checkout"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", "-b", "main", str(author))
    _git(author, "config", "user.name", "Test Author")
    _git(author, "config", "user.email", "test@example.invalid")
    (author / "payload.txt").write_text("A\n", encoding="utf-8")
    (author / "hermes_cli").mkdir()
    (author / "hermes_cli" / "update_rollout_protocol.json").write_text(
        '{"protocol": 1}\n', encoding="utf-8"
    )
    a_sha = _commit(author, "A")
    _git(author, "remote", "add", "origin", str(bare))
    _git(author, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", "-b", "main", str(bare), str(checkout))
    _git(checkout, "config", "user.name", "Test Checkout")
    _git(checkout, "config", "user.email", "checkout@example.invalid")
    (author / "payload.txt").write_text("B\n", encoding="utf-8")
    b_sha = _commit(author, "B")
    _git(author, "push", "origin", "main")
    _git(checkout, "fetch", "origin", "main")
    return checkout, author, a_sha, b_sha


def _request(checkout: Path, target: str, current: str) -> TargetRequest:
    source = SourceBinding(
        str(checkout.resolve()), _git(checkout, "remote", "get-url", "origin"),
        "refs/remotes/origin/main", target, "fixture", "d" * 64, 1,
    )
    return TargetRequest(target, INSTALL_ID, current, source)


def _synthetic_source(root: Path, target: str, branch: str = "main") -> dict:
    return SourceBinding(
        str(root.resolve()), "https://example.test/hermes.git",
        f"refs/remotes/origin/{branch}", target, "fixture", "d" * 64, 1,
    ).to_wire()


def test_post_swap_head_mismatch_is_refused_without_repair(tmp_path):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)

    with pytest.raises(TargetAdmissionError, match="post-swap-head-mismatch"):
        verify_pinned_post_swap(checkout, _request(checkout, b_sha, a_sha))

    assert _git(checkout, "rev-parse", "HEAD") == a_sha


def test_post_swap_install_identity_mismatch_is_refused_without_repair(tmp_path):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)
    apply_pinned_target(checkout, _request(checkout, b_sha, a_sha))
    (Path(os.environ["HERMES_HOME"]) / "install_id").write_text("f" * 32 + "\n", encoding="utf-8")

    with pytest.raises(TargetAdmissionError, match="post-swap-install-id-mismatch"):
        verify_pinned_post_swap(checkout, _request(checkout, b_sha, a_sha))

    assert _git(checkout, "rev-parse", "HEAD") == b_sha


def test_install_identity_is_read_from_authoritative_hermes_home(tmp_path, monkeypatch):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)
    home = tmp_path / "hermes-home"
    (home / "install_id").write_text(INSTALL_ID + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = apply_pinned_target(checkout, _request(checkout, b_sha, a_sha))

    assert result.outcome == "applied"
    assert verify_pinned_post_swap(checkout, _request(checkout, b_sha, a_sha)) == {
        "post_sha": b_sha,
        "post_install_id": INSTALL_ID,
    }


def test_moving_origin_still_applies_reviewed_target_not_new_tip(tmp_path):
    checkout, author, a_sha, b_sha = _remote_fixture(tmp_path)
    (author / "payload.txt").write_text("C\n", encoding="utf-8")
    c_sha = _commit(author, "C")
    _git(author, "push", "origin", "main")

    result = apply_pinned_target(checkout, _request(checkout, b_sha, a_sha))

    assert result.outcome == "applied"
    assert result.post_sha == b_sha
    assert result.post_sha != c_sha
    assert _git(checkout, "rev-parse", "HEAD") == b_sha
    assert (checkout / "payload.txt").read_text(encoding="utf-8") == "B\n"
    assert _git(checkout, "remote") == "origin"


def test_dirty_checkout_refuses_without_moving_head(tmp_path):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)
    (checkout / "payload.txt").write_text("local edit\n", encoding="utf-8")

    with pytest.raises(TargetAdmissionError, match="dirty-checkout"):
        apply_pinned_target(checkout, _request(checkout, b_sha, a_sha))

    assert _git(checkout, "rev-parse", "HEAD") == a_sha


def test_expected_current_sha_mismatch_refuses_before_fetch_or_apply(tmp_path):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)

    with pytest.raises(TargetAdmissionError, match="current-sha-mismatch"):
        apply_pinned_target(checkout, _request(checkout, b_sha, "f" * 40))

    assert _git(checkout, "rev-parse", "HEAD") == a_sha


def test_diverged_target_refuses_instead_of_merge_or_reset(tmp_path):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)
    (checkout / "local.txt").write_text("local\n", encoding="utf-8")
    d_sha = _commit(checkout, "local divergence")

    with pytest.raises(TargetAdmissionError, match="diverged-target"):
        apply_pinned_target(checkout, _request(checkout, b_sha, d_sha))

    assert _git(checkout, "rev-parse", "HEAD") == d_sha
    assert not _git(checkout, "stash", "list")


def test_unreachable_target_refuses_without_code_movement(tmp_path):
    checkout, _author, a_sha, _b_sha = _remote_fixture(tmp_path)

    with pytest.raises(TargetAdmissionError, match="target-unreachable"):
        apply_pinned_target(checkout, _request(checkout, "d" * 40, a_sha))

    assert _git(checkout, "rev-parse", "HEAD") == a_sha


def test_target_removed_from_authorized_origin_branch_is_refused(tmp_path):
    checkout, author, a_sha, b_sha = _remote_fixture(tmp_path)
    bare = Path(_git(author, "remote", "get-url", "origin"))
    # Keep B as a local object, then move origin/main to a sibling C that does
    # not descend from B. The reviewed object is no longer reachable from the
    # authorized branch and must not be substituted with C.
    _git(checkout, "fetch", "origin", "main")
    _git(author, "reset", "--hard", a_sha)
    (author / "payload.txt").write_text("C\n", encoding="utf-8")
    c_sha = _commit(author, "C sibling")
    _git(author, "push", "origin", f"{c_sha}:refs/tmp/c-sibling")
    _git(bare, "update-ref", "refs/heads/main", c_sha)

    with pytest.raises(TargetAdmissionError, match="target-not-on-authorized-origin"):
        apply_pinned_target(checkout, _request(checkout, b_sha, a_sha))

    assert _git(checkout, "rev-parse", "HEAD") == a_sha
    assert c_sha != b_sha


def test_incompatible_target_is_refused_before_code_movement(tmp_path):
    checkout, author, a_sha, _b_sha = _remote_fixture(tmp_path)
    bare = Path(_git(author, "remote", "get-url", "origin"))
    _git(author, "reset", "--hard", a_sha)
    (author / "payload.txt").write_text("incompatible\n", encoding="utf-8")
    (author / "hermes_cli" / "update_rollout_protocol.json").unlink()
    incompatible_sha = _commit(author, "pre-protocol target")
    _git(author, "push", "origin", f"{incompatible_sha}:refs/tmp/incompatible")
    _git(bare, "update-ref", "refs/heads/main", incompatible_sha)

    with pytest.raises(TargetAdmissionError, match="incompatible-target"):
        apply_pinned_target(checkout, _request(checkout, incompatible_sha, a_sha))

    assert _git(checkout, "rev-parse", "HEAD") == a_sha
    assert (checkout / "payload.txt").read_text(encoding="utf-8") == "A\n"


def test_already_equal_target_is_distinct_no_code_change(tmp_path):
    checkout, _author, a_sha, b_sha = _remote_fixture(tmp_path)
    _git(checkout, "merge", "--ff-only", b_sha)

    result = apply_pinned_target(checkout, _request(checkout, b_sha, b_sha))

    assert result.outcome == "already-current"
    assert result.prior_sha == b_sha
    assert result.post_sha == b_sha


def test_receipt_carries_immutable_intent_across_resume_and_writes_once(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    update_receipt._current = None
    intent = {
        "target": "a" * 40,
        "install_id": INSTALL_ID,
        "correlation_id": "c" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
        "source": _synthetic_source(tmp_path, "a" * 40),
    }

    update_receipt.begin_update_receipt(intent=intent)
    update_receipt.record_step("pinned_apply", True, "post_sha=" + intent["target"])
    detached = update_receipt.detach_update_receipt()
    assert detached["update_intent"] == intent

    handoff = update_handoff.write_handoff({
        "receipt": detached,
        "pinned_intent": intent,
        "branch": intent["branch"],
        "pre_pull_sha": intent["prior_sha"],
    })
    loaded = update_handoff.read_handoff(handoff)
    update_receipt.resume_update_receipt(
        loaded["receipt"], handoff_ack_path=loaded["_handoff_ack_path"]
    )
    update_receipt.record_pinned_post_swap(
        post_sha=intent["target"], post_install_id=intent["install_id"], verified=True
    )
    path = update_receipt.finalize_update_receipt("success")

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["update_intent"] == intent
    assert payload["correlation_id"] == intent["correlation_id"]
    assert payload["requested_sha"] == intent["target"]
    assert payload["pre_sha"] == intent["prior_sha"]
    assert len(list((home / "logs" / "update_receipts").glob("update_*.json"))) == 1
    ack = update_receipt.read_handoff_ack(
        loaded["_handoff_ack_path"], correlation_id=intent["correlation_id"]
    )
    assert ack["outcome"] == "success"
    assert update_receipt.finalize_pending_update_receipt(0) is None


@pytest.mark.parametrize("field", ["branch", "pre_pull_sha"])
def test_pinned_handoff_binds_operational_intent(tmp_path, monkeypatch, field):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    intent = {
        "target": "a" * 40,
        "install_id": INSTALL_ID,
        "correlation_id": "e" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
        "source": _synthetic_source(tmp_path, "a" * 40),
    }
    payload = {
        "receipt": {"update_intent": intent},
        "pinned_intent": intent,
        "branch": intent["branch"],
        "pre_pull_sha": intent["prior_sha"],
    }
    handoff = update_handoff.write_handoff(payload)
    tampered = json.loads(handoff.read_text(encoding="utf-8"))
    tampered[field] = "tampered"
    handoff.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="handoff operational intent mismatch"):
        update_handoff.read_handoff(handoff)


def test_pinned_handoff_refuses_dropped_pinned_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    intent = {
        "target": "a" * 40, "install_id": INSTALL_ID,
        "correlation_id": "e" * 32, "prior_sha": "b" * 40, "branch": "main",
        "source": _synthetic_source(tmp_path, "a" * 40),
    }
    handoff = update_handoff.write_handoff({
        "receipt": {"update_intent": intent}, "pinned_intent": intent,
        "branch": "main", "pre_pull_sha": "b" * 40,
    })
    body = json.loads(handoff.read_text(encoding="utf-8"))
    body.pop("pinned_intent")
    handoff.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match="missing pinned intent"):
        update_handoff.read_handoff(handoff)


def test_pinned_handoff_refuses_source_stripped_from_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    intent = {
        "target": "a" * 40, "install_id": INSTALL_ID,
        "correlation_id": "e" * 32, "prior_sha": "b" * 40, "branch": "main",
    }
    with pytest.raises(ValueError, match="source-binding-required"):
        update_handoff.write_handoff({
            "receipt": {"update_intent": intent}, "pinned_intent": intent,
            "branch": "main", "pre_pull_sha": "b" * 40,
        })


def test_pinned_post_swap_argv_refuses_legacy_handoff(tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    handoff = update_handoff.write_handoff({"branch": "main", "receipt": None})
    called = []
    monkeypatch.setattr(update_cmd, "_execute_post_swap", lambda *_args: called.append(True))
    args = SimpleNamespace(
        post_swap=str(handoff), target_request=TargetRequest("a" * 40, INSTALL_ID, "b" * 40),
    )

    with pytest.raises(ValueError, match="missing pinned intent"):
        update_cmd._run_post_swap_phase(args, gateway_mode=False)
    assert called == []


def test_post_swap_refuses_source_changed_between_argv_and_handoff(tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    intent = {
        "target": "a" * 40, "install_id": INSTALL_ID,
        "correlation_id": "e" * 32, "prior_sha": "b" * 40, "branch": "main",
        "source": _synthetic_source(tmp_path, "a" * 40),
    }
    handoff = update_handoff.write_handoff({
        "receipt": {"update_intent": intent}, "pinned_intent": intent,
        "branch": "main", "pre_pull_sha": "b" * 40,
    })
    changed_source = _synthetic_source(tmp_path, "a" * 40)
    changed_source["originUrl"] = "https://other.example.test/hermes.git"
    args = SimpleNamespace(
        post_swap=str(handoff),
        target_request=TargetRequest(
            "a" * 40, INSTALL_ID, "b" * 40, validate_source_binding(changed_source),
        ),
    )
    called = []
    monkeypatch.setattr(update_cmd, "_execute_post_swap", lambda *_args: called.append(True))

    with pytest.raises(ValueError, match="post-swap reviewed source mismatch"):
        update_cmd._run_post_swap_phase(args, gateway_mode=False)
    assert called == []


def test_pinned_command_carries_resulting_branch_into_handoff_without_legacy_prepare(
    tmp_path, monkeypatch
):
    from hermes_cli import update_cmd
    from hermes_cli.update_target import PinnedApplyResult

    request = TargetRequest(
        "a" * 40, INSTALL_ID, "b" * 40,
        SourceBinding(
            str(tmp_path.resolve()), "https://example.test/hermes.git",
            "refs/remotes/origin/release/1", "a" * 40, "fixture", "d" * 64, 1,
        ),
    )
    args = SimpleNamespace(target_request=request, branch=None, post_swap=None, gateway=False)
    captured = {}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    update_receipt._current = None
    monkeypatch.setattr(
        update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=tmp_path)
    )
    monkeypatch.setattr(update_cmd, "current_branch", lambda _root: "release/1")
    monkeypatch.setattr(
        update_cmd,
        "apply_pinned_target",
        lambda _root, _request, branch=None: PinnedApplyResult(
            "applied", request.current_sha, request.revision, "release/1", "origin"
        ),
    )
    monkeypatch.setattr(
        update_cmd,
        "_resolve_update_options",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy preparation entered pinned path")
        ),
    )
    monkeypatch.setattr(
        update_cmd,
        "_hand_off_post_swap",
        lambda _args, **kwargs: captured.update(kwargs),
    )
    try:
        update_cmd._cmd_pinned_update_impl(args, gateway_mode=False)
        assert captured["branch"] == "release/1"
        assert captured["target_request"] == request
        assert captured["correlation_id"]
        assert update_receipt._current.data["update_intent"]["branch"] == "release/1"
    finally:
        update_receipt._current = None


def test_pinned_dependency_warning_is_terminal(monkeypatch, capsys):
    from hermes_cli import update_cmd

    monkeypatch.setattr(
        update_cmd,
        "_sync_python_dependencies_after_pull",
        lambda *args, **kwargs: print("  ⚠ Lazy refresh failed to refresh: fixture"),
    )

    with pytest.raises(RuntimeError, match="pinned dependency sync reported failure"):
        update_cmd._run_pinned_dependency_sync(
            ["git"], "main", "b" * 40, [], [], None
        )

    assert "Lazy refresh failed" in capsys.readouterr().out


@pytest.mark.real_post_swap_handoff
@pytest.mark.parametrize("ack_present", [True, False])
def test_pinned_parent_does_not_duplicate_child_final_receipt(tmp_path, monkeypatch, ack_present):
    from types import SimpleNamespace

    from hermes_cli import update_cmd

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    intent = {
        "target": "a" * 40,
        "install_id": INSTALL_ID,
        "correlation_id": "f" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
        "source": _synthetic_source(tmp_path, "a" * 40),
    }
    ack = home / "logs" / "update_receipts" / "post_swap_123.ack"
    ack.parent.mkdir(parents=True)
    update_receipt._current = None
    update_receipt.begin_update_receipt(intent=intent)
    update_receipt.record_pinned_post_swap(
        post_sha=intent["target"], post_install_id=intent["install_id"], verified=True,
    )
    child_receipt = update_receipt.finalize_update_receipt("success")
    assert child_receipt is not None
    child = json.loads(child_receipt.read_text(encoding="utf-8"))
    if ack_present:
        ack.write_text(json.dumps({
            "schema": 1, "correlation_id": intent["correlation_id"],
            "outcome": "success", "finished_at": child["finished_at"],
            "receipt_path": str(child_receipt),
        }), encoding="utf-8")
    payload = {
        "receipt": {"update_intent": intent}, "pinned_intent": intent,
        "target_intent": intent, "correlation_id": intent["correlation_id"],
        "_handoff_ack_path": str(ack), "_handoff_detached": False,
    }
    monkeypatch.setattr(update_cmd, "_post_swap_payload", lambda **kwargs: payload)
    monkeypatch.setattr(update_cmd._update_handoff, "continue_update_in_fresh_interpreter", lambda *a, **k: 0)

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._hand_off_post_swap(SimpleNamespace(), gateway_mode=False)

    assert exc_info.value.code == 0
    assert list((home / "logs" / "update_receipts").glob("update_*.json")) == [child_receipt]
    assert not ack.exists()


@pytest.mark.live_system_guard_bypass
@pytest.mark.real_post_swap_handoff
def test_real_subprocess_handoff_imports_target_tree_and_preserves_intent(
    tmp_path, monkeypatch
):
    """Exercise Popen, not the inline pytest hand-off fixture."""
    target = tmp_path / "target-tree"
    package = target / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "child-marker.json"
    (package / "main.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "handoff = Path(sys.argv[sys.argv.index('--post-swap') + 1])\n"
        "payload = json.loads(handoff.read_text(encoding='utf-8'))\n"
        "Path(os.environ['CHILD_MARKER']).write_text(json.dumps({\n"
        "  'module': __file__, 'intent': payload['pinned_intent'],\n"
        "  'receipt': payload['receipt']['update_intent'],\n"
        "}), encoding='utf-8')\n"
        "handoff.unlink()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(target))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(target)
    monkeypatch.setenv("CHILD_MARKER", str(marker))
    monkeypatch.setattr(update_handoff, "_running_from_windows_shim", lambda: False)
    monkeypatch.setattr(update_handoff, "post_swap_python", lambda: Path(sys.executable))
    intent = {
        "target": "a" * 40,
        "install_id": INSTALL_ID,
        "correlation_id": "d" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
        "source": _synthetic_source(tmp_path, "a" * 40),
    }
    payload = {
        "receipt": {"update_intent": intent},
        "pinned_intent": intent,
        "branch": intent["branch"],
        "pre_pull_sha": intent["prior_sha"],
    }

    assert update_handoff.continue_update_in_fresh_interpreter(payload, argv_tail=[]) == 0
    child = json.loads(marker.read_text(encoding="utf-8"))
    assert Path(child["module"]).resolve() == (package / "main.py").resolve()
    assert child["intent"] == intent
    assert child["receipt"] == intent
    assert not list((tmp_path / "home").glob("**/post_swap_*.json"))


@pytest.mark.parametrize("failure", ["dependency", "restart", "import"])
def test_pinned_failure_cannot_be_finalized_as_success(tmp_path, monkeypatch, failure):
    home = tmp_path / failure
    monkeypatch.setenv("HERMES_HOME", str(home))
    update_receipt._current = None
    intent = {
        "target": "a" * 40,
        "install_id": INSTALL_ID,
        "correlation_id": (failure[0] * 32),
        "prior_sha": "b" * 40,
        "branch": "main",
    }
    update_receipt.begin_update_receipt(intent=intent)
    update_receipt.record_failure(failure + " failure")

    path = update_receipt.finalize_update_receipt("success")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["outcome"] != "success"
    assert payload["outcome"] == "failed"
    assert payload["failure_reasons"] == [failure + " failure"]
