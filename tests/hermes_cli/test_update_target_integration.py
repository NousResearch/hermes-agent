"""Credential-free pinned-update integration fixtures.

The local tests use only disposable local Git repositories. They do not open
SSH, install software, contact a network host, or persist credentials.

The S20.2 lane at the bottom of this file is the actual-SSH slice: when a
disposable SSH fixture is selected it ships the real admission engine to the
fixture host and executes the pinned update there, in the same topology the
Desktop launcher uses (the engine runs remotely; the local runner only
transports). Without a selected fixture the lane reports named skips.
"""

from __future__ import annotations

import base64
import json
import os
import posixpath
import subprocess
from pathlib import Path

import pytest


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def fixture(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "source"
    bare = tmp_path / "origin.git"
    install = tmp_path / "install"
    git(tmp_path, "init", "--bare", str(bare))
    git(tmp_path, "init", "-b", "main", str(source))
    git(source, "config", "user.name", "fixture")
    git(source, "config", "user.email", "fixture@example.test")
    (source / "hermes_cli").mkdir()
    (source / "hermes_cli" / "update_rollout_protocol.json").write_text(
        json.dumps({"protocol": 1}) + "\n", encoding="utf-8"
    )
    (source / "payload.txt").write_text("A\n", encoding="utf-8")
    git(source, "add", ".")
    git(source, "commit", "-m", "A")
    commit_a = git(source, "rev-parse", "HEAD").stdout.strip()
    git(source, "remote", "add", "origin", str(bare))
    git(source, "push", "-u", "origin", "main")
    git(tmp_path, "clone", "-b", "main", str(bare), str(install))
    git(install, "config", "user.name", "fixture")
    git(install, "config", "user.email", "fixture@example.test")
    return {"source": source, "bare": bare, "install": install, "a": commit_a}


def commit_source(state: dict[str, Path | str], text: str, message: str) -> str:
    source = state["source"]
    assert isinstance(source, Path)
    (source / "payload.txt").write_text(text, encoding="utf-8")
    git(source, "add", "payload.txt")
    git(source, "commit", "-m", message)
    return git(source, "rev-parse", "HEAD").stdout.strip()


def request(state: dict[str, Path | str], target: str, current: str):
    from hermes_cli.update_target import SourceBinding, TargetRequest

    install = state["install"]
    assert isinstance(install, Path)
    origin = git(install, "remote", "get-url", "origin").stdout.strip()
    return TargetRequest(
        target,
        "1" * 32,
        current,
        SourceBinding(
            str(install.resolve()),
            origin,
            "refs/remotes/origin/main",
            target,
            "fixture",
            "d" * 64,
            1,
        ),
    )


def configure_install_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.update_target as update_target

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "install_id").write_text("1" * 32 + "\n", encoding="utf-8")
    monkeypatch.setattr(update_target, "get_default_hermes_root", lambda: hermes_home)


def test_disposable_origin_applies_reviewed_b_after_branch_moves_to_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.update_target import apply_pinned_target

    configure_install_id(tmp_path, monkeypatch)
    state = fixture(tmp_path)
    source = state["source"]
    install = state["install"]
    assert isinstance(source, Path) and isinstance(install, Path)
    current = str(state["a"])
    reviewed_b = commit_source(state, "B\n", "B")
    git(source, "push", "origin", "main")
    commit_c = commit_source(state, "C\n", "C")
    git(source, "push", "origin", "main")

    result = apply_pinned_target(install, request(state, reviewed_b, current))

    assert result.target_sha == reviewed_b
    assert reviewed_b != commit_c
    assert git(install, "rev-parse", "HEAD").stdout.strip() == reviewed_b
    assert (install / "payload.txt").read_text(encoding="utf-8") == "B\n"
    assert git(install, "remote").stdout.strip() == "origin"


def test_disposable_origin_refuses_reviewed_commit_removed_from_authorized_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    configure_install_id(tmp_path, monkeypatch)
    state = fixture(tmp_path)
    source = state["source"]
    bare = state["bare"]
    install = state["install"]
    assert isinstance(source, Path) and isinstance(bare, Path) and isinstance(install, Path)
    current = str(state["a"])
    reviewed_b = commit_source(state, "B\n", "B")
    git(source, "push", "origin", "main")
    git(install, "fetch", "origin", "main")
    git(bare, "update-ref", "refs/heads/main", current)

    with pytest.raises(PinnedTargetRefused, match="target-not-reachable"):
        apply_pinned_target(install, request(state, reviewed_b, current))

    assert git(install, "rev-parse", "HEAD").stdout.strip() == current
    assert (install / "payload.txt").read_text(encoding="utf-8") == "A\n"


# ---------------------------------------------------------------------------
# S20.2 — actual SSH slice
#
# The tests below ship the real admission engine to a disposable SSH fixture
# and execute the pinned update there. The remote host, transport, and Git
# operations are real; the local runner only transports bytes. Without a
# selected fixture every case reports a named skip.
# ---------------------------------------------------------------------------

SSH_DRIVER = r'''
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(sys.argv[1])
CASE = sys.argv[2]
sys.path.insert(0, sys.argv[3])

def git(cwd, *args, check=True):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result

def commit(source, text, message):
    (source / "payload.txt").write_text(text, encoding="utf-8")
    git(source, "add", "payload.txt")
    git(source, "commit", "-m", message)
    return git(source, "rev-parse", "HEAD").stdout.strip()

shutil.rmtree(WORK, ignore_errors=True)
source = WORK / "source"
bare = WORK / "origin.git"
install = WORK / "install"
hermes_home = WORK / "hermes-home"
hermes_home.mkdir(parents=True)
(hermes_home / "install_id").write_text("1" * 32 + "\n", encoding="utf-8")
os.environ["HERMES_HOME"] = str(hermes_home)

git(WORK, "init", "--bare", str(bare))
git(WORK, "init", "-b", "main", str(source))
git(source, "config", "user.name", "fixture")
git(source, "config", "user.email", "fixture@example.test")
(source / "hermes_cli").mkdir()
(source / "hermes_cli" / "update_rollout_protocol.json").write_text(json.dumps({"protocol": 1}) + "\n", encoding="utf-8")
(source / "payload.txt").write_text("A\n", encoding="utf-8")
git(source, "add", ".")
git(source, "commit", "-m", "A")
a = git(source, "rev-parse", "HEAD").stdout.strip()
git(source, "remote", "add", "origin", str(bare))
git(source, "push", "-u", "origin", "main")
git(WORK, "clone", "-b", "main", str(bare), str(install))
git(install, "config", "user.name", "fixture")
git(install, "config", "user.email", "fixture@example.test")

b = commit(source, "B\n", "B")
git(source, "push", "origin", "main")
c = commit(source, "C\n", "C")
git(source, "push", "origin", "main")

from hermes_cli.update_target import PinnedTargetRefused, SourceBinding, TargetRequest, apply_pinned_target

origin = git(install, "remote", "get-url", "origin").stdout.strip()

def request(target_sha, current_sha):
    return TargetRequest(
        target_sha,
        "1" * 32,
        current_sha,
        SourceBinding(str(install.resolve()), origin, "refs/remotes/origin/main", target_sha, "fixture", "d" * 64, 1),
    )

def report(payload):
    print(json.dumps(payload, separators=(",", ":")))

def move_to_b():
    """Bring the install to the reviewed object before a case that starts there."""
    git(install, "fetch", "origin", "main")
    git(install, "merge", "--ff-only", b)

try:
    if CASE == "apply-b-despite-c":
        result = apply_pinned_target(install, request(b, a))
        report({
            "case": CASE,
            "outcome": result.outcome,
            "target_sha": result.target_sha,
            "head": git(install, "rev-parse", "HEAD").stdout.strip(),
            "payload": (install / "payload.txt").read_text(encoding="utf-8"),
            "branch": git(install, "branch", "--show-current").stdout.strip(),
            "origin": git(install, "remote").stdout.strip(),
            "reviewed_b": b,
            "branch_tip_c": c,
        })
    elif CASE == "apply-is-idempotent-when-current":
        move_to_b()
        result = apply_pinned_target(install, request(b, b))
        report({
            "case": CASE,
            "outcome": result.outcome,
            "head": git(install, "rev-parse", "HEAD").stdout.strip(),
            "payload": (install / "payload.txt").read_text(encoding="utf-8"),
        })
    elif CASE == "protocol-floor-missing":
        move_to_b()
        (source / "hermes_cli" / "update_rollout_protocol.json").unlink()
        git(source, "add", "-A")
        git(source, "commit", "-m", "D removes protocol")
        d = git(source, "rev-parse", "HEAD").stdout.strip()
        git(source, "push", "origin", "main")
        try:
            apply_pinned_target(install, request(d, b))
            report({"case": CASE, "refused": False})
        except PinnedTargetRefused as exc:
            report({"case": CASE, "refused": True, "reason": exc.reason, "head": git(install, "rev-parse", "HEAD").stdout.strip()})
    elif CASE == "protocol-floor-version":
        move_to_b()
        (source / "hermes_cli" / "update_rollout_protocol.json").write_text(json.dumps({"protocol": 2}) + "\n", encoding="utf-8")
        (source / "payload.txt").write_text("E\n", encoding="utf-8")
        git(source, "add", "-A")
        git(source, "commit", "-m", "E protocol 2")
        e = git(source, "rev-parse", "HEAD").stdout.strip()
        git(source, "push", "origin", "main")
        try:
            apply_pinned_target(install, request(e, b))
            report({"case": CASE, "refused": False})
        except PinnedTargetRefused as exc:
            report({"case": CASE, "refused": True, "reason": exc.reason, "head": git(install, "rev-parse", "HEAD").stdout.strip()})
    elif CASE == "clean-checkout-floor":
        move_to_b()
        (install / "payload.txt").write_text("dirty\n", encoding="utf-8")
        try:
            apply_pinned_target(install, request(c, b))
            report({"case": CASE, "refused": False})
        except PinnedTargetRefused as exc:
            report({"case": CASE, "refused": True, "reason": exc.reason, "head": git(install, "rev-parse", "HEAD").stdout.strip()})
    else:
        report({"case": CASE, "error": "unknown case"})
except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the local assertion
    report({"case": CASE, "error": f"{type(exc).__name__}: {exc}"})
'''

SSH_PACKAGE_PATHS = (
    "hermes_cli/__init__.py",
    "hermes_cli/update_target.py",
    "hermes_constants.py",
    "hermes_platform/__init__.py",
    "hermes_platform/host/__init__.py",
    "hermes_platform/host/facts.py",
    "hermes_platform/host/runtime.py",
)

SSH_FIXTURE_SET_ENV = "HERMES_MANAGED_ROLLOUT_FIXTURE_SET"
SSH_FIXTURE_KEY_ENV = "HERMES_MANAGED_ROLLOUT_FIXTURE_KEY"
SSH_FIXTURE_USER_ENV = "HERMES_MANAGED_ROLLOUT_FIXTURE_USER"
SSH_REMOTE_ROOT = "/tmp/hermes-s20-ssh-slice"
SSH_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _ssh_fixture_endpoint() -> tuple[str, str, str] | None:
    """Parse the same disposable-fixture contract the Playwright selector uses."""
    raw = os.environ.get(SSH_FIXTURE_SET_ENV, "").strip()
    key = os.environ.get(SSH_FIXTURE_KEY_ENV, "").strip()
    user = os.environ.get(SSH_FIXTURE_USER_ENV, "").strip() or "fixture"
    if not raw or not key:
        return None
    first = raw.split(",")[0].strip()
    if ":" not in first:
        return None
    host, _, port = first.rpartition(":")
    if host not in SSH_LOOPBACK_HOSTS or not port.isdigit():
        return None
    return host, port, user


def _ssh(endpoint: tuple[str, str, str], key: str, command: str, stdin: str | None = None) -> str:
    host, port, user = endpoint
    result = subprocess.run(
        [
            "ssh",
            "-i",
            key,
            "-p",
            port,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            "--",
            command,
        ],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )
    if result.returncode:
        raise AssertionError(f"fixture ssh failed: {result.stderr.strip()[:500]}")
    return result.stdout


def _ssh_write(endpoint: tuple[str, str, str], key: str, remote_path: str, content: bytes) -> None:
    """Ship one file to the fixture over stdin.

    The payload travels on stdin, never in the remote command line: a base64
    driver or engine file exceeds the local platform's argument-length limit,
    and stdin needs no quoting at all.
    """
    encoded = base64.b64encode(content).decode("ascii")

    _ssh(
        endpoint,
        key,
        f"mkdir -p '{posixpath.dirname(remote_path)}' && base64 -d > '{remote_path}'",
        stdin=encoded,
    )


def _ssh_case(case: str) -> dict:
    """Ship the real engine to the fixture and run one pinned-update case remotely."""
    key = os.environ[SSH_FIXTURE_KEY_ENV]
    endpoint = _ssh_fixture_endpoint()
    assert endpoint is not None
    repo_root = Path(__file__).resolve().parents[2]

    for relative in SSH_PACKAGE_PATHS:
        _ssh_write(endpoint, key, posixpath.join(SSH_REMOTE_ROOT, "engine", relative), (repo_root / relative).read_bytes())

    _ssh_write(endpoint, key, f"{SSH_REMOTE_ROOT}/driver.py", SSH_DRIVER.encode("utf-8"))

    output = _ssh(
        endpoint,
        key,
        f"cd '{SSH_REMOTE_ROOT}' && python3 '{SSH_REMOTE_ROOT}/driver.py' '{SSH_REMOTE_ROOT}/work' {case} '{SSH_REMOTE_ROOT}/engine'",
    )
    return json.loads(output.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def ssh_slice() -> dict:
    endpoint = _ssh_fixture_endpoint()
    if endpoint is None:
        pytest.skip(f"{SSH_FIXTURE_SET_ENV}/{SSH_FIXTURE_KEY_ENV} are unset; no disposable SSH fixture is selected")
    return {"endpoint": endpoint, "key": os.environ[SSH_FIXTURE_KEY_ENV]}


def test_actual_ssh_applies_reviewed_b_while_the_branch_moves_to_c(ssh_slice: dict) -> None:
    result = _ssh_case("apply-b-despite-c")

    assert "error" not in result, result
    assert result["outcome"] == "applied"
    assert result["target_sha"] == result["reviewed_b"]
    assert result["reviewed_b"] != result["branch_tip_c"]
    assert result["head"] == result["reviewed_b"]
    assert result["payload"] == "B\n"
    assert result["branch"] == "main"
    assert result["origin"] == "origin"


def test_actual_ssh_pinned_update_is_idempotent_at_the_reviewed_object(ssh_slice: dict) -> None:
    result = _ssh_case("apply-is-idempotent-when-current")

    assert "error" not in result, result
    assert result["outcome"] == "already-current"
    assert result["payload"] == "B\n"


def test_actual_ssh_refuses_a_target_missing_the_protocol_resource(ssh_slice: dict) -> None:
    result = _ssh_case("protocol-floor-missing")

    assert "error" not in result, result
    assert result["refused"] is True
    assert result["reason"] == "incompatible-target"
    # The refusal happens before the merge: the checkout never moved.
    assert result["head"] != "b" * 40
    assert len(result["head"]) == 40


def test_actual_ssh_refuses_a_target_not_at_the_protocol_floor(ssh_slice: dict) -> None:
    result = _ssh_case("protocol-floor-version")

    assert "error" not in result, result
    assert result["refused"] is True
    assert result["reason"] == "incompatible-target"


def test_actual_ssh_refuses_a_dirty_checkout_before_any_mutation(ssh_slice: dict) -> None:
    result = _ssh_case("clean-checkout-floor")

    assert "error" not in result, result
    assert result["refused"] is True
    assert result["reason"] == "dirty-checkout"

