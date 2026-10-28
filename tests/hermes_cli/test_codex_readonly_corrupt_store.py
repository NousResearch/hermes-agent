"""Corrupt auth stores fail closed before readonly Codex recovery or backup I/O."""

import os
import subprocess
import sys

import pytest


_PROBE = r'''
import sys
sys.dont_write_bytecode = True
import io
import logging
import os
from pathlib import Path
import traceback
from agent.codex_auth_policy import bind_codex_auth_policy
from hermes_cli import auth, auth_codex

home = Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
payload = (b'{"synthetic-private-value": ' if sys.argv[1] == "json"
           else b'\xffsynthetic-private-value')
if sys.argv[1] == "bom":
    payload = (b'\xef\xbb\xbf{"providers":{"openai-codex":{"tokens":'
               b'{"access_token":"synthetic","refresh_token":"synthetic-refresh"}}}}')
store = home / "auth.json"
stage = sys.argv[4]
store.write_bytes(payload if stage in {"initial", "copy-error"} else b'{"providers": {}}')
snapshot_root = home
if stage == "global":
    snapshot_root = home.parent.parent
    (snapshot_root / "config.yaml").write_text("model: {}\n")
    store = snapshot_root / "auth.json"
    store.write_bytes(payload)
before_paths = sorted(path.relative_to(snapshot_root) for path in snapshot_root.rglob("*"))
before_bytes = {path: (snapshot_root / path).read_bytes() for path in before_paths
                if (snapshot_root / path).is_file()}
mutation = store.open("r+b") if stage in {"pool", "ratelimit"} else None
mode = sys.argv[2]
if mode != "unbound":
    bind_codex_auth_policy(read_only=mode == "readonly")
if mode != "readonly" and sys.argv[3] != "True":
    copy_logs = io.StringIO()
    auth.logger.addHandler(logging.StreamHandler(copy_logs))
    auth.logger.setLevel(logging.DEBUG)
    if stage == "copy-error":
        def fail_copy(event, args):
            if event == "shutil.copyfile":
                raise ValueError("synthetic copy failure")
        sys.addaudithook(fail_copy)
    assert auth._load_auth_store() == {"version": auth.AUTH_STORE_VERSION, "providers": {}}
    assert store.read_bytes() == payload
    if stage == "copy-error":
        assert not store.with_suffix(".json.corrupt").exists()
        assert "A copy could NOT be preserved at" in copy_logs.getvalue()
        assert "Corrupt file preserved at" not in copy_logs.getvalue()
        assert "ValueError: synthetic copy failure" in copy_logs.getvalue()
    else:
        assert store.with_suffix(".json.corrupt").read_bytes() == payload
    assert not store.with_suffix(".lock").exists()
    sys.exit(0)

attempts = []
def audit(event, args):
    if (event.startswith(("shutil.", "socket.", "subprocess."))
            or event in {"os.rename", "os.remove", "os.mkdir", "os.chmod", "os.utime"}
            or (event == "open" and args[2] &
                (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
        attempts.append(event)
        raise AssertionError("forbidden side effect: " + event)

forbidden = {
    auth._save_auth_store.__code__, auth._auth_store_lock.__wrapped__.__code__,
    auth_codex._pool_codex_credential.__code__,
    auth_codex._codex_pool_rate_limit_status.__code__,
    auth_codex._recover_codex_tokens_from_cli.__code__,
    auth_codex._import_codex_cli_tokens.__code__,
    auth.shutil.copy2.__code__,
}
pool_calls = []
trigger = {"pool": auth_codex._pool_codex_credential.__code__,
           "ratelimit": auth_codex._codex_pool_rate_limit_status.__code__}.get(stage)
def profile(frame, event, arg):
    if event != "call":
        return
    if frame.f_code in {auth_codex._pool_codex_credential.__code__,
                         auth_codex._codex_pool_rate_limit_status.__code__}:
        pool_calls.append(frame.f_code.co_name)
        if frame.f_code == trigger:
            # Simulate an external writer between genuine native reads using a fixture-owned fd.
            mutation.seek(0)
            mutation.write(payload)
            mutation.truncate()
            mutation.flush()
            before_bytes[store.relative_to(snapshot_root)] = payload
        if stage in {"pool", "ratelimit"}:
            return
    if frame.f_code in forbidden:
        attempts.append(frame.f_code.co_name)

logs = io.StringIO()
handler = logging.StreamHandler(logs)
auth.logger.addHandler(handler)
sys.addaudithook(audit)
sys.setprofile(profile)
try:
    result = auth.resolve_codex_runtime_credentials(read_only=sys.argv[3] == "True", force_refresh=True)
except RuntimeError as exc:
    assert sys.argv[1] != "bom"
    assert not attempts, attempts
    assert str(exc) == "Cannot read corrupt auth store with --codex-auth-read-only"
    assert "synthetic-private-value" not in traceback.format_exc()
else:
    assert sys.argv[1] == "bom", "corrupt readonly credentials did not fail closed"
    assert result["api_key"] == "synthetic"
    assert result["source"] == "hermes-auth-store"
finally:
    sys.setprofile(None)
assert not attempts, attempts
assert "synthetic-private-value" not in logs.getvalue()
assert store.read_bytes() == payload
assert sorted(path.relative_to(snapshot_root) for path in snapshot_root.rglob("*")) == before_paths
assert {path: (snapshot_root / path).read_bytes() for path in before_bytes} == before_bytes
expected_calls = {"pool": ["_pool_codex_credential"],
                  "ratelimit": ["_pool_codex_credential", "_codex_pool_rate_limit_status"]}
assert pool_calls == expected_calls.get(stage, []), pool_calls
'''


def _run_probe(tmp_path, kind, mode, explicit_readonly=False, stage="initial"):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, kind, mode, str(explicit_readonly), stage],
        env={**os.environ, "HOME": str(tmp_path),
             "HERMES_HOME": str(tmp_path / "hermes" / "profiles" / "named"
                                if stage == "global" else tmp_path / "hermes")},
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["json", "utf8"])
@pytest.mark.parametrize("explicit_readonly", [False, True])
def test_readonly_resolver_refuses_corruption_without_side_effect_attempts(
    tmp_path, kind, explicit_readonly,
):
    _run_probe(tmp_path, kind, "readonly", explicit_readonly)


@pytest.mark.parametrize("kind", ["json", "utf8"])
@pytest.mark.parametrize("mode", ["unbound", "writable"])
def test_legacy_corruption_still_preserves_exact_backup(tmp_path, kind, mode):
    _run_probe(tmp_path, kind, mode)


@pytest.mark.parametrize("kind", ["json", "utf8"])
@pytest.mark.parametrize("mode", ["unbound", "writable"])
def test_explicit_readonly_refuses_corruption_without_process_policy(tmp_path, kind, mode):
    _run_probe(tmp_path, kind, mode, explicit_readonly=True)


@pytest.mark.parametrize("kind", ["json", "utf8"])
@pytest.mark.parametrize("stage", ["pool", "ratelimit", "global"])
@pytest.mark.parametrize("mode,explicit_readonly", [("unbound", True), ("readonly", False)])
def test_readonly_refusal_survives_later_store_reads(
    tmp_path, kind, stage, mode, explicit_readonly,
):
    _run_probe(tmp_path, kind, mode, explicit_readonly, stage)


@pytest.mark.parametrize("mode", ["unbound", "writable"])
def test_legacy_copy_failure_remains_best_effort_for_non_os_errors(tmp_path, mode):
    _run_probe(tmp_path, "json", mode, stage="copy-error")


@pytest.mark.parametrize("mode,explicit_readonly", [("unbound", True), ("readonly", False)])
def test_readonly_native_resolver_accepts_codex_bom_store_without_writes(
    tmp_path, mode, explicit_readonly,
):
    _run_probe(tmp_path, "bom", mode, explicit_readonly)
