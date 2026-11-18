"""Behaviour contracts: malformed inputs are refused, argv is fixed."""

import contextlib
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from isolated_tests import attestation, docker_argv, inputs, main, policy, probe, proof

# inputs imports pwd and relies on POSIX uid/mode bits, symlinks and AF_UNIX sockets.
pytestmark = pytest.mark.platforms("posix")


def _snapshot(root: Path, extra: dict | None = None) -> Path:
    files = {"scripts/run_tests.sh": b"#!/bin/sh\n", "tests/scripts/isolated_tests/test_x.py": b"x=1\n"}
    files.update(extra or {})
    listing = {}
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        listing[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    (root / inputs.MANIFEST_NAME).write_text(json.dumps({
        "schema": inputs.MANIFEST_SCHEMA, "source": "git-ls-files",
        "base_commit": "3b421578588243f920886e6fe79a6faa87ea026e",
        "candidate_patch_sha256": None, "files": listing}))
    return root


@pytest.fixture
def snap(tmp_path):
    return _snapshot(tmp_path / "snap")


def _canon_sha(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _owned_json(path: Path, doc: dict) -> Path:
    path.write_bytes(json.dumps(doc).encode())
    path.chmod(0o600)  # read_receipt/load_trusted_inputs refuse group/world-writable files
    return path


def _provenance(snap: Path, where: Path) -> tuple[Path, Path]:
    """No-candidate trusted inputs and a VERIFIED receipt bound to them and to
    `snap`, both outside the snapshot (an extra file there breaks the manifest)."""
    manifest_raw = (snap / inputs.MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_raw)
    runner = {"scripts/run_tests.sh": manifest["files"]["scripts/run_tests.sh"]["sha256"]}
    exception_policy = {"allowed": {}, "max_count": 0}
    trusted = _owned_json(where / "trusted-inputs.json", {
        "schema": inputs.TRUSTED_INPUTS_SCHEMA, "base_commit": manifest["base_commit"],
        "candidate_patches": [], "runner_allowlist": runner, "exception_policy": exception_policy})
    receipt = _owned_json(where / "receipt.json", {
        "schema": inputs.RECEIPT_SCHEMA, "result": "VERIFIED",
        "inputs_sha256": hashlib.sha256(trusted.read_bytes()).hexdigest(),
        "base_commit": manifest["base_commit"], "candidate_patches_sha256": [],
        "no_candidate": True, "runner_allowlist_sha256": _canon_sha(runner),
        "exception_policy_sha256": _canon_sha(exception_policy),
        "prepare_limits_sha256": _canon_sha(dict(policy.PREPARE_LIMITS)),
        "applied_exceptions": {}, "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest()})
    return trusted, receipt


@pytest.fixture
def provenance_args(snap, tmp_path):
    trusted, receipt = _provenance(snap, tmp_path)
    return ["--inputs", str(trusted), "--receipt", str(receipt)]


def _cli(phase, snap, tmp_path, provenance_args, runtime="docker", docker="/d"):
    return [phase, "--runtime", runtime, "--docker", docker, "--socket", "/s",
            "--image", "sha256:" + "a" * 64, "--python", "/opt/p",
            "--snapshot", str(snap), "--suite", "isolated-selftests",
            "--out", str(tmp_path / "o"), *provenance_args]


def test_provenance_fixture_binds_snapshot(snap, provenance_args):
    trusted = inputs.load_trusted_inputs(provenance_args[1])
    receipt = inputs.read_receipt(provenance_args[3], trusted)
    assert receipt["applied_exceptions"] == {}
    snapshot = inputs.verify_snapshot(str(snap), receipt["applied_exceptions"])
    assert inputs.verify_receipt(receipt, trusted, snapshot)["no_candidate"] is True


def test_clean_snapshot_accepted(snap):
    assert inputs.verify_snapshot(str(snap)).file_count == 2


@pytest.mark.parametrize("name", [".env", ".git", "id_rsa", "auth.json", "server.pem",
                                  "id_dsa", "id_dsa.pub", ".ENV", "ID_RSA", "Server.PEM"])
def test_denied_names_refused(tmp_path, name):
    root = _snapshot(tmp_path / "s", {name: b"secret"})
    with pytest.raises(inputs.Refused, match="denied name"):
        inputs.verify_snapshot(str(root))


def test_unlisted_file_refused(snap):
    (snap / "extra.txt").write_text("x")
    with pytest.raises(inputs.Refused, match="tree != manifest"):
        inputs.verify_snapshot(str(snap))


def test_modified_file_refused(snap):
    (snap / "tests/scripts/isolated_tests/test_x.py").write_bytes(b"x=2\n")
    with pytest.raises(inputs.Refused, match="content mismatch"):
        inputs.verify_snapshot(str(snap))


def test_symlink_inside_snapshot_refused(snap, tmp_path):
    (tmp_path / "outside").write_text("host")
    os.symlink(tmp_path / "outside", snap / "link")
    with pytest.raises(inputs.Refused, match="symlink"):
        inputs.verify_snapshot(str(snap))


def test_symlinked_snapshot_root_refused(snap, tmp_path):
    os.symlink(snap, tmp_path / "alias")
    with pytest.raises(inputs.Refused, match="symlink"):
        inputs.verify_snapshot(str(tmp_path / "alias"))


def test_broad_roots_refused(tmp_path, monkeypatch):
    # The passwd home need not exist (uid nobody -> /nonexistent), so stand an existing
    # directory in for it: the broad-root rule itself must fire, not a missing-path refusal.
    home = Path(os.path.realpath(tmp_path))
    monkeypatch.setattr(inputs, "account_home", lambda: home)
    for raw in ("/", str(home)):
        with pytest.raises(inputs.Refused, match="broad root"):
            inputs.real_dir(raw, "snapshot")


def test_missing_dir_refused_not_internal(tmp_path):
    base = Path(os.path.realpath(tmp_path))
    with pytest.raises(inputs.Refused, match="does not exist"):
        inputs.real_dir(str(base / "missing"), "snapshot")
    (base / "file").write_text("")
    with pytest.raises(inputs.Refused, match="does not exist"):
        inputs.real_dir(str(base / "file" / "child"), "snapshot")


@pytest.mark.parametrize("image", ["ubuntu:latest", "sha256:abc", "repo@sha256:" + "a" * 64, ""])
def test_image_must_be_local_id(image):
    with pytest.raises(inputs.Refused):
        docker_argv.check_image(image)


@pytest.mark.parametrize("raw", ["relative.sock", "/nonexistent/docker.sock"])
def test_bad_socket_refused(raw):
    with pytest.raises(inputs.Refused):
        inputs.unix_socket(raw)


def test_regular_file_is_not_a_socket(tmp_path):
    (tmp_path / "f").write_text("")
    with pytest.raises(inputs.Refused, match="not a Unix socket"):
        inputs.unix_socket(str(tmp_path / "f"))


def test_real_unix_socket_accepted(tmp_path):
    path = str(tmp_path / "s")
    s = socket.socket(socket.AF_UNIX)
    s.bind(path)
    try:
        assert inputs.unix_socket(path) == Path(os.path.realpath(path))
    finally:
        s.close()


def test_suite_allowlist(snap):
    s = inputs.verify_snapshot(str(snap))
    with pytest.raises(inputs.Refused, match="allowlist"):
        inputs.check_suite(s, "tests/")


def test_run_argv_contract(snap, tmp_path):
    argv = docker_argv.run_argv(docker=Path("/usr/bin/docker"), socket=Path("/x/d.sock"),
                                cli_home=tmp_path, image="sha256:" + "a" * 64,
                                python="/opt/venv/bin/python", snapshot=snap,
                                run_id="0123456789abcdef", script="true")
    for flag in ("--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
                 "--security-opt=no-new-privileges", "--memory-swap=2147483648",
                 "--pids-limit=128", "--cpus=1", "--ulimit=fsize=67108864:67108864"):
        assert flag in argv
    assert argv[argv.index("--host") + 1] == "unix:///x/d.sock"
    assert not any("docker.sock" in a and "mount" in a for a in argv)
    assert sum(a.startswith("--mount") for a in argv) == 1
    assert argv[argv.index("--entrypoint") + 1] == "/usr/bin/env"
    assert argv[argv.index("sha256:" + "a" * 64) + 1] == "-i"


def test_probe_env_cannot_inject_other_vars(snap, tmp_path):
    with pytest.raises(inputs.Refused):
        docker_argv.run_argv(docker=Path("/d"), socket=Path("/s"), cli_home=tmp_path,
                             image="sha256:" + "a" * 64, python="/opt/p", snapshot=snap,
                             run_id="0123456789abcdef", script="true", probe_env=("PATH=/evil",))


def test_host_env_is_allowlisted(tmp_path):
    assert set(docker_argv.host_env(tmp_path)) == {"PATH", "HOME", "DOCKER_CONFIG", "LANG"}


def test_rm_targets_exact_name_only(tmp_path):
    argv = docker_argv.rm_argv(docker=Path("/d"), socket=Path("/s"), cli_home=tmp_path,
                               run_id="0123456789abcdef")
    assert argv[-3:] == ["rm", "--force", "hermes-isotest-0123456789abcdef"]
    with pytest.raises(inputs.Refused):
        docker_argv.container_name("../all")


def test_probe_failure_when_line_missing():
    assert probe.all_passed(probe.parse_probe(b"noise\n")) is False


def test_attestation_missing_and_stale(tmp_path):
    bind = {"a": 1}
    with pytest.raises(attestation.AttestationError, match="missing"):
        attestation.verify(str(tmp_path / "none.json"), bind)
    attestation.write(tmp_path / "att.json", bind, {k: True for k in probe.REQUIRED_CHECKS},
                      "0" * 64, {"timed_out": True, "cleanup_status": "removed"}, "t")
    attestation.verify(str(tmp_path / "att.json"), bind)
    with pytest.raises(attestation.AttestationError, match="stale"):
        attestation.verify(str(tmp_path / "att.json"), {"a": 2})


class _FakeTty:
    """Pipe-backed, as in test_signal_and_tty_deadline: proof._read_line selects and
    os.reads tty.fileno(), so the typed answer must sit in a real fd."""

    def __init__(self, answer, is_tty=True):
        self.is_tty, self.written = is_tty, []
        self.r, w = os.pipe()
        os.write(w, (answer + "\n").encode())
        os.close(w)

    def fileno(self):
        return self.r

    def isatty(self):
        return self.is_tty

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def close(self):
        os.close(self.r)


def test_proof_refuses_without_tty():
    with contextlib.closing(_FakeTty("x", is_tty=False)) as tty:
        with pytest.raises(proof.ProofRefused, match="interactive"):
            proof.confirm(tty, "p")
    with pytest.raises(proof.ProofRefused):
        proof.confirm(None, "p")


def test_proof_refuses_wrong_code():
    with contextlib.closing(_FakeTty("deadbeef-wrong")) as tty:
        with pytest.raises(proof.ProofRefused, match="mismatch"):
            proof.confirm(tty, "p")


def test_unsupported_runtime_exit_code(snap, tmp_path, provenance_args):
    rc = main.main(_cli("plan", snap, tmp_path, provenance_args, runtime="sandbox-exec"))
    assert rc == policy.EXIT_UNSUPPORTED_RUNTIME


def test_run_without_attestation_exit_5(snap, tmp_path, provenance_args, monkeypatch):
    monkeypatch.setattr(inputs, "unix_socket", lambda raw: Path(raw))
    # Only the docker subprocess boundary is canned: the real _bind_runtime parses
    # these through check_image_inspect/check_server_version, then attestation.verify
    # must still refuse before any payload.
    image = "sha256:" + "a" * 64
    canned = {"inspect": json.dumps([{"Id": image, "Config": {"Env": ["PATH=/usr/bin"]},
                                      "Os": "linux", "Architecture": "amd64"}]).encode(),
              "version": json.dumps({"Version": "27.0.0", "Os": "linux", "Arch": "amd64",
                                     "KernelVersion": "6.1.0"}).encode()}
    reads = []
    monkeypatch.setattr(main, "_read", lambda argv, env: reads.append(argv) or
                        canned["inspect" if "inspect" in argv else "version"])
    monkeypatch.setattr(main, "_execute",
                        lambda *a, **k: pytest.fail("payload started without an attestation"))
    # --docker must pass the real executable-file check, but tmp may be a noexec mount
    # (access(X_OK) fails there by design). The running interpreter is an executable
    # regular file on an exec-capable mount; _read is canned, so it is never invoked.
    rc = main.main(_cli("run", snap, tmp_path, provenance_args, docker=sys.executable)
                   + ["--attestation", str(tmp_path / "none.json")])
    assert rc == policy.EXIT_ATTESTATION
    assert len(reads) == 2


@pytest.mark.parametrize("drop", ["--inputs", "--receipt"])
def test_provenance_flags_required(snap, tmp_path, provenance_args, drop):
    argv = _cli("plan", snap, tmp_path, provenance_args)
    i = argv.index(drop)
    with pytest.raises(SystemExit) as exc:
        main.main(argv[:i] + argv[i + 2:])
    assert exc.value.code == 2


def test_unknown_flags_rejected(snap, tmp_path, provenance_args, capsys):
    with pytest.raises(SystemExit) as exc:
        main.main(_cli("plan", snap, tmp_path, provenance_args) + ["--yes"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --yes" in capsys.readouterr().err
