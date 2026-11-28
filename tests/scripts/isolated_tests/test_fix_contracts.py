"""Behaviour contracts for the F1-F8 fixes (NOT_RUN in the authoring lane)."""

import json
import sys
from pathlib import Path

import pytest

from isolated_tests import attestation, docker_argv, image, inputs, main, policy, proof
from isolated_tests.executor import Cleanup, Outcome, cleanup_owned, run_bounded

IMG = "sha256:" + "a" * 64


def _argv():
    return docker_argv.run_argv(docker=Path("/usr/bin/docker"), socket=Path("/s.sock"),
                                cli_home=Path("/h"), image=IMG, python="/opt/py/bin/python3",
                                snapshot=Path("/snap"), run_id="0" * 16, script="true")


def test_log_driver_none_and_fixed_shm():
    argv = _argv()
    assert [a for a in argv if a.startswith("--log-")] == ["--log-driver=none"]
    assert f"--shm-size={policy.LIMITS['shm_bytes']}" in argv


def test_tmpfs_aggregate_is_512mib_and_all_mounted():
    assert policy.tmpfs_total() == policy.TMPFS_TOTAL_MAX
    mounted = [a.split("=", 1)[1].split(":", 1)[0] for a in _argv() if a.startswith("--tmpfs=")]
    assert sorted(mounted) == sorted(policy.TMPFS)


def test_single_test_worker():
    assert "HERMES_TEST_WORKERS=1" in docker_argv.container_env("/opt/py/bin/python3")


def _inspect(**cfg):
    base = {"Env": ["PATH=/usr/bin"], "Volumes": None}
    base.update(cfg)
    return json.dumps([{"Id": IMG, "Config": base, "Os": "linux"}]).encode()


def test_image_with_volume_refused():
    with pytest.raises(inputs.Refused):
        image.check_image_inspect(_inspect(Volumes={"/data": {}}), IMG)


@pytest.mark.parametrize("key", ["LD_PRELOAD", "GCONV_PATH", "PYTHONPATH", "PYTHONSTARTUP"])
def test_image_loader_env_refused(key):
    with pytest.raises(inputs.Refused):
        image.check_image_inspect(_inspect(Env=["PATH=/usr/bin", f"{key}=/x"]), IMG)


def test_image_metadata_hash_changes_with_env():
    a = image.check_image_inspect(_inspect(), IMG)
    b = image.check_image_inspect(_inspect(Env=["PATH=/usr/bin", "LANG=C"]), IMG)
    assert a["metadata_sha256"] != b["metadata_sha256"]


def _outcome(**kw):
    base = dict(returncode=0, timed_out=False, interrupted=None, stdout=b"", stderr=b"",
                stdout_truncated=False, stderr_truncated=False, readers_done=True,
                cleanup_status="absent")
    base.update(kw)
    return Outcome(**base)


@pytest.mark.parametrize("kw", [{"cleanup_status": "unverified"}, {"cleanup_status": "label_mismatch"},
                                {"interrupted": "SIGTERM"}, {"stdout_truncated": True},
                                {"readers_done": False}, {"returncode": 125}])
def test_untrusted_results_are_invalid(kw):
    assert main._map_rc(_outcome(**kw)) == policy.EXIT_INVALID


def _cleanup(tmp_path, label, stderr="Error: No such container: x", inspect_rc=0):
    state = tmp_path / "c"
    state.write_text(label)
    insp = [sys.executable, "-c", "import os,sys\n"
            f"p={str(state)!r}\n"
            f"if not os.path.exists(p): sys.stderr.write({stderr!r}); sys.exit(1)\n"
            f"print(open(p).read()); sys.exit({inspect_rc})"]
    rm = [sys.executable, "-c", f"import os; os.unlink({str(state)!r})"]
    return Cleanup(insp, rm, "mine"), state


def test_label_mismatch_is_never_removed(tmp_path):
    c, state = _cleanup(tmp_path, "someone-else")
    assert cleanup_owned(c, {}, 10) == "label_mismatch" and state.exists()


def test_inspect_error_is_not_absent(tmp_path):
    c, state = _cleanup(tmp_path, "mine", stderr="Cannot connect to the Docker daemon")
    state.unlink()
    assert cleanup_owned(c, {}, 10) == "unverified"


def test_cleanup_runs_after_launch_exception(tmp_path):
    c, state = _cleanup(tmp_path, "mine")
    out = run_bounded(["/nonexistent/docker"], {}, wall_seconds=5, stdout_cap=10, stderr_cap=10,
                      cleanup=c, cleanup_timeout=10)
    assert out.interrupted and out.cleanup_status == "removed" and not state.exists()


def test_tty_expiry_is_hold(tmp_path):
    import os
    r, w = os.pipe()
    tty = os.fdopen(r, "r")

    class T:
        def isatty(self): return True
        def write(self, _s): pass
        def flush(self): pass
        def fileno(self): return tty.fileno()
        def readline(self): return tty.readline()
    with pytest.raises(proof.ProofRefused, match="HOLD"):
        proof.confirm(T(), "x", timeout=0.5)
    os.close(w)


def _bind():
    return {"k": "v"}


@pytest.mark.parametrize("mutate", [
    lambda d: d["probe_results"].update({"no_caps": False}),
    lambda d: d.pop("probe_output_sha256"),
    lambda d: d.update({"wall_canary": {"timed_out": True, "cleanup_status": "unverified"}}),
    lambda d: d.pop("confirmed_by"),
])
def test_damaged_attestation_refused(mutate):
    from isolated_tests.probe import REQUIRED_CHECKS
    doc = {"probe_results": {k: True for k in REQUIRED_CHECKS}, "probe_output_sha256": "0" * 64,
           "wall_canary": {"timed_out": True, "cleanup_status": "removed"},
           "confirmed_by": attestation.CONFIRM_MARKER, "recorded_at": "t"}
    attestation.check_structure(doc)
    mutate(doc)
    with pytest.raises(attestation.AttestationError):
        attestation.check_structure(doc)


def test_runtime_change_makes_binding_stale():
    kw = dict(image=IMG, python="/p", socket="/s", docker="/d", manifest_sha256="m", suite="s",
              image_meta={"image_id": IMG, "metadata_sha256": "x"})
    a = attestation.binding(server={"Version": "1"}, **kw)
    b = attestation.binding(server={"Version": "2"}, **kw)
    assert policy.sha256_json(a) != policy.sha256_json(b)


def test_media_delivery_suite_registered_with_exact_files():
    paths = policy.SUITES["media-delivery"]
    assert len(paths) == len(set(paths)) and all(p.endswith(".py") for p in paths)
