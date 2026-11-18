"""Isolated test runner CLI: plan | prove-human | run.

Launch only as:  python3 -I -S scripts/isolated_tests/main.py <phase> ...
stdlib only; never imports Hermes. No passthrough of docker flags, env,
volumes or entrypoints. See website/docs/developer-guide/isolated-tests.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import secrets
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append: stdlib wins

from isolated_tests import attestation, docker_argv, inputs, policy, probe, proof  # noqa: E402
from isolated_tests import image as image_meta  # noqa: E402
from isolated_tests.executor import CLEAN_OK, Cleanup, run_bounded  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="isolated-tests", allow_abbrev=False)
    sub = p.add_subparsers(dest="phase", required=True)
    for name in ("plan", "prove-human", "run"):
        s = sub.add_parser(name, allow_abbrev=False)
        s.add_argument("--runtime", required=True)
        s.add_argument("--docker", required=True, help="absolute docker CLI path")
        s.add_argument("--socket", required=True, help="absolute local Unix daemon socket")
        s.add_argument("--image", required=True, help="local image ID sha256:<64hex>")
        s.add_argument("--python", required=True, help="offline test interpreter inside image")
        s.add_argument("--snapshot", required=True, help="clean source snapshot with manifest")
        s.add_argument("--suite", required=True)
        s.add_argument("--out", required=True, help="new output dir (must not exist)")
        s.add_argument("--inputs", required=True, help="parent-owned trusted prepare inputs (JSON)")
        s.add_argument("--receipt", required=True, help="prepare.py receipt bound to --inputs")
        if name == "run":
            s.add_argument("--attestation", required=True)
    return p


def _resolve(a) -> dict:
    if a.runtime not in policy.SUPPORTED_RUNTIMES:
        raise _Exit(policy.EXIT_UNSUPPORTED_RUNTIME, f"runtime {a.runtime!r} unsupported (needs_validation)")
    if sys.platform not in policy.SUPPORTED_HOST_OS:
        raise _Exit(policy.EXIT_UNSUPPORTED_RUNTIME, f"host {sys.platform} unsupported")
    # Provenance first: trusted inputs, then the receipt bound to them; only the
    # policy-checked applied_exceptions may relax the denylist for the walk.
    trusted = inputs.load_trusted_inputs(a.inputs)
    receipt = inputs.read_receipt(a.receipt, trusted)
    exceptions = dict(receipt["applied_exceptions"])
    snap = inputs.verify_snapshot(a.snapshot, exceptions)
    inputs.verify_receipt(receipt, trusted, snap)
    ctx = {
        "docker": inputs.docker_binary(a.docker),
        "socket": inputs.unix_socket(a.socket),
        "image": docker_argv.check_image(a.image),
        "python": docker_argv.check_container_python(a.python),
        "snapshot": snap,
        "suite_paths": inputs.check_suite(snap, a.suite),
        "out": inputs.new_output_dir(a.out, snap.root),
        "exceptions": exceptions,
    }
    return ctx


def _bind_runtime(ctx: dict, cli_home: Path, suite: str) -> None:
    """Offline metadata reads before any payload; refuses on any doubt."""
    env = docker_argv.host_env(cli_home)
    common = {"docker": ctx["docker"], "socket": ctx["socket"], "cli_home": cli_home}
    meta = _read(docker_argv.image_inspect_argv(image=ctx["image"], **common), env)
    ver = _read(docker_argv.server_version_argv(**common), env)
    ctx["binding"] = attestation.binding(
        image=ctx["image"], python=ctx["python"], socket=str(ctx["socket"]),
        docker=str(ctx["docker"]), manifest_sha256=ctx["snapshot"].manifest_sha256, suite=suite,
        image_meta=image_meta.check_image_inspect(meta, ctx["image"]),
        server=image_meta.check_server_version(ver))


def _read(argv: list[str], env: dict) -> bytes:
    lim = policy.LIMITS
    out = run_bounded(argv, env, wall_seconds=lim["cleanup_timeout_seconds"], stdout_cap=1 << 20,
                      stderr_cap=1 << 16, cleanup=None, cleanup_timeout=lim["cleanup_timeout_seconds"])
    if out.returncode != 0 or out.stdout_truncated or out.interrupted:
        raise inputs.Refused(f"docker metadata read failed: {argv[-3:]} rc={out.returncode}")
    return out.stdout


class _Exit(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _argv(ctx: dict, cli_home: Path, run_id: str, script: str, probe_env=()) -> list[str]:
    return docker_argv.run_argv(docker=ctx["docker"], socket=ctx["socket"], cli_home=cli_home,
                                image=ctx["image"], python=ctx["python"],
                                snapshot=ctx["snapshot"].root, run_id=run_id,
                                script=script, probe_env=probe_env)


def _execute(ctx: dict, cli_home: Path, run_id: str, argv: list[str], wall=None):
    lim = policy.LIMITS
    common = {"docker": ctx["docker"], "socket": ctx["socket"], "cli_home": cli_home, "run_id": run_id}
    return run_bounded(
        argv, docker_argv.host_env(cli_home), wall_seconds=wall or lim["wall_seconds"],
        stdout_cap=lim["stdout_cap_bytes"], stderr_cap=lim["stderr_cap_bytes"],
        cleanup=Cleanup(docker_argv.label_argv(**common), docker_argv.rm_argv(**common), run_id),
        cleanup_timeout=lim["cleanup_timeout_seconds"])


def _manifest_sha(ctx: dict, rel: str) -> str:
    doc = json.loads((ctx["snapshot"].root / inputs.MANIFEST_NAME).read_bytes())
    return doc["files"][rel]["sha256"]


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _outcome_trusted(outcome) -> bool:
    """Owned cleanup confirmed, no signal, both readers drained, nothing truncated."""
    return (outcome.cleanup_status in CLEAN_OK and not outcome.interrupted and outcome.readers_done
            and not outcome.stdout_truncated and not outcome.stderr_truncated)


def _map_rc(outcome) -> int:
    """INVALID whenever the result cannot be trusted, before any other code."""
    if not _outcome_trusted(outcome):
        return policy.EXIT_INVALID
    if outcome.timed_out:
        return policy.EXIT_TIMEOUT
    if outcome.returncode == policy.CONTAINER_PREREQ_EXIT:
        return policy.EXIT_PREREQUISITE_MISSING
    if outcome.returncode == 125:  # docker CLI/daemon error: unknown container state
        return policy.EXIT_INVALID
    if outcome.returncode in (126, 127):
        return policy.EXIT_PREREQUISITE_MISSING
    return policy.EXIT_OK if outcome.returncode == 0 else policy.EXIT_TARGET_FAILED


def _phase_plan(ctx: dict, run_id: str) -> int:
    cli_home = ctx["out"] / "docker-cli"
    doc = {"binding": ctx["binding"], "policy": policy.policy_document(),
           "run_argv": _argv(ctx, cli_home, run_id, docker_argv.target_script(ctx["suite_paths"])),
           "host_env": docker_argv.host_env(cli_home), "status": "PLANNED_NOT_RUN"}
    print(json.dumps(doc, indent=2, sort_keys=True))
    return policy.EXIT_OK


def _phase_prove(ctx: dict, run_id: str, cli_home: Path, tty) -> int:
    proof.confirm(tty, "About to start the containment canary container (no target code).")
    canary = ctx["out"] / f"host-canary-{secrets.token_hex(8)}"
    _write_new(canary, b"host-only canary\n")
    env = (f"ISOTEST_HOST_CANARY={canary}",
           f"ISOTEST_HOST_USER={pwd.getpwuid(os.getuid()).pw_name}",
           "ISOTEST_LIMITS=" + json.dumps(dict(policy.LIMITS), separators=(",", ":")),
           "ISOTEST_TMPFS=" + json.dumps(dict(policy.TMPFS), separators=(",", ":")),
           "ISOTEST_RESIDUAL=" + json.dumps(list(policy.DOCKER_RESIDUAL_RW), separators=(",", ":")),
           "ISOTEST_EXPECT_ENV=" + json.dumps(sorted(e.split("=", 1)[0] for e in
                                                     docker_argv.container_env(ctx["python"])),
                                              separators=(",", ":")),
           "ISOTEST_SRC_HASHES=" + json.dumps({"scripts/run_tests.sh": _manifest_sha(ctx, "scripts/run_tests.sh")},
                                              separators=(",", ":")))
    outcome = _execute(ctx, cli_home, run_id, _argv(ctx, cli_home, run_id,
                                                    docker_argv.probe_script(probe.PROBE_SOURCE), env))
    rc = _map_rc(outcome)  # same validity rules as a target run; untrusted stdout is never parsed
    if rc == policy.EXIT_INVALID:
        tty.write(json.dumps({"exit": outcome.returncode, "timed_out": outcome.timed_out},
                             sort_keys=True) + "\n")
        tty.write("Canary INVALID: outcome untrusted (cleanup/signal/readers/truncation/rc); "
                  "no attestation written.\n")
        return policy.EXIT_INVALID
    try:
        results = probe.parse_probe(outcome.stdout)
    except ValueError:  # malformed ISOTEST_PROBE JSON proves nothing: a failed proof, not a crash
        results = {}
    tty.write(json.dumps({"exit": outcome.returncode, "timed_out": outcome.timed_out,
                          "checks": results}, indent=2, sort_keys=True) + "\n")
    if rc != policy.EXIT_OK or not probe.all_passed(results):
        tty.write("Canary FAILED: containment not proven; no attestation written.\n")
        return policy.EXIT_PROOF_FAILED
    wall_id = secrets.token_hex(8)
    wall = _execute(ctx, cli_home, wall_id, _argv(ctx, cli_home, wall_id, probe.WALL_CANARY_SCRIPT),
                    wall=probe.WALL_CANARY_SECONDS)
    wall_doc = {"timed_out": wall.timed_out, "cleanup_status": wall.cleanup_status}
    tty.write(json.dumps({"wall_canary": wall_doc}) + "\n")
    # The timeout is the expected result here (returncode None); never promoted to a target PASS.
    if not _outcome_trusted(wall):
        tty.write("Wall-clock kill canary INVALID: outcome untrusted; no attestation written.\n")
        return policy.EXIT_INVALID
    if wall_doc != {"timed_out": True, "cleanup_status": "removed"}:
        tty.write("Wall-clock kill canary FAILED; no attestation written.\n")
        return policy.EXIT_PROOF_FAILED
    proof.confirm(tty, "Review every check above. Confirm only if all are as expected.")
    import datetime
    attestation.write(ctx["out"] / "attestation.json", ctx["binding"], results,
                      probe.output_digest(outcome.stdout), wall_doc,
                      datetime.datetime.now(datetime.timezone.utc).isoformat())
    tty.write(f"Attestation written: {ctx['out'] / 'attestation.json'}\n")
    return policy.EXIT_OK


def _phase_run(ctx: dict, run_id: str, cli_home: Path, att: str) -> int:
    try:
        attestation.verify(att, ctx["binding"])
    except attestation.AttestationError as exc:
        raise _Exit(policy.EXIT_ATTESTATION, str(exc)) from None
    outcome = _execute(ctx, cli_home, run_id, _argv(ctx, cli_home, run_id,
                                                    docker_argv.target_script(ctx["suite_paths"])))
    _write_new(ctx["out"] / "stdout.log", outcome.stdout)
    _write_new(ctx["out"] / "stderr.log", outcome.stderr)
    rc = _map_rc(outcome)
    try:
        after = inputs.verify_snapshot(str(ctx["snapshot"].root), ctx["exceptions"])
        changed = after.manifest_sha256 != ctx["snapshot"].manifest_sha256
    except inputs.Refused:
        changed = True
    if changed:
        rc = policy.EXIT_INVALID
    _write_new(ctx["out"] / "result.json", json.dumps({
        "exit": rc, "container_exit": outcome.returncode, "timed_out": outcome.timed_out,
        "stdout_truncated": outcome.stdout_truncated, "stderr_truncated": outcome.stderr_truncated,
        "cleanup_status": outcome.cleanup_status, "interrupted": outcome.interrupted,
        "snapshot_changed": changed, "valid": rc not in (policy.EXIT_INVALID, policy.EXIT_TIMEOUT)},
        sort_keys=True).encode())
    return rc


def main(argv: list[str] | None = None, *, tty_opener=proof.open_tty) -> int:
    args = _parser().parse_args(argv)
    run_id = secrets.token_hex(8)
    try:
        ctx = _resolve(args)
        if args.phase == "plan":
            ctx["binding"] = {"status": "runtime binding requires docker metadata reads; NOT_RUN in plan"}
            return _phase_plan(ctx, run_id)
        os.mkdir(ctx["out"], 0o700)
        cli_home = ctx["out"] / "docker-cli"
        os.mkdir(cli_home, 0o700)
        os.mkdir(cli_home / "docker-config", 0o700)
        try:
            _bind_runtime(ctx, cli_home, args.suite)
            if args.phase == "prove-human":
                return _phase_prove(ctx, run_id, cli_home, tty_opener())
            return _phase_run(ctx, run_id, cli_home, args.attestation)
        finally:
            shutil.rmtree(cli_home, ignore_errors=True)
    except inputs.ProvenanceInvalid as exc:
        print(f"invalid provenance: {exc}", file=sys.stderr)
        return policy.EXIT_INVALID
    except inputs.Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return policy.EXIT_REFUSED
    except proof.ProofRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return policy.EXIT_PROOF_FAILED
    except _Exit as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001 - never let a launcher bug look like exit 1
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return policy.EXIT_INTERNAL
    except KeyboardInterrupt:  # Ctrl-C outside the executor's signal gate proves nothing
        print("interrupted", file=sys.stderr)
        return policy.EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
