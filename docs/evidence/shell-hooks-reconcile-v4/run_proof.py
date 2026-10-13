"""Run only the two V4 focal files in an offline, disposable Docker snapshot.

Usage: python3 docs/evidence/shell-hooks-reconcile-v4/run_proof.py LABEL [--base-source] [--inflight]
No live checkout, HOME, credentials, or Docker socket is mounted in the container.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

BASE = "db72e58229639b80d68a53cdc1d081f6511457f5"
IMAGE = "sha256:72f85bad31968f2c90fd869e5e23bfcf2160e57dd8610515a1ff8921d2421975"
ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
FILES = ["agent/shell_hooks.py", "tests/agent/test_shell_hooks.py", "tests/tui_gateway/test_profile_shell_hooks.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--base-source", action="store_true")
    parser.add_argument("--inflight", action="store_true")
    parser.add_argument("--scratch", type=Path, default=Path.home() / ".hermes/cache/scratch")
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        parser.error("label must contain letters, numbers or hyphens")
    scratch = args.scratch
    with tempfile.TemporaryDirectory(prefix="v4-proof-", dir=scratch) as temp:
        archive = Path(temp) / "snapshot.tar"
        subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", BASE], cwd=ROOT, check=True)
        hashes = {}
        with tarfile.open(archive, "a") as tar:
            for name in FILES:
                if args.base_source and name == "agent/shell_hooks.py":
                    content = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
                else:
                    content = (ROOT / name).read_bytes()
                hashes[name] = hashlib.sha256(content).hexdigest()
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(content))
        command = (
            "mkdir -p /scratch/home /scratch/tmp && tar xf /snapshot.tar -C /repo && "
            "git init -q && git add agent tests scripts && "
            "bash scripts/run_tests.sh -j 2 --file-retries 0 --file-timeout 120 "
            "tests/agent/test_shell_hooks.py "
        )
        if args.inflight:
            command += "-k 'test_config_reconciliation_preserves_policy_snapshot and inflight' "
        else:
            command += "tests/tui_gateway/test_profile_shell_hooks.py "
        command += "-v --tb=short"
        docker = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "2g",
                  "-e", "HOME=/scratch/home", "-e", "TMP=/scratch/tmp", "-e", "TEMP=/scratch/tmp",
                  "-v", f"{archive}:/snapshot.tar:ro", IMAGE, "sh", "-c", command]
        result = subprocess.run(docker, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        log = EVIDENCE / f"{args.label}.log"
        log.write_bytes(result.stdout)
        receipt = {"base": BASE, "image": IMAGE, "base_source": args.base_source,
                   "source_sha256": hashes, "command": docker, "returncode": result.returncode,
                   "log_sha256": hashlib.sha256(result.stdout).hexdigest()}
        (EVIDENCE / f"{args.label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(result.stdout.decode(errors="replace"))
        print(json.dumps(receipt, indent=2))
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
