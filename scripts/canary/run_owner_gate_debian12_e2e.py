#!/usr/bin/env python3
"""Run the owner-gate offline wheel bootstrap in clean Debian 12 amd64.

The fetch phase verifies every artifact against the canonical runtime lock.
The container is then disconnected from Docker networking before venv creation,
signed-pip execution, target installation, inventory, replay, and hostile-tree
checks.  The disposable host pip wheel is corrupted deliberately to prove it is
not an authority or execution input.

This harness begins at the verified wheelhouse boundary because production
release-signing and owner-authority private keys are intentionally unavailable
to CI.  The behavioral package test materializes a package with ephemeral trust
and hands that exact built layout to Stage0's runtime-lock and wheel pre-parser,
covering the package-to-Stage0 binding without weakening production trust.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import Sequence


DEBIAN_12_IMAGE = (
    "public.ecr.aws/docker/library/debian@"
    "sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818"
)
FETCH_AND_VERIFY = r"""
import hashlib
import json
import pathlib
import shutil
import urllib.request

from scripts.canary import owner_gate_package as package

repo = pathlib.Path("/repo")
lock_raw = (repo / package.RUNTIME_LOCK_RELATIVE).read_bytes()
lock = package.decode_runtime_lock(lock_raw)
wheelhouse = pathlib.Path("/tmp/owner-gate-wheelhouse")
bundle = pathlib.Path("/tmp/owner-gate-bundle")
wheelhouse.mkdir(mode=0o755)
(bundle / "wheels").mkdir(parents=True, mode=0o755)
(bundle / "bootstrap").mkdir(mode=0o755)

bootstrap = {
    key: lock["bootstrap_pip"][key]
    for key in ("filename", "project", "version", "sha256", "size")
}
wheels = [
    {
        key: item[key]
        for key in ("filename", "project", "version", "sha256", "size")
    }
    for item in lock["wheels"]
]
for item in (bootstrap, *wheels):
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{item['project']}/{item['version']}/json",
        timeout=30,
    ) as response:
        release = json.load(response)
    matches = [
        candidate
        for candidate in release["urls"]
        if candidate["filename"] == item["filename"]
    ]
    if len(matches) != 1:
        raise RuntimeError("owner_gate_e2e_artifact_lookup_invalid")
    source = matches[0]
    if (
        source["digests"]["sha256"] != item["sha256"]
        or source["size"] != item["size"]
    ):
        raise RuntimeError("owner_gate_e2e_artifact_metadata_invalid")
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        raw = response.read(item["size"] + 1)
    if (
        len(raw) != item["size"]
        or hashlib.sha256(raw).hexdigest() != item["sha256"]
    ):
        raise RuntimeError("owner_gate_e2e_artifact_digest_invalid")
    target = wheelhouse / item["filename"]
    target.write_bytes(raw)
    target.chmod(0o444)

unsigned = {
    "schema": package.WHEELHOUSE_SCHEMA,
    "python_version": lock["python_version"],
    "platform": lock["platform"],
    "network_required": False,
    "source_build_allowed": False,
    "complete_transitive_closure": True,
    "runtime_lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
    "bootstrap_pip": bootstrap,
    "wheels": wheels,
}
manifest = {
    **unsigned,
    "manifest_sha256": package.foundation.sha256_json(unsigned),
}
verified = package.validate_wheelhouse(
    root=wheelhouse,
    manifest=manifest,
    runtime_lock=lock,
)
if len(verified) != len(wheels):
    raise RuntimeError("owner_gate_e2e_wheel_count_invalid")
for item in wheels:
    shutil.copy2(wheelhouse / item["filename"], bundle / "wheels" / item["filename"])
    (bundle / "wheels" / item["filename"]).chmod(0o444)
shutil.copy2(
    wheelhouse / bootstrap["filename"],
    bundle / "bootstrap" / bootstrap["filename"],
)
(bundle / "bootstrap" / bootstrap["filename"]).chmod(0o444)
(bundle / "wheelhouse-manifest.json").write_bytes(
    package.foundation.canonical_json_bytes(manifest) + b"\n"
)
(bundle / "wheelhouse-manifest.json").chmod(0o444)
print(
    "fetch: exact signed bootstrap pip + "
    f"{len(wheels)} target wheels verified"
)
"""

OFFLINE_INSTALL_AND_REPLAY = r"""
import hashlib
import json
import os
import pathlib
import platform
import shutil
import socket
import subprocess

from scripts.canary import owner_gate_stage0 as stage0

bundle = pathlib.Path("/tmp/owner-gate-bundle")
manifest = json.loads((bundle / "wheelhouse-manifest.json").read_bytes())
bootstrap = stage0._bootstrap_pip_artifact(manifest)
signed_pip = bundle / "bootstrap" / bootstrap["filename"]

os_release = stage0._read_exact_os_release(pathlib.Path("/etc/os-release"))
if os_release.get("ID") != "debian" or os_release.get("VERSION_ID") != "12":
    raise RuntimeError("owner_gate_e2e_os_identity_invalid")
if platform.machine() != "x86_64":
    raise RuntimeError("owner_gate_e2e_architecture_invalid")
python_version = subprocess.run(
    ("/usr/bin/python3", "--version"),
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
).stdout.strip()
if python_version != "Python 3.11.2":
    raise RuntimeError("owner_gate_e2e_python_version_invalid")
if pathlib.Path("/usr/bin/python3").readlink() != pathlib.Path("python3.11"):
    raise RuntimeError("owner_gate_e2e_python_link_invalid")
ownership = subprocess.run(
    ("dpkg-query", "--search", "/usr/bin/python3.11"),
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.strip()
if ownership != "python3.11-minimal: /usr/bin/python3.11":
    raise RuntimeError("owner_gate_e2e_python_ownership_invalid")
venv_ownership = subprocess.run(
    ("dpkg-query", "--search", "/usr/lib/python3.11/venv/__init__.py"),
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.strip()
if venv_ownership != (
    "libpython3.11-stdlib:amd64: /usr/lib/python3.11/venv/__init__.py"
):
    raise RuntimeError("owner_gate_e2e_venv_ownership_invalid")
integrity = subprocess.run(
    (
        "dpkg",
        "--verify",
        "python3-minimal",
        "python3.11-minimal",
        "libpython3.11-minimal",
        "libpython3.11-stdlib",
        "python3.11",
        "python3.11-venv",
        "python3-venv",
    ),
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
).stdout.strip()
allowed_slim_omissions = (
    "missing     /usr/share/doc/",
    "missing     /usr/share/man/",
    "missing     /usr/share/lintian/",
)
if any(
    not line.startswith(allowed_slim_omissions)
    for line in integrity.splitlines()
):
    raise RuntimeError("owner_gate_e2e_python_package_integrity_invalid")
host_wheel_root = pathlib.Path("/usr/share/python-wheels")
host_wheel_root.mkdir(parents=True, exist_ok=True)
host_wheels = tuple(host_wheel_root.glob("pip-*.whl"))
if not host_wheels:
    host_wheels = (host_wheel_root / bootstrap["filename"],)
for host_wheel in host_wheels:
    if host_wheel.exists():
        host_wheel.chmod(0o644)
    host_wheel.write_bytes(b"deliberately tampered host pip wheel\n")
    host_wheel.chmod(0o644)

try:
    socket.create_connection(("pypi.org", 443), timeout=1)
except OSError:
    pass
else:
    raise RuntimeError("owner_gate_e2e_network_not_disabled")

venv = pathlib.Path("/tmp/owner-gate-release/venv")
shutil.rmtree(venv.parent, ignore_errors=True)
stage0.validate_wheel_archives_for_install(bundle, manifest=manifest)
if venv.exists():
    raise RuntimeError("owner_gate_e2e_preparse_launched_venv")
environment = stage0._pip_command_environment()
subprocess.run(
    (
        "/usr/bin/python3",
        "-I",
        "-B",
        "-m",
        "venv",
        "--without-pip",
        "--copies",
        str(venv),
    ),
    check=True,
    env=environment,
)
site_packages = venv / "lib/python3.11/site-packages"
if any(site_packages.rglob("*")):
    raise RuntimeError("owner_gate_e2e_without_pip_not_empty")
wheels = tuple(
    str(bundle / "wheels" / item["filename"])
    for item in manifest["wheels"]
)
subprocess.run(
    stage0._pip_install_argv(venv, signed_pip, wheels),
    check=True,
    env=environment,
)
stage0.purge_generated_site_packages_bytecode(venv)
interpreter_sha256 = hashlib.sha256(
    pathlib.Path("/usr/bin/python3.11").read_bytes()
).hexdigest()
stage0.seal_and_validate_venv_executables(
    venv,
    interpreter_sha256=interpreter_sha256,
    purge_generated_bin=True,
)
stage0.validate_installed_site_packages(bundle, venv=venv, manifest=manifest)
if sorted(item.name for item in (venv / "bin").iterdir()) != ["python"]:
    raise RuntimeError("owner_gate_e2e_venv_bin_not_sealed")

def inventory():
    completed = subprocess.run(
        (
            str(venv / "bin/python"),
            "-I",
            "-B",
            "-c",
            stage0._runtime_inventory_probe_code(),
        ),
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
    )
    return stage0.validate_runtime_inventory(
        completed.stdout,
        venv=venv,
        manifest=manifest,
    )

first = inventory()
if len(first["distributions"]) != len(manifest["wheels"]) + 1:
    raise RuntimeError("owner_gate_e2e_distribution_count_invalid")
stage0.seal_and_validate_venv_executables(
    venv,
    interpreter_sha256=interpreter_sha256,
    purge_generated_bin=False,
)
stage0.validate_installed_site_packages(bundle, venv=venv, manifest=manifest)
if inventory() != first:
    raise RuntimeError("owner_gate_e2e_replay_inventory_changed")

injected = site_packages / "hostile-startup.pth"
injected.write_text("import attacker\n", encoding="utf-8")
injected.chmod(0o644)
invoked = False
original_run = stage0.subprocess.run
def forbidden(*_args, **_kwargs):
    global invoked
    invoked = True
    raise AssertionError("venv launch forbidden during Stage0 tree scan")
stage0.subprocess.run = forbidden
try:
    try:
        stage0.validate_installed_site_packages(
            bundle,
            venv=venv,
            manifest=manifest,
        )
    except stage0.OwnerGateStage0Error as exc:
        if str(exc) != "owner_gate_stage0_site_packages_invalid":
            raise
    else:
        raise RuntimeError("owner_gate_e2e_hostile_pth_accepted")
finally:
    stage0.subprocess.run = original_run
    injected.unlink()
if invoked:
    raise RuntimeError("owner_gate_e2e_scan_invoked_venv")
stage0.validate_installed_site_packages(bundle, venv=venv, manifest=manifest)
if hashlib.sha256(signed_pip.read_bytes()).hexdigest() != bootstrap["sha256"]:
    raise RuntimeError("owner_gate_e2e_signed_pip_changed")
if not all(
    host_wheel.read_bytes() == b"deliberately tampered host pip wheel\n"
    for host_wheel in host_wheels
):
    raise RuntimeError("owner_gate_e2e_host_pip_was_used")
print("offline: signed pip install, exact inventory/replay, host tamper, hostile .pth PASS")
"""


class HarnessError(RuntimeError):
    """Stable local harness failure."""


def _run(
    arguments: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: int = 300,
) -> str:
    try:
        completed = subprocess.run(
            tuple(arguments),
            check=True,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        output = getattr(exc, "stdout", None)
        if isinstance(output, str) and output:
            print(output, end="")
        raise HarnessError("owner_gate_debian12_e2e_failed") from None
    return completed.stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEBIAN_12_IMAGE)
    arguments = parser.parse_args(argv)
    repository = Path(__file__).resolve(strict=True).parents[2]
    container = f"muncho-owner-gate-e2e-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _run(("docker", "version", "--format", "{{.Server.Version}}"), timeout=30)
    try:
        _run((
            "docker",
            "run",
            "--platform",
            "linux/amd64",
            "--name",
            container,
            "--detach",
            "--volume",
            f"{repository}:/repo:ro",
            arguments.image,
            "sleep",
            "infinity",
        ))
        _run((
            "docker",
            "exec",
            container,
            "sh",
            "-c",
            (
                "apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                "ca-certificates python3 python3-venv python3-packaging "
                "python3-cryptography python3-yaml"
            ),
        ))
        print(_run(
            ("docker", "exec", "--interactive", container, "env", "PYTHONPATH=/repo", "python3", "-"),
            input_text=textwrap.dedent(FETCH_AND_VERIFY),
        ), end="")
        _run(("docker", "network", "disconnect", "bridge", container), timeout=30)
        print(_run(
            ("docker", "exec", "--interactive", container, "env", "PYTHONPATH=/repo", "python3", "-"),
            input_text=textwrap.dedent(OFFLINE_INSTALL_AND_REPLAY),
        ), end="")
    finally:
        subprocess.run(
            ("docker", "rm", "--force", container),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
