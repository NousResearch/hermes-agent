from __future__ import annotations

import hashlib
import json
import os
import pwd
import shlex
import subprocess
import sys
import venv
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
DEPLOY_HELPER = ROOT / "ops/muncho/runtime/muncho-auto-deploy-release"
REVISION = "a" * 40
CONFIG_SHA256 = "b" * 64
PR_NUMBER = "101"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _systemd_show(fragment: Path, drop_in: Path | None) -> str:
    return "\n".join(
        (
            "Names=hermes-cloud-gateway.service",
            f"FragmentPath={fragment}",
            "LoadState=loaded",
            "UnitFileState=enabled",
            f"DropInPaths={'' if drop_in is None else drop_in}",
            "NeedDaemonReload=no",
            "TriggeredBy=",
            "Triggers=",
            "",
        )
    )


def _unit_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "fragment": tmp_path / "etc/systemd/system/hermes-cloud-gateway.service",
        "drop_in": (
            tmp_path
            / "etc/systemd/system/hermes-cloud-gateway.service.d"
            / "20-discord-connector.conf"
        ),
        "plan": tmp_path / "cutover/staged/cutover-plan.json",
        "releases": tmp_path / "releases",
        "active": tmp_path / "active",
        "state": tmp_path / "state",
        "reports": tmp_path / "reports",
        "home": tmp_path / "home",
        "operations": tmp_path / "operations.log",
        "show": tmp_path / "systemd-show.txt",
        "bin": tmp_path / "bin",
    }


def _write_pinned_topology(paths: dict[str, Path]) -> dict[str, str]:
    fragment = paths["fragment"]
    drop_in = paths["drop_in"]
    release = paths["releases"] / f"hermes-agent-{REVISION[:12]}"
    interpreter = release / ".venv/bin/python"
    marker = release / ".codex-source-commit"
    fragment.parent.mkdir(parents=True)
    drop_in.parent.mkdir(parents=True)
    marker.parent.mkdir(parents=True)
    marker.write_text(REVISION + "\n", encoding="ascii")
    marker.chmod(0o444)
    fragment.write_text(
        "\n".join(
            (
                "# Exact SHA-bound Cloud Muncho production gateway; do not edit.",
                f"# ReleaseRevision={REVISION}",
                f"# ConfigSHA256={CONFIG_SHA256}",
                "[Unit]",
                f"AssertPathExists={interpreter}",
                f"AssertPathExists={marker}",
                "[Service]",
                f"WorkingDirectory={release}",
                f"Environment=PYTHONPATH={release}",
                (
                    f"ExecStartPre=+{interpreter} -B -P -s -m "
                    "gateway.production_capability_prerequisites collect "
                    f"--revision {REVISION} --config-sha256 {CONFIG_SHA256} "
                    "--lifecycle-phase committed"
                ),
                (
                    f"ExecStart={interpreter} -B -P -s -m gateway.run "
                    "/opt/adventico-ai-platform/hermes-home/config.yaml "
                    "--require-production-model-sovereignty "
                    f"--production-release-revision {REVISION} "
                    f"--production-config-sha256 {CONFIG_SHA256}"
                ).replace(
                    "gateway.run /opt",
                    "gateway.run --config /opt",
                ),
                f"ReadOnlyPaths={release}",
                "",
            )
        ),
        encoding="utf-8",
    )
    fragment.chmod(0o644)
    drop_in.write_text(
        "[Unit]\nBindsTo=muncho-discord-connector.service\n",
        encoding="utf-8",
    )
    drop_in.chmod(0o644)
    fragment_sha256 = hashlib.sha256(fragment.read_bytes()).hexdigest()
    drop_in_sha256 = hashlib.sha256(drop_in.read_bytes()).hexdigest()
    target = {
        "name": "hermes-cloud-gateway.service",
        "fragment_path": str(fragment),
        "fragment_sha256": fragment_sha256,
        "load_state": "loaded",
        "unit_file_state": "enabled",
        "drop_in_paths": [str(drop_in)],
        "drop_in_sha256": {str(drop_in): drop_in_sha256},
        "need_daemon_reload": False,
        "triggered_by": [],
        "triggers": [],
    }
    runtime_unsigned = {
        "schema": "muncho-production-owner-runtime-attestation.v1",
        "revision": REVISION,
        "manifest_sha256": "1" * 64,
        "tree_sha256": "2" * 64,
        "interpreter_sha256": "3" * 64,
        "pyvenv_cfg_sha256": "4" * 64,
        "sys_path_sha256": "5" * 64,
        "required_modules_sha256": "6" * 64,
        "module_origins_release_local": True,
        "ambient_python_environment_present": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    runtime_attestation = {
        **runtime_unsigned,
        "attestation_sha256": hashlib.sha256(
            _canonical(runtime_unsigned)
        ).hexdigest(),
    }
    unsigned = {
        "schema": "muncho-production-legacy-cutover-plan.v2",
        "release_revision": REVISION,
        "gateway_target_identity": target,
        "owner_runtime_attestation": runtime_attestation,
        "secret_material_recorded": False,
    }
    plan = {
        **unsigned,
        "plan_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    paths["plan"].parent.mkdir(parents=True)
    paths["plan"].write_bytes(_canonical(plan))
    paths["plan"].chmod(0o400)
    paths["show"].write_text(
        _systemd_show(fragment, drop_in),
        encoding="utf-8",
    )
    return {
        "fragment_sha256": fragment_sha256,
        "drop_in_sha256": drop_in_sha256,
        "plan_sha256": plan["plan_sha256"],
    }


def _write_legacy_topology(paths: dict[str, Path]) -> None:
    fragment = paths["fragment"]
    fragment.parent.mkdir(parents=True)
    fragment.write_text(
        "\n".join(
            (
                "[Unit]",
                "Description=Legacy Cloud Muncho gateway",
                "[Service]",
                f"WorkingDirectory={paths['active']}",
                (
                    f"ExecStart={paths['active']}/.venv/bin/python "
                    "-m gateway.run --config "
                    "/opt/adventico-ai-platform/hermes-home/config.yaml"
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    fragment.chmod(0o644)
    paths["show"].write_text(
        _systemd_show(fragment, None),
        encoding="utf-8",
    )


def _fake_commands(paths: dict[str, Path]) -> dict[str, str]:
    paths["bin"].mkdir()
    systemctl = paths["bin"] / "systemctl"
    _executable(
        systemctl,
        "#!/bin/sh\n"
        "printf 'systemctl:%s\\n' \"$*\" >>\"$TEST_OPERATIONS\"\n"
        "if [ \"$1\" = show ]; then cat \"$TEST_SYSTEMD_SHOW\"; exit 0; fi\n"
        "exit 97\n",
    )
    _executable(
        paths["bin"] / "systemd-run",
        "#!/bin/sh\n"
        "printf 'systemd-run:%s\\n' \"$*\" >>\"$TEST_OPERATIONS\"\n"
        "exit 0\n",
    )
    system_id = paths["bin"] / "id"
    _executable(
        system_id,
        "#!/bin/sh\n"
        "if [ \"$1\" = -u ]; then printf '%s\\n' \"$TEST_UID\"; exit 0; fi\n"
        "if [ \"$1\" = -g ]; then printf '%s\\n' \"$TEST_GID\"; exit 0; fi\n"
        "exit 95\n",
    )
    for name in ("git", "ln", "mv", "rm"):
        _executable(
            paths["bin"] / name,
            "#!/bin/sh\n"
            f"printf '{name}:%s\\n' \"$*\" >>\"$TEST_OPERATIONS\"\n"
            "exit 96\n",
        )
    _executable(
        paths["bin"] / "sudo",
        "#!/bin/sh\n"
        "if [ \"${TEST_SUDO_PASSTHROUGH:-0}\" = 1 ]; then "
        "shift 3; exec \"$@\"; fi\n"
        "printf 'sudo:%s\\n' \"$*\" >>\"$TEST_OPERATIONS\"\n"
        "exit 96\n",
    )
    return {
        **os.environ,
        "PATH": f"{paths['bin']}:{os.environ['PATH']}",
        "DEPLOY_HELPER": str(DEPLOY_HELPER),
        "TEST_FRAGMENT": str(paths["fragment"]),
        "TEST_DROP_IN": str(paths["drop_in"]),
        "TEST_PLAN": str(paths["plan"]),
        "TEST_RELEASES": str(paths["releases"]),
        "TEST_ACTIVE": str(paths["active"]),
        "TEST_STATE": str(paths["state"]),
        "TEST_REPORTS": str(paths["reports"]),
        "TEST_HOME": str(paths["home"]),
        "TEST_SYSTEMCTL": str(systemctl),
        "TEST_SYSTEM_ID": str(system_id),
        "TEST_SYSTEMD_SHOW": str(paths["show"]),
        "TEST_OPERATIONS": str(paths["operations"]),
        "TEST_UID": str(os.getuid()),
        # macOS temporary directories may inherit a group different from the
        # caller's primary group.  The production contract is exact ownership,
        # so bind the test classifier to the fixture's observed group too.
        "TEST_GID": str(paths["fragment"].stat().st_gid),
        "TEST_OWNER": pwd.getpwuid(os.getuid()).pw_name,
        "TEST_SYSTEM_PYTHON": sys.executable,
        "TEST_SHA": REVISION,
        "TEST_PR": PR_NUMBER,
    }


def _shell_prefix() -> str:
    return r'''
source "$DEPLOY_HELPER"
GATEWAY_FRAGMENT="$TEST_FRAGMENT"
GATEWAY_CONNECTOR_DROP_IN="$TEST_DROP_IN"
CUTOVER_PLAN_PATH="$TEST_PLAN"
RELEASES="$TEST_RELEASES"
ACTIVE_LINK="$TEST_ACTIVE"
STATE_DIR="$TEST_STATE"
REPORT_DIR="$TEST_REPORTS"
HERMES_HOME="$TEST_HOME"
GATEWAY_FRAGMENT_TRUSTED_UID="$TEST_UID"
GATEWAY_FRAGMENT_TRUSTED_GID="$TEST_GID"
SYSTEMCTL="$TEST_SYSTEMCTL"
SYSTEM_PYTHON="$TEST_SYSTEM_PYTHON"
SYSTEM_ID="$TEST_SYSTEM_ID"
OWNER="$TEST_OWNER"
DEPLOY_HEALTH_WAIT_SECONDS=0
'''


def _run_shell(paths: dict[str, Path], body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _shell_prefix() + "\n" + body],
        check=False,
        capture_output=True,
        text=True,
        env=_fake_commands(paths),
        timeout=20,
    )


def _receipt(paths: dict[str, Path]) -> dict[str, object]:
    return json.loads(
        (paths["state"] / "auto-sync-deploy-latest.json").read_text(
            encoding="utf-8"
        )
    )


def _release_site(
    paths: dict[str, Path],
    *,
    short: str = REVISION[:12],
) -> tuple[Path, Path]:
    release = (paths["releases"] / f"hermes-agent-{short}").resolve()
    site = release / ".venv/lib/python3.11/site-packages"
    site.mkdir(parents=True)
    return release, site


def _inherited_release(paths: dict[str, Path]) -> Path:
    inherited = (paths["releases"] / f"hermes-agent-{'c' * 12}").resolve()
    inherited.mkdir(parents=True)
    return inherited


def _write_installed_hermes_identity(
    release: Path,
    site: Path,
    *,
    direct_url: str,
) -> None:
    (release / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.18.2"\n',
        encoding="utf-8",
    )
    dist_info = site / "hermes_agent-0.18.2.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: hermes-agent\nVersion: 0.18.2\n",
        encoding="utf-8",
    )
    (dist_info / "INSTALLER").write_text("pip\n", encoding="utf-8")
    (dist_info / "RECORD").write_text(
        "gateway/__init__.py,,\nrun_agent.py,,\n",
        encoding="utf-8",
    )
    (dist_info / "direct_url.json").write_text(
        json.dumps({"dir_info": {}, "url": direct_url}),
        encoding="utf-8",
    )
    for relative in (
        "gateway/__init__.py",
        "gateway/production_capability_prerequisites.py",
        "run_agent.py",
    ):
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# installed target wheel file\n", encoding="utf-8")


def _run_boundary(
    paths: dict[str, Path],
    action: str,
    release: Path,
    inherited: Path,
) -> subprocess.CompletedProcess[str]:
    _write_legacy_topology(paths)
    return _run_shell(
        paths,
        "release_venv_site_boundary "
        f"{shlex.quote(action)} "
        f"{shlex.quote(str(release))} "
        f"{shlex.quote(str(inherited))}",
    )


def test_release_venv_prepare_removes_only_reviewed_hermes_editable_hooks(
    tmp_path: Path,
) -> None:
    paths = _unit_paths(tmp_path)
    release, site = _release_site(paths)
    inherited = _inherited_release(paths)
    module = "__editable___hermes_agent_0_16_0_finder"
    editable = site / "__editable__.hermes_agent-0.16.0.pth"
    editable.write_text(
        f"import {module}; {module}.install()",
        encoding="utf-8",
    )
    finder = site / f"{module}.py"
    finder.write_text("raise AssertionError('must never execute')\n", encoding="utf-8")
    cache = site / "__pycache__"
    cache.mkdir()
    cached = cache / f"{module}.cpython-311.pyc"
    cached.write_bytes(b"stale editable bytecode")
    orphan_module = "__editable___hermes_agent_0_15_0_finder"
    orphan_finder = site / f"{orphan_module}.py"
    orphan_finder.write_text("# orphaned editable finder\n", encoding="utf-8")
    orphan_cached = cache / f"{orphan_module}.cpython-311.pyc"
    orphan_cached.write_bytes(b"orphaned editable bytecode")
    distutils = site / "distutils-precedence.pth"
    distutils.write_text(
        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
        "os.environ.get(var, 'local') == 'local'; enabled and "
        "__import__('_distutils_hack').add_shim(); \n",
        encoding="utf-8",
    )
    hack = site / "_distutils_hack/__init__.py"
    hack.parent.mkdir()
    hack.write_text("# reviewed internal shim\n", encoding="utf-8")

    completed = _run_boundary(paths, "prepare", release, inherited)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(site)
    assert not editable.exists()
    assert not finder.exists()
    assert not cached.exists()
    assert not orphan_finder.exists()
    assert not orphan_cached.exists()
    assert distutils.exists()
    assert hack.exists()


@pytest.mark.parametrize(
    ("filename", "payload", "reason"),
    (
        (
            "unknown-import.pth",
            "import attacker\n",
            "unreviewed_pth_import_hook",
        ),
        (
            "outside.pth",
            "/var/tmp/outside-release\n",
            "absolute_pth_path_present",
        ),
        (
            "__editable__.other_project-1.0.pth",
            "/var/tmp/outside-release\n",
            "unreviewed_editable_pth_present",
        ),
        (
            "sitecustomize.py",
            "raise AssertionError('must never execute')\n",
            "site_customizer_present",
        ),
    ),
)
def test_release_venv_boundary_blocks_unreviewed_or_external_injection(
    tmp_path: Path,
    filename: str,
    payload: str,
    reason: str,
) -> None:
    paths = _unit_paths(tmp_path)
    release, site = _release_site(paths)
    inherited = _inherited_release(paths)
    (site / filename).write_text(payload, encoding="utf-8")

    completed = _run_boundary(paths, "prepare", release, inherited)

    assert completed.returncode == 12
    assert completed.stdout == ""
    assert (
        f"BLOCKED_RELEASE_VENV_SITE_BOUNDARY:{reason}" in completed.stderr
    )


def test_release_venv_attest_requires_noneditable_target_wheel_identity(
    tmp_path: Path,
) -> None:
    paths = _unit_paths(tmp_path)
    release, site = _release_site(paths)
    inherited = _inherited_release(paths)
    _write_installed_hermes_identity(
        release,
        site,
        direct_url=release.as_uri(),
    )

    completed = _run_boundary(paths, "attest", release, inherited)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(site)


def test_post_rename_attest_accepts_same_sha_staging_metadata_and_final_imports(
    tmp_path: Path,
) -> None:
    paths = _unit_paths(tmp_path)
    release = (paths["releases"] / f"hermes-agent-{REVISION[:12]}").resolve()
    release.mkdir(parents=True)
    venv.EnvBuilder(with_pip=False).create(release / ".venv")
    sites = sorted((release / ".venv/lib").glob("python3.*/site-packages"))
    assert len(sites) == 1
    site = sites[0]
    inherited = _inherited_release(paths)
    staging = release.parent / f".hermes-agent-{REVISION[:12]}.tmp.123"
    _write_installed_hermes_identity(
        release,
        site,
        direct_url=staging.as_uri(),
    )
    _write_legacy_topology(paths)

    completed = _run_shell(
        paths,
        "export TEST_SUDO_PASSTHROUGH=1\n"
        "attest_target_release_venv "
        f"{shlex.quote(str(release))} {shlex.quote(str(inherited))}",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "target_release_venv_imports=ok"


@pytest.mark.parametrize("source_kind", ("different_short", "outside_parent"))
def test_release_venv_attest_rejects_unbound_local_wheel_source(
    tmp_path: Path,
    source_kind: str,
) -> None:
    paths = _unit_paths(tmp_path)
    release, site = _release_site(paths)
    inherited = _inherited_release(paths)
    if source_kind == "different_short":
        source = release.parent / f".hermes-agent-{'b' * 12}.tmp.123"
    else:
        source = tmp_path / "outside" / f".hermes-agent-{REVISION[:12]}.tmp.123"
    _write_installed_hermes_identity(
        release,
        site,
        direct_url=source.resolve().as_uri(),
    )

    completed = _run_boundary(paths, "attest", release, inherited)

    assert completed.returncode == 12
    assert (
        "BLOCKED_RELEASE_VENV_SITE_BOUNDARY:"
        "direct_url_source_outside_target_release" in completed.stderr
    )


def test_exact_signed_sha_pinned_topology_is_detected(tmp_path: Path) -> None:
    paths = _unit_paths(tmp_path)
    identities = _write_pinned_topology(paths)

    completed = _run_shell(paths, "gateway_deploy_topology_json")

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "schema": "muncho-legacy-auto-deploy-topology.v1",
        "classification": "sha_pinned",
        "reason_code": "signed_sha_pinned_topology",
        "fragment_path": str(paths["fragment"]),
        "release_revision": REVISION,
        "release_root": str(
            paths["releases"] / f"hermes-agent-{REVISION[:12]}"
        ),
        "fragment_sha256": identities["fragment_sha256"],
        "connector_drop_in_sha256": identities["drop_in_sha256"],
        "cutover_plan_sha256": identities["plan_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


@pytest.mark.parametrize(
    ("entrypoint", "stage"),
    (("run_deploy", "pre_deploy"), ("start_unit", "pre_unit_start")),
)
def test_pinned_topology_blocks_before_every_legacy_mutation(
    tmp_path: Path,
    entrypoint: str,
    stage: str,
) -> None:
    paths = _unit_paths(tmp_path)
    _write_pinned_topology(paths)
    retained = paths["releases"] / "hermes-agent-111111111111"
    retained.mkdir(parents=True)
    paths["active"].symlink_to(retained, target_is_directory=True)

    completed = _run_shell(
        paths,
        f'{entrypoint} "$TEST_SHA" "$TEST_PR"',
    )

    assert completed.returncode == 8
    assert "BLOCKED_SHA_PINNED_TOPOLOGY" in completed.stderr
    receipt = _receipt(paths)
    assert receipt["status"] == "blocked_legacy_deploy_sha_pinned_topology"
    assert receipt["stage"] == stage
    assert receipt["legacy_symlink_deploy_allowed"] is False
    assert receipt["replacement_gate"] == "owner_approved_signed_cutover_only"
    assert receipt["gateway_topology"]["classification"] == "sha_pinned"
    assert receipt["target_commit"] == REVISION
    assert receipt["pr_number"] == int(PR_NUMBER)
    assert paths["active"].resolve() == retained.resolve()
    assert retained.is_dir()
    operations = paths["operations"].read_text(encoding="utf-8")
    assert operations.count("systemctl:show ") == 1
    assert not any(
        marker in operations
        for marker in (
            "systemctl:restart",
            "systemd-run:",
            "git:",
            "sudo:",
            "ln:",
            "mv:",
            "rm:",
        )
    )
    assert receipt["status"] != "deploy_pass"


@pytest.mark.parametrize("mutation", ("fragment_drift", "missing_plan", "drop_in_drift"))
def test_partial_or_drifted_pinned_topology_blocks_as_ambiguous(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _unit_paths(tmp_path)
    _write_pinned_topology(paths)
    if mutation == "fragment_drift":
        with paths["fragment"].open("a", encoding="utf-8") as handle:
            handle.write("# drift\n")
    elif mutation == "missing_plan":
        paths["plan"].unlink()
    else:
        with paths["drop_in"].open("a", encoding="utf-8") as handle:
            handle.write("# drift\n")

    completed = _run_shell(
        paths,
        'run_deploy "$TEST_SHA" "$TEST_PR"',
    )

    assert completed.returncode == 8
    receipt = _receipt(paths)
    assert receipt["status"] == "blocked_gateway_deploy_topology_ambiguous"
    assert receipt["gateway_topology"]["classification"] == "ambiguous"
    assert receipt["gateway_topology"]["reason_code"] in {
        "cutover_gateway_target_drifted",
        "pinned_topology_cutover_plan_missing",
        "cutover_gateway_drop_in_drifted",
    }
    operations = paths["operations"].read_text(encoding="utf-8")
    assert "systemd-run:" not in operations
    assert "systemctl:restart" not in operations
    assert "sudo:" not in operations


def test_trusted_legacy_symlink_topology_keeps_existing_start_behavior(
    tmp_path: Path,
) -> None:
    paths = _unit_paths(tmp_path)
    _write_legacy_topology(paths)

    classified = _run_shell(paths, "gateway_deploy_topology_json")
    assert classified.returncode == 0, classified.stderr
    assert json.loads(classified.stdout)["classification"] == "legacy"

    paths = _unit_paths(tmp_path / "start")
    _write_legacy_topology(paths)
    completed = _run_shell(
        paths,
        'start_unit "$TEST_SHA" "$TEST_PR"',
    )

    assert completed.returncode == 0, completed.stderr
    receipt = _receipt(paths)
    assert receipt["status"] == "deploy_unit_started"
    operations = paths["operations"].read_text(encoding="utf-8")
    assert operations.count("systemctl:show ") == 1
    assert "systemd-run:--unit=muncho-auto-deploy-aaaaaaaaaaaa-pr101" in operations
    assert "systemctl:restart" not in operations


def test_unknown_legacy_fragment_is_not_treated_as_safe_legacy(
    tmp_path: Path,
) -> None:
    paths = _unit_paths(tmp_path)
    _write_legacy_topology(paths)
    paths["fragment"].write_text(
        "[Service]\nWorkingDirectory=/tmp\nExecStart=/bin/false\n",
        encoding="utf-8",
    )

    completed = _run_shell(
        paths,
        'start_unit "$TEST_SHA" "$TEST_PR"',
    )

    assert completed.returncode == 8
    receipt = _receipt(paths)
    assert receipt["status"] == "blocked_gateway_deploy_topology_ambiguous"
    assert (
        receipt["gateway_topology"]["reason_code"]
        == "legacy_topology_not_symlink_bound"
    )
    assert "systemd-run:" not in paths["operations"].read_text(encoding="utf-8")
