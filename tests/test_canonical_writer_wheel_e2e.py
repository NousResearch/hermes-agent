"""Installed-wheel regression for the privileged Canonical Writer runtime.

The service runs with ``python -B -I`` and therefore cannot import the repository's
unpackaged ``scripts`` namespace.  This test builds the real wheel, installs it
without the source tree, and reaches the first typed PING dispatch so lazy
runtime imports are covered as well as bootstrap imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import textwrap
import tomllib
import venv
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGED_MODULES = {
    "gateway/canonical_canary_bootstrap.py",
    "gateway/canonical_canary_host_identity.py",
    "gateway/canonical_full_canary_coordinator.py",
    "gateway/canonical_full_canary_e2e.py",
    "gateway/canonical_full_canary_live_driver.py",
    "gateway/canonical_full_canary_runtime.py",
    "gateway/canonical_writer_activation.py",
    "gateway/canonical_writer_bootstrap.py",
    "gateway/canonical_writer_config_collector.py",
    "gateway/canonical_writer_deployment_preflight.py",
    "gateway/canonical_writer_gateway_bootstrap.py",
    "gateway/canonical_writer_host_authority.py",
    "gateway/canonical_writer_planner.py",
    "gateway/canonical_writer_preflight_publisher.py",
    "gateway/canonical_writer_readiness.py",
    "gateway/canonical_writer_release_contract.py",
    "gateway/canonical_writer_root_collector.py",
    "gateway/canonical_writer_service.py",
    "gateway/full_canary_discord_edge_bootstrap.py",
    "plugins/muncho_canary_evidence/__init__.py",
    "plugins/muncho_canary_evidence/plugin.yaml",
}
_FORBIDDEN_SCRIPT_MODULES = {
    "scripts/canonical_writer_bootstrap.py",
    "scripts/canonical_writer_service.py",
}


def _bytecode_snapshot(root: Path) -> dict[str, str | None]:
    """Return a content snapshot of every bytecode path below ``root``."""

    snapshot: dict[str, str | None] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            continue
        snapshot[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return snapshot


@pytest.mark.integration
@pytest.mark.skipif(
    os.name == "nt",
    reason="Canonical Writer requires Linux peer credentials",
)
def test_installed_wheel_runs_first_canonical_writer_ping(tmp_path):
    fixture = runpy.run_path(
        str(REPO_ROOT / "tests/gateway/test_canonical_writer_planner.py")
    )
    source_plan = fixture["_final_plan"]()
    source_plan_path = tmp_path / "source-activation-plan.json"
    source_plan_path.write_text(
        json.dumps(
            source_plan.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    source_tree = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source_tree,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "venv",
            "build",
            "dist",
            "node_modules",
            "__pycache__",
            "*.pyc",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), "."],
        cwd=source_tree,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, f"uv build failed:\n{build.stderr}"
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found: {wheels}"
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
    assert _PACKAGED_MODULES <= packaged
    assert not (_FORBIDDEN_SCRIPT_MODULES & packaged)
    assert not any(name.startswith("scripts/") for name in packaged)

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    interpreter = venv_dir / "bin/python"
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required_names = {"cryptography", "pyyaml"}
    bootstrap_requirements = [
        requirement.split(";", 1)[0]
        for requirement in project["project"]["dependencies"]
        if requirement.split("==", 1)[0].split("[", 1)[0].casefold() in required_names
    ]
    assert {
        requirement.split("==", 1)[0].split("[", 1)[0].casefold()
        for requirement in bootstrap_requirements
    } == required_names
    subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "-q",
            *bootstrap_requirements,
        ],
        check=True,
        timeout=300,
    )
    subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        check=True,
        timeout=300,
    )
    site_packages_roots = list((venv_dir / "lib").glob("python*/site-packages"))
    assert len(site_packages_roots) == 1
    bytecode_before = _bytecode_snapshot(site_packages_roots[0])

    probe = textwrap.dedent(
        """
        import json
        import os
        import io
        from contextlib import redirect_stdout
        from pathlib import Path
        from types import SimpleNamespace

        import gateway.canonical_writer_bootstrap as bootstrap_module
        import gateway.canonical_writer_activation as activation_module
        import gateway.canonical_writer_config_collector as config_collector_module
        import gateway.canonical_writer_gateway_bootstrap as gateway_bootstrap_module
        import gateway.canonical_writer_host_authority as host_authority_module
        import gateway.canonical_writer_planner as planner_module
        import gateway.canonical_writer_preflight_publisher as publisher_module
        import gateway.canonical_writer_release_contract as release_contract_module
        import gateway.canonical_writer_service as service_module
        from gateway.canonical_writer_db import QueryResult
        from gateway.canonical_writer_postgres_backend import (
            PRODUCTION_CATALOG_SHA256,
            PRODUCTION_STATEMENT_CATALOG,
        )
        from gateway.canonical_writer_protocol import CanonicalWriterOperation


        class FakeDatabase:
            statement_names = PRODUCTION_STATEMENT_CATALOG.names
            statement_catalog_sha256 = PRODUCTION_CATALOG_SHA256

            def __init__(self, **_kwargs):
                self.attested = False

            def startup_attest(self):
                self.attested = True

            def query_fixed(self, statement_name, parameters):
                assert self.attested
                assert statement_name == "op_ping"
                assert parameters["request"] == {}
                response = json.dumps({"ok": True, "result": {"pong": True}})
                return QueryResult(("response",), ((response,),), "SELECT 1")


        config = SimpleNamespace(
            writer_uid=os.getuid(),
            writer_gid=os.getgid(),
            socket_gid=2345,
            gateway_uid=os.getuid(),
            owner_discord_user_ids=frozenset(),
            gateway_unit="hermes-cloud-gateway.service",
            socket_path=Path("/tmp/canonical-writer-wheel-test.sock"),
            connection_timeout_seconds=2.0,
            max_connections=1,
            database=object(),
            privileges=object(),
            discord_edge_authority=SimpleNamespace(enabled=False),
        )
        assembled = bootstrap_module.build_service(
            config,
            _database_factory=FakeDatabase,
        )
        result = assembled.server.dispatcher.dispatch(
            CanonicalWriterOperation.PING,
            {},
            service_module.DispatchContext(
                request_id="11111111-1111-4111-8111-111111111111",
                sequence=1,
                deadline_unix_ms=1,
                idempotency_key=None,
                peer=service_module.PeerCredentials(
                    pid=os.getpid(),
                    uid=os.getuid(),
                    gid=os.getgid(),
                ),
                runtime={},
            ),
        )
        assert assembled.database.attested is True
        assert result.status == "ok"
        assert result.result == {"pong": True}
        packaged_plan_raw = json.loads(
            Path(os.environ["PACKAGED_ACTIVATION_PLAN"]).read_text(
                encoding="utf-8"
            )
        )
        packaged_plan = activation_module.ActivationPlan.from_mapping(
            packaged_plan_raw
        )
        assert packaged_plan.to_mapping() == packaged_plan_raw
        assert packaged_plan.sha256 == packaged_plan_raw[
            "activation_plan_sha256"
        ]
        native_result = {
            "artifact_sha256": "1" * 64,
            "native_observation_plan_sha256": "2" * 64,
        }
        planner_module.build_and_stage_native_observation_plan = (
            lambda **_arguments: native_result
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert planner_module.main([
                "build-native-plan",
                "--revision",
                "a" * 40,
                "--external-iam-policy-sha256",
                "3" * 64,
                "--config-collector-receipt-sha256",
                "4" * 64,
            ]) == 0
        assert json.loads(stdout.getvalue()) == native_result
        final_result = {
            "activation_plan_sha256": "5" * 64,
            "native_observation_receipt_sha256": "6" * 64,
        }
        planner_module.build_and_stage_final_activation_plan = (
            lambda **_arguments: final_result
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert planner_module.main([
                "build-final-plan",
                "--native-observation-receipt-sha256",
                "6" * 64,
            ]) == 0
        assert json.loads(stdout.getvalue()) == final_result
        assert all(
            key.endswith("_sha256")
            for result in (native_result, final_result)
            for key in result
        )
        assert "/site-packages/gateway/canonical_writer_bootstrap.py" in (
            bootstrap_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_service.py" in (
            service_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_gateway_bootstrap.py" in (
            gateway_bootstrap_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_activation.py" in (
            activation_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_config_collector.py" in (
            config_collector_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_host_authority.py" in (
            host_authority_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_planner.py" in (
            planner_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_preflight_publisher.py" in (
            publisher_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_release_contract.py" in (
            release_contract_module.__file__.replace("\\\\", "/")
        )
        forbidden = (
            "agent",
            "tools",
            "run_agent",
            "gateway.run",
            "gateway.platforms",
            "hermes_cli.config",
            "hermes_cli.env_loader",
            "model_tools",
            "cron",
            "plugins",
            "providers",
            "dotenv",
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in tuple(__import__("sys").modules)
            for prefix in forbidden
        )
        """
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PACKAGED_ACTIVATION_PLAN"] = str(source_plan_path)
    run = subprocess.run(
        [str(interpreter), "-B", "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert run.returncode == 0, (
        "installed Canonical Writer wheel probe failed:\n"
        f"stdout: {run.stdout}\nstderr: {run.stderr}"
    )
    help_run = subprocess.run(
        [
            str(interpreter),
            "-B",
            "-I",
            "-m",
            "gateway.canonical_writer_activation",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "install-approval" in help_run.stdout
    assert "install-external-iam" in help_run.stdout
    assert "observe-native" in help_run.stdout
    assert "validate-plan" in help_run.stdout
    config_help_run = subprocess.run(
        [
            str(interpreter),
            "-B",
            "-I",
            "-m",
            "gateway.canonical_writer_config_collector",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert config_help_run.returncode == 0, config_help_run.stderr
    assert "--release-manifest-file-sha256" in config_help_run.stdout
    assert "--owner-discord-user-id" in config_help_run.stdout
    planner_help_run = subprocess.run(
        [
            str(interpreter),
            "-B",
            "-I",
            "-m",
            "gateway.canonical_writer_planner",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert planner_help_run.returncode == 0, planner_help_run.stderr
    assert "build-native-plan" in planner_help_run.stdout
    assert "build-final-plan" in planner_help_run.stdout
    publisher_help_run = subprocess.run(
        [
            str(interpreter),
            "-B",
            "-I",
            "-m",
            "gateway.canonical_writer_preflight_publisher",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert publisher_help_run.returncode == 0, publisher_help_run.stderr
    assert "plan" in publisher_help_run.stdout
    assert "apply" in publisher_help_run.stdout
    coordinator_help_run = subprocess.run(
        [
            str(interpreter),
            "-B",
            "-I",
            "-m",
            "gateway.canonical_full_canary_coordinator",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert coordinator_help_run.returncode == 0, coordinator_help_run.stderr
    assert "publish-coordinator-input" in coordinator_help_run.stdout
    assert "preflight-owner-launch" in coordinator_help_run.stdout
    assert "preflight-recovery" in coordinator_help_run.stdout
    assert "recover" in coordinator_help_run.stdout
    assert "finalize-recovery" in coordinator_help_run.stdout
    assert "install-discord-token" in coordinator_help_run.stdout
    assert "install-final-approval" in coordinator_help_run.stdout
    assert "stop-and-retire-discord-token" in coordinator_help_run.stdout
    assert _bytecode_snapshot(site_packages_roots[0]) == bytecode_before
