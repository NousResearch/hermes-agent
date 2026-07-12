"""Installed-wheel regression for the privileged Canonical Writer runtime.

The service runs with ``python -I`` and therefore cannot import the repository's
unpackaged ``scripts`` namespace.  This test builds the real wheel, installs it
without the source tree, and reaches the first typed PING dispatch so lazy
runtime imports are covered as well as bootstrap imports.
"""

from __future__ import annotations

import os
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
    "gateway/canonical_writer_bootstrap.py",
    "gateway/canonical_writer_deployment_preflight.py",
    "gateway/canonical_writer_gateway_bootstrap.py",
    "gateway/canonical_writer_readiness.py",
    "gateway/canonical_writer_root_collector.py",
    "gateway/canonical_writer_service.py",
}
_FORBIDDEN_SCRIPT_MODULES = {
    "scripts/canonical_writer_bootstrap.py",
    "scripts/canonical_writer_service.py",
}


@pytest.mark.integration
@pytest.mark.skipif(
    os.name == "nt",
    reason="Canonical Writer requires Linux peer credentials",
)
def test_installed_wheel_runs_first_canonical_writer_ping(tmp_path):
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
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    required_names = {"cryptography", "pyyaml"}
    bootstrap_requirements = [
        requirement.split(";", 1)[0]
        for requirement in project["project"]["dependencies"]
        if requirement.split("==", 1)[0].split("[", 1)[0].casefold()
        in required_names
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

    probe = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path
        from types import SimpleNamespace

        import gateway.canonical_writer_bootstrap as bootstrap_module
        import gateway.canonical_writer_gateway_bootstrap as gateway_bootstrap_module
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
        assert "/site-packages/gateway/canonical_writer_bootstrap.py" in (
            bootstrap_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_service.py" in (
            service_module.__file__.replace("\\\\", "/")
        )
        assert "/site-packages/gateway/canonical_writer_gateway_bootstrap.py" in (
            gateway_bootstrap_module.__file__.replace("\\\\", "/")
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
    run = subprocess.run(
        [str(interpreter), "-I", "-c", probe],
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
