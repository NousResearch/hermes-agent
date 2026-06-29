"""Docker startup contracts for KarinAI managed-runtime containers."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGED_ENV = REPO_ROOT / "karinai" / "docker" / "managed-env.sh"
MAIN_WRAPPER = REPO_ROOT / "docker" / "main-wrapper.sh"
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def test_managed_env_helper_maps_product_paths_to_upstream_env(tmp_path: Path) -> None:
    envdir = tmp_path / "envdir"
    script = f"""
set -eu
. {MANAGED_ENV}
export KARINAI_S6_ENVDIR={envdir}
export KARINAI_MANAGED_RUNTIME=true
export KARINAI_RUNTIME_STATE_DIR=/hermes
export KARINAI_WORKSPACE_DIR=/workspace
karinai_apply_managed_bootstrap_env
printf 'HERMES_HOME=%s\n' "$HERMES_HOME"
printf 'HOME=%s\n' "$HOME"
printf 'TERMINAL_CWD=%s\n' "$TERMINAL_CWD"
printf 'HERMES_WRITE_SAFE_ROOT=%s\n' "$HERMES_WRITE_SAFE_ROOT"
printf 'HERMES_DASHBOARD=%s\n' "$HERMES_DASHBOARD"
"""
    proc = subprocess.run(["/bin/sh", "-c", script], text=True, capture_output=True, check=True)

    lines = dict(line.split("=", 1) for line in proc.stdout.strip().splitlines())
    assert lines == {
        "HERMES_HOME": "/hermes",
        "HOME": "/hermes/home",
        "TERMINAL_CWD": "/workspace",
        "HERMES_WRITE_SAFE_ROOT": "/workspace",
        "HERMES_DASHBOARD": "false",
    }
    assert (envdir / "HERMES_HOME").read_text(encoding="utf-8") == "/hermes"
    assert (envdir / "TERMINAL_CWD").read_text(encoding="utf-8") == "/workspace"
    assert (envdir / "HERMES_WRITE_SAFE_ROOT").read_text(encoding="utf-8") == "/workspace"
    assert (envdir / "HERMES_DASHBOARD").read_text(encoding="utf-8") == "false"


def test_stage2_applies_managed_paths_before_bootstrapping_hermes_home() -> None:
    text = STAGE2_HOOK.read_text(encoding="utf-8")
    assert "karinai/docker/managed-env.sh" in text
    assert "karinai_apply_managed_bootstrap_env" in text
    assert text.index("karinai_apply_managed_bootstrap_env") < text.index('mkdir -p "$HERMES_HOME"')
    assert "KARINAI_DOCKER_MANAGED_RUNTIME" in text
    assert 'mkdir -p "$KARINAI_WORKSPACE_DIR"' in text
    assert "write_managed_model_gateway_config" in text
    assert 'if [ "$KARINAI_DOCKER_MANAGED_RUNTIME" = true ]; then' in text
    assert text.index("docker_config_migrate.py") < text.index("write_managed_model_gateway_config")


def test_main_wrapper_routes_managed_default_to_start_managed() -> None:
    text = MAIN_WRAPPER.read_text(encoding="utf-8")
    assert "karinai/docker/managed-env.sh" in text
    assert "karinai_managed_runtime_enabled" in text
    assert "python -m karinai.runtime.start_managed" in text
    assert text.index("karinai_apply_managed_bootstrap_env") < text.index("if [ $# -eq 0 ]")
    assert text.index("python -m karinai.runtime.start_managed") < text.index("drop hermes")
    assert 'if [ "$1" = "karinai-managed-runtime" ]; then' in text


def test_dockerfile_documents_managed_startup_path() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "KARINAI_MANAGED_RUNTIME=true" in text
    assert "karinai-managed-runtime" in text
    assert "python -m karinai.runtime.start_managed" in text
