"""INCLUDE_BROWSER Dockerfile build arg + slim-image runtime diagnostics.

Static checks always run. A real `docker build --build-arg
INCLUDE_BROWSER=false` smoke test runs when Docker is available
(and HERMES_DOCKER_SMOKE is not set to 0) so the false-branch of the
conditional is actually exercised — pure Dockerfile string assertions
cannot catch a parse/runtime regression.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_exists(dockerfile_text):
    assert dockerfile_text


def test_include_browser_arg_declared(dockerfile_text):
    assert "ARG INCLUDE_BROWSER=true" in dockerfile_text


def test_slim_marker_env_exported(dockerfile_text):
    """Runtime needs HERMES_DOCKER_INCLUDE_BROWSER so browser_tool can tell
    intentional slim images apart from stale full images missing Chromium.
    """
    assert "HERMES_DOCKER_INCLUDE_BROWSER" in dockerfile_text
    assert "ENV HERMES_DOCKER_INCLUDE_BROWSER=${INCLUDE_BROWSER}" in dockerfile_text


def test_playwright_install_is_conditional(dockerfile_text):
    assert 'if [ "$INCLUDE_BROWSER" = "true" ]; then' in dockerfile_text
    install_line = "npx playwright install --with-deps chromium --only-shell"
    assert install_line in dockerfile_text

    if_idx = dockerfile_text.find('if [ "$INCLUDE_BROWSER" = "true" ]; then')
    fi_idx = dockerfile_text.find("fi", if_idx)
    install_idx = dockerfile_text.find(install_line)
    assert if_idx < install_idx < fi_idx


def test_no_unconditional_playwright_install(dockerfile_text):
    install_line = "npx playwright install --with-deps chromium --only-shell"
    occurrences = dockerfile_text.count(install_line)
    assert occurrences == 1, (
        f"Expected exactly 1 occurrence of the playwright install line "
        f"(inside the INCLUDE_BROWSER conditional), found {occurrences}"
    )


def test_slim_image_docker_hint_mentions_include_browser(monkeypatch):
    from tools import browser_tool as bt

    monkeypatch.setenv("HERMES_DOCKER_INCLUDE_BROWSER", "false")
    hint = bt._docker_chromium_missing_hint()
    assert "INCLUDE_BROWSER=false" in hint
    assert "intentionally" in hint.lower()
    assert "pull the latest image" not in hint


def test_full_image_docker_hint_still_suggests_pull(monkeypatch):
    from tools import browser_tool as bt

    monkeypatch.delenv("HERMES_DOCKER_INCLUDE_BROWSER", raising=False)
    hint = bt._docker_chromium_missing_hint()
    assert "docker pull" in hint


def _docker_available() -> bool:
    if os.environ.get("HERMES_DOCKER_SMOKE", "1") == "0":
        return False
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker daemon not available")
def test_slim_docker_build_smoke():
    """Real build of the INCLUDE_BROWSER=false branch + env marker smoke.

    Tags a throwaway image, asserts the build log skipped Playwright, and
    that the resulting image exports HERMES_DOCKER_INCLUDE_BROWSER=false.
    """
    tag = "hermes-agent:slim-smoke-test"
    try:
        proc = subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "INCLUDE_BROWSER=false",
                "-t",
                tag,
                ".",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        assert proc.returncode == 0, f"slim docker build failed:\n{log[-4000:]}"
        assert "skipping Playwright/Chromium install" in log or "INCLUDE_BROWSER=false" in log

        inspect = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
                tag,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        env_blob = inspect.stdout
        assert "HERMES_DOCKER_INCLUDE_BROWSER=false" in env_blob, (
            "slim image did not export HERMES_DOCKER_INCLUDE_BROWSER=false; "
            f"env was:\n{env_blob}"
        )
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
