"""Regression coverage for the minimal installer profile."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _optional_deps() -> dict[str, list[str]]:
    return tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]


def test_pyproject_keeps_minimal_base_light() -> None:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    base = "\n".join(project["dependencies"])

    for optional_dep in [
        "exa-py",
        "firecrawl-py",
        "parallel-web",
        "websockets",
        "fal-client",
        "croniter",
        "edge-tts",
        "fastapi",
        "uvicorn",
        "faster-whisper",
        "sounddevice",
        "slack-bolt",
    ]:
        assert optional_dep not in base


def test_pyproject_defines_profile_and_feature_extras() -> None:
    extras = _optional_deps()
    for name in [
        "minimal",
        "standard",
        "web-search",
        "browser",
        "image-gen",
        "tts",
        "voice",
        "cron",
        "dashboard",
        "web",
        "termux-minimal",
        "termux",
        "termux-all",
        "all",
    ]:
        assert name in extras

    assert extras["minimal"] == []
    assert any(dep.startswith("exa-py") for dep in extras["web-search"])
    assert any(dep.startswith("firecrawl-py") for dep in extras["web-search"])
    assert any(dep.startswith("parallel-web") for dep in extras["web-search"])
    assert any(dep.startswith("websockets") for dep in extras["browser"])
    assert any(dep.startswith("fal-client") for dep in extras["image-gen"])
    assert any(dep.startswith("edge-tts") for dep in extras["tts"])
    assert any(dep.startswith("croniter") for dep in extras["cron"])


def test_install_script_defaults_to_minimal_profile_and_selected_extras() -> None:
    text = INSTALL_SH.read_text()
    assert 'INSTALL_PROFILE="${HERMES_INSTALL_PROFILE:-minimal}"' in text
    assert "WITH_FEATURES=()" in text
    assert "--profile NAME Install profile: minimal (default), standard, full" in text
    assert "resolve_python_extras()" in text
    assert 'minimal) extras+=("minimal") ;;' in text
    assert 'standard) extras+=("standard") ;;' in text
    assert 'has_feature "browser" && extras+=("browser")' in text
    assert 'echo "all"' in text
    assert 'install_target=".[${extras}]"' in text
    assert 'uv pip install -e ".[all]"' not in text


def test_installer_invokes_setup_with_install_profile_not_hermes_profile() -> None:
    text = INSTALL_SH.read_text()
    assert 'setup --install-profile "$INSTALL_PROFILE"' in text
    assert 'setup --profile "$INSTALL_PROFILE"' not in text


def test_setup_install_profile_flag_does_not_require_named_hermes_profile(tmp_path) -> None:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "setup",
            "--install-profile",
            "minimal",
            "--non-interactive",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert "Profile 'minimal' does not exist" not in output
    assert result.returncode == 0, output


def test_install_profile_flag_can_be_combined_with_named_hermes_profile(tmp_path) -> None:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-root")
    env.pop("PYTHONPATH", None)

    create = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "profile", "create", "dev"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert create.returncode == 0, create.stdout + create.stderr

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "--profile",
            "dev",
            "setup",
            "--install-profile",
            "minimal",
            "--non-interactive",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert "Profile 'minimal' does not exist" not in output
    assert result.returncode == 0, output


def test_install_script_gates_heavy_probes_for_minimal_profile() -> None:
    text = INSTALL_SH.read_text()
    assert "should_check_node()" in text
    assert "should_check_ffmpeg()" in text
    assert "should_check_web_network()" in text
    assert "Skipping Node.js check" in text
    assert "Skipping Node.js dependencies" in text
    assert "Skipping ffmpeg check" in text
    assert 'checks=("https://pypi.org/simple/")' in text
    assert 'checks+=("https://duckduckgo.com/")' in text


def test_dashboard_missing_dependencies_message_points_to_feature_extra() -> None:
    text = (REPO_ROOT / "hermes_cli" / "main.py").read_text()
    assert "Dashboard dependencies are not installed." in text
    assert "hermes install-feature dashboard" in text
    assert "pip install 'hermes-agent[web]'" in text


def test_tui_and_install_feature_have_minimal_profile_opt_in_messages() -> None:
    text = (REPO_ROOT / "hermes_cli" / "main.py").read_text()
    assert "def cmd_install_feature(args):" in text
    assert '"browser", "tts", "voice", "dashboard", "tui", "gateway", "web",' in text
    assert "TUI dependencies are not installed" in text
    assert "hermes install-feature tui" in text
    assert "HERMES_TUI_AUTO_INSTALL=1" in text


def test_browser_supervisor_does_not_require_websockets_at_module_import() -> None:
    text = (REPO_ROOT / "tools" / "browser_supervisor.py").read_text()
    assert "except ImportError:  # Browser/CDP support is optional" in text
    assert "def _require_websockets():" in text
    assert "Install them with `hermes install-feature browser`" in text

    import importlib

    importlib.import_module("tools.browser_supervisor")


def test_image_generation_tool_does_not_import_fal_client_at_module_import() -> None:
    text = (REPO_ROOT / "tools" / "image_generation_tool.py").read_text()
    assert "import fal_client\n" not in text
    assert "def _require_fal_client():" in text
    assert "Install them with `hermes install-feature image-gen`" in text
