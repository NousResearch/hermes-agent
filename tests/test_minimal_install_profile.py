"""Regression coverage for Hermes installer install options."""

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


def test_pyproject_defines_install_option_and_feature_extras() -> None:
    extras = _optional_deps()
    for name in [
        "minimal",
        # Backcompat extra: still resolvable, but not a public install option.
        "standard",
        "web-search",
        "browser",
        "image-gen",
        "tts",
        "voice",
        "cron",
        "dashboard",
        # Hidden/backcompat alias for dashboard deps.
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
    assert extras["dashboard"] == ["hermes-agent[web]"]


def test_install_script_defaults_to_default_full_and_custom_minimal_extras() -> None:
    text = INSTALL_SH.read_text()
    assert 'INSTALL_OPTION="${HERMES_INSTALL_OPTION:-default}"' in text
    assert "INSTALL_OPTION_EXPLICIT=false" in text
    assert "WITH_FEATURES=()" in text
    assert "--install-option NAME  Install option: default (full), minimal, minimalTUI" in text
    assert "--minimal-tui  Alias for --install-option minimalTUI" in text
    assert "--full         Backward-compatible alias for --install-option default" in text
    assert "--profile NAME Install profile" not in text
    assert "resolve_python_extras()" in text
    assert 'if [ "$INSTALL_OPTION" = "default" ] || has_feature "all"; then' in text
    assert 'minimal|minimalTUI) extras+=("minimal" "web-search") ;;' in text
    assert 'has_feature "dashboard" && extras+=("dashboard")' in text
    assert 'has_feature "browser" && extras+=("browser")' in text
    assert 'echo "all"' in text
    assert 'install_target=".[${extras}]"' in text
    assert 'uv pip install -e ".[all]"' not in text


def test_install_script_public_features_and_hidden_web_alias() -> None:
    text = INSTALL_SH.read_text()
    assert "Valid features: browser, tts, voice, dashboard, tui, gateway, web-search, image-gen, cron, file, terminal, all" in text
    assert "--with web is deprecated; using --with dashboard instead" in text
    assert "web-search, image-gen, cron, file, terminal, all" in text
    assert "browser, tts, voice, dashboard, tui, gateway," in text
    assert "web feature" not in text.lower()


def test_default_config_uses_install_option_not_install_profile() -> None:
    text = (REPO_ROOT / "hermes_cli" / "config.py").read_text()
    assert '"install_option": "default"' in text
    assert '"toolsets": ["hermes-cli"]' in text
    assert '"install_profile"' not in text


def test_installer_invokes_setup_with_install_option_not_hermes_profile() -> None:
    text = INSTALL_SH.read_text()
    assert 'setup --install-option "$INSTALL_OPTION"' in text
    assert 'setup --install-profile "$INSTALL_OPTION"' not in text
    assert 'setup --profile "$INSTALL_OPTION"' not in text


def test_local_checkout_installer_uses_checkout_branch_and_remote_for_target_update() -> None:
    """Running scripts/install.sh from a fork branch must install that branch, not upstream main."""
    text = INSTALL_SH.read_text()
    assert "SOURCE_REPO_URL" in text
    assert "BRANCH_EXPLICIT=false" in text
    assert "detect_installer_source()" in text
    assert 'source_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"' in text
    assert 'branch_remote="$(git -C "$source_root" config "branch.${source_branch}.remote" 2>/dev/null || true)"' in text
    assert 'BRANCH="$source_branch"' in text
    assert 'git remote set-url origin "$SOURCE_REPO_URL"' in text
    assert 'git fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"' in text
    assert 'git checkout -B "$BRANCH" "origin/$BRANCH"' in text


def test_setup_install_option_flag_does_not_require_named_hermes_profile(tmp_path) -> None:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "setup",
            "--install-option",
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


def test_setup_install_option_accepts_minimal_tui_alias(tmp_path) -> None:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "setup",
            "--install-option",
            "minimal-tui",
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
    assert "Unknown install option" not in output
    assert result.returncode == 0, output
    config_text = (Path(env["HERMES_HOME"]) / "config.yaml").read_text()
    assert "install_option: minimalTUI" in config_text
    assert "hermes-minimal" in config_text


def test_install_option_flag_can_be_combined_with_named_hermes_profile(tmp_path) -> None:
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
            "--install-option",
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


def test_install_script_gates_heavy_probes_for_minimal_option() -> None:
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
    assert "pip install 'hermes-agent[dashboard]'" in text
    assert "pip install 'hermes-agent[web]'" not in text


def test_tui_and_install_feature_have_minimal_option_opt_in_messages() -> None:
    text = (REPO_ROOT / "hermes_cli" / "main.py").read_text()
    assert "def cmd_install_feature(args):" in text
    assert '"browser", "tts", "voice", "dashboard", "tui", "gateway",' in text
    assert '"web-search", "image-gen", "cron", "full", "all",' in text
    assert 'aliases = {"web": "dashboard"}' in text
    assert "Feature 'web' is deprecated; installing 'dashboard' instead." in text
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
