import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRAPLING_DIR = REPO_ROOT / "optional-skills" / "research" / "scrapling"
SETUP_SCRIPT = SCRAPLING_DIR / "scripts" / "setup_runtime.py"
EXTRACT_SCRIPT = SCRAPLING_DIR / "scripts" / "scrapling_extract.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_setup_runtime_help_is_available():
    result = subprocess.run(
        [sys.executable, str(SETUP_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--install-browsers" in result.stdout
    assert "--python" in result.stdout
    assert "--runtime-dir" in result.stdout


def test_setup_runtime_build_plan_keeps_browser_install_opt_in(tmp_path):
    module = load_module(SETUP_SCRIPT, "scrapling_setup_runtime_test")

    plan = module.build_setup_plan(
        python_executable="/custom/python3.11",
        runtime_dir=tmp_path / "runtime",
        requirements_file=SCRAPLING_DIR / "requirements.txt",
        install_browsers=False,
    )

    rendered = [" ".join(map(str, command)) for command in plan.commands]
    assert any("venv" in command for command in rendered)
    assert any("pip" in command and "install" in command for command in rendered)
    assert not any("scrapling install" in command for command in rendered)
    assert plan.browser_install_requested is False


def test_setup_runtime_build_plan_includes_browser_install_only_when_requested(tmp_path):
    module = load_module(SETUP_SCRIPT, "scrapling_setup_runtime_browser_test")

    plan = module.build_setup_plan(
        python_executable="/custom/python3.11",
        runtime_dir=tmp_path / "runtime",
        requirements_file=SCRAPLING_DIR / "requirements.txt",
        install_browsers=True,
    )

    rendered = [" ".join(map(str, command)) for command in plan.commands]
    assert rendered[-1].endswith("bin/scrapling install")
    assert plan.browser_install_requested is True


def test_setup_runtime_dry_run_prints_json_plan(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SETUP_SCRIPT),
            "--dry-run",
            "--python",
            "/custom/python3.11",
            "--runtime-dir",
            str(tmp_path / "runtime"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["backend"] == "scrapling"
    assert payload["action"] == "setup_runtime"
    assert payload["dry_run"] is True
    assert payload["browser_install_requested"] is False
    assert payload["results"] == []
    assert payload["errors"] == []
    assert any("-m venv" in command for command in payload["commands"])
    assert not any("scrapling install" in command for command in payload["commands"])


def test_extract_help_is_available():
    result = subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--url" in result.stdout
    assert "--selector" in result.stdout
    assert "--mode" in result.stdout


def test_extract_receipt_schema_success_with_injected_fetcher():
    module = load_module(EXTRACT_SCRIPT, "scrapling_extract_test")

    def fake_fetcher(*, url, selector, selector_type, mode, timeout, wait_selector, network_idle, max_chars):
        assert url == "https://example.com/page"
        assert selector == "article"
        assert selector_type == "css"
        assert mode == "static"
        return "Extracted article text"

    receipt = module.run_extract(
        url="https://example.com/page",
        selector="article",
        selector_type="css",
        mode="static",
        timeout=20,
        wait_selector=None,
        network_idle=False,
        max_chars=1000,
        fallback_reason="selector_required",
        fetcher=fake_fetcher,
    )

    assert receipt["backend"] == "scrapling"
    assert receipt["mode"] == "static"
    assert receipt["url"] == "https://example.com/page"
    assert receipt["selector"] == "article"
    assert receipt["selector_type"] == "css"
    assert receipt["content"] == "Extracted article text"
    assert isinstance(receipt["elapsed_ms"], int)
    assert receipt["fallback_reason"] == "selector_required"
    assert receipt["errors"] == []


def test_extract_cli_returns_structured_json_error_for_missing_dependency(monkeypatch, capsys):
    module = load_module(EXTRACT_SCRIPT, "scrapling_extract_missing_dep_test")

    def missing_dependency(**kwargs):
        raise ModuleNotFoundError("No module named 'scrapling'")

    exit_code = module.main(
        [
            "--url",
            "https://example.com/page",
            "--selector",
            "article",
            "--mode",
            "static",
        ],
        fetcher=missing_dependency,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["backend"] == "scrapling"
    assert payload["mode"] == "static"
    assert payload["url"] == "https://example.com/page"
    assert payload["selector"] == "article"
    assert payload["content"] == ""
    assert payload["errors"]
    assert payload["errors"][0]["type"] == "ModuleNotFoundError"
