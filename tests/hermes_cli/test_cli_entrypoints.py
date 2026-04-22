from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_hermes_agent_console_script_uses_real_cli_entrypoint():
    content = PYPROJECT.read_text(encoding="utf-8")
    assert 'hermes-agent = "hermes_cli.main:main"' in content
    assert 'hermes-agent = "run_agent:main"' not in content
