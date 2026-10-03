from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_mcp_transport_dependencies_are_core_runtime_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["dependencies"])

    assert "mcp==2.0.0; python_version >= '3.14'" in dependencies
    assert "httpx2==2.7.0; python_version >= '3.14'" in dependencies
    assert "starlette==1.3.1; python_version >= '3.14'" in dependencies
