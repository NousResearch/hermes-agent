import os

from hermes_os_integration.health import check_runtime_health
from hermes_os_integration.wrapper import default_launcher_path


def test_runtime_health_degrades_for_missing_launcher(tmp_path):
    status = check_runtime_health(str(tmp_path / "missing-hermes-agent"))

    assert status.available is False
    assert status.provider == "official-hermes-agent"


def test_hermes_agent_launcher_exists_without_replacing_hermes():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    assert os.path.exists(os.path.join(repo_root, "hermes"))
    assert os.path.exists(default_launcher_path())
    assert os.path.basename(default_launcher_path()) == "hermes-agent"
