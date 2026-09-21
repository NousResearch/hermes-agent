import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "feature-completion-notify.py"


def _module(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("feature_completion_notify", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_paths_and_state_are_scoped_to_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    module = _module(monkeypatch, hermes_home)

    assert module.BOARDS_ROOT == hermes_home / "kanban" / "boards"
    assert module.STATE_FILE == hermes_home / "data" / "feature-completion-state.json"

    module.save_state({})

    assert module.STATE_FILE.read_text() == "{}"
