from agent.turn_explainers import TurnExplainersMixin as M
from hermes_constants import get_scratch_dir


def test_scratch_paths_dropped_real_paths_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scratch = get_scratch_dir(prune=False) / "pages.py"
    real = tmp_path / "notes" / "brief.md"
    failed = {str(scratch): {"tool": "write_file"}, str(real): {"tool": "patch"}}
    assert list(M._file_mutations_still_failed(failed)) == [str(real)]
