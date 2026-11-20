"""Static analysis closes source handles while retaining its metrics."""

import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("read_error", [False, True])
def test_source_handle_closed(tmp_path, monkeypatch, read_error):
    script = Path(__file__).resolve().parents[2] / "evals/codebase_navigability/static_metrics.py"
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "support.py").write_text("def support():\n    return 1\n")
    (tree / "tests").mkdir()
    (tree / "tests/test_fixture.py").write_text("def test_fixture():\n    pass\n")
    (tree / "sample.py").write_text("import os\n# comment\n\ndef foo():\n    return 1\n")
    monkeypatch.setattr(sys, "argv", [str(script), str(tree), "test"])
    monkeypatch.setenv("NAV_OUT", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, stdout="{}")))
    spec = importlib.util.spec_from_file_location("static_metrics_close_test", script)
    module = importlib.util.module_from_spec(spec)
    handles = []
    real_open = builtins.open

    def tracked_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if Path(path) == tree / "sample.py":
            handles.append(handle)
            if read_error:
                handle.read = Mock(side_effect=OSError("read interrupted"))
        return handle

    module.open = tracked_open
    spec.loader.exec_module(module)

    assert handles
    assert all(handle.closed for handle in handles)
    if not read_error:
        metrics = module.analyse([str(tree / "sample.py")])
        assert metrics["lines"] == 5
        assert metrics["code"] == 3
        assert all(handle.closed for handle in handles)
