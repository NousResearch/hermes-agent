"""Prerequisite behavior for the full and no-LaTeX rendering paths."""
import importlib.util
from pathlib import Path
from unittest.mock import patch


def checker():
    path = Path(__file__).resolve().parents[2] / "skills/creative/manim-video/scripts/check_prerequisites.py"
    spec = importlib.util.spec_from_file_location("manim_skill_prerequisites", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_latex_mode_still_requires_manim_and_ffmpeg():
    module = checker()
    with patch.object(module.importlib.util, "find_spec", return_value=None), patch.object(module.shutil, "which", return_value=None):
        assert module.missing_requirements(no_latex=True) == ["manim", "ffmpeg"]
    with patch.object(module.importlib.util, "find_spec", return_value=object()), patch.object(module.shutil, "which", side_effect=lambda name: "/bin/ffmpeg" if name == "ffmpeg" else None):
        assert module.missing_requirements(no_latex=True) == []
        assert module.missing_requirements() == ["latex", "dvisvgm"]


def test_checker_reports_failure_and_success(capsys):
    module = checker()
    with patch("sys.argv", ["check_prerequisites.py", "--no-latex"]), patch.object(module, "missing_requirements", return_value=["manim"]):
        assert module.main() == 1
        assert "Missing: manim" in capsys.readouterr().out
    with patch("sys.argv", ["check_prerequisites.py"]), patch.object(module, "missing_requirements", return_value=[]):
        assert module.main() == 0
        assert "Prerequisites available" in capsys.readouterr().out
