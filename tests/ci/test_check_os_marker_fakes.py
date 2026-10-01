"""The unmarked-macOS-fake guard (#111866): flags fakes, ignores honest reads and marked files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

from check_os_marker_fakes import find_unmarked_fakes  # noqa: E402


def _write(root: Path, name: str, body: str) -> None:
    (root / name).write_text(body, encoding="utf-8")


def test_unmarked_fake_is_flagged_marked_and_opted_out_are_not(tmp_path):
    _write(tmp_path, "test_fake.py", "def test_x(monkeypatch):\n"
           "    monkeypatch.setattr(gw, 'is_macos', lambda: True)\n"
           "    monkeypatch.setattr(sys, 'platform', 'darwin')\n")
    _write(tmp_path, "test_marked.py", "import pytest\npytestmark = pytest.mark.platforms(\"macos\")\n"
           "def test_x(monkeypatch):\n    monkeypatch.setattr(gw, 'is_macos', lambda: True)\n")
    _write(tmp_path, "test_opted.py", "def test_x(monkeypatch):\n"
           "    patch('m.is_macos', return_value=True)  # os-marker: ok — pure data mapping\n")

    hits = find_unmarked_fakes(tmp_path, tmp_path)

    assert set(hits) == {"test_fake.py"}
    assert [n for n, _ in hits["test_fake.py"]] == [2, 3]


def test_host_honest_platform_read_is_not_a_fake(tmp_path):
    _write(tmp_path, "test_read.py", "import sys\n"
           "def test_x():\n    expected = sys.platform == 'darwin'\n"
           "    if sys.platform == 'darwin':\n        pass\n"
           "    payload = {'platform': 'darwin'}\n"
           "    # monkeypatch.setattr(gw, 'is_macos', lambda: True) in a comment\n")

    assert find_unmarked_fakes(tmp_path, tmp_path) == {}
