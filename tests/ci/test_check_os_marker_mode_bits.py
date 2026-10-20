"""The unmarked mode-bit assertion guard: flags Windows-runnable assertions, ignores the rest."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

from check_os_marker_mode_bits import added_lines, scan_file

# The shape that fails on Windows: the mode is read into a name and asserted on the next line
# (tests/tools/test_browser_vault.py before #141909).
UNMARKED = '''\
import os
import stat


def test_file_permissions_0600(tmp_path):
    mode = stat.S_IMODE(os.stat(tmp_path / "vault.json.enc").st_mode)
    assert mode == 0o600, f"has mode {oct(mode)}"
'''


def _names(source: str) -> list[str]:
    return [hit.func for hit in scan_file("tests/test_x.py", source)]


def test_unmarked_mode_bit_assertion_is_flagged():
    assert _names(UNMARKED) == ["test_file_permissions_0600"]


def test_posix_and_single_host_markers_clear_the_test():
    for marker in ('platforms("posix")', 'platforms("linux")', 'platforms("macos")',
                   'platforms("not windows")'):
        source = f"import pytest\nimport stat\n\n\n@pytest.mark.{marker}\ndef test_x(p):\n    assert stat.S_IMODE(p.stat().st_mode) == 0o600\n"
        assert _names(source) == [], marker


def test_windows_admitting_marker_is_still_flagged():
    for marker in ('platforms("windows")', 'platforms("any")', 'platforms("posix", "windows")',
                   'platforms("not macos")'):
        source = f"import pytest\nimport stat\n\n\n@pytest.mark.{marker}\ndef test_x(p):\n    assert p.stat().st_mode & 0o777 == 0o600\n"
        assert _names(source) == ["test_x"], marker


def test_module_level_marker_clears_every_test_in_the_file():
    source = 'import pytest\nimport stat\n\npytestmark = pytest.mark.platforms("posix")\n\n\ndef test_x(p):\n    assert stat.S_IMODE(p.stat().st_mode) == 0o600\n'
    assert _names(source) == []


def test_opt_out_line_clears_the_test():
    source = 'import stat\n\n\ndef test_x(stub):\n    # os-marker: ok — the test builds its own stat_result, no real bits involved\n    assert stat.S_IMODE(stub.st_mode) == 0o600\n'
    assert _names(source) == []


def test_mode_free_and_octal_free_tests_are_ignored():
    # No st_mode/S_IMODE read: nothing here can fail on Windows.
    assert _names("def test_x(p):\n    assert oct(p.stat().st_size)\n") == []
    # Mode bits read as data (a stat_result stub built by the test) without an assertion literal.
    assert _names("import stat\n\n\ndef test_x():\n    st = os.stat_result((0, 0, 0, 1, 0, 0, 0, 0, 0, 0))\n    return st.st_mode\n") == []


def test_class_method_is_reported_at_its_own_line():
    source = 'import stat\n\n\nclass TestVault:\n    def test_other(self):\n        pass\n\n    def test_owner_only(self, p):\n        assert p.stat().st_mode & 0o077 == 0\n'
    hits = scan_file("tests/test_x.py", source)
    # The def sits on line 8 (lines 2-3 and 7 are blank); the hit carries the function's own line.
    assert [(h.func, h.lineno) for h in hits] == [("test_owner_only", 8)]


def test_head_revision_bytes_are_judged_not_the_checkout(tmp_path, monkeypatch):
    """A run that names --head must judge that revision's bytes, not whatever is checked out."""
    import subprocess

    from check_os_marker_mode_bits import main

    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "tests" / "tools").mkdir(parents=True)
    target = tmp_path / "tests" / "tools" / "test_sample.py"
    target.write_text("def test_placeholder():\n    pass\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "base")
    target.write_text(UNMARKED.replace("test_file_permissions_0600", "test_sample"), encoding="utf-8")
    git("commit", "-qam", "unmarked mode-bit test")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
                          capture_output=True, text=True).stdout.strip()

    # The checkout is the FIXED side; only --head holds the violation, so a disk read would miss it.
    target.write_text('import pytest\nimport stat\n\n\n@pytest.mark.platforms("posix")\n'
                      "def test_sample(p):\n    assert stat.S_IMODE(p.stat().st_mode) == 0o600\n",
                      encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    assert main(["--base", f"{head}~1", "--head", head, "--repo", str(tmp_path)]) == 1
    assert main(["--base", f"{head}~1", "--repo", str(tmp_path)]) == 0  # the working tree is clean


def test_added_lines_reads_post_image_numbers():
    diff = (
        "diff --git a/tests/test_x.py b/tests/test_x.py\n"
        "--- a/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n"
        "@@ -10,0 +11,2 @@ def test_x():\n"
        "+    assert p.stat().st_mode & 0o777 == 0o600\n"
        "+    assert not p.stat().st_mode & 0o077\n"
        "@@ -40 +42 @@ def test_y():\n"
        "-    old\n"
        "+    new\n"
    )
    assert added_lines(diff) == {"tests/test_x.py": {11, 12, 42}}
