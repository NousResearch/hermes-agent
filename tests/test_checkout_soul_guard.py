"""The run-wide backstop for a Hermes home seeded inside the checkout (``tests/conftest.py``)."""

from pathlib import Path

import tests.conftest as root_conftest


def test_the_checkout_scan_sees_ignored_homes_and_paths_with_spaces(monkeypatch, tmp_path):
    import subprocess

    tmp_path = tmp_path / "checkout"  # the fixture's own Hermes home sits in tmp_path itself
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("tmp/\nprofiles/\n")
    for rel in ("tmp/SOUL.md", "profiles/fixture/SOUL.md", "test home/SOUL.md", "SOUL.md"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    monkeypatch.setattr(root_conftest, "PROJECT_ROOT", Path(tmp_path))
    assert root_conftest._checkout_souls() == {"tmp/SOUL.md", "profiles/fixture/SOUL.md", "test home/SOUL.md",
                                               "SOUL.md"}
