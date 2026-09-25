"""Cleanup needs a complete idle scan before deleting a scratch tree."""

import os
import time

import pytest

import hermes_constants_scratch as scratch


@pytest.mark.parametrize("failed_probe", ["root-stat", "directory-scan", "child-stat"])
def test_unreadable_subtree_is_preserved_until_it_can_be_scanned(tmp_path, monkeypatch, failed_probe):
    root = tmp_path / "scratch"
    entry = root / "lane"
    entry.mkdir(parents=True)
    result = entry / "result.txt"
    result.write_text("keep until activity can be checked", encoding="utf-8")
    old = time.time() - 48 * 3600
    os.utime(result, (old, old))
    os.utime(entry, (old, old))
    # Process inspection is unrelated to determining whether the files are idle.
    monkeypatch.setattr(scratch, "reap_processes_rooted_in", lambda *_: 0)
    real_lstat, real_scandir = os.lstat, os.scandir
    with monkeypatch.context() as probe:
        if failed_probe == "root-stat":
            def lstat(path, *args, **kwargs):
                if path == entry:
                    raise PermissionError("cannot inspect entry")
                return real_lstat(path, *args, **kwargs)
            probe.setattr(scratch.os, "lstat", lstat)
        else:
            def scandir(path):
                if path == str(entry):
                    if failed_probe == "directory-scan":
                        raise PermissionError("cannot inspect subtree")

                    class UnreadableChild:
                        name = "result.txt"

                        def stat(self, **_kwargs):
                            raise OSError("cannot inspect child activity")

                        def is_dir(self, **_kwargs):
                            return False

                    class Scan:
                        def __enter__(self):
                            return iter([UnreadableChild()])

                        def __exit__(self, *_args):
                            pass

                    return Scan()
                return real_scandir(path)
            probe.setattr(scratch.os, "scandir", scandir)
        assert scratch.prune_idle_entries(root, 24, frozenset()) == 0
        assert result.read_text(encoding="utf-8") == "keep until activity can be checked"
    # Once observation succeeds, the same genuinely idle tree is reclaimable.
    assert scratch.prune_idle_entries(root, 24, frozenset()) == 1
    assert not entry.exists()
