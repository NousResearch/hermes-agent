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


@pytest.mark.parametrize("failed_probe", ["root-stat", "directory-scan", "child-stat"])
def test_post_selection_scan_failure_keeps_entry_with_honest_audit(
    tmp_path, monkeypatch, failed_probe
):
    """P3 on #134173: the mid-prune re-validation is a fail-closed keep, and its record
    must not dress a failed re-scan up as observed activity. Selection completes on a
    genuinely idle entry, then the re-validation scan fails: the entry is kept with its
    bytes and mtime intact, and the audit names the failed re-scan — it never claims
    the entry was touched. Once the scan works again, the entry is reclaimable."""
    root = tmp_path / "scratch"
    entry = root / "lane"
    entry.mkdir(parents=True)
    result = entry / "result.txt"
    result.write_text("keep until activity can be checked", encoding="utf-8")
    old = time.time() - 48 * 3600
    os.utime(result, (old, old))
    os.utime(entry, (old, old))
    log_file = tmp_path / "logs" / "scratch-prune.log"

    state = {"selected": False}

    def reap_after_selection(*_args, **_kwargs):
        # The doomed list is built; every scan past this point is the re-validation walk.
        state["selected"] = True
        return 0

    monkeypatch.setattr(scratch, "reap_processes_rooted_in", reap_after_selection)
    real_lstat, real_scandir = os.lstat, os.scandir
    with monkeypatch.context() as probe:
        if failed_probe == "root-stat":
            def lstat(path, *args, **kwargs):
                if state["selected"] and path == entry:
                    raise PermissionError("cannot inspect entry")
                return real_lstat(path, *args, **kwargs)
            probe.setattr(scratch.os, "lstat", lstat)
        else:
            def scandir(path):
                if state["selected"] and path == str(entry):
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
        # Kept, never counted as removed, original bytes and mtime intact: no payload
        # write occurred, and none may be invented by the audit.
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert result.read_text(encoding="utf-8") == "keep until activity can be checked"
        assert result.stat().st_mtime == old
    log = log_file.read_text(encoding="utf-8-sig")
    assert "touched since selection" not in log, log
    assert f"kept {str(entry)!r} " in log, log
    assert "activity re-scan failed, left untouched" in log, log
    # A later prune that can scan the entry reclaims it: fail-closed, not fail-forever.
    assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
    assert not entry.exists()
