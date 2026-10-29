"""Live memory headroom for a process (hermes_platform/memory.py)."""

from __future__ import annotations

from hermes_platform.memory import cgroup_headroom

MB = 1024 * 1024


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_cgroup_headroom_uses_tightest_ancestor_and_credits_page_cache(tmp_path):
    # Container root capped at 1 GiB; the gateway's own leaf is uncapped. 900 MiB charged, of
    # which 300 MiB is reclaimable inactive file cache -> 424 MiB really left.
    _write(tmp_path / "memory.max", str(1024 * MB))
    _write(tmp_path / "memory.current", str(900 * MB))
    _write(tmp_path / "memory.stat", f"anon 1\ninactive_file {300 * MB}\n")
    leaf = tmp_path / "svc" / "gw"
    _write(leaf / "memory.max", "max")
    _write(leaf / "memory.current", str(500 * MB))
    headroom, limit = cgroup_headroom("/svc/gw", root=tmp_path)
    assert (headroom // MB, limit // MB) == (424, 1024)
