"""Download progress describes actual whole-plan bytes on fresh and resumed runs."""

from __future__ import annotations

from pm.downloader import Download, Source
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


def test_progress_counts_cached_files_and_distinguishes_equal_basenames(tmp_path, dl_server):
    bodies = {"/a": b"first", "/b": b"second-file"}
    RangeHandler.payloads = bodies
    sources = [Source(url(dl_server, path), tmp_path / path[1:] / "model.bin") for path in bodies]
    expected = sum(map(len, bodies.values()))
    for attempt in range(2):
        ticks = []
        Download(sources, partials_dir=tmp_path / "partials").run(
            progress=lambda done, total, ranges: ticks.append((done, total, ranges)))
        assert ticks, f"attempt {attempt} did not report its completed files"
        done, total, ranges = ticks[-1]
        assert done == total == expected
        assert len(ranges) == len(sources)
        assert sum(end - start for rows in ranges.values() for start, end in rows) == done


def test_unknown_length_sources_do_not_lose_completed_bytes(tmp_path, dl_server, monkeypatch):
    RangeHandler.payloads = {"/a": b"first", "/b": b"second-file"}
    RangeHandler.no_range = True
    send_header = RangeHandler.send_header

    def without_length(self, name, value):
        if name.lower() != "content-length":
            send_header(self, name, value)

    monkeypatch.setattr(RangeHandler, "send_header", without_length)
    ticks = []
    sources = [Source(url(dl_server, path), tmp_path / path[1:]) for path in RangeHandler.payloads]
    Download(sources, partials_dir=tmp_path / "partials").run(
        progress=lambda done, total, ranges: ticks.append((done, total, ranges)))
    expected = sum(source.dest.stat().st_size for source in sources)
    assert ticks[-1][:2] == (expected, expected)
    assert all(sum(end - start for rows in ranges.values() for start, end in rows) == done
               for done, _, ranges in ticks)
    assert [done for done, _, _ in ticks] == sorted(done for done, _, _ in ticks)
