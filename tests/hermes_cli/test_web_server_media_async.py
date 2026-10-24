"""Dashboard media reads must not run on the event-loop thread."""

import threading
from pathlib import Path

import pytest

from hermes_cli.web_routers import files


@pytest.mark.asyncio
async def test_get_media_reads_and_encodes_off_event_loop(monkeypatch, tmp_path):
    """The bounded media payload is read from a worker thread."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage-data")

    monkeypatch.setattr(files, "get_hermes_home", lambda: tmp_path)
    original_read_bytes = Path.read_bytes
    read_threads = []

    def _tracking_read_bytes(path):
        read_threads.append(threading.current_thread())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _tracking_read_bytes)
    event_loop_thread = threading.current_thread()

    response = await files.get_media(str(image))

    assert response["data_url"].startswith("data:image/png;base64,")
    assert read_threads
    assert all(thread is not event_loop_thread for thread in read_threads)
