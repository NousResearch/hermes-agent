"""A materialized data-URL image must be readable back by the vision resolver.

``AIAgent._materialize_data_url_for_vision`` writes the decoded image to a temp
file and hands the path to ``vision_analyze``. Under a non-local terminal
backend, ``tools.image_source`` confines host reads to ``_media_cache_roots()``
and otherwise re-reads the path *inside the sandbox* — where a host temp file
was never written. Writing outside those roots therefore makes every inbound
data-URL image fail with "not reachable inside the sandbox and no active
sandbox session is available to read it", with no other symptom.

The assertion here is the invariant (produced path is host-readable per the
resolver's own allowlist), not a literal directory, so it keeps holding if the
cache layout is reorganised.
"""

import base64
import importlib
from pathlib import Path

import pytest


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    importlib.reload(hermes_constants)
    return home


def _media_cache_roots():
    import tools.image_source as isrc
    importlib.reload(isrc)
    return [Path(r).resolve() for r in isrc._media_cache_roots()]


def _under_a_media_root(path: Path) -> bool:
    resolved = Path(path).resolve()
    for root in _media_cache_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def test_data_url_lands_where_vision_may_read_it(hermes_home):
    from run_agent import AIAgent

    path_str, cleanup = AIAgent._materialize_data_url_for_vision(DATA_URL)
    try:
        assert path_str, "materialization returned no path"
        produced = Path(path_str)
        assert produced.is_file()
        assert produced.read_bytes() == PNG_BYTES
        assert _under_a_media_root(produced), (
            f"{produced} is outside the media-cache roots, so image_source will "
            f"refuse the host read and then look for it inside the sandbox"
        )
    finally:
        if cleanup is not None and Path(cleanup).exists():
            Path(cleanup).unlink()


def test_cleanup_path_is_returned_so_the_caller_can_unlink(hermes_home):
    """delete=False means the caller owns removal; it needs the path back."""
    from run_agent import AIAgent

    path_str, cleanup = AIAgent._materialize_data_url_for_vision(DATA_URL)
    assert cleanup is not None
    assert str(cleanup) == path_str
    cleanup.unlink()
    assert not Path(path_str).exists()


def test_oversized_data_url_is_skipped(hermes_home):
    """The size guard runs before any file is created."""
    from run_agent import AIAgent

    oversized = "data:image/png;base64," + "A" * (
        AIAgent._MAX_DATA_URL_BASE64_BYTES + 1
    )
    path_str, cleanup = AIAgent._materialize_data_url_for_vision(oversized)
    assert path_str == ""
    assert cleanup is None
