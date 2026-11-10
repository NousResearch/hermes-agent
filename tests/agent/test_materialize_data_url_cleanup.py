"""Regression test: temp file cleanup when materializing data URLs for vision.

`_materialize_data_url_for_vision` creates a `NamedTemporaryFile(delete=False)`
so the path can be handed to vision backends.  If `base64.b64decode` raises on
a corrupt/unsupported data URL the temp file would otherwise persist forever
on disk, leaking once per failed call.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import pytest

from run_agent import AIAgent


def _list_anthropic_tmpfiles(tmpdir: str) -> list[str]:
    return [
        name for name in os.listdir(tmpdir)
        if name.startswith("anthropic_image_")
    ]


def test_b64decode_failure_does_not_leak_tempfile(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    bad_url = "data:image/png;base64,!!!not-valid-base64!!!"
    with pytest.raises(Exception):
        AIAgent._materialize_data_url_for_vision(bad_url)

    leftovers = _list_anthropic_tmpfiles(str(tmp_path))
    assert leftovers == [], f"leaked temp files after decode failure: {leftovers}"


def test_successful_decode_returns_path_to_existing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # a few bytes is enough
    encoded = base64.b64encode(payload).decode("ascii")
    good_url = f"data:image/png;base64,{encoded}"

    path_str, path_obj = AIAgent._materialize_data_url_for_vision(good_url)

    assert isinstance(path_obj, Path)
    assert path_obj.exists()
    assert path_obj.read_bytes() == payload
    assert path_str == str(path_obj)
    # Caller is responsible for cleanup; mimic that here so the test leaves
    # no artifacts behind.
    path_obj.unlink()
