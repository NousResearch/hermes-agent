"""Tests for interrupted-install self-heal (the ``.update-incomplete`` marker).

Covers the breadcrumb lifecycle so a ``hermes update`` killed mid-install
(Ctrl-C, terminal close, WSL OOM) leaves a marker the next launch can act on.
"""

from __future__ import annotations

import hermes_cli.main as m


def test_marker_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    marker = m._update_marker_path()
    assert marker == tmp_path / ".update-incomplete"
    assert not marker.exists()

    m._write_update_incomplete_marker()
    assert marker.exists()
    body = marker.read_text()
    assert "started=" in body
    assert "pid=" in body

    m._clear_update_incomplete_marker()
    assert not marker.exists()
