"""Case-insensitive filesystem coverage for the gateway media-delivery denylist.

The media-delivery guard (gateway/platforms/base.py) is the read/exfil twin of
agent/file_safety.py's write guard: its own comment notes the two must stay in
lockstep so "the delivery side can't trail the write side". On a case-insensitive
filesystem (macOS/APFS, Windows/NTFS) a model-emitted MEDIA path like
``~/.SSH/id_rsa`` resolves to the same credential file as ``~/.ssh/id_rsa`` but
preserves the typed case, so the denylist comparison must casefold there too.
Otherwise a case variant auto-attaches a credential to a chat reply. On
case-sensitive Linux the variants are distinct files and behavior is unchanged.

Run with: python -m pytest tests/gateway/test_media_delivery_case_insensitive.py -v
"""

import os
from pathlib import Path

import pytest

import gateway.platforms.base as base
from gateway.platforms.base import _path_under_denied_prefix


def _force_case_insensitive(monkeypatch, value):
    monkeypatch.setattr(base, "_fs_case_insensitive", lambda: value)


def _home_path(*parts):
    return Path(os.path.expanduser("~")).resolve(strict=False).joinpath(*parts)


class TestMediaDeliveryDenyCaseInsensitive:
    def test_ssh_exact_case_is_denied(self, monkeypatch):
        _force_case_insensitive(monkeypatch, False)
        assert _path_under_denied_prefix(_home_path(".ssh", "id_rsa")) is True

    def test_ssh_case_variant_denied_on_case_insensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, True)
        assert _path_under_denied_prefix(_home_path(".SSH", "id_rsa")) is True

    def test_aws_case_variant_denied_on_case_insensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, True)
        assert _path_under_denied_prefix(_home_path(".AWS", "credentials")) is True

    def test_ssh_case_variant_not_denied_on_case_sensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, False)
        # Distinct file on Linux, so it must not be blocked (no over-blocking).
        assert _path_under_denied_prefix(_home_path(".SSH", "id_rsa")) is False

    def test_plain_home_file_still_deliverable(self, monkeypatch):
        # A normal deliverable in the user's home stays allowed even with
        # case-folding on, so the fix must not over-block ordinary files.
        _force_case_insensitive(monkeypatch, True)
        assert _path_under_denied_prefix(_home_path("report.pdf")) is False
