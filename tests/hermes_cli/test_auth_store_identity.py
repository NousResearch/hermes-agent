"""Unit tests for three-way auth-store identity classification."""
from __future__ import annotations

import os
from pathlib import Path

from hermes_cli.auth_store_identity import (
    AUTH_STORE_IDENTITY_DIFFERENT,
    AUTH_STORE_IDENTITY_SAME,
    AUTH_STORE_IDENTITY_UNKNOWN,
    classify_auth_store_identity,
    paths_are_same_auth_store,
)


def test_classify_same_path_and_distinct_files(tmp_path):
    left = tmp_path / "a" / "auth.json"
    right = tmp_path / "b" / "auth.json"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text('{"version": 1}\n', encoding="utf-8")
    right.write_text('{"version": 1}\n', encoding="utf-8")
    assert classify_auth_store_identity(left, left) == AUTH_STORE_IDENTITY_SAME
    assert classify_auth_store_identity(left, right) == AUTH_STORE_IDENTITY_DIFFERENT
    assert paths_are_same_auth_store(left, left)
    assert not paths_are_same_auth_store(left, right)


def test_classify_missing_counterpart_is_different(tmp_path):
    left = tmp_path / "auth.json"
    left.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "gone" / "auth.json"
    assert classify_auth_store_identity(left, missing) == AUTH_STORE_IDENTITY_DIFFERENT


def test_classify_symlink_and_hardlink_are_same(tmp_path):
    root = tmp_path / "auth.json"
    root.write_text('{"version": 1}\n', encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(root)
    assert classify_auth_store_identity(root, alias) == AUTH_STORE_IDENTITY_SAME
    hard = tmp_path / "hard.json"
    os.link(root, hard)
    assert classify_auth_store_identity(root, hard) == AUTH_STORE_IDENTITY_SAME


def test_classify_probe_failure_is_unknown(tmp_path, monkeypatch):
    root = tmp_path / "auth.json"
    root.write_text("{}\n", encoding="utf-8")
    hard = tmp_path / "hard.json"
    os.link(root, hard)

    def boom_resolve(self, strict=False):
        raise OSError("injected resolve failure")

    def boom_samefile(self, other):
        raise OSError("injected samefile failure")

    monkeypatch.setattr(Path, "resolve", boom_resolve)
    monkeypatch.setattr(Path, "samefile", boom_samefile)
    assert classify_auth_store_identity(root, hard) == AUTH_STORE_IDENTITY_UNKNOWN
    assert not paths_are_same_auth_store(root, hard)
