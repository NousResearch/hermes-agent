"""Review-9 (a): native media capture keeps the Hermes-home provenance of its store for elevated grants.

``native_inputs_root()`` resolves the store, which erases a link planted below the home
(``cache/documents`` -> elsewhere). An elevated grant on that resolved path landed outside the home
with no home fence (``expected=None``). Only the elevation fact and the native writer are
replaced (no host faking); capture, mkdir and the links are real disk operations.
"""
import os
from pathlib import Path

import pytest

import hermes_dir_acl
from gateway import session_ingress_media as media


def _elevated(monkeypatch, home):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    grants = []
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32",
                        lambda path, is_dir, expected=None: grants.append((os.path.realpath(path), expected)))
    return grants


def test_capture_never_grants_through_a_link_below_the_home(tmp_path, monkeypatch):
    home, outside = tmp_path / "home", tmp_path / "outside"
    (home / "cache").mkdir(parents=True)
    (outside / "native-inputs").mkdir(parents=True)
    (home / "cache" / "documents").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "sample.txt"
    source.write_text("native-media sentinel")
    grants = _elevated(monkeypatch, home)

    references = media.capture_native_media([source])

    assert Path(references[0]["path"]).is_file()  # capture itself still works (the user's link)
    escaped = [g for g in grants if Path(g[0]).is_relative_to(outside)]
    assert escaped == [], f"an elevated grant followed the link out of the home: {escaped}"


def test_capture_grants_the_real_store_with_a_home_fence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    source = tmp_path / "sample.txt"
    source.write_text("bytes")
    grants = _elevated(monkeypatch, home)

    references = media.capture_native_media([source])

    digest_dir = Path(references[0]["path"]).parent
    granted = dict(grants)
    assert str(digest_dir) in granted and granted[str(digest_dir)] == str(digest_dir)


def test_icacls_fallback_refuses_a_grant_with_no_trusted_root(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_dir_acl, "_whoami_sid", lambda: "S-1-5-21-1-2-3-1001")
    calls = []
    monkeypatch.setattr(hermes_dir_acl.subprocess, "run", lambda argv, **_k: calls.append(argv))
    with pytest.raises(hermes_dir_acl.ReparsePointRefused):
        hermes_dir_acl._grant_with_icacls(tmp_path, True, None)
    assert calls == []
