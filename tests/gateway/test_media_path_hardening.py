"""Media delivery hardening: $HOME denylist root, backslash terminator, NUL paths."""

from pathlib import Path

from gateway.platforms import base
from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE, BasePlatformAdapter

BS = chr(92)


def _isolate_hermes_root(tmp_path, monkeypatch):
    """The credential roots are module constants resolved at import; point them at a temp home
    so the denylist build never scans the machine's real Hermes profiles."""
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setattr(base, "_HERMES_HOME", hermes_home)
    monkeypatch.setattr(base, "_HERMES_ROOT", hermes_home)


def test_denylist_home_follows_home_env(tmp_path, monkeypatch):
    home = tmp_path / "operator-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile-home"))
    _isolate_hermes_root(tmp_path, monkeypatch)
    denied = base._media_delivery_denied_paths()
    assert home / ".ssh" in denied


def test_backslash_terminates_a_media_tag_path():
    text = "see MEDIA:C:" + BS + "out" + BS + "a.png" + BS + "n done"
    assert [m.group("path") for m in MEDIA_TAG_CLEANUP_RE.finditer(text)] == [
        "C:" + BS + "out" + BS + "a.png"
    ]
    media, cleaned = BasePlatformAdapter.extract_media(text)
    assert [p for p, _ in media] == ["C:" + BS + "out" + BS + "a.png"]
    assert "MEDIA:" not in cleaned


def test_nul_path_is_dropped():
    media, cleaned = BasePlatformAdapter.extract_media("MEDIA:/tmp/x" + chr(0) + "y.png ok")
    assert media == []


def test_plain_path_still_delivered():
    """Positive control for the NUL case: the same tag without NUL delivers."""
    media, _ = BasePlatformAdapter.extract_media("MEDIA:/tmp/xy.png ok")
    assert [p for p, _ in media] == ["/tmp/xy.png"]


def test_denylist_keeps_the_native_home_when_home_differs(tmp_path, monkeypatch):
    """Adding the operator ``$HOME`` must not drop the platform-native home's credential dirs."""
    native, configured = tmp_path / "native-home", tmp_path / "operator-home"
    real_expand = base.os.path.expanduser
    monkeypatch.setattr(base.os.path, "expanduser",
                        lambda p: str(native) + p[1:] if p.startswith("~") else real_expand(p))
    monkeypatch.setenv("HOME", str(configured))
    _isolate_hermes_root(tmp_path, monkeypatch)
    denied = base._media_delivery_denied_paths()
    assert native / ".ssh" in denied
    assert configured / ".ssh" in denied


def test_backslash_inside_a_windows_path_is_a_separator_not_a_terminator():
    """A directory named like a file (``album.png``, ``old.zip``) is still a directory."""
    for path in ("C:" + BS + "out" + BS + "album.png" + BS + "photo.jpg",
                 "C:" + BS + "old.zip" + BS + "v1.png" + BS + "report.pdf"):
        media, cleaned = BasePlatformAdapter.extract_media("MEDIA:" + path)
        assert media == [(path, False)]
        assert cleaned == ""


def test_dropped_nul_tag_is_still_removed_from_the_caption():
    media, cleaned = BasePlatformAdapter.extract_media("MEDIA:/tmp/x" + chr(0) + "y.png ok")
    assert media == []
    assert "MEDIA:" not in cleaned
    assert chr(0) not in cleaned
    assert cleaned == "ok"


def test_escaped_newline_before_the_next_tag_still_splits_the_two_paths():
    text = "MEDIA:C:" + BS + "out" + BS + "a.png" + BS + "nMEDIA:C:" + BS + "out" + BS + "b.png"
    media, _ = BasePlatformAdapter.extract_media(text)
    assert [p for p, _ in media] == ["C:" + BS + "out" + BS + "a.png", "C:" + BS + "out" + BS + "b.png"]


def _rehomed_profile(tmp_path, monkeypatch, *, strict: bool):
    """A profile-mode child: HOME (and USERPROFILE) point at ``{HERMES_HOME}/home`` while
    ``HERMES_REAL_HOME`` keeps the OS account's home. Returns ``(real_home, profile_home)``."""
    hermes_home = tmp_path / "hermes-home"
    profile_home, real_home = hermes_home / "home", tmp_path / "real-home"
    profile_home.mkdir(parents=True)
    real_home.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(profile_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
    _isolate_hermes_root(tmp_path, monkeypatch)
    monkeypatch.setattr(base, "_media_delivery_allowed_roots", lambda: [])
    monkeypatch.setattr(base, "_translate_docker_container_media_path", lambda *a, **k: None)
    monkeypatch.setattr("gateway.media_policy.media_delivery_strict", lambda: strict)
    monkeypatch.setattr("gateway.media_policy.media_delivery_trust_recent", lambda: True)
    monkeypatch.setattr("gateway.media_policy.media_delivery_trust_recent_seconds", lambda: "")
    return real_home, profile_home


def _fixture(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic nonsecret fixture")
    return str(path)


def test_real_home_credentials_stay_denied_when_home_is_the_profile_home(tmp_path, monkeypatch):
    """Re-homed children keep the account home only in ``HERMES_REAL_HOME``; its credential
    dirs must stay denied in default mode and, when fresh, in strict/recency mode."""
    for strict in (False, True):
        real_home, profile_home = _rehomed_profile(tmp_path / str(strict), monkeypatch, strict=strict)
        assert base.validate_media_delivery_path(_fixture(real_home / ".ssh" / "id_ed25519")) is None
        assert base.validate_media_delivery_path(_fixture(profile_home / ".ssh" / "id_ed25519")) is None
        artifact = _fixture(tmp_path / str(strict) / "out" / "chart.png")  # positive control
        assert base.validate_media_delivery_path(artifact) == str(Path(artifact).resolve())


def test_escaped_control_letter_needs_a_boundary_after_it():
    """``\n`` / ``\r`` / ``\t`` ends a path only when a boundary follows the letter; otherwise
    it is a separator starting a directory (``\notes``), and the tag is not cut at ``a.png``."""
    for tail in ("notes" + BS + "README", "reports" + BS + "data.weird", "tmp" + BS + "script.py"):
        text = "MEDIA:C:" + BS + "out" + BS + "a.png" + BS + tail
        assert [m.group("path") for m in MEDIA_TAG_CLEANUP_RE.finditer(text)] == []
        media, cleaned = BasePlatformAdapter.extract_media(text)
        assert media == []
        assert cleaned == text  # the unvalidated tag stays whole, no stray tail
    for suffix in ("n done", "n", "r" + BS + "n done", "t, next"):  # positive controls
        text = "MEDIA:C:" + BS + "out" + BS + "a.png" + BS + suffix
        assert [m.group("path") for m in MEDIA_TAG_CLEANUP_RE.finditer(text)] == [
            "C:" + BS + "out" + BS + "a.png"]
