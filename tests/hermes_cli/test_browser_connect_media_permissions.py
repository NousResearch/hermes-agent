"""The chrome-debug Chromium profile and Hermes media-cache dirs are created owner-only.

``$HERMES_HOME/chrome-debug`` is a real Chromium user-data dir (Cookies, Login Data,
Local Storage). A bare ``os.makedirs`` inherited the umask and landed 0755, so on the
documented ``HERMES_HOME_MODE=0701`` web-server hatch every other local account could
read those stores (#77579 / #77486). The same held for the lazily-created media caches
that hold user content — vision/computer-use captures, downloaded media, gateway inbound
media. These tests pin the creation contract: 0700 dirs at creation time, a 0600 launch
stderr log, and 0600 media files, with managed (NixOS) installs deliberately left to
the configured umask/setgid.
"""
import os
import stat

import pytest

import hermes_cli.browser_connect as bc
import tools.computer_use.tool as cu
import tools.vision_tools as vt


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _no_managed(monkeypatch, home):
    """Force the not-managed branch: no HERMES_MANAGED env, no marker in ``home``."""
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    monkeypatch.delenv("HERMES_UID", raising=False)
    monkeypatch.delenv("HERMES_GID", raising=False)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)


def test_chrome_debug_data_dir_created_owner_only(tmp_path, monkeypatch):
    """A fresh chrome-debug profile dir lands 0700, not the umask-derived 0755."""
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
    bc._ensure_chrome_debug_data_dir(bc.chrome_debug_data_dir())
    assert _mode(home / "chrome-debug") == 0o700


def test_chrome_debug_data_dir_reconciles_legacy_world_readable(tmp_path, monkeypatch):
    """A profile an older Hermes left at 0755 is tightened on the next launch — the
    exposure is already on disk, so creation-time hardening alone is not enough."""
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
    profile = home / "chrome-debug"
    profile.mkdir(parents=True, mode=0o755)
    bc._ensure_chrome_debug_data_dir(str(profile))
    assert _mode(profile) == 0o700


def test_chrome_debug_data_dir_managed_install_leaves_mode_to_umask(tmp_path, monkeypatch):
    """Managed (NixOS) installs share $HERMES_HOME group-wise by design; the profile
    is not pre-created by tmpfiles rules, so the creation mode must stay with the
    configured umask/setgid rather than a hardcoded 0700."""
    home = tmp_path / "hh"
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
    monkeypatch.setattr(bc, "_managed_install", lambda: True)
    profile = home / "chrome-debug"
    bc._ensure_chrome_debug_data_dir(str(profile))
    # mkdir with no explicit mode: umask applies (0o777 & ~umask); the point is that
    # the helper did NOT force 0700 — group bits survive under the service umask.
    assert _mode(profile) == 0o777 & ~_umask()


def _umask() -> int:
    current = os.umask(0o022)
    os.umask(current)
    return current


def test_launch_stderr_log_owner_only(tmp_path, monkeypatch):
    """The launch stderr log (fixed, guessable name inside the profile dir) is created
    0600, and an older 0644 log is reconciled, not left exposed on upgrade."""
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
    log = home / "chrome-debug" / bc._LAUNCH_STDERR_LOG
    log.parent.mkdir(parents=True, mode=0o700)
    with bc._open_launch_stderr_log(str(log)) as handle:
        handle.write(b"stderr tail")
    assert _mode(log) == 0o600
    # Legacy exposure on disk: the reconcile must tighten it after the open.
    os.chmod(log, 0o644)
    with bc._open_launch_stderr_log(str(log)) as handle:
        handle.write(b"again")
    assert _mode(log) == 0o600
    assert log.read_bytes() == b"again"  # O_TRUNC overwrite semantics preserved


def test_vision_media_cache_dirs_created_owner_only(tmp_path, monkeypatch):
    """cache/vision and cache/video land 0700 at creation, and a legacy 0755 dir is
    healed (PR #77579's media-cache half)."""
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(vt, "get_hermes_dir", lambda new, old: home / new)
    for sub in ("cache/vision", "cache/video"):
        assert _mode(vt._secure_cache_dir(sub, "legacy")) == 0o700
    legacy = home / "cache" / "vision"
    os.chmod(legacy, 0o755)
    assert _mode(vt._secure_cache_dir("cache/vision", "legacy")) == 0o700


def test_vision_media_cache_files_owner_only(tmp_path, monkeypatch):
    """Media bytes written into the hardened cache land 0600, not the umask default."""
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(vt, "get_hermes_dir", lambda new, old: home / new)
    path = vt._secure_cache_dir("cache/vision", "legacy") / "capture.png"
    vt._write_private_bytes(path, b"screen contents")
    assert path.read_bytes() == b"screen contents"
    assert _mode(path) == 0o600


def test_computer_use_cache_files_owner_only(tmp_path, monkeypatch):
    """computer_use's capture caches (screenshots, vision temps) are created 0700 and
    the capture bytes land 0600 — a capture is as sensitive as the screen it came
    from."""
    import hermes_constants

    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    # computer_use resolves the cache dir through the lazily-imported
    # hermes_constants.get_hermes_dir (see _cache_file), so patch it there —
    # the module attribute the import binds to, not the consumer module.
    monkeypatch.setattr(hermes_constants, "get_hermes_dir", lambda new, old: home / new)
    path = cu._cache_file("cache/vision", "temp_vision_images", "capture.png")
    cu._write_private_bytes(path, b"frame")
    assert path.read_bytes() == b"frame"
    assert _mode(path.parent) == 0o700
    assert _mode(path) == 0o600


@pytest.mark.parametrize("getter_name", ["get_image_cache_dir", "get_video_cache_dir",
                                         "get_audio_cache_dir", "get_document_cache_dir",
                                         "get_screenshot_cache_dir"])
def test_gateway_media_cache_dirs_created_owner_only(tmp_path, monkeypatch, getter_name):
    """The gateway's inbound-media caches (photos/voice notes/documents users sent)
    land 0700 at creation and heal a legacy 0755 dir."""
    import gateway.platforms.base as gw
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(gw, "get_hermes_dir", lambda new, old: home / new)
    for name in gw._CACHE_DIR_IMPORT_DEFAULTS:
        monkeypatch.setattr(gw, name, None, raising=False)
    cache_dir = getattr(gw, getter_name)()
    assert _mode(cache_dir) == 0o700
    os.chmod(cache_dir, 0o755)
    assert _mode(getattr(gw, getter_name)()) == 0o700


def test_gateway_media_cache_files_owner_only(tmp_path, monkeypatch):
    """Inbound media bytes written into the gateway cache land 0600."""
    import gateway.platforms.base as gw
    home = tmp_path / "hh"
    _no_managed(monkeypatch, home)
    monkeypatch.setattr(gw, "get_hermes_dir", lambda new, old: home / new)
    for name in gw._CACHE_DIR_IMPORT_DEFAULTS:
        monkeypatch.setattr(gw, name, None, raising=False)
    path_str = gw.cache_image_from_bytes(b"\x89PNG\r\n\x1a\n" + b"photo bytes" * 8)
    assert _mode(path_str) == 0o600
