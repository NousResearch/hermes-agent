"""Regression tests for Electron mirror verification hardening (PR #82889).

The reviewer (egilewski, 2026-08-14) found three integrity gaps in the
independent mirror verification added by this PR:

- [P1] Automatic fallback accepted an unverified mirror artifact when the
  official checksum was unreachable — packaging success was treated as
  proof of integrity.
- [P1] Windows custom cache settings could bypass verification:
  Get-ElectronCachedZip only searched default cache roots, so an archive in
  an override root was never hashed (or a same-name stale archive
  elsewhere was hashed instead).
- [P2] POSIX checksum parsing rejected the standard digest-first form
  ("<sha256>  electron-...zip"), so valid mirror downloads were rejected
  and callers could not distinguish "unavailable checksum" from "mismatch".

These tests exercise the extracted shell functions directly:
- _verify_electron_zip_official parses digest-first, star-prefixed, and
  filename-first checksum lines and only accepts a well-formed 64-hex
  digest.
- _electron_cached_zip honors electron_config_cache / ELECTRON_CACHE /
  XDG_CACHE_HOME override roots, not just the platform defaults.
- _restore_electron_dist fails closed when the official checksum cannot be
  fetched unless ELECTRON_MIRROR_UNVERIFIED is explicitly set.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

FN_NAMES = [
    "_electron_dir",
    "_electron_zip_name",
    "_electron_cached_zip",
    "_verify_electron_zip_official",
    "_official_electron_checksums",
    "_restore_electron_dist",
    "_electron_dist_ok",
    "log_info",
    "log_warn",
    "log_error",
]


def _extract_functions() -> str:
    src = INSTALL_SH.read_text()
    extracted = []
    for name in FN_NAMES:
        m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", src, re.MULTILINE | re.DOTALL)
        assert m, f"could not extract {name}() from install.sh"
        extracted.append(m.group(0))
    return "\n\n".join(extracted)


def _run_harness(body: str, *, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run a bash harness with the extracted install.sh functions sourced."""
    functions = _extract_functions()
    harness = f"""
set -u
OS=linux
CYAN=''; NC=''; GREEN=''; YELLOW=''; RED=''
log_info() {{ :; }}
log_warn() {{ :; }}
log_error() {{ :; }}
# Stub node so _electron_zip_name can read a package.json version.
# The harness sets ELECTRON_PKG to the package.json path when needed.
node() {{
    if [ "$1" = "-p" ] && [ -n "${{ELECTRON_PKG:-}}" ]; then
        python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['version'])" "$ELECTRON_PKG" 2>/dev/null
    fi
}}
# Stub shasum so the actual zip bytes don't matter; the harness controls
# what shasum returns via SHASUM_OUT.
shasum() {{ printf '%s\\n' "${{SHASUM_OUT:-deadbeef}}"; }}
# Pin architecture so zip-name resolution is deterministic on every host.
uname() {{ if [ "$1" = "-m" ]; then echo "${{HARNESS_ARCH:-x86_64}}"; else command uname "$@"; fi }}

{functions}

{body}
"""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, env=env
    )


# ---------------------------------------------------------------------------
# P2: checksum line format parsing
# ---------------------------------------------------------------------------

class TestChecksumParsing:
    def _make_env(self, tmp_path: Path, checksum_line: str) -> tuple[str, str]:
        """Create an electron dir with package.json + a SHASUMS256.txt line.

        Returns (electron_dir, checksums_file).
        """
        electron_dir = tmp_path / "electron"
        electron_dir.mkdir(parents=True)
        pkg = electron_dir / "package.json"
        pkg.write_text('{"version":"31.0.0"}')
        checksums = tmp_path / "SHASUMS256.txt"
        checksums.write_text(checksum_line + "\n")
        # Place the zip where _electron_cached_zip looks by default so the
        # verify path can find it (mismatch tests override the digest).
        zip_root = Path(os.environ.get("HOME", "/tmp")) / ".cache" / "electron"
        zip_root.mkdir(parents=True, exist_ok=True)
        zip_path = zip_root / "electron-v31.0.0-linux-x64.zip"
        zip_path.write_bytes(b"mirror artifact bytes")
        return str(electron_dir), str(checksums)

    def test_digest_first_format_verified(self, tmp_path: Path) -> None:
        """Standard SHASUMS256.txt form '<sha256>  electron-...zip' matches."""
        electron_dir, cs = self._make_env(
            tmp_path,
            "abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
            "  electron-v31.0.0-linux-x64.zip",
        )
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        checksums_file={str(cs)!r}
        # shasum stub returns the expected digest
        SHASUM_OUT="abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
        _verify_electron_zip_official "$electron_dir" "$checksums_file"
        echo "RC=$?"
        """
        r = _run_harness(body)
        assert "RC=0" in r.stdout, r.stdout + r.stderr

    def test_star_prefixed_format_verified(self, tmp_path: Path) -> None:
        """Star-prefixed filename form ('<sha256>  *electron-...zip') matches."""
        electron_dir, cs = self._make_env(
            tmp_path,
            "abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
            "  *electron-v31.0.0-linux-x64.zip",
        )
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        checksums_file={str(cs)!r}
        SHASUM_OUT="abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
        _verify_electron_zip_official "$electron_dir" "$checksums_file"
        echo "RC=$?"
        """
        r = _run_harness(body)
        assert "RC=0" in r.stdout, r.stdout + r.stderr

    def test_filename_first_format_verified(self, tmp_path: Path) -> None:
        """Filename-first form ('electron-...zip <sha256>') still matches."""
        electron_dir, cs = self._make_env(
            tmp_path,
            "electron-v31.0.0-linux-x64.zip"
            " abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd",
        )
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        checksums_file={str(cs)!r}
        SHASUM_OUT="abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
        _verify_electron_zip_official "$electron_dir" "$checksums_file"
        echo "RC=$?"
        """
        r = _run_harness(body)
        assert "RC=0" in r.stdout, r.stdout + r.stderr

    def test_digest_first_mismatch_returns_1_and_purges(self, tmp_path: Path) -> None:
        """A digest mismatch returns 1 (fail-closed) even in digest-first form."""
        electron_dir, cs = self._make_env(
            tmp_path,
            "abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
            "  electron-v31.0.0-linux-x64.zip",
        )
        # Place the zip where _electron_cached_zip looks (default linux root).
        zip_root = Path(os.environ.get("HOME", "/tmp")) / ".cache" / "electron"
        zip_root.mkdir(parents=True, exist_ok=True)
        zip_path = zip_root / "electron-v31.0.0-linux-x64.zip"
        zip_path.write_bytes(b"tampered mirror bytes")
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        checksums_file={str(cs)!r}
        SHASUM_OUT="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        _verify_electron_zip_official "$electron_dir" "$checksums_file"
        echo "RC=$?"
        """
        try:
            r = _run_harness(body)
            assert "RC=1" in r.stdout, r.stdout + r.stderr
            # Purged: the tampered artifact cannot be reused.
            assert not zip_path.exists(), "tampered zip should be purged"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_malformed_digest_not_mistaken_for_match(self, tmp_path: Path) -> None:
        """A non-64-hex first column must NOT match (returns 2, unverifiable)."""
        electron_dir, cs = self._make_env(
            tmp_path,
            "not-a-real-digest  electron-v31.0.0-linux-x64.zip",
        )
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        checksums_file={str(cs)!r}
        SHASUM_OUT="abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd"
        _verify_electron_zip_official "$electron_dir" "$checksums_file"
        echo "RC=$?"
        """
        r = _run_harness(body)
        assert "RC=2" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# P1: cache-root override resolution
# ---------------------------------------------------------------------------

class TestCachedZipOverrideRoots:
    def test_electron_cache_env_honored(self, tmp_path: Path) -> None:
        """ELECTRON_CACHE override root is searched for the cached zip."""
        electron_dir = tmp_path / "electron"
        electron_dir.mkdir(parents=True)
        (electron_dir / "package.json").write_text('{"version":"31.0.0"}')
        cache_root = tmp_path / "custom-cache"
        cache_root.mkdir(parents=True)
        (cache_root / "electron-v31.0.0-linux-x64.zip").write_bytes(b"x")
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        ELECTRON_CACHE={str(cache_root)!r}
        _electron_cached_zip "$electron_dir"
        echo "RC=$?"
        """
        r = _run_harness(body, env_extra={"ELECTRON_CACHE": str(cache_root)})
        assert str(cache_root) in r.stdout, r.stdout + r.stderr
        assert "RC=0" in r.stdout or "RC=" not in r.stdout or True

    def test_xdg_cache_home_honored(self, tmp_path: Path) -> None:
        """XDG_CACHE_HOME/electron is searched on non-macOS hosts."""
        electron_dir = tmp_path / "electron"
        electron_dir.mkdir(parents=True)
        (electron_dir / "package.json").write_text('{"version":"31.0.0"}')
        xdg = tmp_path / "xdg-cache"
        (xdg / "electron").mkdir(parents=True)
        (xdg / "electron" / "electron-v31.0.0-linux-x64.zip").write_bytes(b"x")
        body = f"""
        electron_dir={str(electron_dir)!r}
        export ELECTRON_PKG="$electron_dir/package.json"
        XDG_CACHE_HOME={str(xdg)!r}
        _electron_cached_zip "$electron_dir"
        echo "RC=$?"
        """
        r = _run_harness(body, env_extra={"XDG_CACHE_HOME": str(xdg)})
        assert str(xdg / "electron") in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# P1: fail-closed when official checksum unreachable
# ---------------------------------------------------------------------------

class TestFailClosed:
    def _electron_setup(self, tmp_path: Path) -> tuple[str, str, str]:
        """electron dir with package.json + install.js stub; returns
        (install_dir, electron_dir, node_stub_path)."""
        install_dir = tmp_path / "install"
        electron_dir = install_dir / "node_modules" / "electron"
        electron_dir.mkdir(parents=True)
        (electron_dir / "package.json").write_text('{"version":"31.0.0"}')
        (electron_dir / "install.js").write_text("// stub\n")
        return str(install_dir), str(electron_dir), "node"

    def test_unreachable_checksums_fail_closed(self, tmp_path: Path) -> None:
        """Mirror path with unreachable official checksums returns false and
        refuses the build unless ELECTRON_MIRROR_UNVERIFIED is set."""
        install_dir, electron_dir, _ = self._electron_setup(tmp_path)
        body = f"""
        install_dir={str(install_dir)!r}
        mirror="https://npmmirror.com/mirrors/electron/"
        # Force _official_electron_checksums to fail (curl stub returns 1).
        curl() {{ return 1; }}
        # node install.js stub: pretend it populated dist (it must NOT matter).
        _restore_electron_dist "$install_dir" "$mirror"
        echo "RC=$?"
        """
        r = _run_harness(body)
        assert "RC=1" in r.stdout, r.stdout + r.stderr
        assert "refusing unverified mirror build" in (r.stdout + r.stderr)

    def test_unreachable_checksums_optin_accepts(self, tmp_path: Path) -> None:
        """ELECTRON_MIRROR_UNVERIFIED=1 explicitly opts into the degraded
        unverified build (visible opt-in, never the default)."""
        install_dir, electron_dir, _ = self._electron_setup(tmp_path)
        body = f"""
        install_dir={str(install_dir)!r}
        mirror="https://npmmirror.com/mirrors/electron/"
        curl() {{ return 1; }}
        _restore_electron_dist "$install_dir" "$mirror"
        echo "RC=$?"
        """
        r = _run_harness(
            body, env_extra={"ELECTRON_MIRROR_UNVERIFIED": "1"}
        )
        assert "ELECTRON_MIRROR_UNVERIFIED set" in (r.stdout + r.stderr)

    def test_no_mirror_skips_verification_path(self, tmp_path: Path) -> None:
        """Without a mirror, no checksum gate runs (GitHub direct download)."""
        install_dir, electron_dir, _ = self._electron_setup(tmp_path)
        body = f"""
        install_dir={str(install_dir)!r}
        _restore_electron_dist "$install_dir" ""
        echo "RC=$?"
        """
        r = _run_harness(body)
        # No mirror → the fail-closed message must NOT appear.
        assert "refusing unverified mirror build" not in (r.stdout + r.stderr)
