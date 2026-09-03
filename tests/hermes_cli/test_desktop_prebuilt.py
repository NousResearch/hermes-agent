"""Prebuilt Desktop artifact ladder for ``hermes update``.

Invariants: exact SHA wins; nearest in-window ancestor is the fallback;
checksum mismatch and zip-slip never touch the live app; a successful
prebuilt install skips the source rebuild.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from hermes_cli.desktop_prebuilt import (
    DEFAULT_COMPATIBILITY_WINDOW,
    ArtifactIndex,
    DesktopArtifact,
    ancestor_distance,
    extract_zip_bytes,
    fetch_index_from_github_releases,
    flatten_single_wrapper,
    host_arch,
    host_platform,
    install_artifact,
    merge_indexes,
    parse_index,
    resolve_artifact,
    try_install_prebuilt_desktop,
    verify_sha256,
)
from hermes_cli.update_cmd import _rebuild_desktop_after_update


def _sha(n: int = 1) -> str:
    return f"{n:040x}"


def _art(
    commit: str,
    *,
    platform: str = "linux",
    arch: str = "x64",
    url: str = "https://example.invalid/a.zip",
    sha256: str | None = None,
    tag: str = "",
) -> DesktopArtifact:
    digest = sha256 or ("ab" * 32)
    return DesktopArtifact(
        commit=commit,
        platform=platform,
        arch=arch,
        url=url,
        sha256=digest,
        tag=tag,
        filename="a.zip",
    )


def _index(*arts: DesktopArtifact, window: int = 64) -> ArtifactIndex:
    return ArtifactIndex(1, window, list(arts))


def test_parse_index_drops_invalid_rows():
    parsed = parse_index(
        {
            "schema_version": 1,
            "compatibility_window": 8,
            "artifacts": [
                {
                    "commit": _sha(1),
                    "platform": "linux",
                    "arch": "x64",
                    "url": "https://example.invalid/ok.zip",
                    "sha256": "cd" * 32,
                },
                {"commit": "short", "platform": "linux", "arch": "x64", "url": "u", "sha256": "ab" * 32},
                {"commit": _sha(2), "platform": "linux", "arch": "x64", "url": "u"},
                "not-a-row",
            ],
        }
    )
    assert parsed.compatibility_window == 8
    assert len(parsed.artifacts) == 1
    assert parsed.artifacts[0].commit == _sha(1)
    assert parsed.artifacts[0].sha256 == "cd" * 32


def test_merge_indexes_last_writer_wins_identity():
    first = _index(_art(_sha(1), url="https://example.invalid/old.zip"))
    second = _index(_art(_sha(1), url="https://example.invalid/new.zip", sha256="ef" * 32))
    merged = merge_indexes([first, second])
    assert len(merged.artifacts) == 1
    assert merged.artifacts[0].url.endswith("new.zip")
    assert merged.artifacts[0].sha256 == "ef" * 32


def test_resolve_prefers_exact_sha_over_nearer_tag():
    exact = _art(_sha(9), tag="v-exact")
    older = _art(_sha(1), tag="v-old")
    chosen = resolve_artifact(
        _index(older, exact),
        commit=_sha(9),
        platform_name="linux",
        arch="x64",
        distance_fn=lambda _a, _b: 1,
        window=64,
    )
    assert chosen is exact


def test_resolve_nearest_tag_within_window():
    near = _art(_sha(2), tag="v-near")
    far = _art(_sha(3), tag="v-far")

    def distance(ancestor: str, _head: str) -> int | None:
        return { _sha(2): 3, _sha(3): 40 }[ancestor]

    chosen = resolve_artifact(
        _index(far, near, window=16),
        commit=_sha(9),
        platform_name="linux",
        arch="x64",
        distance_fn=distance,
    )
    assert chosen is near


def test_resolve_skips_outside_window_and_other_arch():
    too_far = _art(_sha(2), tag="v-far")
    other = _art(_sha(3), arch="arm64", tag="v-arm")
    chosen = resolve_artifact(
        _index(too_far, other, window=4),
        commit=_sha(9),
        platform_name="linux",
        arch="x64",
        distance_fn=lambda _a, _b: 10,
    )
    assert chosen is None


def test_host_platform_and_arch_aliases():
    assert host_platform("linux") == "linux"
    assert host_platform("darwin") == "darwin"
    assert host_platform("win32") == "win32"
    assert host_arch("x86_64") == "x64"
    assert host_arch("amd64") == "x64"
    assert host_arch("aarch64") == "arm64"


def test_verify_sha256_and_extract_rejects_slip(tmp_path: Path):
    payload = b"hello-desktop"
    digest = hashlib.sha256(payload).hexdigest()
    assert verify_sha256(payload, digest)
    assert not verify_sha256(payload, "00" * 32)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError, match="slip"):
        extract_zip_bytes(buf.getvalue(), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_and_flatten_wrapper(tmp_path: Path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("wrap/linux-unpacked/Hermes", b"exe")
    dest = tmp_path / "staging"
    extract_zip_bytes(buf.getvalue(), dest)
    flatten_single_wrapper(dest)
    assert (dest / "linux-unpacked" / "Hermes").read_bytes() == b"exe"


def _linux_unpacked_zip(exe_bytes: bytes = b"fake-hermes") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("linux-unpacked/Hermes", exe_bytes)
    return buf.getvalue()


def test_install_artifact_checksum_mismatch_leaves_live_app(tmp_path: Path):
    desktop = tmp_path / "apps" / "desktop"
    live = desktop / "release" / "linux-unpacked"
    live.mkdir(parents=True)
    (live / "Hermes").write_bytes(b"LIVE")
    blob = _linux_unpacked_zip(b"NEW")
    art = _art(_sha(1), url="https://example.invalid/a.zip", sha256="00" * 32)
    ok = install_artifact(
        art,
        desktop,
        project_root=tmp_path,
        fetch_bytes=lambda _url: blob,
    )
    assert ok is False
    assert (live / "Hermes").read_bytes() == b"LIVE"


def test_install_artifact_swaps_on_valid_checksum(tmp_path: Path):
    desktop = tmp_path / "apps" / "desktop"
    live = desktop / "release" / "linux-unpacked"
    live.mkdir(parents=True)
    (live / "Hermes").write_bytes(b"LIVE")
    blob = _linux_unpacked_zip(b"NEW")
    art = _art(
        _sha(1),
        url="https://example.invalid/a.zip",
        sha256=hashlib.sha256(blob).hexdigest(),
    )
    ok = install_artifact(
        art,
        desktop,
        project_root=tmp_path,
        fetch_bytes=lambda _url: blob,
    )
    assert ok is True
    assert (live / "Hermes").read_bytes() == b"NEW"


def test_try_install_uses_exact_sha_and_injected_index(tmp_path: Path):
    desktop = tmp_path / "apps" / "desktop"
    live = desktop / "release" / "linux-unpacked"
    live.mkdir(parents=True)
    (live / "Hermes").write_bytes(b"LIVE")
    blob = _linux_unpacked_zip(b"SHA-HIT")
    head = _sha(7)
    art = _art(
        head,
        url="https://example.invalid/hit.zip",
        sha256=hashlib.sha256(blob).hexdigest(),
    )

    def run_git(args: list[str], _cwd: Path) -> str | None:
        if args[:2] == ["rev-parse", "HEAD"]:
            return head + "\n"
        return None

    ok = try_install_prebuilt_desktop(
        desktop,
        project_root=tmp_path,
        fetch_bytes=lambda _url: blob,
        run_git=run_git,
        platform_name="linux",
        arch="x64",
        index=_index(art),
    )
    assert ok is True
    assert (live / "Hermes").read_bytes() == b"SHA-HIT"


def test_try_install_misses_without_git_head(tmp_path: Path):
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    ok = try_install_prebuilt_desktop(
        desktop,
        project_root=tmp_path,
        run_git=lambda *_a, **_k: None,
        index=_index(_art(_sha(1))),
    )
    assert ok is False


def test_ancestor_distance_zero_and_counts(tmp_path: Path):
    calls: list[list[str]] = []

    def run_git(args: list[str], _cwd: Path) -> str | None:
        calls.append(args)
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return ""
        if args[:2] == ["rev-list", "--count"]:
            return "5\n"
        return None

    assert ancestor_distance(_sha(1), _sha(1), cwd=tmp_path, run_git=run_git) == 0
    assert ancestor_distance(_sha(1), _sha(2), cwd=tmp_path, run_git=run_git) == 5


class _Result:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def test_rebuild_skips_source_when_prebuilt_installs(tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    spawned: list[list[str]] = []

    class _FakeMain:
        PROJECT_ROOT = tmp_path

        @staticmethod
        def _resolve_node_runtime_npm():
            return "/fake/npm"

        @staticmethod
        def _desktop_build_needed(*_a, **_kw):
            return True

        @staticmethod
        def _run_logged_subprocess(cmd, cwd=None, env=None):
            spawned.append(cmd)
            return _Result(0)

    monkeypatch.setattr(update_cmd, "_m", lambda: _FakeMain)
    monkeypatch.setattr(
        "hermes_cli.desktop_prebuilt.try_install_prebuilt_desktop",
        lambda *_a, **_k: True,
    )
    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    assert spawned == []


def test_rebuild_falls_back_to_source_when_prebuilt_misses(tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    spawned: list[list[str]] = []

    class _FakeMain:
        PROJECT_ROOT = tmp_path

        @staticmethod
        def _resolve_node_runtime_npm():
            return "/fake/npm"

        @staticmethod
        def _desktop_build_needed(*_a, **_kw):
            return True

        @staticmethod
        def _run_logged_subprocess(cmd, cwd=None, env=None):
            spawned.append(cmd)
            return _Result(0)

    monkeypatch.setattr(update_cmd, "_m", lambda: _FakeMain)
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path", lambda: {}, raising=False
    )
    monkeypatch.setattr(
        "hermes_cli.desktop_prebuilt.try_install_prebuilt_desktop",
        lambda *_a, **_k: False,
    )
    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    assert len(spawned) == 1


def test_live_github_releases_discovery_hits_real_api():
    """Live GET of the public releases list (no credential). Empty is OK."""
    statuses: list[int] = []

    def capturing_fetch(url: str) -> bytes:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "hermes-cli-desktop-prebuilt-test",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            statuses.append(int(resp.status))
            return resp.read()

    index = fetch_index_from_github_releases(fetch_bytes=capturing_fetch, per_page=5)
    assert statuses and statuses[0] == 200
    assert index.schema_version >= 1
    assert isinstance(index.artifacts, list)
    assert index.compatibility_window >= 0
    # No desktop-index.json is published yet; discovery must still return a
    # valid empty-or-populated index rather than raise.
    for art in index.artifacts:
        assert art.platform in {"linux", "darwin", "win32"}
        assert len(art.sha256) == 64


def test_default_window_constant_is_positive():
    assert DEFAULT_COMPATIBILITY_WINDOW > 0
