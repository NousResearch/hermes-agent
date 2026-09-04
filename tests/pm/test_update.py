"""Pure unit tests for pm.update (no network, no store, no lockfile writes).

The upstream index helpers (github_release_tags, npm_dist_tags, ...) are
network I/O — covered by design, never in tests. The resolution engine is
pure: candidate lists in, Resolved decision out.
"""

from __future__ import annotations

from pm.update import (
    Resolved,
    best_in_minor,
    minor_of,
    resolve_best,
    resolve_package,
    version_key,
)


def _pkg(pkg_name="node", version_style="semver", **latest):
    """A stub package whose latest_versions returns canned per-target lists."""

    class _P:
        def __init__(self):
            self.name = pkg_name
            self.version_style = version_style
            self._latest = dict(latest)

        def latest_versions(self, target, locked=None):
            return list(self._latest.get(target, []))

    return _P()


# ── version parsing / comparison ───────────────────────────────────────────


def test_version_key_sorts_numeric_and_suffixes():
    assert version_key("2.53.0+5") > version_key("2.53.0+3")
    assert version_key("3.11.16+20260814") > version_key("3.11.16+20260801")
    assert version_key("26.8.1") > version_key("26.7.0")
    assert version_key("10362") > version_key("10361")
    # prerelease-ish segments sort after numerics
    assert version_key("9.0.1") < version_key("9.0.1-rc1")


def test_minor_of():
    assert minor_of("26.7.0") == (26, 7)
    assert minor_of("3.11.16+20260814") == (3, 11)
    assert minor_of("10362") is None  # single component


def test_best_in_minor():
    versions = ["9.0.3", "9.0.1", "9.1.0", "8.4.9"]
    assert best_in_minor(versions, (9, 0)) == "9.0.3"
    assert best_in_minor(versions, (9, 1)) == "9.1.0"
    assert best_in_minor(versions, (10, 0)) is None


# ── semver resolution ──────────────────────────────────────────────────────


def test_semver_picks_highest_shared_version():
    r = resolve_best(
        "node",
        ["win32-x64", "linux-x64"],
        {"win32-x64": ["26.8.1", "26.7.0"], "linux-x64": ["26.8.1", "26.7.0"]},
        locked="26.7.0",
        style="semver",
    )
    assert r.changed
    assert r.version == "26.8.1"
    assert r.per_target == {"win32-x64": "26.8.1", "linux-x64": "26.8.1"}


def test_semver_no_update_when_up_to_date():
    r = resolve_best(
        "node",
        ["win32-x64"],
        {"win32-x64": ["26.7.0"]},
        locked="26.7.0",
        style="semver",
    )
    assert not r.changed
    assert r.version == "26.7.0"


def test_semver_divergent_targets_no_shared_version():
    r = resolve_best(
        "gh",
        ["win32-x64", "linux-x64"],
        {"win32-x64": ["2.97.0"], "linux-x64": ["2.96.0"]},
        locked="2.95.0",
        style="semver",
    )
    assert r.version is None
    assert r.reason == "no shared version"
    assert not r.changed


# ── minor-style (ffmpeg) resolution ───────────────────────────────────────


def test_minor_style_shared_minor_with_per_target_patches():
    """The cadence-mismatch case: posix and win32 drift in PATCH. The update
    moves to the highest shared major.minor; each target pins its own patch."""
    r = resolve_best(
        "ffmpeg",
        ["linux-x64", "win32-x64"],
        {"linux-x64": ["9.1.2", "9.0.1"], "win32-x64": ["9.1.0", "9.0.1"]},
        locked="9.0.1",
        style="minor",
    )
    assert r.changed
    assert r.version == "9.1"
    assert r.per_target == {"linux-x64": "9.1.2", "win32-x64": "9.1.0"}


def test_minor_style_no_shared_minor_blocks_update():
    """posix on 9.1, win32 still on 9.0 — no minor every target serves."""
    r = resolve_best(
        "ffmpeg",
        ["linux-x64", "win32-x64"],
        {"linux-x64": ["9.1.2"], "win32-x64": ["9.0.3"]},
        locked="9.0.1",
        style="minor",
    )
    assert r.version is None
    assert r.reason == "no shared minor"
    assert not r.changed


def test_minor_style_patch_drift_within_shared_minor_is_up_to_date():
    """Same minor, patch drift only — the lockfile version label doesn't
    move (patches live in per-target urls)."""
    r = resolve_best(
        "ffmpeg",
        ["linux-x64", "win32-x64"],
        {"linux-x64": ["9.1.2"], "win32-x64": ["9.1.0"]},
        locked="9.1",
        style="minor",
    )
    assert not r.changed
    assert r.version == "9.1"


# ── source availability ────────────────────────────────────────────────────


def test_no_source_anywhere():
    r = resolve_best("chromium", ["win32-x64"], {"win32-x64": []}, locked="1208+145", style="semver")
    assert r.version is None
    assert r.reason == "no source"
    assert not r.changed


def test_missing_source_for_one_target_skipped():
    """A target whose index is unreachable must not block the others."""
    r = resolve_best(
        "node",
        ["win32-x64", "linux-x64"],
        {"win32-x64": ["26.8.1"], "linux-x64": []},
        locked="26.7.0",
        style="semver",
    )
    assert r.changed
    assert r.version == "26.8.1"


# ── resolve_package wiring ─────────────────────────────────────────────────


def test_resolve_package_calls_latest_versions_per_target():
    pkg = _pkg(
        "node",
        **{
            "win32-x64": ["26.8.1", "26.7.0"],
            "linux-x64": ["26.8.1", "26.7.0"],
        },
    )
    r = resolve_package(pkg, ["win32-x64", "linux-x64"], locked="26.7.0")
    assert isinstance(r, Resolved)
    assert r.changed
    assert r.version == "26.8.1"


# ── llama.app installer bucket helpers (pure, monkeypatched) ───────────────


def test_llama_app_latest_parses_build_number(monkeypatch):
    from pm import update as u

    calls = []
    monkeypatch.setattr(
        u,
        "_get_text",
        lambda url, headers=None: (calls.append((url, headers)) or "b10679\n"),
    )
    assert u.llama_app_latest() == "10679"
    url, headers = calls[0]
    assert url.endswith("/resolve/latest")
    assert headers == {}  # no HF_TOKEN → no auth header


def test_llama_app_latest_honors_hf_token(monkeypatch):
    from pm import update as u

    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    seen = {}
    monkeypatch.setattr(u, "_get_text", lambda url, headers=None: (seen.update(headers) or "b10612\n"))
    assert u.llama_app_latest() == "10612"
    assert seen.get("Authorization") == "Bearer hf-secret"


def test_llama_app_latest_unreachable_returns_none(monkeypatch):
    from pm import update as u

    monkeypatch.setattr(u, "_get_text", lambda url, headers=None: (_ for _ in ()).throw(OSError("down")))
    assert u.llama_app_latest() is None


def test_llama_app_bucket_versions_dedupes_and_sorts(monkeypatch):
    from pm import update as u

    # The HF tree API ignores offset and returns only the FIRST page —
    # the oldest ~1000 paths sorted by path ascending. The helper must
    # dedupe (the same build appears on many paths) and sort by build
    # number descending so the newest of what the tree can see comes first.
    page = [
        {"path": "b10326/aarch64/linux/cpu/kk/llama-app.zst"},
        {"path": "b10098/x86_64/linux/vulkan/kk/llama-app.zst"},
        {"path": "b10326/x86_64/windows/cuda/13/kk/llama-app.exe.zst"},  # dup
        {"path": "b9733/aarch64/macos/metal/kk/llama-app.zst"},
        {"path": "latest/whatever"},  # non-build path must be ignored
    ]
    monkeypatch.setattr(u, "_get_json", lambda url: page)
    assert u.llama_app_bucket_versions() == ["10326", "10098", "9733"]


def test_llama_app_bucket_versions_unreachable_returns_empty(monkeypatch):
    from pm import update as u

    monkeypatch.setattr(u, "_get_json", lambda url: (_ for _ in ()).throw(OSError("down")))
    assert u.llama_app_bucket_versions() == []


def test_llamacpp_latest_versions_prefers_bucket(monkeypatch):
    """The real LlamaCppCpu class: latest from the llama.app `latest`
    pointer first, then the bucket tree, GitHub only as a fallback.
    Patching targets the from-imported names in pm.packages (a from-import
    copies the reference at import time — patching pm.update would miss)."""
    import pm.packages as pkgs

    monkeypatch.setattr(pkgs, "llama_app_latest", lambda: "10679")
    monkeypatch.setattr(pkgs, "llama_app_bucket_versions", lambda: ["10679", "10612", "10362"])
    called = []
    monkeypatch.setattr(pkgs, "github_release_tags", lambda *a, **k: (called.append(a) or ["99999"]))

    pkg = pkgs.LlamaCppCpu()
    versions = pkg.latest_versions("linux-x64")
    assert versions == ["10679", "10679", "10612", "10362"]
    assert called == []  # GitHub never consulted when the bucket answered


def test_llamacpp_latest_versions_github_fallback(monkeypatch):
    """Bucket unreachable → fall back to the GitHub releases tags."""
    import pm.packages as pkgs

    monkeypatch.setattr(pkgs, "llama_app_latest", lambda: None)
    monkeypatch.setattr(pkgs, "llama_app_bucket_versions", lambda: [])
    monkeypatch.setattr(pkgs, "github_release_tags", lambda *a, **k: ["10362", "10217"])

    pkg = pkgs.LlamaCppCpu()
    assert pkg.latest_versions("linux-x64") == ["10362", "10217"]
