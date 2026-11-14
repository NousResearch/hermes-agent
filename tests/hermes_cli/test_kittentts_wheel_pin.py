"""Regression: the kittentts extra's direct wheel URL must be integrity-pinned.

Supply-chain guard (PR #82891, re-adapted to the pm/extras architecture after
main decomposed the old inline installers): the KittenTTS wheel ships from a
third-party GitHub release, whose asset BYTES ARE MUTABLE IN PLACE — same URL,
different digest, no signal. Before the pin, both install flows
(`hermes setup tts` and `hermes tools post-setup kittentts`) resolved the
extra through this manifest line with zero integrity, so a tampered or
replaced release was installed silently.

PEP 508 direct references support a `#sha256=` fragment and uv enforces it on
every resolution/sync of that URL, which makes the manifest the right place
for the anchor: it is consulted by BOTH flows, and `uv lock --check` stays
green because uv.lock already mirrors the same wheel digest.

These assertions could not have passed before the fix — the manifest line
carried no fragment at all (red proof: delete `#sha256=` from the extra and
test_every_direct_url_wheel_in_kittentts_extra_is_sha256_pinned fails).
"""

from __future__ import annotations

import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"

_SHA256_FRAG = re.compile(r"#sha256=([0-9a-f]{64})(?= |$)")

# sha256 of the official kittentts-0.8.1-py3-none-any.whl, cross-checked
# against uv.lock's recorded wheel hash (single anchor, two layers).
_KITTENTTS_WHEEL = "kittentts-0.8.1-py3-none-any.whl"


def _kittentts_extra_entries() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["kittentts"]


def test_every_direct_url_wheel_in_kittentts_extra_is_sha256_pinned() -> None:
    """Every https direct-reference in the extra carries a 64-hex #sha256=.

    Mutable release assets (github.com/.../releases/download/...) MUST be
    digest-pinned; content-addressed refs (git+https pinned at a commit SHA)
    are already immutable-by-identity and are checked by a separate guard.
    """
    offenders: list[str] = []
    seen_direct = False
    for entry in _kittentts_extra_entries():
        if "@" not in entry:
            continue  # plain name==version / specifier pin
        ref = entry.split("@", 1)[1].split(";", 1)[0].strip()
        if ref.startswith("git+"):
            continue  # git URL resolved at an immutable commit SHA
        seen_direct = True
        if not _SHA256_FRAG.search(ref):
            offenders.append(entry)
        assert "releases/download" not in ref or _SHA256_FRAG.search(ref), (
            f"mutable GitHub-release wheel must carry #sha256=: {entry}"
        )
    assert seen_direct, "extra lost its direct wheel reference (guard went stale)"
    assert not offenders, f"unpinned direct URLs in kittentts extra: {offenders}"


def test_pinned_digest_matches_official_release() -> None:
    """The manifest anchors the exact official 0.8.1 wheel digest.

    If upstream bumps the version, the URL and the fragment must be updated
    TOGETHER (this test fails on a copy-paste of either half). The digest was
    verified against uv.lock's recorded wheel hash and the 2026-08 release
    asset recorded when the pin was first computed (PR #82891).
    """
    entry = next(
        (e for e in _kittentts_extra_entries() if _KITTENTTS_WHEEL in e), None
    )
    assert entry is not None, f"kittentts extra no longer references {_KITTENTTS_WHEEL}"
    m = _SHA256_FRAG.search(entry)
    assert m, f"missing #sha256= fragment: {entry}"
    assert m.group(1) == (
        "482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851"
    ), "pinned digest drifted from the official release anchor"


def test_uv_lock_mirrors_the_manifest_digest() -> None:
    """Manifest fragment and lock hash must not disagree.

    uv.lock records the wheel under its own `hash = "sha256:..."` field; if
    the two layers drift, `uv lock --check` and installs diverge — one
    anchor, stated twice. Parse the lock text (no tomllib round-trip of the
    whole 4k-line file needed).
    """
    lock = UV_LOCK.read_text(encoding="utf-8")
    m = re.search(
        _KITTENTTS_WHEEL
        + r'", hash = "sha256:([0-9a-f]{64})"',
        lock,
    )
    assert m, f"uv.lock no longer records a sha256 for {_KITTENTTS_WHEEL}"
    entry = next(e for e in _kittentts_extra_entries() if _KITTENTTS_WHEEL in e)
    frag = _SHA256_FRAG.search(entry)
    assert frag and frag.group(1) == m.group(1), (
        "pyproject #sha256= fragment and uv.lock wheel hash disagree"
    )
