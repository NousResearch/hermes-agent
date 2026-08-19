"""Tests for the extended style menu (style_registry) + content_engine seam.

Content-engine-scoped tests: registry package + image_jobs + image_job_service.
The runner/gateway tests live in the repo-root suite (tests/test_style_registry_runner.py)
where tools/ and gateway/ are importable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure content_engine is importable (tests run from repo root or content_engine).
_REPO = Path(__file__).resolve().parents[1]
_CE = _REPO / "content_engine"
if str(_CE) not in sys.path:
    sys.path.insert(0, str(_CE))

from style_registry import (  # noqa: E402
    RegistryError,
    assert_neutral,
    blend,
    get_registry,
    resolve_fragment,
    resolve_style,
    search,
)


# ---------------------------------------------------------------------------
# (a) style_registry package
# ---------------------------------------------------------------------------

def test_registry_loads_full():
    reg = get_registry()
    assert reg["count"] >= 9000
    assert len(reg["styles"]) >= 9000


def test_resolve_style_known():
    entry = resolve_style("steampunk")
    assert entry["style_label"]
    assert entry["prompt_fragment"]
    # The registry surface must never carry the protected source name or SREF.
    assert "name" not in entry
    assert "sref" not in json.dumps(entry)


def test_resolve_style_unknown_fails_fast():
    with pytest.raises(RegistryError):
        resolve_style("definitely-not-a-style-slug")


def test_blend_deterministic():
    a = blend(["steampunk", "synthwave"], seed=7)
    b = blend(["steampunk", "synthwave"], seed=7)
    assert a == b
    assert "prompt_fragment" in a
    assert "resolution_notes" in a


def test_blend_requires_two():
    with pytest.raises(RegistryError):
        blend(["steampunk"])


def test_search_by_traits():
    results = search(["editorial"], limit=5)
    assert len(results) >= 1
    for r in results:
        assert "style_label" in r
        assert "slug" in r


def test_assert_neutral_rejects_protected_name():
    with pytest.raises(RegistryError):
        assert_neutral("painting in the style of Claude Monet")


def test_assert_neutral_rejects_sref():
    with pytest.raises(RegistryError):
        assert_neutral("abstract --sref 1234567890")


def test_assert_neutral_accepts_clean():
    assert_neutral("abstract painting, vivid palette, soft lighting")


def test_resolve_fragment_none_when_no_style():
    assert resolve_fragment(style_slug=None, blend_slugs=None) is None


def test_resolve_fragment_single_style():
    frag = resolve_fragment(style_slug="steampunk", blend_slugs=None)
    assert frag and "sref" not in frag.lower()


def test_resolve_fragment_blend():
    frag = resolve_fragment(style_slug=None, blend_slugs=["steampunk", "synthwave"], seed=7)
    assert frag and "sref" not in frag.lower()


# ---------------------------------------------------------------------------
# (b) image_jobs.prepare_image_request
# ---------------------------------------------------------------------------

def test_prepare_image_request_accepts_blend():
    from image_jobs import prepare_image_request

    prepared = prepare_image_request(
        prompt="a brass airship over a neon city",
        style="mythic-tech-codex",
        blend=["steampunk", "synthwave"],
    )
    assert prepared.registry_traits
    assert prepared.registry_slugs == ("steampunk", "synthwave")
    # base style still resolved (backward compatible)
    assert prepared.style_id == "mythic-tech-codex"


def test_prepare_image_request_no_blend_backward_compatible():
    from image_jobs import prepare_image_request

    prepared = prepare_image_request(prompt="a sunset", style="mythic-tech-codex")
    assert prepared.registry_traits is None
    assert prepared.registry_slugs == ()


def test_prepare_image_request_unknown_blend_fails_fast():
    from image_jobs import ImageRequestError, prepare_image_request

    with pytest.raises(ImageRequestError):
        prepare_image_request(
            prompt="a sunset",
            style="mythic-tech-codex",
            blend=["steampunk", "not-a-real-slug"],
        )


def test_prepare_image_request_single_slug_style_trait():
    from image_jobs import prepare_image_request

    # A single registry slug is now a valid single-style additive trait
    # (not a blend). This supports the extended menu for auto-routed jobs
    # where the variation picks one style with no blend partner.
    prepared = prepare_image_request(
        prompt="a sunset",
        style="mythic-tech-codex",
        blend=["steampunk"],
    )
    assert prepared.registry_slugs == ("steampunk",)
    assert prepared.registry_traits
    assert "sref" not in (prepared.registry_traits or "").lower()


# ---------------------------------------------------------------------------
# (c) image_job_service._native_prompt merges neutral traits
# ---------------------------------------------------------------------------

def test_native_prompt_merges_neutral_registry_traits():
    from image_job_service import _native_prompt
    from image_jobs import prepare_image_request

    prepared = prepare_image_request(
        prompt="a brass airship over a neon city",
        style="mythic-tech-codex",
        blend=["steampunk", "synthwave"],
    )
    prompt_text = _native_prompt(prepared, ())
    assert "Additional neutral style traits:" in prompt_text
    # The merged fragment is the registry's neutral blend, never protected/SREF.
    assert "Claude Monet" not in prompt_text
    assert "sref" not in prompt_text.lower()


def test_native_prompt_no_registry_traits_when_absent():
    from image_job_service import _native_prompt
    from image_jobs import prepare_image_request

    prepared = prepare_image_request(prompt="a sunset", style="mythic-tech-codex")
    prompt_text = _native_prompt(prepared, ())
    assert "Additional neutral style traits:" not in prompt_text
