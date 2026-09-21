"""Provider-free contracts for the P10 image-generation convergence lane."""
from __future__ import annotations

import pytest

from image_jobs import (
    ImageRequestError,
    canonical_layouts,
    get_style_profile,
    prepare_image_request,
)


def test_prepare_image_defaults_to_codex_for_a_named_style() -> None:
    prepared = prepare_image_request(
        prompt="An editorial map of a multi-agent research system.",
        style="Data Atlas",
    )

    assert prepared.backend == "codex"
    assert prepared.style_id == "data-atlas"
    assert prepared.references == ()


def test_prepare_image_uses_local_only_when_explicitly_selected() -> None:
    prepared = prepare_image_request(
        prompt="A visual quality comparison of two local workflows.",
        style="Technical Diorama",
        backend="local",
    )

    assert prepared.backend == "local"


def test_prepare_image_rejects_unapproved_backend_without_a_fallback() -> None:
    with pytest.raises(ImageRequestError, match="backend"):
        prepare_image_request(
            prompt="A system map.",
            style="Data Atlas",
            backend="fal",
        )


def test_prepare_image_resolves_style_ids_and_human_labels() -> None:
    by_id = prepare_image_request(
        prompt="A dependable operations dashboard.",
        style="signal-hud",
    )
    by_label = prepare_image_request(
        prompt="A dependable operations dashboard.",
        style="Dark Cyberpunk HUD",
    )

    assert by_id.style_id == by_label.style_id == "signal-hud"


def test_prepare_image_rejects_unknown_style_instead_of_guessing() -> None:
    with pytest.raises(ImageRequestError, match="unknown style"):
        prepare_image_request(
            prompt="A system map.",
            style="Whatever Looks Good",
        )


def test_supplied_public_links_are_preserved_for_text_and_visual_processing() -> None:
    prepared = prepare_image_request(
        prompt="A comparison chart for the article.",
        style="Baoyu Infographic",
        references=(
            "https://example.com/research/report.pdf",
            "https://images.example.com/reference.png",
        ),
    )

    assert [reference.url for reference in prepared.references] == [
        "https://example.com/research/report.pdf",
        "https://images.example.com/reference.png",
    ]
    assert all(
        reference.requested_roles == frozenset({"written_inspiration", "visual_reference"})
        for reference in prepared.references
    )


def test_style_catalogue_keeps_style_specific_layout_grammar() -> None:
    profile = get_style_profile("Data Atlas")

    assert profile.style_id == "data-atlas"
    assert profile.label == "Data Atlas"
    assert "Sankey" in profile.layout_options
    assert "network graph" in profile.layout_options


def test_style_catalogue_accepts_legacy_social_studio_aliases() -> None:
    profile = get_style_profile("dark-cyberpunk-hud")

    assert profile.style_id == "signal-hud"


def test_canonical_layout_catalogue_preserves_legacy_social_layout_ids() -> None:
    layouts = canonical_layouts()

    assert "poster" in layouts
    assert "flow_diagram" in layouts
    assert "architectural cross-section" in layouts


def test_prepare_image_caps_distinct_reference_fanout_before_staging() -> None:
    with pytest.raises(ImageRequestError, match="at most"):
        prepare_image_request(
            prompt="A secure reference-aware image.",
            style="Data Atlas",
            references=tuple(f"https://example.com/source-{index}" for index in range(9)),
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:pass@example.com/private.png",
        "http://127.0.0.1:8188/system_stats",
        "http://[::1]/internal.png",
        "http://localhost/private.png",
    ],
)
def test_prepare_image_rejects_unsafe_reference_urls(url: str) -> None:
    with pytest.raises(ImageRequestError, match="reference"):
        prepare_image_request(
            prompt="A secure reference-aware image.",
            style="Data Atlas",
            references=(url,),
        )
