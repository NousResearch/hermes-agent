"""Regression: MiniMax API-key overlays must use the /anthropic base.

`anthropic_messages` transport POSTs to `{base}/v1/messages`. MiniMax serves
that only under `/anthropic`; the bare `/v1` host is OpenAI-only and 404s the
Anthropic transport. Any overlay that forces anthropic_messages for MiniMax
must therefore pin a `base_url_override` ending in `/anthropic`.

Guards against the regression where `minimax` / `minimax-cn` shipped no
override and fell back to the models.dev `/v1` base → 404 page not found.
"""
import pytest

from hermes_cli.providers import HERMES_OVERLAYS


@pytest.mark.parametrize(
    "provider_id, expected_base",
    [
        ("minimax", "https://api.minimax.io/anthropic"),
        ("minimax-oauth", "https://api.minimax.io/anthropic"),
        ("minimax-cn", "https://api.minimaxi.com/anthropic"),
    ],
)
def test_minimax_overlay_pins_anthropic_base(provider_id, expected_base):
    ov = HERMES_OVERLAYS[provider_id]
    assert ov.transport == "anthropic_messages"
    assert ov.base_url_override == expected_base, (
        f"{provider_id!r} forces anthropic_messages but base_url_override="
        f"{ov.base_url_override!r}; the anthropic transport 404s on the /v1 "
        f"OpenAI path. Expected {expected_base!r}."
    )


def test_no_minimax_overlay_falls_back_to_v1():
    """Any minimax-family anthropic overlay must NOT rely on the /v1 default."""
    for pid, ov in HERMES_OVERLAYS.items():
        if "minimax" in pid and ov.transport == "anthropic_messages":
            assert ov.base_url_override.endswith("/anthropic"), (
                f"{pid!r} would resolve to the models.dev /v1 base and 404."
            )
