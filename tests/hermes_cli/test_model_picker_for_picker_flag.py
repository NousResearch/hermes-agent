"""Desktop model picker must use the picker credential posture (#124510).

``build_model_options_payload`` backs every GUI picker surface (API server
``GET /api/model/options``, dashboard Models page, TUI picker). It forwarded
to ``build_models_payload`` without ``for_picker``, so OAuth-subscription
providers (``openai-codex`` via the external ``~/.codex/auth.json`` store,
exhausted-pool rows) were dropped while the CLI picker (``for_picker=True``)
listed them fine.

These tests pin the behavior at the payload seam with a
``list_authenticated_providers`` stand-in that reproduces the real branching:
the ``openai-codex`` row exists only under the picker posture. Enrichment
steps (pricing/capabilities/featured) are stubbed as no-ops; they are
orthogonal to row visibility.
"""

from unittest.mock import patch

from hermes_cli.inventory import ConfigContext, build_model_options_payload


def _ctx() -> ConfigContext:
    return ConfigContext(
        current_provider="",
        current_model="",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=[],
    )


def _fake_list_authenticated_providers(**kwargs):
    rows = [
        {
            "slug": "anthropic",
            "name": "Anthropic",
            "models": ["claude-opus-4-6"],
            "total_models": 1,
        }
    ]
    if kwargs.get("for_picker"):
        rows.append(
            {
                "slug": "openai-codex",
                "name": "ChatGPT or Codex Subscription",
                "models": ["gpt-5.5"],
                "total_models": 1,
            }
        )
    return [dict(r) for r in rows]


def _payload(**kwargs):
    with (
        patch(
            "hermes_cli.model_switch.list_authenticated_providers",
            side_effect=_fake_list_authenticated_providers,
        ),
        patch("hermes_cli.inventory._apply_pricing", lambda *a, **k: None),
        patch("hermes_cli.inventory._apply_capabilities", lambda *a, **k: None),
        patch("hermes_cli.inventory._apply_featured", lambda *a, **k: None),
    ):
        return build_model_options_payload(_ctx(), **kwargs)


def test_picker_payload_keeps_oauth_subscription_row():
    slugs = {r["slug"] for r in _payload()["providers"]}
    assert "openai-codex" in slugs, (
        "GUI payload dropped the Codex subscription row; "
        "build_model_options_payload must forward for_picker=True"
    )


def test_picker_payload_keeps_posture_on_explicit_refresh():
    slugs = {r["slug"] for r in _payload(refresh=True)["providers"]}
    assert "openai-codex" in slugs, (
        "explicit refresh dropped the Codex subscription row; "
        "for_picker=True must survive refresh=True"
    )
