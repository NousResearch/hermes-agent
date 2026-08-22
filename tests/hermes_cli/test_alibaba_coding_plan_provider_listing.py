"""Test that alibaba-coding-plan requires its own key and does not fall back
to DASHSCOPE_API_KEY.

Regression for #70361: alibaba-coding-plan previously accepted
DASHSCOPE_API_KEY as a valid credential (both in the plugin env_vars and the
static auth.py registry), so it appeared as configured for users who only
held a generic DashScope key. Coding Plan is a separate billing SKU with its
own endpoint and must require ALIBABA_CODING_PLAN_API_KEY.
"""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


# -- Only DASHSCOPE_API_KEY set ---------------------------------------------


@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-dashscope-fake"}, clear=False)
def test_alibaba_coding_plan_hidden_when_only_dashscope_key_set():
    """alibaba-coding-plan must NOT appear when only DASHSCOPE_API_KEY is set."""
    providers = list_authenticated_providers(current_provider="alibaba")

    # alibaba-coding-plan must NOT be listed (no dedicated key)
    cp = next((p for p in providers if p["slug"] == "alibaba-coding-plan"), None)
    assert cp is None, (
        "alibaba-coding-plan should NOT appear when only DASHSCOPE_API_KEY is set"
    )

    # alibaba itself must still appear (DASHSCOPE_API_KEY belongs to it)
    alibaba = next((p for p in providers if p["slug"] == "alibaba"), None)
    assert alibaba is not None, (
        "alibaba should appear when DASHSCOPE_API_KEY is set"
    )


# -- Only ALIBABA_CODING_PLAN_API_KEY set -----------------------------------


@patch.dict(os.environ, {"ALIBABA_CODING_PLAN_API_KEY": "sk-cp-fake"}, clear=False)
def test_alibaba_coding_plan_appears_when_own_key_set():
    """alibaba-coding-plan should appear when ALIBABA_CODING_PLAN_API_KEY is set."""
    providers = list_authenticated_providers(current_provider="alibaba-coding-plan")

    cp = next((p for p in providers if p["slug"] == "alibaba-coding-plan"), None)
    assert cp is not None, (
        "alibaba-coding-plan should appear when ALIBABA_CODING_PLAN_API_KEY is set"
    )
