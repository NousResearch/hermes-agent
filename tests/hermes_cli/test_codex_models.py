"""Behavioral tests for the pure helper functions in hermes_cli.codex_models.

_add_context_variants imports from agent.model_metadata which may not be
available in isolation, so that function is tested via integration only when
the import succeeds; the rest are pure list transforms with no heavy deps.
"""

import base64
import json

import pytest

from hermes_cli.codex_models import (
    DEFAULT_CODEX_MODELS,
    _FORWARD_COMPAT_TEMPLATE_MODELS,
    _add_forward_compat_models,
    _extract_chatgpt_account_id,
    _finalize_codex_models,
)


# ── _add_forward_compat_models ────────────────────────────────────────────────

class TestAddForwardCompatModels:
    def test_no_synthesis_when_all_models_present(self):
        all_ids = [m for m, _ in _FORWARD_COMPAT_TEMPLATE_MODELS]
        result = _add_forward_compat_models(all_ids)
        # All synthetic models were already present, no duplication
        assert result == list(dict.fromkeys(all_ids))

    def test_synthesizes_gpt56_sol_when_gpt55_present(self):
        live = ["gpt-5.5"]
        result = _add_forward_compat_models(live)
        assert "gpt-5.6-sol" in result
        assert "gpt-5.6-terra" in result
        assert "gpt-5.6-luna" in result

    def test_synthesizes_spark_when_gpt53_codex_present(self):
        live = ["gpt-5.3-codex"]
        result = _add_forward_compat_models(live)
        assert "gpt-5.3-codex-spark" in result

    def test_no_duplication_of_existing_models(self):
        live = ["gpt-5.5", "gpt-5.6-sol"]
        result = _add_forward_compat_models(live)
        assert result.count("gpt-5.6-sol") == 1

    def test_preserves_input_order_before_synthetics(self):
        live = ["gpt-5.4", "gpt-5.3-codex"]
        result = _add_forward_compat_models(live)
        assert result[0] == "gpt-5.4"
        assert result[1] == "gpt-5.3-codex"
        # synthetic entries come after
        for synthetic in result[2:]:
            assert synthetic not in live

    def test_empty_input(self):
        assert _add_forward_compat_models([]) == []

    def test_unknown_models_passed_through(self):
        live = ["some-future-model-xyz"]
        result = _add_forward_compat_models(live)
        assert "some-future-model-xyz" in result


# ── _extract_chatgpt_account_id ───────────────────────────────────────────────

def _make_jwt(payload: dict) -> str:
    """Build a minimal (unsigned) JWT with the given payload."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{body}."


class TestExtractChatgptAccountId:
    def test_extracts_account_id_from_jwt(self):
        token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-abc123"}})
        result = _extract_chatgpt_account_id(token)
        assert result == "acct-abc123"

    def test_returns_none_when_claim_absent(self):
        token = _make_jwt({"sub": "user-xyz"})
        assert _extract_chatgpt_account_id(token) is None

    def test_returns_none_for_malformed_jwt(self):
        assert _extract_chatgpt_account_id("notajwt") is None

    def test_returns_none_for_empty_string(self):
        assert _extract_chatgpt_account_id("") is None

    def test_returns_none_for_empty_account_id(self):
        token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": ""}})
        assert _extract_chatgpt_account_id(token) is None


# ── DEFAULT_CODEX_MODELS invariants ──────────────────────────────────────────

def test_default_codex_models_non_empty():
    assert len(DEFAULT_CODEX_MODELS) >= 1


def test_default_codex_models_no_duplicates():
    assert len(DEFAULT_CODEX_MODELS) == len(set(DEFAULT_CODEX_MODELS))


def test_default_codex_models_are_strings():
    for m in DEFAULT_CODEX_MODELS:
        assert isinstance(m, str) and m


def test_forward_compat_template_references_exist_in_defaults_or_curated():
    """Every template in _FORWARD_COMPAT_TEMPLATE_MODELS references at least
    one model that either appears in DEFAULT_CODEX_MODELS or is itself a
    synthetic entry (present as a key in the template list).
    """
    all_known = set(DEFAULT_CODEX_MODELS) | {m for m, _ in _FORWARD_COMPAT_TEMPLATE_MODELS}
    for synthetic, templates in _FORWARD_COMPAT_TEMPLATE_MODELS:
        for t in templates:
            assert t in all_known, f"Template '{t}' for '{synthetic}' not in known model set"
