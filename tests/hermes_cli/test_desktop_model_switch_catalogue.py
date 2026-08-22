"""Desktop picker catalogue proof for the live model-switch fast path."""

import secrets
import time
from unittest.mock import patch

import pytest

from hermes_cli.model_switch import switch_model
from tui_gateway import server as gateway_server


@pytest.fixture
def catalogue_proof():
    session_ids: list[str] = []

    def issue(
        entries: frozenset[tuple[str, str]], served_at: float | None = None
    ) -> str:
        proof = secrets.token_urlsafe(32)
        session_id = f"test-catalogue-proof-{proof}"
        with gateway_server._sessions_lock:
            gateway_server._sessions[session_id] = {
                "model_options_catalogue": entries,
                "model_options_catalogue_at": (
                    time.monotonic() if served_at is None else served_at
                ),
                "model_options_catalogue_proof": proof,
            }
        session_ids.append(session_id)
        return proof

    yield issue

    with gateway_server._sessions_lock:
        for session_id in session_ids:
            gateway_server._sessions.pop(session_id, None)


def test_catalogue_proof_switch_skips_redundant_remote_model_probe(
    catalogue_proof,
):
    """A server-proven picker pair must not repeat the provider /models call."""
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.models.validate_requested_model") as validate,
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": "https://provider.example/v1",
                "api_mode": "chat_completions",
            },
        ),
    ):
        result = switch_model(
            raw_input="anthropic/claude-sonnet-4.6",
            current_provider="openrouter",
            current_model="old-model",
            current_base_url="https://provider.example/v1",
            current_api_key="test-key",
            explicit_provider="openrouter",
            catalogue_proof=catalogue_proof(
                frozenset({("openrouter", "anthropic/claude-sonnet-4.6")})
            ),
        )

    assert result.success is True
    assert result.new_model == "anthropic/claude-sonnet-4.6"
    validate.assert_not_called()


def test_unproven_switch_retains_remote_model_validation():
    """Typed/non-picker model switches preserve the existing validation gate."""
    accepted = {
        "accepted": True,
        "persist": True,
        "recognized": True,
        "message": None,
    }
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value=accepted,
        ) as validate,
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": "https://provider.example/v1",
                "api_mode": "chat_completions",
            },
        ),
    ):
        result = switch_model(
            raw_input="anthropic/claude-sonnet-4.6",
            current_provider="openrouter",
            current_model="old-model",
            current_base_url="https://provider.example/v1",
            current_api_key="test-key",
            explicit_provider="openrouter",
        )

    assert result.success is True
    validate.assert_called_once()


def test_unmatched_or_stale_catalogue_evidence_retains_remote_validation(
    catalogue_proof,
):
    """Unrelated or stale catalogue evidence cannot bypass validation."""
    accepted = {"accepted": True, "persist": True, "recognized": True, "message": None}
    valid = catalogue_proof(
        frozenset({("openrouter", "anthropic/claude-sonnet-4.6")})
    )
    orphaned = catalogue_proof(
        frozenset({("openrouter", "anthropic/claude-sonnet-4.6")})
    )
    with gateway_server._sessions_lock:
        gateway_server._sessions.pop(f"test-catalogue-proof-{orphaned}")
    proofs = (
        catalogue_proof(
            frozenset({("other-provider", "anthropic/claude-sonnet-4.6")})
        ),
        catalogue_proof(frozenset({("openrouter", "other-model")})),
        catalogue_proof(
            frozenset({("openrouter", "anthropic/claude-sonnet-4.6")}),
            time.monotonic() - 301,
        ),
        f"{valid}x",
        orphaned,
        "not-a-proof",
    )
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.models.validate_requested_model", return_value=accepted) as validate,
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"api_key": "test-key", "base_url": "https://provider.example/v1", "api_mode": "chat_completions"},
        ),
    ):
        for proof in proofs:
            result = switch_model(
                raw_input="anthropic/claude-sonnet-4.6",
                current_provider="openrouter",
                current_model="old-model",
                current_base_url="https://provider.example/v1",
                current_api_key="test-key",
                explicit_provider="openrouter",
                catalogue_proof=proof,
            )
            assert result.success is True

    assert validate.call_count == len(proofs)
