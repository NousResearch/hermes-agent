"""End-to-end no-exfiltration gate: applied secret values must never reach the provider.

Issue #77162. Shape-based redaction (vendor prefixes, ``KEY=value`` assignments,
auth headers) cannot catch an opaque value applied from an external secret
source under a non-credential name (``DATABASE_URL``, an arbitrary 1Password
item key). Those values are masked by exact-value match instead:

* tool-result construction (``make_tool_result_message``) — the content that is
  appended to the provider-bound ``messages`` list, and
* the pre-send sanitizers (``redact_sensitive_text``).

Each test here asserts on the bytes that actually leave the process, not on an
intermediate helper's return value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402

from agent import redact  # noqa: E402
from agent.redact import redact_sensitive_text  # noqa: E402
from agent.tool_dispatch_helpers import make_tool_result_message  # noqa: E402

# No vendor prefix, no assignment shape, no URL/query structure — every
# shape-based pass in agent/redact.py lets it through verbatim.
OPAQUE_SOURCE_VALUE = "pg-prod-9f2c1a77be4d3051"
OPAQUE_ENV_VALUE = "mw-opaque-token-4471bde2"


def _active_home() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()).resolve()


def _seed_secret_source(name: str, value: str) -> None:
    """Simulate a value applied from Bitwarden / 1Password / a command source."""
    env_loader._SECRET_SOURCE_VALUES_BY_HOME[str(_active_home())] = {name: value}


@pytest.fixture(autouse=True)
def _redaction_enabled_and_clean_sources():
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader.reset_secret_source_cache()
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader.reset_secret_source_cache()


@pytest.fixture(autouse=True)
def _redaction_on(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestToolResultEgress:
    """``make_tool_result_message`` output is what lands in provider-bound messages."""

    def test_arbitrary_secret_source_value_masked_in_tool_result(self):
        _seed_secret_source("DATABASE_URL", OPAQUE_SOURCE_VALUE)
        msg = make_tool_result_message(
            "terminal", f"connection ok for {OPAQUE_SOURCE_VALUE}", "call_1"
        )
        assert OPAQUE_SOURCE_VALUE not in msg["content"]
        assert "***" in msg["content"]

    def test_credential_suffixed_env_value_masked_in_tool_result(self, monkeypatch):
        monkeypatch.setenv("MY_SERVICE_TOKEN", OPAQUE_ENV_VALUE)
        msg = make_tool_result_message(
            "terminal", f"upstream rejected {OPAQUE_ENV_VALUE}", "call_2"
        )
        assert OPAQUE_ENV_VALUE not in msg["content"]
        assert "***" in msg["content"]

    def test_content_parts_masked_in_tool_result(self):
        _seed_secret_source("FOO", OPAQUE_SOURCE_VALUE)
        parts = [
            {"type": "text", "text": f"header x-secret: {OPAQUE_SOURCE_VALUE}"},
            {"type": "text", "text": "clean part"},
        ]
        msg = make_tool_result_message("web_extract", parts, "call_3")
        wire = json.dumps(msg["content"], ensure_ascii=False)
        assert OPAQUE_SOURCE_VALUE not in wire
        assert "***" in wire

    def test_wire_serialization_of_provider_bound_message_carries_no_value(self):
        """The assertion the issue asks for: the serialized provider-bound
        message must not contain the applied value, only the mask."""
        _seed_secret_source("DATABASE_URL", OPAQUE_SOURCE_VALUE)
        msg = make_tool_result_message(
            "terminal", f"dump -> {OPAQUE_SOURCE_VALUE}", "call_4"
        )
        wire = json.dumps([msg], ensure_ascii=False, default=str)
        assert OPAQUE_SOURCE_VALUE not in wire
        assert "***" in wire

    def test_unrelated_content_is_untouched(self):
        _seed_secret_source("DATABASE_URL", OPAQUE_SOURCE_VALUE)
        msg = make_tool_result_message(
            "terminal", "42 files changed, all tests passed", "call_5"
        )
        assert msg["content"] == "42 files changed, all tests passed"


class TestPreSendSanitizers:
    """``redact_sensitive_text`` is what the pre-send sanitizers call."""

    def test_force_pass_masks_applied_secret_source_value(self):
        _seed_secret_source("DATABASE_URL", OPAQUE_SOURCE_VALUE)
        out = redact_sensitive_text(f"connecting to {OPAQUE_SOURCE_VALUE} now", force=True)
        assert OPAQUE_SOURCE_VALUE not in out
        assert "***" in out

    def test_default_pass_masks_applied_secret_source_value(self):
        _seed_secret_source("FOO", OPAQUE_SOURCE_VALUE)
        out = redact_sensitive_text(f"value={OPAQUE_SOURCE_VALUE}")
        assert OPAQUE_SOURCE_VALUE not in out

    def test_credential_suffixed_env_value_masked(self, monkeypatch):
        monkeypatch.setenv("DB_PASSWORD", OPAQUE_ENV_VALUE)
        out = redact_sensitive_text(f"auth failed for {OPAQUE_ENV_VALUE}", force=True)
        assert OPAQUE_ENV_VALUE not in out

    def test_terminal_output_path_masks_applied_value(self):
        from agent.redact import redact_terminal_output

        _seed_secret_source("SOME_TOKEN", OPAQUE_SOURCE_VALUE)
        out = redact_terminal_output(f"printed {OPAQUE_SOURCE_VALUE}", "echo hi")
        assert OPAQUE_SOURCE_VALUE not in out


class TestSafetyValves:
    """The exact-value pass must not mangle ordinary content or ignore the opt-out."""

    def test_short_values_are_not_masked(self):
        _seed_secret_source("MODE", "prod9")
        out = redact_sensitive_text("running in prod9 mode", force=True)
        assert out == "running in prod9 mode"
        msg = make_tool_result_message("terminal", "running in prod9 mode", "call_s")
        assert msg["content"] == "running in prod9 mode"

    def test_values_from_another_profile_home_are_not_masked(self, tmp_path):
        other_home = tmp_path / "profile-b"
        other_home.mkdir()
        env_loader._SECRET_SOURCE_VALUES_BY_HOME[str(other_home.resolve())] = {
            "DATABASE_URL": OPAQUE_SOURCE_VALUE
        }
        out = redact_sensitive_text(f"peer dump {OPAQUE_SOURCE_VALUE}", force=True)
        assert OPAQUE_SOURCE_VALUE in out

    def test_redaction_opt_out_passes_tool_result_through(self, monkeypatch):
        """``security.redact_secrets: false`` covers this pass too: the tool-result
        constructor has no force boundary of its own, so it follows the switch."""
        _seed_secret_source("DATABASE_URL", OPAQUE_SOURCE_VALUE)
        monkeypatch.setattr("agent.redact._redact_enabled", lambda: False)
        msg = make_tool_result_message(
            "terminal", f"connecting to {OPAQUE_SOURCE_VALUE}", "call_o"
        )
        assert OPAQUE_SOURCE_VALUE in msg["content"]
        # The public helper follows the same switch.
        assert OPAQUE_SOURCE_VALUE in redact.mask_exact_secret_values(
            f"connecting to {OPAQUE_SOURCE_VALUE}"
        )

    def test_force_boundary_masks_even_with_redaction_disabled(self, monkeypatch):
        """``force=True`` is documented as "must never return raw secrets
        regardless" — the exact-value pass inherits that, so a pre-send
        sanitizer cannot leak an applied value under the opt-out either."""
        _seed_secret_source("DATABASE_URL", OPAQUE_SOURCE_VALUE)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        monkeypatch.setattr("agent.redact._redact_enabled", lambda: False)
        out = redact_sensitive_text(f"connecting to {OPAQUE_SOURCE_VALUE}", force=True)
        assert OPAQUE_SOURCE_VALUE not in out
        assert "***" in out

    def test_public_helper_is_a_no_op_without_values(self):
        helper = getattr(redact, "mask_exact_secret_values", None)
        assert helper is not None, "agent.redact.mask_exact_secret_values is missing (#77162)"
        assert helper("plain text with nothing to hide") == (
            "plain text with nothing to hide"
        )
