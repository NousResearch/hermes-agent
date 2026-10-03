"""Picker current-model injection guards — ``_finalize_picker_rows``.

``_finalize_picker_rows`` injects ``model.default`` into the current provider's
row so a custom/uncurated model set via ``/model <provider>/<name>`` stays
visible in pickers. The injection must not surface a ``model.default`` left
over from a *different* provider's family — the everyday state after
``hermes profile create --clone`` + ``config set model.provider <x>`` (#125640).

Contracts under test:

- a model id that positively identifies as another single-vendor provider's
  family is NOT injected into the current row (nor inflates ``total_models``);
- an unrecognized id (custom fine-tune) and aggregator/custom-endpoint rows
  keep the historical inject behavior;
- vendor- and scope-prefixed ids (``anthropic/claude-…``, ``hf:org/GLM-…``)
  are normalized before the family match.
"""

from hermes_cli.model_switch_providers import _finalize_picker_rows


def _row(slug, models, *, is_current=True, is_user_defined=False, **extra):
    row = {
        "slug": slug,
        "name": slug,
        "is_current": is_current,
        "is_user_defined": is_user_defined,
        "models": list(models),
        "total_models": len(models),
        "source": "built-in",
    }
    row.update(extra)
    return row


def test_foreign_family_model_not_injected_into_current_row():
    row = _row("openai-codex", ["gpt-5.2"])
    out = _finalize_picker_rows([row], {}, "claude-sonnet-5")
    assert out[0]["models"] == ["gpt-5.2"]
    assert out[0]["total_models"] == 1


def test_reverse_mismatch_not_injected():
    row = _row("anthropic", ["claude-opus-4-6"])
    out = _finalize_picker_rows([row], {}, "gpt-5.2")
    assert out[0]["models"] == ["claude-opus-4-6"]


def test_vendor_prefixed_foreign_id_not_injected():
    row = _row("openai-codex", ["gpt-5.2"])
    out = _finalize_picker_rows([row], {}, "anthropic/claude-sonnet-5")
    assert out[0]["models"] == ["gpt-5.2"]


def test_scope_prefixed_foreign_id_not_injected():
    row = _row("openai-codex", ["gpt-5.2"])
    out = _finalize_picker_rows([row], {}, "hf:zai-org/GLM-5.3-Flash")
    assert out[0]["models"] == ["gpt-5.2"]


def test_uncurated_same_family_model_still_injected():
    row = _row("zai", ["glm-5.1"])
    out = _finalize_picker_rows([row], {}, "glm-5.3-flash")
    assert out[0]["models"] == ["glm-5.3-flash", "glm-5.1"]
    assert out[0]["total_models"] == 2


def test_same_vendor_sibling_slug_still_injects():
    row = _row("openai-api", ["gpt-5.2"])
    out = _finalize_picker_rows([row], {}, "codex-mini-latest")
    assert out[0]["models"] == ["codex-mini-latest", "gpt-5.2"]


def test_unknown_custom_model_still_injected():
    row = _row("openai-codex", ["gpt-5.2"])
    out = _finalize_picker_rows([row], {}, "my-finetune-ft-001")
    assert out[0]["models"] == ["my-finetune-ft-001", "gpt-5.2"]


def test_aggregator_row_not_gated():
    row = _row("openrouter", ["anthropic/claude-sonnet-5"])
    out = _finalize_picker_rows([row], {}, "claude-sonnet-5")
    assert out[0]["models"] == ["claude-sonnet-5", "anthropic/claude-sonnet-5"]


def test_custom_endpoint_row_not_gated():
    row = _row("custom", ["llama-local"], is_user_defined=True, api_url="http://127.0.0.1:11434/v1")
    out = _finalize_picker_rows([row], {}, "claude-sonnet-5")
    assert out[0]["models"] == ["claude-sonnet-5", "llama-local"]


def test_blank_slug_row_still_injects():
    row = _row("", ["m1"])
    out = _finalize_picker_rows([row], {}, "claude-sonnet-5")
    assert out[0]["models"] == ["claude-sonnet-5", "m1"]


def test_astra_account_discovery_guard_still_wins():
    # The pre-existing guard must keep firing before any family logic: an Astra
    # id on an account-discovery provider aborts injection entirely.
    row = _row("openai-codex", ["gpt-5.6-sol"])
    out = _finalize_picker_rows([row], {}, "gpt-6-astra")
    assert out[0]["models"] == ["gpt-5.6-sol"]
