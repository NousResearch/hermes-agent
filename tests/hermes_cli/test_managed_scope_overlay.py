"""apply_managed_overlay() — the shared helper used by every standalone loader."""
import textwrap

import pytest


@pytest.fixture
def managed(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return md


def _write(md, body):
    (md / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()


def test_overlay_noop_without_scope(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "nope"))
    managed_scope.invalidate_managed_cache()
    src = {"display": {"skin": "user"}}
    assert managed_scope.apply_managed_overlay(src) == {"display": {"skin": "user"}}


def test_overlay_preserves_user_siblings(managed):
    from hermes_cli import managed_scope

    _write(managed, "display:\n  skin: charizard\n")
    out = managed_scope.apply_managed_overlay(
        {"display": {"skin": "user", "show_reasoning": True}}
    )
    assert out["display"]["skin"] == "charizard"
    assert out["display"]["show_reasoning"] is True


def _write_flat_moa(md):
    _write(
        md,
        """
        moa:
          reference_models:
            - provider: anthropic
              model: claude-fable-5
          aggregator:
            provider: openai-codex
            model: gpt-5.6-sol
        """,
    )


def _assert_flat_moa_resolves(out):
    from hermes_cli.moa_config import resolve_moa_preset

    resolved = resolve_moa_preset(out["moa"])
    assert resolved["reference_models"] == [
        {"provider": "anthropic", "model": "claude-fable-5", "enabled": True}
    ]
    assert resolved["aggregator"]["provider"] == "openai-codex"
    assert resolved["aggregator"]["model"] == "gpt-5.6-sol"


def test_flat_managed_moa_replaces_inherited_named_presets(managed):
    from hermes_cli import managed_scope

    _write_flat_moa(managed)
    out = managed_scope.apply_managed_overlay(
        {
            "moa": {
                "default_preset": "custom",
                "presets": {
                    "custom": {
                        "reference_models": [
                            {"provider": "openrouter", "model": "old/reference"}
                        ],
                        "aggregator": {
                            "provider": "openrouter",
                            "model": "old/aggregator",
                        },
                    }
                },
            }
        }
    )

    _assert_flat_moa_resolves(out)


def test_flat_managed_moa_replaces_malformed_user_value(managed):
    from hermes_cli import managed_scope

    _write_flat_moa(managed)
    out = managed_scope.apply_managed_overlay({"moa": "invalid"})

    _assert_flat_moa_resolves(out)


