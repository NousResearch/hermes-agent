"""Tests that the ``hermes model`` picker does not list a configured provider
twice when its ``provider_key`` is also a canonical provider slug.

The CLI picker (``hermes_cli.main.select_provider_and_model``) appends
configured rows after the ``CANONICAL_PROVIDERS`` rows, so a provider whose
``providers:`` key is a canonical slug used to appear in both lists. See #7524.
"""

from unittest.mock import patch

import pytest
import yaml


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal config."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: old-model\nproviders: {}\n")
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_MODEL",
        "LLM_MODEL",
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


def _write_config(home, providers):
    cfg = {"model": "old-model", "providers": providers}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _capture_provider_labels(config_home):
    """Drive ``select_provider_and_model`` and return the provider-menu labels
    shown to the user (the first ``_prompt_provider_choice`` call). Cancels
    immediately after capturing."""
    from hermes_cli.main import select_provider_and_model

    captured: dict = {}

    def _capture_and_cancel(labels, default=0, title=None):
        if "labels" not in captured:
            captured["labels"] = list(labels)
        return None  # cancel

    with patch(
        "hermes_cli.main._prompt_provider_choice", side_effect=_capture_and_cancel
    ), patch("builtins.print"):
        select_provider_and_model()

    return captured.get("labels", [])


def test_canonical_provider_key_is_not_listed_twice(config_home):
    """A provider configured under a canonical slug must appear once (from the
    built-in catalog), not again as a custom row."""
    _write_config(
        config_home,
        {
            "openrouter": {
                "name": "My OpenRouter",
                "base_url": "https://openrouter.example/v1",
            }
        },
    )

    labels = _capture_provider_labels(config_home)
    assert labels, "provider menu was empty"
    assert not any("openrouter.example" in lbl for lbl in labels), (
        f"custom row duplicates the canonical OpenRouter entry; labels={labels}"
    )
    assert sum("OpenRouter" in lbl for lbl in labels) == 1, (
        f"OpenRouter should be listed exactly once; labels={labels}"
    )


def test_canonical_provider_key_casing_variant_is_suppressed(config_home):
    """``provider_key`` casing must not defeat the deduplication — provider ids
    are normalized to lowercase elsewhere in the CLI."""
    _write_config(
        config_home,
        {
            "OpenRouter": {
                "name": "My OpenRouter",
                "base_url": "https://openrouter.example/v1",
            }
        },
    )

    labels = _capture_provider_labels(config_home)
    assert labels, "provider menu was empty"
    assert not any("openrouter.example" in lbl for lbl in labels), (
        f"casing variant should be deduplicated too; labels={labels}"
    )


def test_non_canonical_custom_provider_still_listed(config_home):
    """Deduplication must not hide a genuinely custom provider."""
    _write_config(
        config_home,
        {
            "my-private-llm": {
                "name": "My Private LLM",
                "base_url": "https://private.example/v1",
            }
        },
    )

    labels = _capture_provider_labels(config_home)
    assert any("private.example" in lbl for lbl in labels), (
        f"non-canonical custom provider should remain available; labels={labels}"
    )
