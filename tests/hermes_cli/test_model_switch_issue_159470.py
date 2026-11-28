"""Regression tests for GH #159470.

1. A ``/model`` switch to a custom provider whose display name contains spaces
   mints the ``custom:<raw name>`` provider id. The resolver's alias set must
   accept that exact token, not just the dash-normalised/slug forms.
2. ``persist_model_selection`` must refuse a suspicious ``new_model`` (non
   printable / overlong picker text) instead of corrupting ``model.default``.
"""

import pytest

import hermes_yaml as yaml

from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection
from hermes_cli.providers import custom_provider_aliases, resolve_custom_provider


_SPACED_NAME = "My Model (Server 1)"
_CUSTOM_PROVIDERS = [
    {"name": _SPACED_NAME, "base_url": "https://gateway.example.test/v1", "api_key": "k"}
]


def test_custom_aliases_accept_custom_prefixed_raw_name():
    """The gateway rejects a token Hermes itself minted for a spaced provider name."""
    aliases = custom_provider_aliases(_SPACED_NAME)
    assert f"custom:{_SPACED_NAME.lower()}" in aliases


def test_resolve_custom_provider_accepts_custom_prefixed_raw_name():
    """The resolver must accept ``custom:<raw name>`` from switch_model (PATH B)."""
    resolved = resolve_custom_provider(f"custom:{_SPACED_NAME.lower()}", _CUSTOM_PROVIDERS)
    assert resolved is not None


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "gpt-oss-120b " + "\u2500" * 124,  # picker box-drawing overflow from the report
        "model\nwith-control",
        "m" * 129,
    ],
)
def test_persist_refuses_suspicious_model_id(candidate, tmp_path, monkeypatch):
    """A corrupted model id must not be written into ``model.default``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"default": "old-model", "provider": "openai"}}), encoding="utf-8")

    result = ModelSwitchResult(
        success=True,
        new_model=candidate,
        target_provider="openai",
        is_global=True,
    )
    persist_model_selection(result, path)

    stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert stored["model"]["default"] == "old-model"
