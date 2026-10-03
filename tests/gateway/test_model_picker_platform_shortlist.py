"""Regression tests for the per-platform ``/model`` picker shortlist (#89809).

The Matrix ``/model`` picker is reaction-based and hard-capped at ten slots
(``_MATRIX_MODEL_PICKER_REACTIONS`` in ``plugins/platforms/matrix/adapter.py``), so a provider
with more models than that silently loses the tail — and the only lever over *which* ten appear
was the shared ``providers.<name>.models`` allowlist, which also trims the CLI/desktop and
Telegram/Discord/Slack pickers.  ``<platform>.model_allowlist`` narrows the list for that one
platform (and only its picker; the CLI/desktop pickers never read it).

These tests drive the real ``_handle_model_command`` against a real temp ``HERMES_HOME`` with a
fake picker-capable adapter that records the provider rows the picker was handed.
"""

import types

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

# Two providers, the first with more models than the Matrix picker has slots.
_LISTED_ROWS = [
    {
        "slug": "ollama-cloud",
        "name": "Ollama Cloud",
        "is_user_defined": True,
        "models": ["deepseek-v4-pro", "glm-5.2", "kimi-k3", "qwen3-coder", "llama-4-scout"],
        "total_models": 19,
        "source": "user-config",
    },
    {
        "slug": "openrouter",
        "name": "OpenRouter",
        "models": ["openai/gpt-5.5"],
        "total_models": 120,
        "source": "builtin",
    },
]


class _CapturingPickerAdapter:
    """Minimal adapter that looks picker-capable and records the rows it received."""

    def __init__(self):
        self.captured_providers = None

    async def send_model_picker(
        self, *, chat_id, providers, current_model, current_provider, session_key,
        on_model_selected, metadata=None,
    ):
        self.captured_providers = providers
        return types.SimpleNamespace(success=True)


def _make_runner(adapter, platform):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _event(platform, text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="12345", chat_type="dm"),
    )


def _isolated_home(tmp_path, monkeypatch, extra_config):
    """Write a temp config.yaml, point the gateway at it, and stub the live listing."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config = {"model": {"default": "deepseek-v4-pro", "provider": "ollama-cloud"}, "providers": {}}
    config.update(extra_config)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    # Deterministic inventory: the live listing probes credentials and disk caches.
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **kw: [dict(row) for row in _LISTED_ROWS],
    )
    return hermes_home


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_config",
    [
        {"matrix": {"model_allowlist": ["GLM-5.2", "kimi-k3"]}},
        {"platforms": {"matrix": {"model_allowlist": ["GLM-5.2", "kimi-k3"]}}},
    ],
    ids=["matrix-block", "platforms-matrix-block"],
)
async def test_matrix_picker_narrows_to_configured_allowlist(tmp_path, monkeypatch, extra_config):
    """A ``matrix.model_allowlist`` shows only those models in the Matrix picker.

    Rows with no allowed model are dropped entirely rather than rendered empty, and the
    per-provider count follows the narrowed list so the reply cannot advertise 19 models
    while offering two. Both platform-block spellings the config loader accepts are covered.
    """
    adapter = _CapturingPickerAdapter()
    _isolated_home(tmp_path, monkeypatch, extra_config)

    assert await _make_runner(adapter, Platform.MATRIX)._handle_model_command(
        _event(Platform.MATRIX)
    ) is None, "picker path should have handled the command"

    rows = adapter.captured_providers
    assert rows, "picker was not sent the provider rows"
    assert [row["slug"] for row in rows] == ["ollama-cloud"], (
        "the provider with no allowlisted model should be dropped, got %r" % ([r["slug"] for r in rows],)
    )
    assert rows[0]["models"] == ["glm-5.2", "kimi-k3"], (
        "picker should list exactly the allowlisted models, got %r" % (rows[0]["models"],)
    )
    assert rows[0]["total_models"] == 2


@pytest.mark.asyncio
async def test_shortlist_does_not_leak_into_other_platforms(tmp_path, monkeypatch):
    """The same config leaves every other platform's picker on the full provider list."""
    adapter = _CapturingPickerAdapter()
    _isolated_home(tmp_path, monkeypatch, {"matrix": {"model_allowlist": ["glm-5.2"]}})

    assert await _make_runner(adapter, Platform.TELEGRAM)._handle_model_command(
        _event(Platform.TELEGRAM)
    ) is None

    rows = {row["slug"]: row for row in adapter.captured_providers or []}
    assert set(rows) == {"ollama-cloud", "openrouter"}
    assert rows["ollama-cloud"]["models"] == _LISTED_ROWS[0]["models"], (
        "Telegram must keep the provider's full model list"
    )
    assert rows["ollama-cloud"]["total_models"] == 19
