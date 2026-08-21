"""Provider-plugin transport metadata must reach auxiliary vision clients."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


_PLUGIN_SOURCE = """
from providers import ProviderProfile, register_provider

register_provider(ProviderProfile(
    name="responses-profile",
    api_mode="codex_responses",
    display_name="Responses Profile",
    env_vars=("RESPONSES_PROFILE_API_KEY",),
    base_url="http://127.0.0.1:9/v1",
    default_aux_model="gpt-test-vision",
    supports_vision=True,
))

register_provider(ProviderProfile(
    name="responses-runtime",
    api_mode="chat_completions",
    display_name="Responses Runtime",
    env_vars=("RESPONSES_RUNTIME_API_KEY",),
    base_url="http://127.0.0.1:9/v1",
    default_aux_model="gpt-test-vision",
    supports_vision=True,
))
"""


_PROBE_SOURCE = """
import json

from agent.auxiliary_client import (
    AsyncCodexAuxiliaryClient,
    resolve_provider_client,
    resolve_vision_provider_client,
)

profile_provider, profile_client, profile_model = resolve_vision_provider_client(
    provider="auto",
    async_mode=True,
    main_runtime={
        "provider": "responses-profile",
        "model": "gpt-test-vision",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "profile-test-key",
    },
)

runtime_provider, runtime_client, runtime_model = resolve_vision_provider_client(
    provider="auto",
    async_mode=True,
    main_runtime={
        "provider": "responses-runtime",
        "model": "gpt-test-vision",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "runtime-test-key",
        "api_mode": "codex_responses",
    },
)

explicit_client, explicit_model = resolve_provider_client(
    "responses-profile",
    "gpt-test-vision",
    async_mode=True,
    api_mode="chat_completions",
)

print(json.dumps({
    "profile": {
        "provider": profile_provider,
        "model": profile_model,
        "client": type(profile_client).__name__,
        "responses": isinstance(profile_client, AsyncCodexAuxiliaryClient),
    },
    "runtime": {
        "provider": runtime_provider,
        "model": runtime_model,
        "client": type(runtime_client).__name__,
        "responses": isinstance(runtime_client, AsyncCodexAuxiliaryClient),
    },
    "explicit": {
        "model": explicit_model,
        "client": type(explicit_client).__name__,
        "responses": isinstance(explicit_client, AsyncCodexAuxiliaryClient),
    },
}))
"""


def test_user_provider_transport_reaches_real_vision_resolution(tmp_path: Path) -> None:
    """Exercise discovery -> auth registry -> vision routing -> client adapter."""
    hermes_home = tmp_path / "hermes-home"
    plugin = hermes_home / "plugins" / "model-providers" / "responses-only"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        textwrap.dedent(_PLUGIN_SOURCE), encoding="utf-8"
    )
    (plugin / "plugin.yaml").write_text(
        "name: responses-only\nkind: model-provider\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "RESPONSES_PROFILE_API_KEY": "profile-test-key",
            "RESPONSES_RUNTIME_API_KEY": "runtime-test-key",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE_SOURCE)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["profile"] == {
        "provider": "responses-profile",
        "model": "gpt-test-vision",
        "client": "AsyncCodexAuxiliaryClient",
        "responses": True,
    }
    assert result["runtime"] == {
        "provider": "responses-runtime",
        "model": "gpt-test-vision",
        "client": "AsyncCodexAuxiliaryClient",
        "responses": True,
    }
    assert result["explicit"]["model"] == "gpt-test-vision"
    assert result["explicit"]["client"] == "AsyncOpenAI"
    assert result["explicit"]["responses"] is False
