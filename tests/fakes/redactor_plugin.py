"""A redaction plugin for ``transform_llm_output`` stream-hold tests.

``redactor_home(home, base_url, hold=...)`` writes a loopback-routed ``HERMES_HOME`` with a user
plugin that rewrites every ``SECRET-<digits>`` in the final reply to ``[REDACTED]``, registered
with ``hold_stream=True`` when ``hold`` is set. ``redactor_turn()`` is the scripted model turn: a
tool call carrying commentary (model text between tool calls) and then the streamed answer.
"""

from __future__ import annotations

from pathlib import Path

from tests.fakes.fake_llm_provider import Text, ToolCall, write_hermes_home

RAW = "Here is the config. Your api key is SECRET-123 and the backup key is SECRET-456. Keep them safe."
REWRITE = "Here is the config. Your api key is [REDACTED] and the backup key is [REDACTED]. Keep them safe."
COMMENTARY = "Checking where SECRET-777 is stored first."

_PLUGIN = """import re


def _redact(response_text, **_):
    return re.sub(r"SECRET-\\d+", "[REDACTED]", response_text)


def register(ctx):
    ctx.register_hook("transform_llm_output", _redact{hold})
"""

_CONFIG = (
    "updates:\n  check: false\n"
    "models_dev:\n  url: http://127.0.0.1:9/api.json\n"
    # The background memory/skill review would fork a second agent against the same stub.
    "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"
    # ``todo_list`` sits in the request's tools instead of behind tool_search.
    'tools:\n  tool_search:\n    enabled: "off"\n'
    "platform_toolsets:\n  api_server: [todo]\n"
    "plugins:\n  enabled:\n    - redactor\n"
)


def redactor_home(home: Path, base_url: str, *, hold: bool) -> Path:
    plugin = home / "plugins" / "redactor"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text("name: redactor\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN.format(hold=", hold_stream=True" if hold else ""), encoding="utf-8")
    return write_hermes_home(home, base_url, extra_config=_CONFIG)


def redactor_turn(*, delay_per_chunk: float = 0.0) -> list:
    return [ToolCall("todo_list", {}, text=COMMENTARY), Text(RAW, chunk_chars=6, delay_per_chunk=delay_per_chunk)]


def raw_leaks(texts) -> list[str]:
    """Every user-visible text that carried an unredacted secret."""
    return [t for t in texts if isinstance(t, str) and "SECRET-" in t]
