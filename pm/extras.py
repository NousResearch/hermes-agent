"""Extras: the runtime features of the python venv.

Feature names ARE pyproject extra names. This table maps each extra to the
import that proves it (the anchor), so availability is one find_spec — no
package-manager machinery on the hot path. sync_venv([extra]) makes an extra
true; pm owns HOW (uv sync inside the venv package).
"""

from __future__ import annotations

import importlib.util


# extra name -> module that proves it is installed
ANCHORS: dict[str, str | tuple[str, ...]] = {
    "anthropic": "anthropic",
    "bedrock": "boto3",
    "vertex": "google.auth",
    "azure-identity": "azure.identity",
    "exa": "exa_py",
    "firecrawl": "firecrawl",
    "parallel-web": "parallel",
    "otlp": "opentelemetry.sdk",
    "mistral": "mistralai",
    "edge-tts": "edge_tts",
    "tts-premium": "elevenlabs",
    "voice": "faster_whisper",
    "stt-whisper": "faster_whisper",
    "audio-io": ("sounddevice", "numpy"),
    "silk": "pilk",
    "wake": "openwakeword",
    "wake-openwakeword": "openwakeword",
    "wake-sherpa": "sherpa_onnx",
    "wake-porcupine": "pvporcupine",
    "wake-tflite": "ai_edge_litert",
    "fal": "fal_client",
    "honcho": "honcho",
    "hindsight": "hindsight",
    "supermemory": "supermemory",
    "mem0": "mem0",
    "messaging": "telegram",
    "telegram": "telegram",
    "discord": "discord",
    "slack": "slack_bolt",
    "matrix": ("mautrix", "asyncpg", "aiosqlite", "markdown", "aiohttp_socks"),
    "dingtalk": "dingtalk_stream",
    "feishu": "lark_oapi",
    "wecom": "defusedxml",
    "teams": "microsoft.teams.apps",
    "modal": "modal",
    "daytona": "daytona",
    "vercel": "vercel",
    "google": "googleapiclient",
    "google-chat": "google.cloud.pubsub_v1",
    "youtube": "youtube_transcript_api",
    "acp": "acp",
    "web": "fastapi",
    "doc-extract": "anydoc",
    "computer-use": "mcp",
    "trace-upload": "huggingface_hub",
    # Pillow resize recovery for vision tools (the `vision` extra is a no-op
    # back-compat alias — Pillow is core — but ensure_import("vision") must
    # still resolve an anchor so availability checks work).
    "vision": "PIL",
}


def _anchors(extra: str) -> tuple[str, ...]:
    got = ANCHORS.get(extra, extra.replace("-", "_"))
    return got if isinstance(got, tuple) else (got,)


def _importable(anchor: str) -> bool:
    """An anchor already present in sys.modules counts even without a
    findable spec — tests fake SDKs by inserting modules there, and the
    caller's import right after this check resolves the same way."""
    import sys

    if anchor in sys.modules:
        return True
    try:
        return importlib.util.find_spec(anchor) is not None
    except (ImportError, ValueError):
        return False


def available(extra: str) -> bool:
    """Fast, side-effect-free: are all of this extra's anchors importable?
    A platform-gated extra whose gate excludes this machine reads as
    unavailable — the anchors are absent by design, not by accident."""
    if not extra_supported(extra):
        return False
    return all(_importable(a) for a in _anchors(extra))


def _platform_gates() -> dict[str, str]:
    """The [tool.hermes.extras-platforms] table from pyproject.toml:
    extra -> PEP 508 marker string. Cached per process."""
    global _PLATFORM_GATES
    if _PLATFORM_GATES is not None:
        return _PLATFORM_GATES
    import tomllib

    from pm.paths import repo_root

    gates: dict[str, str] = {}
    try:
        with (repo_root() / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        table = data.get("tool", {}).get("hermes", {}).get("extras-platforms", {})
        if isinstance(table, dict):
            gates = {str(k): str(v) for k, v in table.items()}
    except (OSError, ValueError):
        pass
    _PLATFORM_GATES = gates
    return gates


_PLATFORM_GATES: dict[str, str] | None = None


def extra_supported(extra: str) -> bool:
    """Is this extra installable on THIS platform? True when the extra
    carries no gate, or its marker matches the running platform. An
    extra that IS present on this machine (anchors importable) is always
    supported — an installed override beats the table (dev machines,
    hand-synced venvs)."""
    if all(_importable(a) for a in _anchors(extra)):
        return True
    marker = _platform_gates().get(extra)
    if marker is None:
        return True
    import os
    import platform
    import sys

    from packaging.markers import Marker

    environment = {
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "os_name": os.name,
    }
    try:
        return bool(Marker(marker).evaluate(environment=environment))
    except Exception:
        # A malformed marker must never brick availability — treat as
        # ungated and let the resolver be the authority.
        return True


def ensure_import(extra: str) -> None:
    """Make an extra available: no-op when the anchor imports, otherwise
    sync the venv with the extra enabled. Raises InstallError on failure
    — including when a platform gate excludes this machine."""
    if available(extra):
        return
    if not extra_supported(extra):
        from pm.package import InstallError

        marker = _platform_gates().get(extra, "")
        raise InstallError(
            "venv",
            f"extra {extra!r} is not supported on this platform "
            f"(gate: {marker!r}); the adapter degrades without it",
        )
    from pm.ensure import sync_venv

    sync_venv([extra])


def ensure_and_bind(extra, importer, target_globals) -> bool:
    """ensure_import + rebind module-level names after a mid-process install.
    importer returns {name: value}; bound into target_globals on success."""
    try:
        ensure_import(extra)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("extra %r unavailable: %s", extra, exc)
        return False
    try:
        bindings = importer()
    except ImportError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "import after installing %r failed: %s", extra, exc
        )
        return False
    target_globals.update(bindings)
    return True


def missing(extra: str) -> tuple[str, ...]:
    return tuple(a for a in _anchors(extra) if not _importable(a))
