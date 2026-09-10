"""Canonical Groupchat settings, validation and one-time Matrix migration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _env_boolean(path: Path, name: str, default: bool) -> bool:
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                normalized = value.strip().strip("'\"").lower()
                return normalized not in {"false", "0", "no", "off"}
    except OSError:
        pass
    return default


def matrix_require_mention(profile_home: Path, config: dict[str, Any]) -> bool:
    """Return the saved effective Matrix mention requirement for a profile."""
    matrix = ((config.get("platforms") or {}).get("matrix") or {})
    configured = matrix.get("require_mention")
    if configured is None:
        configured = ((config.get("matrix") or {}).get("require_mention"))
    if configured is not None:
        if isinstance(configured, str):
            return configured.strip().lower() not in {"false", "0", "no", "off"}
        return bool(configured)
    return _env_boolean(profile_home / ".env", "MATRIX_REQUIRE_MENTION", True)


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["mistral", "openrouter", "openai-codex"] = "mistral"
    model: str = "mistral-small-latest"


class FilterModel(ModelChoice):
    fallbacks: list[ModelChoice] = Field(default_factory=lambda: [
        ModelChoice(provider="openrouter", model="mistralai/mistral-small-3.2-24b-instruct"),
        ModelChoice(provider="openrouter", model="mistralai/mistral-small-24b-instruct-2506"),
        ModelChoice(provider="openai-codex", model=""),
    ], max_length=5)


class RelevanceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    score_delays: dict[int, int] = Field(default_factory=lambda: {5: 0, 4: 3, 3: 10, 2: 30})
    max_context_messages: int = Field(5, ge=1, le=100)
    context_cooldown_seconds: float = Field(3600, ge=0, le=86400)
    typing_delay: float = Field(10, ge=0, le=300)
    info_delay: float = Field(300, ge=0, le=86400)
    system_patterns: list[str] | None = None
    multiline_patterns: list[str] | None = None
    literal_phrases: list[str] | None = None
    context_file: str = "RELEVANCE_CONTEXT.xml"
    transcript_entries: int = Field(15, ge=1, le=200)
    voice_delay_min: float = Field(0, ge=0, le=60)
    voice_delay_max: float = Field(5, ge=0, le=60)
    eval_interval: float = Field(5, ge=0, le=300)
    burst_window: float = Field(10, gt=0, le=300)
    burst_threshold: int = Field(10, ge=1, le=1000)


class PingpongSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    min_chars: int = Field(60, ge=1, le=4000)
    silence_patterns: list[str] | None = None


class GroupchatSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    platforms: list[str] = Field(default_factory=lambda: available_platforms())
    filter_model: FilterModel = Field(default_factory=FilterModel)
    relevance: RelevanceSettings = Field(default_factory=RelevanceSettings)
    pingpong_guard: PingpongSettings = Field(default_factory=PingpongSettings)


def available_platforms():
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    registered = {entry.name for entry in platform_registry.all_entries()}
    return sorted(({platform.value for platform in Platform} | Platform._scan_bundled_plugin_platforms() | registered) - {"local"})


def validate_settings(raw):
    raw = deepcopy(raw or {})
    relevance = raw.get('relevance') or {}
    if 'interrupt_notice' in relevance:
        previous = relevance.pop('interrupt_notice')
        relevance.setdefault('literal_phrases', None if previous is None else ([previous] if previous else []))
    settings = GroupchatSettings.model_validate(raw or {})
    from gateway.config import Platform
    for platform in settings.platforms:
        if not platform or platform == "local" or Platform(platform).value != platform:
            raise ValueError("Invalid Groupchat platform")
    from .relevance import DEFAULT_SYSTEM_PATTERNS, _SYSTEM_EMOJIS, _INTERRUPT_NOTICE
    from .pingpong_guard import SILENCE_PATTERNS
    if settings.relevance.system_patterns is None:
        settings.relevance.system_patterns = list(DEFAULT_SYSTEM_PATTERNS)
    if settings.relevance.multiline_patterns is None:
        settings.relevance.multiline_patterns = [r'^\s*' + _SYSTEM_EMOJIS + '(?:\uFE0F)?\\s']
    if settings.relevance.literal_phrases is None:
        settings.relevance.literal_phrases = [_INTERRUPT_NOTICE]
    if settings.pingpong_guard.silence_patterns is None:
        settings.pingpong_guard.silence_patterns = list(SILENCE_PATTERNS)
    delays = settings.relevance.score_delays
    # Before score 1 became passive context it had a delivery delay. Accept
    # and discard that legacy key so existing configurations migrate in place.
    if set(delays) == {1, 2, 3, 4, 5}:
        delays.pop(1)
    if set(delays) != {2, 3, 4, 5} or any(v < 0 or v > 86400 for v in delays.values()):
        raise ValueError("score_delays must map scores 2–5 to delays between 0 and 86400 seconds")
    if settings.relevance.voice_delay_max < settings.relevance.voice_delay_min:
        raise ValueError("voice_delay_max must not be less than voice_delay_min")
    import re
    for label, patterns in [('system', settings.relevance.system_patterns),
                            ('multiline', settings.relevance.multiline_patterns),
                            ('silence', settings.pingpong_guard.silence_patterns),
                            ('literal', settings.relevance.literal_phrases)]:
        if len(patterns) > 100:
            raise ValueError(f'Too many {label} patterns (maximum 100)')
        for index, pattern in enumerate(patterns, 1):
            if not pattern.strip() or len(pattern) > 2000 or '\n' in pattern or '\r' in pattern:
                raise ValueError(f'Invalid {label} pattern at line {index}: use one non-empty regex per line, maximum 2000 characters')
            if pattern.lstrip().startswith('#') or label == 'literal':
                continue
            try:
                re.compile(pattern, re.IGNORECASE if label == 'silence' else 0)
            except re.error as exc:
                raise ValueError(f'Invalid {label} pattern at line {index}: {exc.msg}') from exc
    for choice in [settings.filter_model, *settings.filter_model.fallbacks]:
        if not choice.model.strip() and choice.provider != "openai-codex":
            raise ValueError("A model ID is required (only Codex may use the profile model)")
    return settings.model_dump()


def migrate_legacy(config, env):
    """Return a new config; never read/write files or mutate the input.

    Run once before deployment. Existing Groupchat values always win. Secrets
    are never copied; only the explicit behavioral-variable allowlist is read.
    """
    config = deepcopy(config)
    if "groupchat" in config:
        return config
    from .relevance import _parse_score_delays
    from gateway.config import _coerce_bool
    settings = GroupchatSettings(enabled=True, platforms=["matrix"]).model_dump()
    relevance = settings["relevance"]
    matrix = config.get("matrix", {}) or {}
    relevance["enabled"] = _coerce_bool(env.get("MATRIX_INTELLIGENT_REACTION", matrix.get("intelligent_reaction")), False)
    prefix = "MATRIX_INTELLIGENT_REACTION_"
    for key in relevance:
        value = env.get(prefix + key.upper())
        if value is None or key == "enabled":
            continue
        if key == "score_delays":
            value = _parse_score_delays(value)
        elif key == "system_patterns":
            value = [part for part in value.split(",") if part]
        relevance[key] = value
    primary = settings["filter_model"]
    if env.get(prefix + "MODEL"):
        primary["fallbacks"][0]["model"] = env[prefix + "MODEL"]
    if env.get(prefix + "BACKUP_MODEL"):
        primary["fallbacks"][1]["model"] = env[prefix + "BACKUP_MODEL"]
    if env.get(prefix + "CODEX_MODEL"):
        primary["fallbacks"][2]["model"] = env[prefix + "CODEX_MODEL"]
    if env.get("PINGPONG_GUARD_MIN_CHARS"):
        settings["pingpong_guard"]["min_chars"] = env["PINGPONG_GUARD_MIN_CHARS"]
    config["groupchat"] = validate_settings(settings)
    plugins = config.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])
    if "groupchat" not in enabled:
        enabled.append("groupchat")
    # The obsolete non-secret YAML activation flag is no longer a control.
    if isinstance(config.get("matrix"), dict):
        config["matrix"].pop("intelligent_reaction", None)
    return config
