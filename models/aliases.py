"""Pure model-alias semantics over caller-supplied candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from providers import is_aggregator, normalize_provider


@dataclass(frozen=True, slots=True)
class ModelAliasPattern:
    """Vendor/family pattern used to resolve a short model alias."""

    vendor: str
    family: str


MODEL_ALIASES: dict[str, ModelAliasPattern] = {
    "sonnet": ModelAliasPattern("anthropic", "claude-sonnet"),
    "opus": ModelAliasPattern("anthropic", "claude-opus"),
    "haiku": ModelAliasPattern("anthropic", "claude-haiku"),
    "claude": ModelAliasPattern("anthropic", "claude"),
    "gpt5": ModelAliasPattern("openai", "gpt-5"),
    "gpt": ModelAliasPattern("openai", "gpt"),
    "codex": ModelAliasPattern("openai", "codex"),
    "o3": ModelAliasPattern("openai", "o3"),
    "o4": ModelAliasPattern("openai", "o4"),
    "gemini": ModelAliasPattern("google", "gemini"),
    "deepseek": ModelAliasPattern("deepseek", "deepseek-chat"),
    "grok": ModelAliasPattern("x-ai", "grok"),
    "llama": ModelAliasPattern("meta-llama", "llama"),
    "qwen": ModelAliasPattern("qwen", "qwen"),
    "minimax": ModelAliasPattern("minimax", "minimax"),
    "nemotron": ModelAliasPattern("nvidia", "nemotron"),
    "kimi": ModelAliasPattern("moonshotai", "kimi"),
    "glm": ModelAliasPattern("z-ai", "glm"),
    "step": ModelAliasPattern("stepfun", "step"),
    "mimo": ModelAliasPattern("xiaomi", "mimo"),
    "trinity": ModelAliasPattern("arcee-ai", "trinity"),
}


class AmbiguousModelAliasError(ValueError):
    """Raised when identity matching leaves more than one valid candidate."""

    def __init__(self, alias: str, provider: str, candidates: Iterable[str]) -> None:
        self.alias = str(alias or "").strip().lower()
        self.provider = normalize_provider(provider)
        self.candidates = tuple(candidates)
        super().__init__(
            f"{self.alias!r} matches {len(self.candidates)} models on "
            f"{self.provider}: {', '.join(self.candidates)}"
        )


def _split_version_suffix(rest: str) -> tuple[list[float], str]:
    nums: list[float] = []
    run, pos = "", 0

    def flush() -> None:
        nonlocal run
        try:
            nums.append(float(run.rstrip(".")))
        except ValueError:
            pass
        run = ""

    while pos < len(rest):
        ch = rest[pos]
        if ch in "-_.":
            pos += 1
            continue
        if not (ch in "vV" or ch.isdigit()):
            break
        if ch in "vV":
            pos += 1
        while pos < len(rest) and (rest[pos].isdigit() or rest[pos] == "."):
            if rest[pos] == "." and "." in run:
                flush()
            else:
                run += rest[pos]
            pos += 1
        flush()
        if pos < len(rest) and rest[pos] not in "-_":
            break
    return nums, rest[pos:]


def model_alias_sort_key(model_id: str, prefix: str) -> tuple:
    """Stable best-guess ordering for ambiguity display only."""

    rest = model_id[len(prefix):].removeprefix("/").lstrip("-").strip()
    nums, suffix_buf = _split_version_suffix(rest)
    suffix = suffix_buf.lower().strip("-_.").strip()
    version_key = tuple(-n for n in nums if n < 19_000_101)
    date_stamp = max((n for n in nums if n >= 19_000_101), default=0.0)
    date_key = (0.0, 0.0) if date_stamp == 0.0 else (1.0, -date_stamp)
    suffix_rank = 0 if suffix in (
        "pro", "max", "plus", "turbo", "sol", "astra"
    ) else 1
    return version_key + (suffix_rank, suffix) + date_key


def resolve_declared_model_id(
    typed: str,
    provider: str,
    candidates: Iterable[str],
    *,
    provider_aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve provider-declared aliases, exact IDs, or one unique ID prefix."""

    wanted = str(typed or "").strip().lower()
    declared_aliases = {
        str(key).strip().lower(): str(value).strip()
        for key, value in (provider_aliases or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if wanted in declared_aliases:
        return declared_aliases[wanted]

    available = [str(value).strip() for value in candidates if str(value).strip()]
    exact = next((model for model in available if model.lower() == wanted), None)
    if exact is not None:
        return exact
    matches = [model for model in available if model.lower().startswith(wanted)]
    if len(matches) > 1:
        raise AmbiguousModelAliasError(wanted, provider, matches)
    return matches[0] if matches else None


def resolve_model_alias(
    alias: str,
    provider: str,
    candidates: Iterable[str],
    aliases: Mapping[str, ModelAliasPattern] = MODEL_ALIASES,
    *,
    provider_aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve one semantic alias against caller-supplied candidates."""

    key = str(alias or "").strip().lower()
    declared = {
        str(name).strip().lower(): str(model).strip()
        for name, model in (provider_aliases or {}).items()
        if str(name).strip() and str(model).strip()
    }
    if key in declared:
        return declared[key]

    pattern = aliases.get(key)
    if pattern is None:
        return None
    prefix = pattern.family
    if is_aggregator(provider):
        prefix = f"{pattern.vendor}/{pattern.family}"

    matches = [
        candidate
        for candidate in (str(value or "").strip() for value in candidates)
        if candidate and candidate.lower().startswith(prefix.lower())
    ]
    matches.sort(key=lambda model: model_alias_sort_key(model, prefix))
    if len(matches) > 1:
        raise AmbiguousModelAliasError(key, provider, matches)
    return matches[0] if matches else None
