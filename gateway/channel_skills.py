"""Resolve static channel-to-skill bindings and profile inheritance.

This module intentionally contains no file I/O or skill loading. It turns the platform
configuration into an ordered list of skill names; the gateway owns loading the resulting
skills into a new session.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_MAX_PROFILE_DEPTH = 16


class ChannelSkillResolutionError(ValueError):
    """Base class for invalid channel-skill configuration."""


class ChannelSkillProfileCycleError(ChannelSkillResolutionError):
    """Raised when profile inheritance contains a cycle."""

    def __init__(self, chain: tuple[str, ...]):
        self.chain = chain
        super().__init__(f"Channel skill profile cycle detected: {' -> '.join(chain)}")


class ChannelSkillProfileDepthError(ChannelSkillResolutionError):
    """Raised when profile inheritance exceeds the configured limit."""

    def __init__(self, profile_name: str, max_depth: int):
        self.profile_name = profile_name
        self.max_depth = max_depth
        super().__init__(
            f"Channel skill profile inheritance exceeds maximum depth "
            f"{max_depth} at {profile_name!r}"
        )


class UnknownChannelSkillProfileError(ChannelSkillResolutionError):
    """Raised when a binding or parent references a missing profile."""

    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        super().__init__(f"Unknown channel skill profile: {profile_name}")


@dataclass(frozen=True)
class ChannelSkillProfile:
    """Normalized profile configuration."""

    name: str
    skills: tuple[str, ...] = ()
    base_skills: tuple[str, ...] = ()
    extends: str | None = None


@dataclass(frozen=True)
class ChannelSkillBinding:
    """Normalized channel binding, supporting both legacy and profile forms."""

    channel_id: str
    profile: str | None = None
    skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedChannelSkills:
    """Ordered result of resolving a channel and its optional parent."""

    skills: tuple[str, ...]
    profile_chain: tuple[str, ...] = ()
    matched_channel_id: str | None = None
    inherited_from_parent: bool = False


class ChannelSkillResolver:
    """Resolve channel bindings into ordered, deduplicated skill names."""

    def __init__(
        self,
        config_extra: Mapping[str, Any] | None,
        *,
        max_profile_depth: int = DEFAULT_MAX_PROFILE_DEPTH,
    ) -> None:
        self._config_extra = config_extra if isinstance(config_extra, Mapping) else {}
        self.max_profile_depth = max(1, int(max_profile_depth))
        self._profiles = self._parse_profiles(self._config_extra.get("channel_skill_profiles"))
        self._bindings = self._parse_bindings(self._config_extra.get("channel_skill_bindings"))

    @property
    def profiles(self) -> Mapping[str, ChannelSkillProfile]:
        """Normalized profiles, primarily useful for diagnostics and tests."""
        return self._profiles

    @property
    def bindings(self) -> tuple[ChannelSkillBinding, ...]:
        """Normalized bindings, primarily useful for diagnostics and tests."""
        return self._bindings

    def resolve(
        self,
        channel_id: str | None,
        parent_id: str | None = None,
    ) -> ResolvedChannelSkills | None:
        """Resolve a channel, preferring its exact binding over its parent binding.

        Invalid profile references are fail-open for the gateway: the caller receives
        ``None`` and a warning is logged, so one malformed channel cannot stop a
        platform adapter from serving other channels.
        """
        binding, matched_id = self._find_binding(channel_id, parent_id)
        if binding is None:
            return None

        try:
            if binding.profile:
                profile_chain = self.resolve_profile_chain(binding.profile)
                skills = self._skills_for_profile_chain(profile_chain)
            else:
                profile_chain = ()
                skills = list(binding.skills)
        except ChannelSkillResolutionError as exc:
            logger.warning(
                "[Gateway] Failed to resolve channel skills for channel %s: %s",
                matched_id,
                exc,
            )
            return None

        skills = _deduplicate(skills)
        if not skills:
            return None
        return ResolvedChannelSkills(
            skills=tuple(skills),
            profile_chain=profile_chain,
            matched_channel_id=matched_id,
            inherited_from_parent=bool(parent_id and matched_id == str(parent_id)),
        )

    def resolve_skills(
        self,
        channel_id: str | None,
        parent_id: str | None = None,
    ) -> list[str] | None:
        """Compatibility API returning the ordered skill names only."""
        result = self.resolve(channel_id, parent_id)
        return list(result.skills) if result else None

    def resolve_profile_skills(self, profile_name: str) -> list[str]:
        """Resolve a named profile without requiring a channel binding."""
        chain = self.resolve_profile_chain(profile_name)
        return _deduplicate(self._skills_for_profile_chain(chain))

    def resolve_profile_chain(self, profile_name: str) -> tuple[str, ...]:
        """Return profile names in parent-to-child order."""
        normalized_name = _normalize_name(profile_name)
        if not normalized_name:
            raise UnknownChannelSkillProfileError(str(profile_name))
        return self._resolve_profile_chain(normalized_name, (), ())

    def _resolve_profile_chain(
        self,
        profile_name: str,
        visiting: tuple[str, ...],
        visited: tuple[str, ...],
    ) -> tuple[str, ...]:
        if profile_name in visiting:
            cycle_start = visiting.index(profile_name)
            raise ChannelSkillProfileCycleError(
                (*visiting[cycle_start:], profile_name)
            )
        if profile_name in visited:
            return ()
        if len(visiting) >= self.max_profile_depth:
            raise ChannelSkillProfileDepthError(profile_name, self.max_profile_depth)

        profile = self._profiles.get(profile_name)
        if profile is None:
            raise UnknownChannelSkillProfileError(profile_name)

        parent_chain: tuple[str, ...] = ()
        if profile.extends:
            parent_chain = self._resolve_profile_chain(
                profile.extends,
                (*visiting, profile_name),
                (*visited, profile_name),
            )
        return (*parent_chain, profile_name)

    def _skills_for_profile_chain(self, profile_chain: tuple[str, ...]) -> list[str]:
        skills: list[str] = []
        for profile_name in profile_chain:
            profile = self._profiles[profile_name]
            skills.extend(profile.base_skills)
            skills.extend(profile.skills)
        return skills

    def _find_binding(
        self,
        channel_id: str | None,
        parent_id: str | None,
    ) -> tuple[ChannelSkillBinding | None, str | None]:
        candidates = [str(channel_id)] if channel_id else []
        if parent_id:
            candidates.append(str(parent_id))

        for candidate in candidates:
            for binding in self._bindings:
                if binding.channel_id == candidate:
                    return binding, candidate
        return None, None

    @staticmethod
    def _parse_profiles(value: Any) -> dict[str, ChannelSkillProfile]:
        if not isinstance(value, Mapping):
            return {}

        profiles: dict[str, ChannelSkillProfile] = {}
        for raw_name, raw_profile in value.items():
            name = _normalize_name(raw_name)
            if not name or not isinstance(raw_profile, Mapping):
                continue
            profiles[name] = ChannelSkillProfile(
                name=name,
                skills=_as_names(raw_profile.get("skills", raw_profile.get("skill"))),
                base_skills=_as_names(
                    raw_profile.get("base_skills", raw_profile.get("base_skill"))
                ),
                extends=_normalize_name(raw_profile.get("extends")),
            )
        return profiles

    @staticmethod
    def _parse_bindings(value: Any) -> tuple[ChannelSkillBinding, ...]:
        if not isinstance(value, list):
            return ()

        bindings: list[ChannelSkillBinding] = []
        for raw_binding in value:
            if not isinstance(raw_binding, Mapping):
                continue
            raw_id = raw_binding.get("id")
            if raw_id is None:
                continue
            channel_id = str(raw_id)
            profile = _normalize_name(raw_binding.get("profile"))
            skills = _as_names(raw_binding.get("skills", raw_binding.get("skill")))
            if profile:
                if skills:
                    logger.warning(
                        "[Gateway] Channel skill binding %s defines both profile and skills; "
                        "using profile %s",
                        channel_id,
                        profile,
                    )
                skills = ()
            bindings.append(
                ChannelSkillBinding(
                    channel_id=channel_id,
                    profile=profile,
                    skills=skills,
                )
            )
        return tuple(bindings)


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _as_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        name = _normalize_name(value)
        return (name,) if name else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(name for item in value if (name := _normalize_name(item)))


def _deduplicate(names: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(name for name in names if name))
