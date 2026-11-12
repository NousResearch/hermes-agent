"""Method factories for the Slack adapter's ``config.extra``-else-env readers.

Re-exported from ``plugins.platforms.slack.adapter`` (callers and tests import them there).
"""
from typing import Callable


def _extra_or_env_flag_getter(key: str, env_var: str, *, strip: bool = False) -> Callable[..., bool]:
    """Method factory: ``self._extra_or_env_flag(key, env_var, strip=strip)``."""

    def getter(self) -> bool:
        return self._extra_or_env_flag(key, env_var, strip=strip)

    getter.__name__ = f"_slack_{key}"
    return getter


def _extra_or_env_channel_set_getter(
    key: str, env_var: str, *, coerce_scalar: bool = False) -> Callable[..., set]:
    """Method factory: ``self._extra_or_env_channel_set(key, env_var, coerce_scalar=...)``."""

    def getter(self) -> set:
        return self._extra_or_env_channel_set(key, env_var, coerce_scalar=coerce_scalar)

    getter.__name__ = f"_slack_{key}"
    return getter
