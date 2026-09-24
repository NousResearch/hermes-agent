"""Typed field extraction for spec documents.

Hand-written rather than delegating to a validation library, for three reasons: the
platform layer stays importable with only PyYAML (see ``nova/__init__.py``); the error
messages name the customer's own field paths; and unknown keys are rejected, so a typo
in a customer bundle fails loudly at load instead of being silently ignored and
producing an agent that is subtly not what was asked for.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from nova.errors import SpecError

# ``${VAR}`` or ``${VAR:-default}``. Deliberately not ``$VAR``: a bare dollar is common
# in prose fields (prompts, descriptions) and silently eating it would be surprising.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# An identifier usable as a directory name, a runtime profile name, and a URL segment.
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")


def join(prefix: str, key: str) -> str:
    """Dotted field path for error messages."""
    return f"{prefix}.{key}" if prefix else key


def interpolate_env(
    value: str, *, field: str, source: Optional[Path], env: Optional[Mapping[str, str]] = None
) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` against the environment.

    A missing variable with no default is an error rather than an empty string: a
    silently blank model name or API base produces a confusing runtime failure much
    later, in a different process, usually in front of a customer.
    """
    environ = os.environ if env is None else env

    def _sub(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        got = environ.get(name)
        if got is not None and got != "":
            return got
        if default is not None:
            return default
        raise SpecError(
            f"environment variable {name!r} is referenced but not set, and has no "
            f"default (write ${{{name}:-fallback}} to allow one)",
            field=field,
            source=source,
        )

    return _ENV_PATTERN.sub(_sub, value)


class Doc:
    """A mapping being read as a spec document, with path-aware accessors.

    Every getter records the key it consumed; :meth:`reject_unknown` then fails on
    anything left over.
    """

    def __init__(
        self,
        data: Any,
        *,
        source: Optional[Path] = None,
        prefix: str = "",
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise SpecError(
                f"expected a mapping, got {type(data).__name__}", field=prefix or None, source=source
            )
        self._data: Mapping[str, Any] = data
        self._source = source
        self._prefix = prefix
        self._env = env
        self._seen: set[str] = set()

    @property
    def source(self) -> Optional[Path]:
        return self._source

    def _fail(self, key: str, message: str) -> SpecError:
        return SpecError(message, field=join(self._prefix, key), source=self._source)

    def _raw(self, key: str, default: Any = None) -> Any:
        self._seen.add(key)
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        """Declared keys, in document order. Consuming them stays the caller's job."""
        return list(self._data)

    def str_(
        self,
        key: str,
        *,
        required: bool = False,
        default: str = "",
        expand: bool = True,
        allow_empty: bool = False,
    ) -> str:
        value = self._raw(key)
        if value is None:
            if required:
                raise self._fail(key, "is required")
            return default
        if not isinstance(value, str):
            raise self._fail(key, f"must be a string, got {type(value).__name__}")
        if expand:
            value = interpolate_env(
                value, field=join(self._prefix, key), source=self._source, env=self._env
            )
        value = value.strip()
        if not value and not allow_empty:
            if required:
                raise self._fail(key, "is required and must not be empty")
            return default
        return value

    def identifier(self, key: str, *, required: bool = True) -> str:
        """A lowercase, hyphen-separated id safe as a directory and profile name."""
        value = self.str_(key, required=required)
        if not value:
            return value
        if not _ID_PATTERN.match(value):
            raise self._fail(
                key,
                f"{value!r} is not a valid id — use lowercase letters, digits and hyphens, "
                "starting and ending alphanumeric (max 64 characters)",
            )
        return value

    def bool_(self, key: str, *, default: bool = False) -> bool:
        value = self._raw(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise self._fail(key, f"must be true or false, got {type(value).__name__}")
        return value

    def int_(
        self,
        key: str,
        *,
        default: Optional[int] = None,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> Optional[int]:
        value = self._raw(key)
        if value is None:
            return default
        # bool is an int subclass; a stray `true` here is a mistake, not a 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._fail(key, f"must be a whole number, got {type(value).__name__}")
        if minimum is not None and value < minimum:
            raise self._fail(key, f"must be at least {minimum}, got {value}")
        if maximum is not None and value > maximum:
            raise self._fail(key, f"must be at most {maximum}, got {value}")
        return value

    def money(self, key: str, *, default: Optional[float] = None) -> Optional[float]:
        """A non-negative amount of money, e.g. a budget in USD. Whole or decimal."""
        value = self._raw(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._fail(key, f"must be an amount such as 50 or 12.5, got {type(value).__name__}")
        if value < 0 or value != value:  # negative, or NaN
            raise self._fail(key, f"must be zero or more, got {value}")
        return float(value)

    def str_list(
        self,
        key: str,
        *,
        default: Optional[Sequence[str]] = None,
        expand: bool = True,
        unique: bool = True,
    ) -> list[str]:
        value = self._raw(key)
        if value is None:
            return list(default or ())
        if isinstance(value, str) or not isinstance(value, Iterable):
            raise self._fail(key, "must be a list of strings")
        out: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise SpecError(
                    f"must be a string, got {type(item).__name__}",
                    field=f"{join(self._prefix, key)}[{index}]",
                    source=self._source,
                )
            text = item
            if expand:
                text = interpolate_env(
                    text,
                    field=f"{join(self._prefix, key)}[{index}]",
                    source=self._source,
                    env=self._env,
                )
            text = text.strip()
            if not text:
                raise SpecError(
                    "must not be empty", field=f"{join(self._prefix, key)}[{index}]", source=self._source
                )
            out.append(text)
        if unique:
            seen: set[str] = set()
            for item in out:
                if item in seen:
                    raise self._fail(key, f"contains {item!r} more than once")
                seen.add(item)
        return out

    def child(self, key: str, *, required: bool = False) -> Optional["Doc"]:
        """A nested mapping as its own :class:`Doc`, or None when absent."""
        value = self._raw(key)
        if value is None:
            if required:
                raise self._fail(key, "is required")
            return None
        return Doc(
            value, source=self._source, prefix=join(self._prefix, key), env=self._env
        )

    def choice(self, key: str, allowed: Sequence[str], *, default: Optional[str] = None) -> Optional[str]:
        value = self.str_(key)
        if not value:
            return default
        if value not in allowed:
            raise self._fail(key, f"must be one of {', '.join(sorted(allowed))}, got {value!r}")
        return value

    def reject_unknown(self) -> None:
        """Fail on any key no getter consumed.

        A customer bundle with ``instuctions:`` should not quietly produce an agent
        with no instructions.
        """
        unknown = sorted(set(self._data) - self._seen)
        if unknown:
            where = self._prefix or "(document root)"
            raise SpecError(
                f"unknown field(s): {', '.join(unknown)}",
                field=where if self._prefix else None,
                source=self._source,
            )
