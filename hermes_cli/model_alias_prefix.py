"""``+<alias> rest``: send one message on a configured model alias (``display.model_alias_prefix``).

The one parser the classic CLI and the messaging gateway share. A message whose first word is
``+<name>``, where ``<name>`` is a configured alias (``model_aliases:`` or ``model.aliases:``, the
same table ``/model <alias>`` resolves), runs as ``/model <name> --once`` followed by ``rest``.
Anything else that starts with ``+`` (``+1``, ``+unknownword``, a bare ``+fast`` with nothing to
send) is ordinary text. Off by default: the flag is read before any alias lookup.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli.model_switch import _load_direct_aliases
from utils import is_truthy_value

_PREFIX_RE = re.compile(r"\+(\S+)\s+(\S.*)", re.DOTALL)


@dataclass(frozen=True)
class AliasPrefix:
    """A matched ``+<alias> rest`` message."""

    alias: str
    rest: str

    @property
    def model_command(self) -> str:
        """The ``/model`` command the prefix stands for."""
        return f"/model {self.alias} --once"


def split_alias_prefix(
    text: Any, load_config: Callable[[], Mapping[str, Any]]
) -> Optional[AliasPrefix]:
    """The ``+<alias>`` prefix of *text*, or None when it is ordinary text.

    *load_config* returns the owning profile's config; it is called only when *text* has the
    prefix shape, so plain messages never read config.
    """
    match = _PREFIX_RE.fullmatch(text.strip()) if isinstance(text, str) else None
    if match is None:
        return None
    cfg = load_config() or {}
    display = cfg.get("display")
    if not isinstance(display, Mapping) or not is_truthy_value(display.get("model_alias_prefix")):
        return None
    alias = match.group(1).lower()
    if alias not in _load_direct_aliases(dict(cfg)):
        return None
    return AliasPrefix(alias=alias, rest=match.group(2).strip())
