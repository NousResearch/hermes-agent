"""Feishu per-group admission rules hot-reload: ~/.hermes/feishu_group_rules.json.

Entries use the same shape as ``platforms.feishu.extra.group_rules`` in config.yaml
(policy / allowlist / blacklist / require_mention) and overlay the boot-time rules
per chat at admission time, so flipping a group's gating no longer needs a gateway
restart. Malformed entries are skipped with a warning (the last valid state keeps
serving), mirroring the feishu_comment_rules.json mtime-cached hot reload.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home
from plugins.platforms.feishu.feishu_comment_rules import _MtimeCache

logger = logging.getLogger(__name__)

# Resolved at import time: this module is lazy-imported by the adapter's admission
# path, long after profile/HERMES_HOME overrides have been applied, so freezing is safe.
GROUP_RULES_FILE = get_hermes_home() / "feishu_group_rules.json"

_group_rules_cache = _MtimeCache(GROUP_RULES_FILE)

_VALID_POLICIES = ("open", "allowlist", "blacklist", "admin_only", "disabled")


def _validate_entry(chat_id: str, raw: Any) -> Dict[str, Any]:
    """Normalize one rule entry; raises ValueError on a shape we refuse to guess."""
    if not isinstance(raw, dict):
        raise ValueError(f"entry for {chat_id!r} is not an object")
    entry: Dict[str, Any] = {}
    if "policy" in raw:
        policy = str(raw["policy"]).strip().lower()
        if policy not in _VALID_POLICIES:
            raise ValueError(f"entry for {chat_id!r} has unknown policy {policy!r}")
        entry["policy"] = policy
    for key in ("allowlist", "blacklist"):
        if key in raw:
            values = raw[key]
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"entry for {chat_id!r} has non-list {key}")
            entry[key] = {str(u).strip() for u in values if str(u).strip()}
    if "require_mention" in raw:
        value = raw["require_mention"]
        entry["require_mention"] = value is True or value == 1 or value == "true"
    return entry


def load_group_rules() -> Dict[str, Dict[str, Any]]:
    """Read hot-reload group rules (mtime-cached); malformed entries are skipped."""
    rules: Dict[str, Dict[str, Any]] = {}
    for chat_id, raw in _group_rules_cache.load().items():
        try:
            rules[str(chat_id)] = _validate_entry(chat_id, raw)
        except ValueError as exc:
            logger.warning("[Feishu-GroupRules] Skipping %s", exc)
    return rules


def _main() -> int:
    """Print the effective hot-reload rules (debugging aid for operators)."""
    rules = load_group_rules()
    if not rules:
        print(f"No hot-reload group rules loaded (file: {GROUP_RULES_FILE})")
        return 0
    for chat_id, entry in sorted(rules.items()):
        fields = ", ".join(
            f"{key}={sorted(value) if isinstance(value, set) else value!r}"
            for key, value in entry.items()
        )
        print(f"  {chat_id}: {fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
