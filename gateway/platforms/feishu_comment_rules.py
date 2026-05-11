"""
Feishu document comment access-control rules.

3-tier rule resolution: exact doc > wildcard "*" > top-level > code defaults.
Each field (enabled/policy/allow_from) falls back independently.
Config: ~/.hermes/feishu_comment_rules.json (mtime-cached, hot-reload).
Pairing store: ~/.hermes/feishu_comment_pairing.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# Uses the canonical ``get_hermes_home()`` helper (HERMES_HOME-aware and
# profile-safe). Resolved at import time; this module is lazy-imported by
# the Feishu comment event handler, which runs long after profile overrides
# have been applied, so freezing paths here is safe.

RULES_FILE = get_hermes_home() / "feishu_comment_rules.json"
PAIRING_FILE = get_hermes_home() / "feishu_comment_pairing.json"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_VALID_POLICIES = ("allowlist", "pairing")


@dataclass(frozen=True)
class CommentDocumentRule:
    """Per-document rule.  ``None`` means 'inherit from lower tier'."""
    enabled: Optional[bool] = None
    policy: Optional[str] = None
    allow_from: Optional[frozenset] = None


@dataclass(frozen=True)
class CommentsConfig:
    """Top-level comment access config."""
    enabled: bool = True
    policy: str = "pairing"
    allow_from: frozenset = field(default_factory=frozenset)
    documents: Dict[str, CommentDocumentRule] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedCommentRule:
    """Fully resolved rule after field-by-field fallback."""
    enabled: bool
    policy: str
    allow_from: frozenset
    match_source: str  # e.g. "exact:docx:xxx" | "wildcard" | "top" | "default"


# ---------------------------------------------------------------------------
# Mtime-cached file loading
# ---------------------------------------------------------------------------

class _MtimeCache:
    """Generic mtime-based file cache.  ``stat()`` per access, re-read only on change."""

    def __init__(self, path: Path):
        self._path = path
        self._mtime: float = 0.0
        self._data: Optional[dict] = None

    def load(self) -> dict:
        try:
            st = self._path.stat()
            mtime = st.st_mtime
        except FileNotFoundError:
            self._mtime = 0.0
            self._data = {}
            return {}

        if mtime == self._mtime and self._data is not None:
            return self._data

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, OSError):
            logger.warning("[Feishu-Rules] Failed to read %s, using empty config", self._path)
            data = {}

        self._mtime = mtime
        self._data = data
        return data


_rules_cache = _MtimeCache(RULES_FILE)
_pairing_cache = _MtimeCache(PAIRING_FILE)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_frozenset(raw: Any) -> Optional[frozenset]:
    """Parse a list of strings into a frozenset; return None if key absent."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return frozenset(str(u).strip() for u in raw if str(u).strip())
    return None


def _parse_document_rule(raw: dict) -> CommentDocumentRule:
    enabled = raw.get("enabled")
    if enabled is not None:
        enabled = bool(enabled)
    policy = raw.get("policy")
    if policy is not None:
        policy = str(policy).strip().lower()
        if policy not in _VALID_POLICIES:
            policy = None
    allow_from = _parse_frozenset(raw.get("allow_from"))
    return CommentDocumentRule(enabled=enabled, policy=policy, allow_from=allow_from)


def load_config() -> CommentsConfig:
    """Load comment rules from disk (mtime-cached)."""
    raw = _rules_cache.load()
    if not raw:
        return CommentsConfig()

    documents: Dict[str, CommentDocumentRule] = {}
    raw_docs = raw.get("documents", {})
    if isinstance(raw_docs, dict):
        for key, rule_raw in raw_docs.items():
            if isinstance(rule_raw, dict):
                documents[str(key)] = _parse_document_rule(rule_raw)

    policy = str(raw.get("policy", "pairing")).strip().lower()
    if policy not in _VALID_POLICIES:
        policy = "pairing"

    return CommentsConfig(
        enabled=raw.get("enabled", True),
        policy=policy,
        allow_from=_parse_frozenset(raw.get("allow_from")) or frozenset(),
        documents=documents,
    )


# ---------------------------------------------------------------------------
# Rule resolution  (§8.4 field-by-field fallback)
# ---------------------------------------------------------------------------

def has_wiki_keys(cfg: CommentsConfig) -> bool:
    """Check if any document rule key starts with 'wiki:'."""
    return any(k.startswith("wiki:") for k in cfg.documents)


def resolve_rule(
    cfg: CommentsConfig,
    file_type: str,
    file_token: str,
    wiki_token: str = "",
) -> ResolvedCommentRule:
    """Resolve effective rule: exact doc → wiki key → wildcard → top-level → defaults."""
    exact_key = f"{file_type}:{file_token}"

    exact = cfg.documents.get(exact_key)
    exact_src = f"exact:{exact_key}"
    if exact is None and wiki_token:
        wiki_key = f"wiki:{wiki_token}"
        exact = cfg.documents.get(wiki_key)
        exact_src = f"exact:{wiki_key}"

    wildcard = cfg.documents.get("*")

    layers = []
    if exact is not None:
        layers.append((exact, exact_src))
    if wildcard is not None:
        layers.append((wildcard, "wildcard"))

    def _pick(field_name: str):
        for layer, source in layers:
            val = getattr(layer, field_name)
            if val is not None:
                return val, source
        return getattr(cfg, field_name), "top"

    enabled, en_src = _pick("enabled")
    policy, pol_src = _pick("policy")
    allow_from, _ = _pick("allow_from")

    # match_source = highest-priority tier that contributed any field
    priority_order = {"exact": 0, "wildcard": 1, "top": 2}
    best_src = min(
        [en_src, pol_src],
        key=lambda s: priority_order.get(s.split(":")[0], 3),
    )

    return ResolvedCommentRule(
        enabled=enabled,
        policy=policy,
        allow_from=allow_from,
        match_source=best_src,
    )


# ---------------------------------------------------------------------------
# Pairing store
# ---------------------------------------------------------------------------

def _load_pairing_approved() -> set:
    """Return set of approved user open_ids (mtime-cached)."""
    data = _pairing_cache.load()
    approved = data.get("approved", {})
    if isinstance(approved, dict):
        return set(approved.keys())
    if isinstance(approved, list):
        return set(str(u) for u in approved if u)
    return set()


def _save_pairing(data: dict) -> None:
    PAIRING_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PAIRING_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(PAIRING_FILE)
    # Invalidate cache so next load picks up change
    _pairing_cache._mtime = 0.0
    _pairing_cache._data = None


def pairing_add(user_open_id: str) -> bool:
    """Add a user to the pairing-approved list. Returns True if newly added."""
    data = _pairing_cache.load()
    approved = data.get("approved", {})
    if not isinstance(approved, dict):
        approved = {}
    if user_open_id in approved:
        return False
    approved[user_open_id] = {"approved_at": time.time()}
    data["approved"] = approved
    _save_pairing(data)
    return True


def pairing_remove(user_open_id: str) -> bool:
    """Remove a user from the pairing-approved list. Returns True if removed."""
    data = _pairing_cache.load()
    approved = data.get("approved", {})
    if not isinstance(approved, dict):
        return False
    if user_open_id not in approved:
        return False
    del approved[user_open_id]
    data["approved"] = approved
    _save_pairing(data)
    return True


def pairing_list() -> Dict[str, Any]:
    """Return the approved dict  {user_open_id: {approved_at: ...}}."""
    data = _pairing_cache.load()
    approved = data.get("approved", {})
    return dict(approved) if isinstance(approved, dict) else {}


# ---------------------------------------------------------------------------
# Access check  (public API for feishu_comment.py)
# ---------------------------------------------------------------------------

def is_user_allowed(rule: ResolvedCommentRule, user_open_id: str) -> bool:
    """Check if user passes the resolved rule's policy gate."""
    if user_open_id in rule.allow_from:
        return True
    if rule.policy == "pairing":
        return user_open_id in _load_pairing_approved()
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_status() -> None:
    cfg = load_config()
    print(f"Rules file: {RULES_FILE}")
    print(f"  exists: {RULES_FILE.exists()}")
    print(f"Pairing file: {PAIRING_FILE}")
    print(f"  exists: {PAIRING_FILE.exists()}")
    print()
    print(f"Top-level:")
    print(f"  enabled:    {cfg.enabled}")
    print(f"  policy:     {cfg.policy}")
    print(f"  allow_from: {sorted(cfg.allow_from) if cfg.allow_from else '[]'}")
    print()
    if cfg.documents:
        print(f"Document rules ({len(cfg.documents)}):")
        for key, rule in sorted(cfg.documents.items()):
            parts = []
            if rule.enabled is not None:
                parts.append(f"enabled={rule.enabled}")
            if rule.policy is not None:
                parts.append(f"policy={rule.policy}")
            if rule.allow_from is not None:
                parts.append(f"allow_from={sorted(rule.allow_from)}")
            print(f"  [{key}] {', '.join(parts) if parts else '(empty — inherits all)'}")
    else:
        print("Document rules: (none)")
    print()
    approved = pairing_list()
    print(f"Pairing approved ({len(approved)}):")
    for uid, meta in sorted(approved.items()):
        ts = meta.get("approved_at", 0)
        print(f"  {uid}  (approved_at={ts})")


def _do_check(doc_key: str, user_open_id: str) -> None:
    cfg = load_config()
    parts = doc_key.split(":", 1)
    if len(parts) != 2:
        print(f"Error: doc_key must be 'fileType:fileToken', got '{doc_key}'")
        return
    file_type, file_token = parts
    rule = resolve_rule(cfg, file_type, file_token)
    allowed = is_user_allowed(rule, user_open_id)
    print(f"Document:     {doc_key}")
    print(f"User:         {user_open_id}")
    print(f"Resolved rule:")
    print(f"  enabled:      {rule.enabled}")
    print(f"  policy:       {rule.policy}")
    print(f"  allow_from:   {sorted(rule.allow_from) if rule.allow_from else '[]'}")
    print(f"  match_source: {rule.match_source}")
    print(f"Result:       {'ALLOWED' if allowed else 'DENIED'}")


def _main() -> int:
    import sys

    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
    except Exception:
        pass

    usage = (
        "Usage: python -m gateway.platforms.feishu_comment_rules <command> [args]\n"
        "\n"
        "Commands:\n"
        "  status                              Show rules config and pairing state\n"
        "  check <fileType:token> <user>        Simulate access check\n"
        "  pairing add <user_open_id>           Add user to pairing-approved list\n"
        "  pairing remove <user_open_id>        Remove user from pairing-approved list\n"
        "  pairing list                         List pairing-approved users\n"
        "\n"
        f"Rules config file: {RULES_FILE}\n"
        "  Edit this JSON file directly to configure policies and document rules.\n"
        "  Changes take effect on the next comment event (no restart needed).\n"
    )

    args = sys.argv[1:]
    if not args:
        print(usage)
        return 1

    cmd = args[0]

    if cmd == "status":
        _print_status()

    elif cmd == "check":
        if len(args) < 3:
            print("Usage: check <fileType:fileToken> <user_open_id>")
            return 1
        _do_check(args[1], args[2])

    elif cmd == "pairing":
        if len(args) < 2:
            print("Usage: pairing <add|remove|list> [args]")
            return 1
        sub = args[1]
        if sub == "add":
            if len(args) < 3:
                print("Usage: pairing add <user_open_id>")
                return 1
            if pairing_add(args[2]):
                print(f"Added: {args[2]}")
            else:
                print(f"Already approved: {args[2]}")
        elif sub == "remove":
            if len(args) < 3:
                print("Usage: pairing remove <user_open_id>")
                return 1
            if pairing_remove(args[2]):
                print(f"Removed: {args[2]}")
            else:
                print(f"Not in approved list: {args[2]}")
        elif sub == "list":
            approved = pairing_list()
            if not approved:
                print("(no approved users)")
            for uid, meta in sorted(approved.items()):
                print(f"  {uid}  approved_at={meta.get('approved_at', '?')}")
        else:
            print(f"Unknown pairing subcommand: {sub}")
            return 1
    else:
        print(f"Unknown command: {cmd}\n")
        print(usage)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
