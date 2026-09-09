"""``/access`` — manage this gateway's DM/group allowlists from chat.

Owner convenience command: ``/access allow user 62812…``, ``/access deny group "Kelas XI-C"``,
``/access list``.  Platform-agnostic by contract — the core never hardcodes an identity
format.  Adapters opt into richer input handling by exposing:

- ``resolve_access_ref(ref, scope, event)`` (async) → :class:`AccessResolution` — turn a
  human-supplied reference (phone in any local format, ``@Name``, group name, ``<@id>``
  mention) into the canonical id the platform's allowlist stores.  Absent → the core's
  generic fallbacks (event mentions, reply-to author, mention-wrapper stripping,
  pass-through).
- ``ACCESS_ALLOWLIST_ENV_KEYS`` — ``{"user": (...), "group": (...)}`` env vars that also
  carry the allowlist (env-gated adapters).  Absent → only config + live attrs are written.

Writes land in three layers so the change is effective immediately AND survives restart:
the live adapter's in-memory sets (WhatsApp-family gates read them per message), the
adapter's ``config.extra`` (the authz union path), and the routed profile's ``config.yaml``
(source of truth; the shared-key bridge re-seeds ``extra`` on the next load).  Env vars are
updated only for env-seeded adapters and never under multiplex, where ``os.environ`` holds
the default profile's values (#72348).

Authorization is delegated to the standard slash-access policy: the command is runnable by
``allow_admin_from`` / ``group_allow_admin_from`` members only (gateway/run_busy.py dispatch
+ gateway/slash_access.py).  Without an admin list the gating is disabled platform-wide —
operators must set one for ``/access`` to be owner-only.
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple



logger = logging.getLogger(__name__)

# Mention-wrapper syntaxes seen across platforms: Discord ``<@123>``, Slack ``<@U123>``,
# WhatsApp mention text ``@<digits>``.  Stripped by the generic fallback before
# pass-through so ``/access allow user <@123>`` stores ``123``.
_MENTION_WRAPPER_RE = re.compile(r"^<@!?(\w+)>$")
_MENTION_AT_RE = re.compile(r"^@([\w.\-]+)$")

_ACCESS_USAGE = (
    "Usage:\n"
    "/access list\n"
    "/access allow user <phone | @name | id>\n"
    "/access deny user <phone | @name | id>\n"
    "/access allow group <name | id>\n"
    "/access deny group <name | id>"
)


@dataclass(frozen=True)
class AccessResolution:
    """One reference resolved against a platform's identity space.

    ``canonical`` is the id to store; ``label`` is a human-readable name for the
    confirmation reply.  ``candidates`` (non-empty) means the reference matched several
    identities — the command replies asking the sender to disambiguate instead of
    guessing (a wrong guess silently stores a dead entry)."""

    canonical: str = ""
    label: str = ""
    candidates: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


class GatewayAccessCommandsMixin:
    """``/access`` handlers for GatewayRunner (see module docstring for the contract)."""

    async def _handle_access_command(self, event) -> str:
        from gateway.slash_access import policy_for_source

        # Fail closed: without an admin list the central slash gate lets
        # everyone through, which would let any chatter rewrite the
        # allowlists (including allowlisting themselves). Same policy object
        # the dispatch gate used, so the two can never disagree.
        if not policy_for_source(getattr(self, "config", None), event.source).enabled:
            return ("`/access` is disabled: no admin list is configured for this scope. "
                    "Set `allow_admin_from` (DMs) or `group_allow_admin_from` (groups) "
                    "in the platform block, then retry.")
        parts = (event.get_command_args() or "").strip().split()
        if not parts or parts[0].lower() not in ("list", "allow", "deny"):
            return _ACCESS_USAGE
        op = parts[0].lower()
        if op == "list" and len(parts) > 1:
            return _ACCESS_USAGE
        source = event.source
        adapter = self.adapters.get(source.platform) if source else None
        if adapter is None:
            return "This platform is not available right now."

        if op == "list":
            return self._access_render_list(adapter, source)

        if len(parts) < 3:
            return _ACCESS_USAGE
        scope = self._access_scope(parts[1])
        if scope is None:
            return _ACCESS_USAGE
        kind = "DM" if scope == "user" else "group"
        policy = self._access_effective_policy(adapter, scope)
        if policy in ("open", "pairing", "disabled"):
            return (f"Stored nothing: the {kind} policy here is *{policy}* — the allowlist "
                    f"is not consulted, so allow/deny changes nothing. "
                    f"Switch the {kind} policy to allowlist for /access to take effect.")
        ref = " ".join(parts[2:]).strip().strip('"').strip("'")
        if not ref:
            return _ACCESS_USAGE

        resolution = await self._access_resolve(adapter, event, scope, ref)
        if resolution is None:
            return (f"Could not resolve {ref!r} on {source.platform.value}. "
                    f"Give a phone number (any local format), @name, or the platform id.")
        if resolution.candidates:
            lines = [f"{ref!r} matches several contacts — reply with one:"]
            lines += [f"• {cid} — {label}" for cid, label in resolution.candidates]
            return "\n".join(lines)
        canonical = resolution.canonical
        if not canonical:
            return f"Could not resolve {ref!r} on {source.platform.value}."
        if not self._access_would_match(adapter, scope, canonical):
            hint = ("Give a phone number (any local format), tap-mention the person, "
                    "or paste the platform id."
                    if scope == "user" else
                    "Paste the full group id (…@g.us) — names only work when the "
                    "group lookup finds them.")
            return (f"Could not resolve {ref!r} to a usable id on {source.platform.value}. "
                    f"{hint}")

        added = self._access_apply(adapter, source, scope, canonical, op)
        label = resolution.label or canonical
        verb = "added to" if op == "allow" else "removed from"
        state = "now allowed" if op == "allow" else "no longer allowed"
        if added:
            return f"✅ {label} ({canonical}) {verb} the {kind} allowlist — {state}."
        if op == "allow":
            return f"• {label} ({canonical}) was already on the {kind} allowlist."
        return f"• {label} ({canonical}) was not on the {kind} allowlist."

    # ------------------------------------------------------------------ parsing

    @staticmethod
    def _access_scope(word: str) -> Optional[str]:
        lowered = word.strip().lower()
        if lowered in ("user", "dm", "contact"):
            return "user"
        if lowered in ("group", "grup", "chat"):
            return "group"
        return None

    @staticmethod
    def _access_effective_policy(adapter, scope: str) -> str:
        """The allowlist policy actually enforced for *scope*, or "" when unknown.

        Mirrors the precedence `_access_render_list` shows: config.extra first,
        then the adapter's live attribute. Unknown means "old behavior" — the
        caller only blocks on policies known to ignore the allowlist.
        """
        extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
        if scope == "user":
            value = extra.get("dm_policy") or getattr(adapter, "_dm_policy", None)
        else:
            value = extra.get("group_policy") or getattr(adapter, "_group_policy", None)
        return str(value or "").strip().lower()

    def _access_would_match(self, adapter, scope: str, canonical: str) -> bool:
        """Would *canonical* ever match this adapter's gate? Guards dead entries.

        Scoped to WhatsApp-family adapters, whose gate shapes are fully known
        (bare digits fold to the same form the gate compares, group chats are
        ``@g.us``). Other platforms keep the old behavior — their resolvers own
        honesty there, and refusing unknown shapes could block legit ids.
        """
        if canonical == "*":
            return True
        if not self._access_is_whatsapp_family(adapter):
            return True
        if "@" in canonical:
            return bool(canonical.split("@", 1)[0])
        return canonical.isdigit() and len(canonical) >= 7

    # ------------------------------------------------------------------ resolution

    async def _access_resolve(self, adapter, event, scope: str, ref: str) -> Optional[AccessResolution]:
        """Adapter resolver first (richest identity knowledge), then generic fallbacks."""
        resolver = getattr(adapter, "resolve_access_ref", None)
        if callable(resolver):
            try:
                resolution = await resolver(ref, scope=scope, event=event)
            except Exception:
                logger.warning("[access] %s resolver failed for %r",
                               getattr(adapter, "name", adapter.__class__.__name__), ref, exc_info=True)
                resolution = None
            if resolution is not None:
                return resolution
        return self._access_generic_resolve(adapter, event, scope, ref)

    def _access_generic_resolve(self, adapter, event, scope: str, ref: str) -> Optional[AccessResolution]:
        """Platform-agnostic fallbacks: reply-to author, event mentions, mention-wrapper
        stripping, raw pass-through.  WhatsApp-family adapters get phone normalization
        through their own ``resolve_access_ref``; this path only handles ids that are
        already canonical or trivially wrappable — a bare phone-like number is NOT
        passed through (local formats must be normalized by a platform that knows the
        country; storing it raw would dead-end the allowlist)."""
        if scope == "user":
            if ref.lower() in ("reply", "^") and getattr(event, "reply_to_author_id", None):
                return AccessResolution(canonical=str(event.reply_to_author_id), label="reply target")
            metadata_mentions = self._access_event_mentions(event)
            if metadata_mentions:
                needle = ref.lstrip("@").lower()
                for mid, mlabel in metadata_mentions:
                    if ref.lstrip("@") == mid or (mlabel and needle == mlabel.lower()):
                        return AccessResolution(canonical=mid, label=mlabel or mid)
            stripped = self._access_strip_mention(ref)
            if stripped:
                return AccessResolution(canonical=stripped)
            if self._access_is_whatsapp_family(adapter) and re.fullmatch(r"\+?[\d\s().\-]{7,}", ref):
                # Phone-shaped input on a WhatsApp-family adapter: normalize against the
                # session's own country (shared mixin knowledge, not a guess).  Unresolvable
                # formats reply with usage help instead of storing a dead entry.
                phone_jid = adapter._access_phone_jid(ref)
                if not phone_jid:
                    return AccessResolution()
                return AccessResolution(canonical=phone_jid)
            return AccessResolution(canonical=ref)
        # group scope: mention wrappers carry channel/group ids on some platforms.
        stripped = self._access_strip_mention(ref)
        return AccessResolution(canonical=stripped or ref)

    @staticmethod
    def _access_event_mentions(event) -> list:
        """``[(id, label), …]`` from the event's mention metadata, when the adapter
        populated it.  Malformed entries are skipped, not trusted."""
        metadata = getattr(event, "metadata", None) or {}
        raw = metadata.get("mentions")
        if not isinstance(raw, list):
            return []
        out = []
        for entry in raw:
            if isinstance(entry, dict) and entry.get("id"):
                out.append((str(entry["id"]), str(entry.get("label") or "")))
            elif isinstance(entry, (str, int)):
                out.append((str(entry), ""))
        return out

    @staticmethod
    def _access_strip_mention(ref: str) -> str:
        match = _MENTION_WRAPPER_RE.match(ref.strip())
        if match:
            return match.group(1)
        at = _MENTION_AT_RE.match(ref.strip())
        if at:
            return at.group(1)
        return ""

    @staticmethod
    def _access_is_whatsapp_family(adapter) -> bool:
        """WhatsApp-family adapters (via ``WhatsAppBehaviorMixin._access_phone_jid``)
        store phone JIDs, so a bare digit string is a local phone format that MUST be
        normalized (never stored raw).  Other platforms (Telegram/Discord numeric ids)
        treat digit strings as canonical ids."""
        return callable(getattr(adapter, "_access_phone_jid", None))

    # ------------------------------------------------------------------ apply + persist

    def _access_current_ids(self, adapter, scope: str) -> list:
        """The live allowlist for *scope*: the adapter's in-memory set when it keeps one
        (WhatsApp-family gates read it per message), else ``config.extra``, else env."""
        attr = "_allow_from" if scope == "user" else "_group_allow_from"
        live = getattr(adapter, attr, None)
        if isinstance(live, (set, frozenset, list, tuple)):
            return [str(item) for item in live]
        extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
        key = "allow_from" if scope == "user" else "group_allow_from"
        value = extra.get(key)
        return [str(item) for item in value] if isinstance(value, (list, tuple, set)) else []

    def _access_apply(self, adapter, source, scope: str, canonical: str, op: str) -> bool:
        """Add/remove *canonical*, then persist.

        The whole read-compute-write runs under one cross-process lock, and
        the working set is the union of the live set and the file list — so
        two concurrent edits (threads or processes) merge instead of one
        silently clobbering the other. Persist-first: a failed write leaves
        memory untouched, so the reply never claims an effect that did not
        happen.
        """
        from gateway.run import _gateway_config_home
        from hermes_cli.config import atomic_config_write, read_user_config_raw

        key = "allow_from" if scope == "user" else "group_allow_from"
        config_path = _gateway_config_home() / "config.yaml"
        with self._access_config_lock(config_path):
            raw = read_user_config_raw(config_path)
            block = self._access_platform_block(raw, source.platform.value)
            filed = block.get(key)
            filed = [str(item) for item in filed] if isinstance(filed, (list, tuple, set)) else []
            live = self._access_current_ids(adapter, scope)
            current = live + [item for item in filed if item not in live]
            present = canonical in current or self._access_id_in_list(current, canonical)
            if op == "allow":
                if present:
                    return False
                updated = current + [canonical]
            else:
                if not present:
                    return False
                updated = [item for item in current
                           if item != canonical and not self._access_id_in_list([item], canonical)]
            block[key] = sorted(updated)
            atomic_config_write(config_path, raw)
            self._access_mutate_live(adapter, scope, updated)
            self._access_audit(source, scope, canonical, op)
            return True

    @staticmethod
    def _access_audit(source, scope: str, canonical: str, op: str) -> None:
        """Append-only audit trail beside the profile config (best-effort —
        an audit failure must never break the command itself)."""
        try:
            import datetime
            import json

            from gateway.run import _gateway_config_home

            platform = getattr(getattr(source, "platform", None), "value", "?")
            entry = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "actor": getattr(source, "user_id", "?"),
                "platform": platform,
                "scope": scope,
                "op": op,
                "canonical": canonical,
            }
            path = _gateway_config_home() / "access_audit.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            logger.warning("[access] audit append failed", exc_info=True)

    @staticmethod
    def _access_id_in_list(items, canonical: str) -> bool:
        """Loose membership for WhatsApp-family lists where the same human may be stored
        as ``62812…@c.us``, ``62812…@s.whatsapp.net`` or bare digits."""
        from gateway.whatsapp_identity import normalize_whatsapp_identifier
        target = normalize_whatsapp_identifier(canonical)
        return any(normalize_whatsapp_identifier(item) == target for item in items)

    @staticmethod
    @contextlib.contextmanager
    def _access_config_lock(config_path):
        """Best-effort cross-process mutex for the /access read-modify-write.

        Without it two concurrent edits interleave (both read, both write) and
        one entry is silently lost. POSIX uses flock; elsewhere the atomic
        write inside is the only guard.
        """
        try:
            import fcntl
        except ImportError:
            yield
            return
        with open(config_path.with_name(config_path.name + ".lock"), "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _access_platform_block(raw: dict, platform_name: str) -> dict:
        """The platform's YAML block — the same one the loader reads.

        `platform_section` (gateway/config_loader.py) gives a top-level
        `<name>:` block precedence over `platforms.<name>`; writing anywhere
        else would be silently ignored on restart. When both blocks carry
        allowlists, union them into the winner and drop the loser's keys so
        the two locations can never diverge.
        """
        top = raw.get(platform_name)
        top = top if isinstance(top, dict) else None
        platforms = raw.get("platforms")
        nested = None
        if isinstance(platforms, dict) and isinstance(platforms.get(platform_name), dict):
            nested = platforms[platform_name]
        if top is not None and nested is not None:
            for key in ("allow_from", "group_allow_from"):
                union = [str(i) for i in (top.get(key) or [])]
                union += [str(i) for i in (nested.get(key) or []) if str(i) not in union]
                if union:
                    top[key] = sorted(set(union))
                nested.pop(key, None)
            return top
        if top is not None:
            return top
        if nested is not None:
            return nested
        if not isinstance(platforms, dict):
            platforms = raw["platforms"] = {}
        block = {}
        platforms[platform_name] = block
        return block

    def _access_mutate_live(self, adapter, scope: str, updated: list) -> None:
        """Make *updated* effective on the running gateway without a restart."""
        attr = "_allow_from" if scope == "user" else "_group_allow_from"
        current = getattr(adapter, attr, None)
        if isinstance(current, (set, frozenset)):
            setattr(adapter, attr, set(updated))
        elif isinstance(current, list):
            setattr(adapter, attr, list(updated))
        extra = getattr(getattr(adapter, "config", None), "extra", None)
        if isinstance(extra, dict):
            extra["allow_from" if scope == "user" else "group_allow_from"] = list(updated)
        self._access_mutate_env(adapter, scope, updated)

    @staticmethod
    def _access_mutate_env(adapter, scope: str, updated: list) -> None:
        """Env-gated adapters read their allowlist from ``os.environ`` per message; keep
        it in sync.  Skipped under multiplex — ``os.environ`` there holds the DEFAULT
        profile's values and a write would leak one profile's list into another (#72348)."""
        try:
            from agent.secret_scope import is_multiplex_active
            if is_multiplex_active():
                return
        except Exception:
            pass
        if getattr(getattr(adapter, "_dm_allowlist_source", "config"), "startswith", None) is None:
            return
        source = getattr(adapter, "_dm_allowlist_source", None)
        if scope == "user" and isinstance(source, str) and source and source != "config":
            # Env-seeded DM allowlist: the live read path re-reads this var per message.
            import os
            os.environ[source] = ",".join(updated)
            return
        env_keys = getattr(adapter, "ACCESS_ALLOWLIST_ENV_KEYS", None) or {}
        key = "user" if scope == "user" else "group"
        for env_name in env_keys.get(key, ()):
            import os
            os.environ[env_name] = ",".join(updated)

    # ------------------------------------------------------------------ rendering

    def _access_render_list(self, adapter, source) -> str:
        extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
        lines = [f"Access lists for *{source.platform.value}* (this profile):"]
        dm_policy = extra.get("dm_policy") or getattr(adapter, "_dm_policy", "?")
        group_policy = extra.get("group_policy") or getattr(adapter, "_group_policy", "?")
        dm_source = getattr(adapter, "_dm_allowlist_source", None) or "config"
        users = self._access_current_ids(adapter, "user")
        groups = self._access_current_ids(adapter, "group")
        lines.append(f"DM policy: *{dm_policy}* (source: {dm_source}) — {len(users)} allowed")
        lines.append(self._access_render_ids(users))
        lines.append(f"Group policy: *{group_policy}* — {len(groups)} allowed")
        lines.append(self._access_render_ids(groups))
        return "\n".join(lines)

    @staticmethod
    def _access_render_ids(items: list, limit: int = 30) -> str:
        # No id→name API exists, so entries render as raw ids; long lists are
        # cut with a remainder note instead of flooding the chat.
        if not items:
            return "• (none)"
        shown = "\n".join(f"• {item}" for item in items[:limit])
        if len(items) > limit:
            shown += f"\n• …and {len(items) - limit} more"
        return shown
