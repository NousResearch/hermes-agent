"""Recall/read-path cluster for HonchoSessionManager: context, cards, search, conclusions, dialectic."""

from __future__ import annotations

import logging
from typing import Any, Callable

from plugins.memory.honcho.session_auth import HonchoAuthError

logger = logging.getLogger("plugins.memory.honcho.session")

_FAILED = object()  # sentinel: a guarded call raised (distinct from a legitimately empty/None result)


class SessionContextMixin:
    """Peer context / card / search / conclusion / dialectic operations against Honcho."""

    def _guarded(self, fn: Callable[[], Any], default: Any, level: int, msg: str, *args: Any) -> Any:
        """Run ``fn``; re-raise HonchoAuthError, log (exc appended to args) and return default otherwise."""
        try:
            return fn()
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.log(level, msg, *args, e)
            return default

    def _guarded_authed(self, label: str, fn: Callable[[], Any], default: Any, level: int, msg: str, *args: Any) -> Any:
        """``_guarded`` around ``_authed_call(label, fn)`` (401 -> forced refresh + one retry)."""
        return self._guarded(lambda: self._authed_call(label, fn), default, level, msg, *args)

    def _guarded_session(
        self, session_key: str, fn: Callable[[Any], Any], default: Any, level: int, msg: str, *args: Any,
    ) -> Any:
        """``_guarded`` over ``fn(session)`` for the cached session; ``default`` when no session is cached."""
        session = self._cache.get(session_key)
        return self._guarded(lambda: fn(session), default, level, msg, *args) if session else default

    @staticmethod
    def _normalize_card(card: Any) -> list[str]:
        """Normalize Honcho card payloads into a plain list of strings."""
        return ([str(item) for item in card if item] if isinstance(card, list) else [str(card)]) if card else []

    @staticmethod
    def _target_kwargs(target: str | None) -> dict[str, str]:
        """SDK peer getters take ``target=`` only when observing another peer."""
        return {} if target is None else {"target": target}

    def _fetch_peer_card(self, peer_id: str, *, target: str | None = None) -> list[str]:
        """Fetch a peer card from the peer object (session.context() can return an empty card)."""
        def _get_card() -> Any:
            peer = self._get_or_create_peer(peer_id)
            getters = (getattr(peer, n, None) for n in ("get_card", "card"))  # "card" is the legacy SDK getter
            getter = next((g for g in getters if callable(g)), None)
            return getter(**self._target_kwargs(target)) if getter else None
        return self._normalize_card(self._authed_call("peer card fetch", _get_card))

    def _fetch_peer_context(
        self, peer_id: str, search_query: str | None = None, *, target: str | None = None,
    ) -> dict[str, Any]:
        """Fetch representation + peer card from a peer object; when peer.context() leaves either
        empty, fall back to the dedicated representation / card getters. Raises HonchoAuthError when
        auth is dead or a 401 survives the forced refresh; the fallback chain would just repeat it."""
        context_kwargs: dict[str, Any] = self._target_kwargs(target)
        if search_query is not None:
            context_kwargs["search_query"] = search_query
        peer = lambda: self._get_or_create_peer(peer_id)  # noqa: E731
        failed = "Direct %s failed for '%%s': %%s"
        ctx = self._guarded_authed(
            "peer context fetch", lambda: peer().context(**context_kwargs), None, logging.DEBUG,
            failed % "peer.context()", peer_id,
        )
        representation = getattr(ctx, "representation", None) or getattr(ctx, "peer_representation", None) or ""
        card = self._normalize_card(getattr(ctx, "peer_card", None))
        if not representation:
            representation = self._guarded_authed(
                "peer representation fetch", lambda: peer().representation(**self._target_kwargs(target)),
                "", logging.DEBUG, failed % "peer.representation()", peer_id,
            ) or ""
        if not card:
            card = self._guarded(
                lambda: self._fetch_peer_card(peer_id, target=target), [], logging.DEBUG,
                failed % "peer card fetch", peer_id,
            )
        return {"representation": representation, "card": card}

    def _peer_context_strings(self, peer_id: str, search_query: str | None = None, *, target: str | None = None):
        """``_fetch_peer_context`` flattened to ``(representation, newline-joined card)`` for prompt injection."""
        ctx = self._fetch_peer_context(peer_id, search_query, target=target)
        return ctx["representation"], "\n".join(ctx["card"])

    def get_prefetch_context(self, session_key: str, user_message: str | None = None) -> dict[str, str]:
        """Pre-fetch user + AI peer context (representation, card) plus the session summary.
        ``user_message`` is passed as search_query so Honcho returns topic-relevant conclusions.
        Stops early (returning what it has) once auth is dead."""
        session = self._cache.get(session_key)
        if not session:
            return {}
        result: dict[str, str] = {}

        def _summary() -> None:
            if session.honcho_session_id not in self._sessions_cache:
                return
            ctx = self._authed_call(
                "session summary fetch", lambda: self._sdk_session(session.honcho_session_id).context(summary=True),
            )
            if ctx.summary and getattr(ctx.summary, "content", None):
                result["summary"] = ctx.summary.content

        def _user() -> None:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, "user")
            result["representation"], result["card"] = self._peer_context_strings(
                observer_peer_id, search_query=user_message or None, target=target_peer_id or session.user_peer_id,
            )

        def _ai() -> None:
            result["ai_representation"], result["ai_card"] = self._peer_context_strings(
                session.assistant_peer_id, target=session.assistant_peer_id,
            )
        for step, level, msg in (
            (_summary, logging.DEBUG, "Failed to fetch session summary from Honcho: %s"),
            (_user, logging.WARNING, "Failed to fetch user context from Honcho: %s"),
            (_ai, logging.DEBUG, "Failed to fetch AI peer context from Honcho: %s"),
        ):
            try:
                step()
            except HonchoAuthError:
                break  # Auth is dead; the pop_auth_notice path tells the model why context is missing.
            except Exception as e:
                logger.log(level, msg, e)
        return result

    def get_session_context(self, session_key: str, peer: str = "user") -> dict[str, Any]:
        """Fetch session-level context (summary, representation, card, recent messages).
        Raises HonchoAuthError so callers can tell rejected credentials from no context."""
        session = self._cache.get(session_key)
        if not session:
            return {}
        if session.honcho_session_id not in self._sessions_cache:
            # Fall back to peer-level context, respecting the requested peer.
            peer_id = self._resolve_peer_id(session, peer)
            return self._fetch_peer_context(peer_id, target=peer_id)

        def _fetch() -> dict[str, Any]:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            ctx = self._authed_call(
                "session context fetch",
                lambda: self._sdk_session(session.honcho_session_id).context(
                    summary=True, peer_target=target_peer_id or observer_peer_id, peer_perspective=observer_peer_id,
                ),
            )
            result: dict[str, Any] = {}
            if ctx.summary:
                result["summary"] = ctx.summary.content
            if ctx.peer_representation:
                result["representation"] = ctx.peer_representation
            if ctx.peer_card:
                result["card"] = "\n".join(ctx.peer_card)
            if ctx.messages:
                result["recent_messages"] = [
                    {"role": getattr(m, "peer_id", "unknown"), "content": (m.content or "")[:500]}
                    for m in ctx.messages[-10:]
                ]
            return result
        return self._guarded(_fetch, {}, logging.DEBUG, "Session context fetch failed: %s")

    def get_peer_card(self, session_key: str, peer: str = "user") -> list[str]:
        """Fetch a peer card (curated facts, no LLM). [] if unavailable; raises HonchoAuthError."""
        def _fetch(session: Any) -> list[str]:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            card = self._fetch_peer_card(observer_peer_id, target=target_peer_id)
            # Some backends store cards on the target peer, not the observer-target slot.
            return card or (self._fetch_peer_card(target_peer_id) if target_peer_id else [])
        return self._guarded_session(session_key, _fetch, [], logging.DEBUG, "Failed to fetch peer card from Honcho: %s")

    def search_context(self, session_key: str, query: str, max_tokens: int = 800, peer: str = "user") -> str:
        """Hybrid search over raw messages visible from ``peer``'s perspective, all sessions. Snippets
        accumulate until ``max_tokens`` (~4 chars/token) is exhausted. Returns "" when nothing matches;
        raises HonchoAuthError on rejected credentials."""
        session = self._cache.get(session_key)
        q = (query or "").strip()[:4000]  # Honcho caps query length for the embedding model.
        if not session or not q:
            return ""
        peer_id = self._resolve_peer_id(session, peer)
        char_budget = max(200, int(max_tokens) * 4)
        limit = max(3, min(20, char_budget // 300))
        messages = self._guarded_authed(
            "message search", lambda: self.honcho.search(q, filters={"peer_perspective": peer_id}, limit=limit),
            _FAILED, logging.DEBUG, "Honcho message search failed (peer_perspective=%s): %s", peer_id,
        )
        if messages is _FAILED:
            # Older Honcho versions lack the perspective filter; fall back to peer-authored search.
            messages = self._guarded_authed(
                "peer search", lambda: self._get_or_create_peer(peer_id).search(q, limit=limit),
                None, logging.DEBUG, "Honcho peer search fallback also failed: %s",
            )
        if not messages:
            return ""
        # Author labels distinguish user-stated facts from assistant-derived ones.
        lines: list[str] = []
        for m in messages:
            content = (getattr(m, "content", "") or "").strip()
            if not content:
                continue
            author = getattr(m, "peer_id", "") or "unknown"
            who = "assistant" if author == session.assistant_peer_id else author
            sess = getattr(m, "session_id", "") or ""
            entry = f"[{who}{f' · {sess}' if sess else ''}] {content[:1200]}"
            # Budget left after the joined snippets so far plus the separator this entry would need.
            remaining = char_budget - len("\n\n".join(lines)) - (2 if lines else 0)
            if remaining <= 0:
                break
            truncated = len(entry) > remaining
            entry = entry[:remaining].rstrip()
            if not entry:
                break
            lines.append(entry)
            if truncated:
                break
        return "\n\n".join(lines)

    def _conclusions_scope(self, session: Any, target_peer_id: str) -> Any:
        """ConclusionScope for observing target_peer_id; shared by create/delete/list."""
        ai_observes = target_peer_id == session.assistant_peer_id or self._ai_observe_others
        observer = self._get_or_create_peer(session.assistant_peer_id if ai_observes else target_peer_id)
        return observer.conclusions_of(target_peer_id)

    def create_conclusion(self, session_key: str, content: str, peer: str = "user") -> bool:
        """Write a conclusion (durable fact) about ``peer`` back to Honcho."""
        if not content or not content.strip():
            return False
        session = self._cache.get(session_key)
        if not session:
            logger.warning("No session cached for '%s', skipping conclusion", session_key)
            return False

        def _create() -> bool:
            target_peer_id = self._resolve_peer_id(session, peer)
            if target_peer_id is None:
                logger.warning("Could not resolve conclusion peer '%s' for session '%s'", peer, session_key)
                return False
            payload = [{"content": content.strip(), "session_id": session.honcho_session_id}]
            self._authed_call("conclusion create", lambda: self._conclusions_scope(session, target_peer_id).create(payload))
            logger.info("Created conclusion about %s for %s: %s", target_peer_id, session_key, content[:80])
            return True
        return self._guarded(_create, False, logging.ERROR, "Failed to create conclusion: %s")

    def delete_conclusion(self, session_key: str, conclusion_id: str, peer: str = "user") -> bool:
        """Delete a conclusion by ID. Use only for PII removal."""
        def _delete(session: Any) -> bool:
            target_peer_id = self._resolve_peer_id(session, peer)
            self._authed_call(
                "conclusion delete", lambda: self._conclusions_scope(session, target_peer_id).delete(conclusion_id),
            )
            logger.info("Deleted conclusion %s for %s", conclusion_id, session_key)
            return True
        return self._guarded_session(
            session_key, _delete, False, logging.ERROR, "Failed to delete conclusion %s: %s", conclusion_id,
        )

    def list_conclusions(self, session_key: str, query: str | None = None, peer: str = "user", limit: int = 20):
        """List (or semantically search with ``query``) conclusions as {"id", "content"} dicts."""
        def _list(session: Any) -> list[dict]:
            target_peer_id = self._resolve_peer_id(session, peer)
            if target_peer_id is None:
                return []

            def _fetch() -> Any:
                scope = self._conclusions_scope(session, target_peer_id)
                return scope.query(query, top_k=limit) if query else scope.list(size=limit).items
            return [{"id": c.id, "content": c.content} for c in self._authed_call("conclusion list", _fetch)]
        return self._guarded_session(session_key, _list, [], logging.DEBUG, "Honcho list_conclusions failed: %s")

    def set_peer_card(self, session_key: str, card: list[str], peer: str = "user") -> list[str] | None:
        """Replace a peer's card. Returns the updated card, or None on failure."""
        def _update(session: Any) -> list[str] | None:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            if observer_peer_id is None:
                logger.warning("Could not resolve peer '%s' for set_peer_card in session '%s'", peer, session_key)
                return None
            result = self._authed_call(
                "peer card update",
                lambda: self._get_or_create_peer(observer_peer_id).set_card(card, **self._target_kwargs(target_peer_id)),
            )
            logger.info("Updated peer card observer=%s target=%s (%d facts)",
                        observer_peer_id, target_peer_id or observer_peer_id, len(card))
            return result
        return self._guarded_session(session_key, _update, None, logging.ERROR, "Failed to set peer card: %s")

    def seed_ai_identity(self, session_key: str, content: str, source: str = "manual") -> bool:
        """Seed the AI peer's representation from text (SOUL.md, exported chats, ...), sent as an
        assistant-peer message so Honcho's reasoning model incorporates it. Unlike the other
        operations, auth failures are logged and swallowed here too."""
        if not content or not content.strip():
            return False
        session = self._cache.get(session_key)
        if not session:
            logger.warning("No session cached for '%s', skipping AI seed", session_key)
            return False
        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping AI seed", session_key)
            return False
        wrapped = f"<ai_identity_seed>\n<source>{source}</source>\n\n{content.strip()}\n</ai_identity_seed>"

        def _seed() -> bool:
            assistant_peer = self._get_or_create_peer(session.assistant_peer_id)
            self._sdk_session(session.honcho_session_id).add_messages([assistant_peer.message(wrapped)])
            logger.info("Seeded AI identity from '%s' into %s", source, session_key)
            return True
        try:
            return self._authed_call("identity seed", _seed)
        except Exception as e:
            logger.error("Failed to seed AI identity: %s", e)
            return False

    def get_ai_representation(self, session_key: str) -> dict[str, str]:
        """Fetch the AI peer's representation + card ("" values if unavailable)."""
        def _fetch(session: Any) -> dict[str, str]:
            rep, card = self._peer_context_strings(session.assistant_peer_id, target=session.assistant_peer_id)
            return {"representation": rep, "card": card}
        return self._guarded_session(
            session_key, _fetch, {"representation": "", "card": ""}, logging.DEBUG,
            "Failed to fetch AI representation: %s",
        )

    def dialectic_query(
        self, session_key: str, query: str, reasoning_level: str | None = None, peer: str = "user",
        apply_injection_cap: bool = True, raise_errors: bool = False,
    ) -> str:
        """Ask Honcho's dialectic endpoint about a peer (LLM on the backend; run off-thread).
        ``reasoning_level`` is honored only when dialecticDynamic is true. ``apply_injection_cap``
        clips to ``dialecticMaxChars`` (automatic injection only). ``raise_errors`` re-raises backend
        failures instead of returning "" so explicit tool calls can tell a timeout from an empty answer.
        Raises HonchoAuthError when credentials are rejected after a forced refresh and one retry.

        Args: session_key: The session key to query against. query: Natural language question.
        reasoning_level: Override the configured default (dialecticReasoningLevel). If None or
        dialecticDynamic is false, uses the configured default. peer: Which peer to query — "user" (default)
        or "ai". apply_injection_cap: Clip automatic injections to ``dialecticMaxChars``. Explicit
        ``honcho_reasoning`` calls pass False because Honcho already bounds their output. raise_errors:
        Re-raise backend failures instead of returning "". Explicit tool calls pass True so a timeout or
        server error surfaces as an error, not as "no result" (#36098 issue 4: collapsing failures to ""
        made auth errors, timeouts, and genuinely-empty answers indistinguishable).
        """
        session = self._cache.get(session_key)
        target_peer_id = self._resolve_peer_id(session, peer) if session else None
        if target_peer_id is None:
            return ""
        if len(query) > self._dialectic_max_input_chars:
            query = query[:self._dialectic_max_input_chars].rsplit(" ", 1)[0]
        level = reasoning_level if (self._dialectic_dynamic and reasoning_level) else self._dialectic_reasoning_level

        def _chat_once() -> str:
            # The AI peer observes others when allowed; otherwise each peer queries its own context.
            if self._ai_observe_others and target_peer_id != session.assistant_peer_id:
                observer = self._get_or_create_peer(session.assistant_peer_id)
                return observer.chat(query, target=target_peer_id, reasoning_level=level) or ""
            return self._get_or_create_peer(target_peer_id).chat(query, reasoning_level=level) or ""
        try:
            result = self._authed_call("dialectic query", _chat_once)
            cap = self._dialectic_max_chars if apply_injection_cap else 0
            return result[:cap].rsplit(" ", 1)[0] + " …" if result and cap and len(result) > cap else result
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.warning("Honcho dialectic query failed: %s", e)
            if raise_errors:
                raise
            return ""
