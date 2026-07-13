"""Turbohaul Manager — first-class inference provider with per-role session-ID lineage injection."""

from typing import Any
import uuid

from providers import register_provider
from providers.base import ProviderProfile


class TurbohaulProfile(ProviderProfile):
    """Turbohaul Manager provider — injects per-role thread_id for KV cache reuse.

    The shipped Turbohaul manager reads top-level ``thread_id`` from the request
    payload (chat_completion.py:401).  The OpenAI SDK flattens ``extra_body``
    keys to the top level, so returning ``{"thread_id": ...}`` from
    ``build_extra_body`` lands exactly where the manager reads it.

    Thread-ID scheme:
        - Main agent:   ``hermes-main-<session_id>``  (stable across turns → continuation = KV reuse)
        - Sub-agents:   ``hermes-sub-<session_id>``   (distinct per sub-agent → isolation, no cross-restore)
        - Curator:      ``hermes-main-<session_id>``  (SAME as main → reuses main's warm KV ~26% savings)
        - Compression:  same main thread_id + smaller context (manager infers from ctx_len drop)

    The ``session_id`` is already unique per AIAgent instance — the main agent
    keeps the same session_id across turns, each sub-agent gets a distinct one.
    We prefix with the role tag so the manager can distinguish roles even when
    session_id rotation happens (e.g. /new, compression).
    """

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Inject per-role ``thread_id`` and turn-0 metadata into the request.

        The OpenAI SDK flattens extra_body keys to the top level, so
        ``thread_id`` lands where the Turbohaul manager reads it
        (payload.get("thread_id")).

        Role detection (via params threaded through build_kwargs →
        build_extra_body):

        - Sub-agents carry ``parent_session_id`` (set by the harness on sub-agent construction).
        - Curator carries ``persist_disabled=True`` (set on the curator / title-gen path).
        - Compression carries ``is_compression=True`` (set on the compression path).
        - Main agent: neither flag set → stable thread_id across turns.

        Turn-0 stability metadata (model, provider, pinned conversation date)
        are carried as structured JSON so the Turbohaul manager recognizes them
        OUTSIDE the engine's tokenized/hashed prompt — keeping common_prefix
        stable across sessions/days/model-swaps.

        CRITICAL: Always emit client_meta role flags (is_main/is_sub_agent/
        is_curator/is_compression) even if session_id is missing. The manager
        uses these for R2B_REQ_IDENTITY classification. Returning an empty body
        when session_id is absent causes sub-agent turns to arrive "all-None"
        (no session_id, no is_sub_agent, nothing) — the root cause of the
        turbohaul sub-agent tagging gap.

        MOD-1: When session_id is None for a sub-agent, synthesize a distinct
        session_id from parent_session_id + per-instance nonce to preserve
        sub-agent isolation (no two sub-agents collide on hermes-sub-unknown).

        MOD-2: Contradiction guard — if parent_session_id AND is_compression
        both present, that's definitionally suspicious (real compression should
        not carry parent_session_id). Log and downgrade to sub_agent.

        MOD-3: Single role computation drives both thread_id prefix and flags.
        """
        body: dict[str, Any] = {}

        # ── Role detection (SINGLE SOURCE OF TRUTH) ──────────────────────
        # Order matters: curator (persist_disabled) MUST be checked BEFORE
        # parent_session_id because the curator fork sets BOTH flags.
        # Curator: reuses main's session_id to reuse warm KV cache (~26% savings).
        is_compression = context.get("is_compression", False)
        parent_session_id = context.get("parent_session_id")
        persist_disabled = context.get("persist_disabled")

        # MOD-2: Contradiction guard — parent_session_id + is_compression is suspicious
        if parent_session_id and is_compression:
            import logging
            logging.warning(
                "Turbohaul: contradictory flags — parent_session_id=%s with is_compression=true. "
                "Real compression turns should not carry parent_session_id. Downgrading to sub_agent.",
                parent_session_id
            )
            # Downgrade: treat as sub_agent, not compression
            is_compression = False

        if persist_disabled and not is_compression:
            # Curator: shares main's thread_id to reuse warm KV (~26% savings).
            # Same thread_id as main agent → manager reuses main's prefix cache.
            # is_curator flag in client_meta provides observability.
            role = "curator"
        elif parent_session_id and not persist_disabled:
            # Sub-agent: distinct ID per spawned agent → sub-agent isolation
            role = "sub_agent"
        elif is_compression:
            # Compression turn: same main prefix, smaller context
            role = "compression"
        else:
            # Main agent: stable across turns → continuation = KV reuse
            role = "main"

        # ── Compute effective session_id (MOD-1) ───────────────────────────
        # If session_id missing for sub-agent, synthesize from parent_session_id + nonce
        effective_session_id = session_id
        if effective_session_id is None and role == "sub_agent" and parent_session_id:
            # Use parent_session_id + short nonce to guarantee uniqueness per sub-agent instance
            nonce = uuid.uuid4().hex[:8]
            effective_session_id = f"{parent_session_id}-sub-{nonce}"
        elif effective_session_id is None and role == "main":
            # Main agent without session_id shouldn't happen in practice, but guard anyway
            effective_session_id = "main-unknown"
        elif effective_session_id is None and role == "curator":
            # Curator without session_id — fall back to main-unknown (shares main's prefix)
            effective_session_id = "main-unknown"
        elif effective_session_id is None and role == "compression":
            effective_session_id = "main-unknown"

        # ── Derive thread_id from role + effective_session_id ─────────────
        if role == "curator":
            thread_id = f"hermes-main-{effective_session_id}"
        elif role == "sub_agent":
            thread_id = f"hermes-sub-{effective_session_id}"
        elif role == "compression":
            thread_id = f"hermes-main-{effective_session_id}"
        else:
            thread_id = f"hermes-main-{effective_session_id}"

        # Always emit thread_id with effective_session_id
        body["thread_id"] = thread_id

        # ── Turn-0 stability metadata (recognized by Turbohaul manager) ──
        # These are structured JSON the manager keys on; they are NOT tokenized
        # into the engine's prompt, so common_prefix doesn't collapse to ~3.
        meta: dict[str, Any] = {}
        if context.get("model_name"):
            meta["model"] = context["model_name"]
        if context.get("provider_name"):
            meta["provider"] = context["provider_name"]
        if context.get("pinned_conversation_date"):
            meta["conversation_started"] = context["pinned_conversation_date"]
        if meta:
            body["turn0_meta"] = meta

        # ── Explicit classification labels for Turbohaul manager (the Turbohaul Manager spec) ──
        # Structured JSON carried outside engine's hashed prompt for manager-side
        # routing/observability. Keys mirror the Turbohaul Manager spec:
        # {ip, session_id, model, is_main, is_sub_agent, is_compression, is_curator}
        # ip is filled by manager from request.client.host; we pass session_id + role flags.
        # Mutually exclusive: exactly one of is_main/is_sub_agent/is_curator/is_compression is true.
        is_main = role == "main"
        is_sub_agent = role == "sub_agent"
        is_curator = role == "curator"
        is_compression = role == "compression"

        client_meta: dict[str, Any] = {}
        # Always include effective_session_id for correlation
        client_meta["session_id"] = effective_session_id
        # Role flags (redundant with thread_id prefix but explicit for manager consumption)
        # Exactly-one-true contract so the classify_request can trust the label.
        client_meta["is_main"] = is_main
        client_meta["is_sub_agent"] = is_sub_agent
        client_meta["is_curator"] = is_curator
        client_meta["is_compression"] = is_compression
        # model already in turn0_meta.model; not duplicated here.
        if client_meta:
            body["client_meta"] = client_meta

        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Turbohaul: no special top-level kwargs needed beyond thread_id in extra_body."""
        return {}, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Turbohaul: fetch model list from the manager's /v1/models endpoint."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


turbohaul = TurbohaulProfile(
    name="turbohaul",
    aliases=("turbohaul-manager",),
    display_name="Turbohaul Manager",
    description="Turbohaul Manager — first-class inference provider with per-role session-ID lineage",
    env_vars=("OPENAI_API_KEY",),
    base_url="http://localhost:11401/v1",
    auth_type="api_key",
    default_max_tokens=65536,
)

register_provider(turbohaul)
