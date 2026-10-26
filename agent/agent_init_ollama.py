"""Init-time local-server (Ollama) num_ctx detection and the compressor clamp to it."""

from __future__ import annotations

from agent.agent_runtime_helpers import _ra
from agent.model_metadata import is_local_endpoint, query_ollama_num_ctx


def _configure_ollama_num_ctx(agent, _model_cfg, _config_context_length):
    # Ollama defaults num_ctx to 2048, so detect the max window and send num_ctx per request.
    # model.ollama_num_ctx overrides; model.context_length caps the detected value (VRAM).
    agent._ollama_num_ctx: int | None = None
    _override = _model_cfg.get("ollama_num_ctx") if isinstance(_model_cfg, dict) else None
    if _override is not None:
        try:
            agent._ollama_num_ctx = int(_override)
        except (TypeError, ValueError):
            _ra().logger.debug("Invalid ollama_num_ctx config value: %r", _override)
    if agent._ollama_num_ctx is None and agent.base_url and is_local_endpoint(agent.base_url):
        try:
            # api_key may be a callable (Entra token provider); detection needs a string.
            _key = agent.api_key if isinstance(agent.api_key, str) else ""
            _detected = query_ollama_num_ctx(agent.model, agent.base_url, api_key=_key or "")
            if _detected and _detected > 0:
                agent._ollama_num_ctx = _detected
        except Exception as exc:
            _ra().logger.debug("Local server num_ctx detection failed: %s", exc)
    # Cap auto-detected num_ctx to the explicit context_length (GGUF metadata can advertise
    # 256K+ and Ollama would allocate that much VRAM); never override an explicit num_ctx.
    if (
        agent._ollama_num_ctx
        and _config_context_length
        and _override is None
        and agent._ollama_num_ctx > _config_context_length
    ):
        _ra().logger.info(
            "Ollama num_ctx capped: %d -> %d (model.context_length override)",
            agent._ollama_num_ctx, _config_context_length,
        )
        agent._ollama_num_ctx = _config_context_length
    if agent._ollama_num_ctx and not agent.quiet_mode:
        # Name the real source: a config override is honoured on any local server, /api/show is Ollama-only.
        _ra().logger.info(
            "Local server num_ctx: will request %d tokens (%s)",
            agent._ollama_num_ctx,
            "model.ollama_num_ctx" if _override is not None else "model max from Ollama /api/show",
        )


def _clamp_compressor_to_ollama_num_ctx(agent):
    # Recalibrate the compressor to the served window: every request runs at num_ctx, so a
    # trigger derived from the probed model window could sit above it and never fire.
    # A config that sets only model.ollama_num_ctx (without model.context_length) previously left the
    # compressor targeting the probed window while the server truncated/rejected at num_ctx — the compaction
    # trigger could sit several times ABOVE the real served window and never fire. Clamp the compressor's
    # window to the effective num_ctx so threshold math operates on the context the server actually serves.
    # (Overlaps #60103's silent-clamp dead zone; this is the init-order half.)
    _cc_window = getattr(agent.context_compressor, "context_length", 0) or 0
    if agent._ollama_num_ctx and agent._ollama_num_ctx > 0 and _cc_window and agent._ollama_num_ctx < _cc_window:
        _ra().logger.info(
            "Compressor window clamped to Ollama num_ctx: %d -> %d",
            _cc_window, agent._ollama_num_ctx,
        )
        agent.context_compressor.update_model(
            model=agent.model, context_length=agent._ollama_num_ctx, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )
