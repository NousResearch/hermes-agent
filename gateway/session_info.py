"""Session-banner presentation and provenance collected by the live route resolver."""

from dataclasses import dataclass
import logging

from agent.i18n import t
from utils import base_url_hostname

logger = logging.getLogger(__name__)


@dataclass
class SessionRouteInfo:
    """Display-only metadata, kept separate from kwargs passed to AIAgent."""

    source: str = "global default"
    model: str = ""
    provider: str = ""
    problem: str = ""

    def select(self, source, model, provider):
        self.source, self.model, self.provider = source, model or "", provider or ""

    def channel(self, override, model, runtime):
        """Record the resolver's selected channel, unless a session route already owns it."""
        if self.source != "session override" and (override.model or override.provider):
            self.select("channel override", override.model or model, override.provider or runtime.get("provider"))

    def resolve_provider(self, provider, *, target_model=None):
        from gateway.run import _resolve_runtime_agent_kwargs_for_provider
        try:
            return _resolve_runtime_agent_kwargs_for_provider(provider, target_model=target_model)
        except Exception:
            # Provider exceptions can contain endpoints or secrets. Give actionable, safe copy
            # rather than echoing their arbitrary text into a shared channel.
            self.problem = f"Could not resolve provider credentials for {provider}."
            raise

    def unavailable_banner(self):
        route = " / ".join(part for part in (self.provider, self.model) if part)
        return (
            f"◆ Route unavailable: {self.source}" + (f" ({route})" if route else "")
            + f"\n{self.problem or 'Could not resolve the route.'} "
            "Check provider configuration and credentials with `hermes auth`, "
            "or change the route with `/model` / channel_overrides."
        )


def format_session_info(runner, source=None):
    """Resolve once; never describe a global route as an unavailable override."""
    from gateway.run_model_context import _resolve_gateway_model_context
    info = SessionRouteInfo()
    try:
        if source is None:
            resolved = _resolve_gateway_model_context()
        else:
            model, runtime = runner._resolve_session_agent_runtime(source=source, route_info=info)
            if info.problem:
                return info.unavailable_banner()
            resolved = _resolve_gateway_model_context(model, runtime_snapshot=runtime)
        return _format_resolved(resolved, info.source if source is not None else "global default")
    except Exception:
        logger.debug("Session-info resolution failed", exc_info=True)
        return info.unavailable_banner()


def _format_resolved(resolved, route_source):
    context_length = resolved.context_length
    ctx_source = {
        "config": "config",
        "default": "default — set model.context_length in config to override",
    }.get(resolved.context_source, "detected")
    ctx_display = (
        f"{context_length / 1_000_000:.1f}M" if context_length >= 1_000_000
        else f"{context_length // 1_000}K" if context_length >= 1_000 else str(context_length)
    )
    lines = [
        t("gateway.session.info_model", model=resolved.model),
        t("gateway.session.info_provider", provider=resolved.provider or "unknown"),
        t("gateway.session.info_context", tokens=ctx_display, source=ctx_source),
    ]
    if route_source != "global default":
        lines.append(f"◆ Route: {route_source}")
    if (resolved.provider or "") == "moa":
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config
        agg = normalize_moa_config(load_config().get("moa"))["presets"].get(resolved.model, {}).get("aggregator") or {}
        if agg:
            lines.append(t("gateway.session.info_acting_model", provider=agg.get("provider"), model=agg.get("model")))
    base_url = resolved.base_url
    if base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1", "0.0.0.0"):
        lines.append(t("gateway.session.info_endpoint", url=base_url))
    return "\n".join(lines)
