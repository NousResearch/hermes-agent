"""WebTmux plugin registration entry point."""

import logging

logger = logging.getLogger(__name__)


def register(ctx):
    """Register the WebTmux platform adapter with Hermes."""
    from .adapter import WebTmuxAdapter, check_webtmux_requirements, standalone_send_webtmux

    def _is_connected(cfg) -> bool:
        return bool(
            cfg.enabled
            and check_webtmux_requirements()
        )

    ctx.register_platform(
        name="webtmux",
        label="WebTmux",
        adapter_factory=lambda cfg: WebTmuxAdapter(cfg),
        check_fn=check_webtmux_requirements,
        validate_config=None,
        is_connected=_is_connected,
        emoji="🖥️",
        allow_update_command=True,
        allow_all_env="WEBTMUX_ALLOW_ALL",
        platform_hint=(
            "You are on WebTmux — a web-based terminal multiplexer interface. "
            "The user can see tmux panes in their browser (hermes chat, nvim, logs). "
            "Respond concisely and assume the user may be on a mobile device."
        ),
        standalone_sender_fn=standalone_send_webtmux,
    )
    logger.info("WebTmux plugin registered")
