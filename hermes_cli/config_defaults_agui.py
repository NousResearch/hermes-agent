"""Pure-data factories for the standalone AG-UI adapter's configuration."""

from hermes_cli.config_defaults_env import _setting


def agui_defaults():
    # Standalone AG-UI HTTP/SSE adapter. Credentials remain in .env; blank
    # model/provider overrides use the normal Hermes runtime resolver.
    return {
        "host": "127.0.0.1",
        "port": 8000,
        "toolsets": ["hermes-acp"],
        "provider": "",
        "model": "",
        "api_mode": "",
        "base_url": "",
        "frontend_tool_timeout": 60,  # seconds to wait for a browser tool result
        "max_conversations": 16,  # active + idle agents; refuse excess admission
        "conversation_idle_timeout": 1800,  # seconds before closing an idle agent
    }


def agui_token_metadata():
    return _setting(
        "Session token required for a non-loopback `hermes agui` listener. "
        "Generate with `openssl rand -hex 32`.",
        "AG-UI session token (for a networked bind)", None,
        password=True, advanced=True)
