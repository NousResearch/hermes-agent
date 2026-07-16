"""Built-in Hermes CLI top-level subcommand names.

Shared by ``main()`` (fast-path plugin discovery gating) and plugin
registration (rejecting CLI names that would collide with core subparsers).
"""

from __future__ import annotations

BUILTIN_SUBCOMMANDS = frozenset(
    {
        "acp",
        "auth",
        "backup",
        "bundles",
        "checkpoints",
        "claw",
        "completion",
        "computer-use",
        "config",
        "console",
        "cron",
        "curator",
        "dashboard",
        "serve",
        "debug",
        "doctor",
        "dump",
        "fallback",
        "gateway",
        "hooks",
        "import",
        "insights",
        "gui",
        "desktop",
        "kanban",
        "login",
        "logout",
        "logs",
        "lsp",
        "mcp",
        "memory",
        "migrate",
        "moa",
        "journey",
        "memory-graph",
        "learning",
        "model",
        "pairing",
        "pets",
        "plugins",
        "portal",
        "postinstall",
        "profile",
        "project",
        "proxy",
        "prompt-size",
        "send",
        "sessions",
        "setup",
        "skills",
        "slack",
        "status",
        "tools",
        "uninstall",
        "update",
        "version",
        "webhook",
        "whatsapp",
        "whatsapp-cloud",
        "chat",
        "secrets",
        "security",
        "harness",
        # Help-ish invocations — plugin commands not being listed in
        # top-level --help is an acceptable trade-off for skipping an
        # expensive eager import of every bundled plugin module.
        "help",
    }
)
