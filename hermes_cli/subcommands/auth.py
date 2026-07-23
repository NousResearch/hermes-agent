"""``hermes auth`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_auth_parser(subparsers, *, cmd_auth: Callable) -> None:
    """Attach the ``auth`` subcommand to ``subparsers``."""
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage pooled provider credentials",
    )
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action")
    auth_add = auth_subparsers.add_parser("add", help="Add a pooled credential")
    auth_add.add_argument(
        "provider",
        help="Provider id (for example: anthropic, openai-codex, openrouter)",
    )
    auth_add.add_argument(
        "--type",
        dest="auth_type",
        choices=["oauth", "api-key", "api_key"],
        help="Credential type to add",
    )
    auth_add.add_argument("--label", help="Optional display label")
    auth_add.add_argument(
        "--api-key", help="API key value (otherwise prompted securely)"
    )
    auth_add.add_argument("--portal-url", help="Nous portal base URL")
    auth_add.add_argument("--inference-url", help="Nous inference base URL")
    auth_add.add_argument("--client-id", help="OAuth client id")
    auth_add.add_argument("--scope", help="OAuth scope override")
    auth_add.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser for OAuth login",
    )
    auth_add.add_argument(
        "--timeout", type=float, help="OAuth/network timeout in seconds"
    )
    auth_add.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for OAuth login",
    )
    auth_add.add_argument("--ca-bundle", help="Custom CA bundle for OAuth login")
    auth_add.add_argument(
        "--shared",
        action="store_true",
        help="Stage credential into the machine-wide Anthropic shared OAuth pool",
    )
    auth_list = auth_subparsers.add_parser("list", help="List pooled credentials")
    auth_list.add_argument("provider", nargs="?", help="Optional provider filter")
    auth_list.add_argument("--shared", action="store_true", help="Show shared Anthropic pool")
    auth_remove = auth_subparsers.add_parser(
        "remove", help="Remove a pooled credential by index, id, or label"
    )
    auth_remove.add_argument("provider", help="Provider id")
    auth_remove.add_argument(
        "target", help="Credential index, entry id, or exact label"
    )
    auth_remove.add_argument("--shared", action="store_true", help="Mutate shared Anthropic pool")
    auth_reset = auth_subparsers.add_parser(
        "reset", help="Clear exhaustion status for all credentials for a provider"
    )
    auth_reset.add_argument("provider", help="Provider id")
    auth_reset.add_argument("--shared", action="store_true", help="Reset shared Anthropic pool statuses")
    auth_status = auth_subparsers.add_parser(
        "status", help="Show auth status for a provider"
    )
    auth_status.add_argument("provider", help="Provider id")
    auth_status.add_argument("--shared", action="store_true", help="Show shared Anthropic pool status")
    auth_logout = auth_subparsers.add_parser(
        "logout", help="Log out a provider and clear stored auth state"
    )
    auth_logout.add_argument("provider", help="Provider id")
    auth_logout.add_argument("--shared", action="store_true", help="Clear shared Anthropic pool")
    auth_spotify = auth_subparsers.add_parser(
        "spotify", help="Authenticate Hermes with Spotify via PKCE"
    )
    auth_spotify.add_argument(
        "spotify_action",
        nargs="?",
        choices=["login", "status", "logout"],
        default="login",
    )
    auth_spotify.add_argument(
        "--client-id", help="Spotify app client_id (or set HERMES_SPOTIFY_CLIENT_ID)"
    )
    auth_spotify.add_argument(
        "--redirect-uri",
        help="Allow-listed localhost redirect URI for your Spotify app",
    )
    auth_spotify.add_argument("--scope", help="Override requested Spotify scopes")
    auth_spotify.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open the browser automatically",
    )
    auth_spotify.add_argument(
        "--timeout", type=float, help="Callback/token exchange timeout in seconds"
    )

    # Machine-wide Anthropic shared OAuth pool
    auth_scope = auth_subparsers.add_parser(
        "scope", help="Show or change Anthropic credential scope (profile|shared)"
    )
    auth_scope.add_argument("provider", help="Provider id (anthropic)")
    auth_scope.add_argument(
        "scope_mode",
        nargs="?",
        choices=["profile", "shared", "repair"],
        help="Target scope, or repair for malformed marker",
    )
    auth_scope.add_argument(
        "--attest-distinct-accounts",
        action="store_true",
        help="Operator attestation that three OAuth grants are distinct Anthropic accounts",
    )
    auth_scope.add_argument(
        "--shared",
        action="store_true",
        help="Required for repair of the shared scope marker",
    )
    auth_scope.add_argument("--yes", action="store_true", help="Confirm destructive repair")

    auth_backup = auth_subparsers.add_parser(
        "backup", help="Backup shared Anthropic pool (owner-only archive)"
    )
    auth_backup.add_argument("provider", help="Provider id (anthropic)")
    auth_backup.add_argument("--shared", action="store_true", required=False)
    auth_backup.add_argument("--output", required=True, help="Absolute output path")

    auth_restore = auth_subparsers.add_parser(
        "restore", help="Restore shared Anthropic pool from archive"
    )
    auth_restore.add_argument("provider", help="Provider id (anthropic)")
    auth_restore.add_argument("--shared", action="store_true", required=False)
    auth_restore.add_argument("--input", required=True, help="Absolute input archive path")
    auth_restore.add_argument("--yes", action="store_true", help="Confirm restore")

    auth_parser.set_defaults(func=cmd_auth)
