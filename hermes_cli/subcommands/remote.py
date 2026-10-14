"""``hermes remote`` subcommand — remote gateway plugins.

The concrete consumer of the ``gateway.remote_plugins`` registry:

* ``hermes remote list``            — which transports are installed
* ``hermes remote serve``           — accept side: expose this gateway
  over  a remote plugin's transport and print the pairing artifact
* ``hermes remote connect <url>``   — dial side: turn a saved
  ``scheme://…`` url into a live local access point and hold it open

Plain http(s) remotes are core and need no remote plugin.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def build_remote_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach the ``remote`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "remote",
        help="Remote gateway plugins: serve this gateway / dial a remote over plugin transports",
        description="Manage remote gateway plugins — scheme-registered transports "
        "(any non-https transport) that pair a Hermes desktop/CLI with a gateway without "
        "port-forwarding. Plain https remotes are core and need no remote plugin.")
    actions = parser.add_subparsers(dest="remote_action", required=True)

    actions.add_parser("list", aliases=["ls"], help="List installed remote plugins and what they do")

    serve = actions.add_parser("serve", help="Expose the local gateway over  a remote plugin transport")
    serve.add_argument("--scheme", required=True, help="RemotePlugin scheme to serve with (see 'hermes remote list')")
    serve.add_argument("--gateway-url", default="http://127.0.0.1:9119",
                       help="Local gateway/dashboard URL to expose (default: %(default)s)")

    connect = actions.add_parser("connect", help="Dial a saved scheme://… url to a local access point")
    connect.add_argument("url", help="Saved plugin URL, e.g. example://<ticket>")

    parser.set_defaults(func=cmd_remote)


def cmd_remote(args: argparse.Namespace) -> int:
    from gateway.remote_plugins import RemotePlugin, get_remote_plugin, list_remote_plugins, scheme_of
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    action = args.remote_action

    if action in ("list", "ls"):
        plugins = list_remote_plugins()

        if not plugins:
            print("No remote gateway plugins installed.")
            print("Install one as a hermes plugin, or use plain https remotes.")
            return 0

        for scheme, plugin in sorted(plugins.items()):
            hint = f" — {plugin.hint}" if plugin.hint else ""
            dials = "serves+dials" if type(plugin).connect is not RemotePlugin.connect else "serves"
            print(f"  {scheme}://  [{dials}]{hint}")

        return 0

    if action == "serve":
        scheme = args.scheme.strip().lower()
        plugin = get_remote_plugin(scheme)

        if plugin is None:
            print(f"No plugin registered for scheme '{scheme}://'. Run 'hermes remote list'.", file=sys.stderr)
            return 2

        # core locates the remote plugin's own config block; the remote plugin interprets it.
        from hermes_cli.config import load_config

        entries = ((load_config() or {}).get("plugins") or {}).get("entries") or {}
        config = entries.get(plugin.plugin_id or plugin.scheme) or None

        return _run_forever(plugin.serve(args.gateway_url, config), mode="serve")

    if action == "connect":
        scheme = scheme_of(args.url)

        if scheme in ("http", "https", ""):
            print("Plain http(s) urls need no remote plugin — point the app at them directly.", file=sys.stderr)
            return 2

        plugin = get_remote_plugin(args.url)

        if plugin is None:
            print(f"No plugin registered for scheme '{scheme}://'. Run 'hermes remote list'.", file=sys.stderr)
            return 2

        return _run_forever(plugin.connect(args.url), mode="connect")

    return 2


def _run_forever(coro, *, mode: str) -> int:
    """Run a serve/connect coroutine until Ctrl-C, always tearing the handle down."""

    async def _main() -> int:
        handle = await coro

        if mode == "serve":
            ticket = getattr(handle, "ticket", "") or "(no ticket)"
            print(f"serving {handle.target_url} — pair with:\n  {ticket}")
            print("Ctrl-C to stop.")
        else:
            label = getattr(handle, "host_label", "") or ""
            print(f"dialled: access point {handle.base_url}" + (f" ({label})" if label else ""))
            print("Ctrl-C to close.")

        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop = getattr(handle, "stop", None)

            if stop is not None:
                await stop()

        return 0

    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return 0
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
