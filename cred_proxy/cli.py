"""CLI for the credential proxy daemon.

Standalone entry point (``hermes-cred-proxy``):
  hermes-cred-proxy start
  hermes-cred-proxy stop
  hermes-cred-proxy status
  hermes-cred-proxy add <name>     (prompts for value, never echoes it)
  hermes-cred-proxy list

Also callable from the main hermes CLI as ``hermes cred-proxy <subcommand>``.
Use dispatch(args) for that path where args.cred_proxy_command is set.
"""

import argparse
import getpass
import http.client
import json
import socket
import sys


# ---------------------------------------------------------------------------
# Daemon HTTP client — talks to the management API over Unix socket
# ---------------------------------------------------------------------------

class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection subclass that connects via a Unix domain socket."""

    def __init__(self, socket_path: str) -> None:
        # host is required by HTTPConnection but irrelevant for Unix sockets
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(self._socket_path)


def _mgmt_request(method: str, path: str, body: dict | None = None) -> dict:
    """Send a request to the daemon's management API and return parsed JSON."""
    from cred_proxy.daemon import get_socket_path, is_running

    if not is_running():
        print("Error: credential proxy is not running. Start it with: hermes cred-proxy start")
        sys.exit(1)

    conn = _UnixHTTPConnection(get_socket_path())
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(data))

    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        result = json.loads(resp.read())
        if resp.status >= 400:
            print(f"Error: {result.get('error', f'HTTP {resp.status}')}")
            sys.exit(1)
        return result
    except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
        print(f"Error: cannot connect to credential proxy: {exc}")
        sys.exit(1)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Individual command implementations
# ---------------------------------------------------------------------------

def cmd_start(args=None) -> None:
    from cred_proxy.daemon import start
    start()


def cmd_stop(args=None) -> None:
    from cred_proxy.daemon import stop
    stop()


def cmd_status(args=None) -> None:
    from cred_proxy.daemon import status
    info = status()
    if info["running"]:
        print(f"running (PID {info['pid']})")
    else:
        print("stopped")
    print(f"socket: {info['socket_path']}")


def cmd_add(args) -> None:
    name = args.name
    try:
        value = getpass.getpass(f"Value for {name!r}: ")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(1)
    if not value:
        print("Error: empty value not allowed.")
        sys.exit(1)
    result = _mgmt_request("POST", "/_cred/add", {"name": name, "value": value})
    if result.get("stored"):
        print(f"Stored credential {name!r}.")


def cmd_list(args=None) -> None:
    result = _mgmt_request("GET", "/_cred/list")
    names = result.get("names", [])
    if not names:
        print("(no credentials stored)")
    else:
        for n in names:
            print(n)


# ---------------------------------------------------------------------------
# Dispatcher (used by hermes cred-proxy subcommand in main.py)
# ---------------------------------------------------------------------------

def dispatch(args) -> None:
    """Route args.cred_proxy_command to the appropriate handler."""
    cmd = getattr(args, "cred_proxy_command", None)
    if cmd == "start":
        cmd_start(args)
    elif cmd == "stop":
        cmd_stop(args)
    elif cmd == "status":
        cmd_status(args)
    elif cmd == "add":
        cmd_add(args)
    elif cmd == "list":
        cmd_list(args)
    else:
        # No subcommand: print help
        print("Usage: hermes cred-proxy {start,stop,status,add,list}")
        print("       hermes-cred-proxy {start,stop,status,add,list}")


# ---------------------------------------------------------------------------
# Standalone argparse CLI (hermes-cred-proxy entry point)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-cred-proxy",
        description="Hermes credential proxy — store and inject secrets into tool subprocesses",
    )
    subs = parser.add_subparsers(dest="subcommand", help="Command")

    subs.add_parser("start", help="Start the credential proxy daemon")
    subs.add_parser("stop", help="Stop the credential proxy daemon")
    subs.add_parser("status", help="Show daemon status")

    add_p = subs.add_parser("add", help="Add or update a named credential")
    add_p.add_argument("name", help="Credential name (used in hermes-proxy://<name>)")

    subs.add_parser("list", help="List stored credential names")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand == "start":
        cmd_start(args)
    elif args.subcommand == "stop":
        cmd_stop(args)
    elif args.subcommand == "status":
        cmd_status(args)
    elif args.subcommand == "add":
        cmd_add(args)
    elif args.subcommand == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
