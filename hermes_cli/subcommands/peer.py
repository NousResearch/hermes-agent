"""``hermes peer`` — bot-to-bot DMs across machines/gateways.

A *peer* is another Hermes gateway (any machine: homelab, Spark, Hermes
Cloud) running the ``api_server`` platform. Registering it here gives every
bot on THIS machine a transport to message bots on THAT machine:

    hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
    hermes peer dm spark "Message from 🤖 dixie (@dixie): disk status?"
    hermes peer dm spark/researcher "..."      # named profile (multiplexed peer)
    hermes peer run spark --idempotency-key ticket-123 < /tmp/long-task.txt
    hermes peer status spark run_abc123

``dm`` resolves the remote agent's canonical "Bot Chat" session (by title,
creating it when missing), runs ONE synchronous agent turn over the peer's
existing ``POST /api/sessions/{id}/chat`` endpoint, and prints the reply on
stdout — the exact cross-machine twin of the local
``hermes -p <bot> chat --in ~ -c "Bot Chat" ...`` bot-messaging command, so
the Bot Mode protocol composes over it unchanged.

``run`` starts the same canonical-session turn through the asynchronous Runs
API and returns a ``run_id`` immediately. ``status`` polls that handle without
holding the original HTTP connection open. Use this pair for long turns.

Design notes:
- No new server surface: the peer's stock api_server is the transport.
- Peer labels/URLs live in config.yaml (``bot_peers``); the peer's
  API_SERVER_KEY is a credential and lives in ``~/.hermes/.env`` as
  ``HERMES_PEER_<NAME>_KEY``.
- Named-profile targets use the peer's ``/p/<profile>/`` multiplex mirror;
  the bare target is the peer gateway's own (launch) profile.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

BOT_CHAT_TITLE = "Bot Chat"
_PEER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# One synchronous agent turn can legitimately take minutes.
DM_TIMEOUT_S = 600
LIST_TIMEOUT_S = 30


def _peer_key_env(name: str) -> str:
    return f"HERMES_PEER_{name.upper().replace('-', '_')}_KEY"


def _load_peers() -> dict:
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    peers = cfg.get("bot_peers")
    return peers if isinstance(peers, dict) else {}


def _save_peers(peers: dict) -> None:
    from hermes_cli.config import load_config, save_config

    cfg = load_config() or {}
    cfg["bot_peers"] = peers
    save_config(cfg)


def _peer_secret(name: str) -> str:
    """The peer's API key: profile-scoped secret store first, raw env fallback."""
    env_name = _peer_key_env(name)
    try:
        from agent.secret_scope import get_secret

        return (get_secret(env_name, "") or "").strip()
    except Exception:
        import os

        return (os.environ.get(env_name) or "").strip()


def _request(
    url: str,
    key: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: int = LIST_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> dict:
    from hermes_cli.urllib_security import open_credentialed_url
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "hermes-peer-dm",
    }
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=request_headers,
    )
    # The peer URL is user-registered (``hermes peer add``); a redirect to a
    # different origin must not carry the Authorization: Bearer key with it —
    # a compromised/MITM'd peer could otherwise harvest it. open_credentialed_url
    # strips non-safelisted headers across a cross-origin redirect.
    with open_credentialed_url(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8", "replace")
    try:
        parsed = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError(f"Peer returned non-JSON response: {payload[:200]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Peer returned a non-object JSON response")
    return parsed


def _base_url(peer: dict, profile: str | None) -> str:
    url = str(peer.get("url") or "").rstrip("/")
    if profile:
        # Multiplex mirror: same handlers, scoped to the named profile.
        return f"{url}/p/{urllib.parse.quote(profile, safe='')}"
    return url


def _find_bot_chat(base: str, key: str) -> str | None:
    """The remote canonical Bot Chat's session id, or None.

    Bot Mode always HIDES canonical chats, so the plain listing (which
    excludes hidden sessions) misses an existing Bot Chat and the caller
    would try to create a duplicate that the peer's UNIQUE(title) guard
    rejects (issue #91583). Newer peers support an exact-title lookup with
    ``include_hidden=1``; older peers ignore the unknown query params and
    return the ordinary visible listing, so this single request degrades
    to exactly the previous behavior against them.
    """
    query = urllib.parse.urlencode({"limit": 200, "title": BOT_CHAT_TITLE, "include_hidden": 1})
    listing = _request(f"{base}/api/sessions?{query}", key)
    for session in listing.get("data") or []:
        if isinstance(session, dict) and (session.get("title") or "").strip() == BOT_CHAT_TITLE:
            return str(session.get("id") or "") or None
    return None


def _ensure_bot_chat(base: str, key: str) -> str:
    existing = _find_bot_chat(base, key)
    if existing:
        return existing
    try:
        created = _request(
            f"{base}/api/sessions",
            key,
            method="POST",
            body={"title": BOT_CHAT_TITLE, "source": "bot_peer_dm"},
        )
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc)
        if exc.code == 400 and "title" in detail.lower():
            # Older peer (no title/include_hidden lookup support): its
            # canonical Bot Chat exists but is hidden, so we couldn't see it
            # and the create collided with the UNIQUE(title) guard.
            raise RuntimeError(
                f"Peer already has a '{BOT_CHAT_TITLE}' session but it is hidden and the "
                f"peer's gateway is too old to expose hidden sessions to this lookup "
                f"(HTTP 400: {detail}). Update the peer's hermes-agent, or unhide the "
                f"session there: PATCH /api/sessions/<id> {{\"hidden\": false}}."
            ) from exc
        raise
    # Real api_server wraps the row: {"object": "hermes.session", "session": {...}}.
    session = created.get("session") if isinstance(created.get("session"), dict) else created
    session_id = str(session.get("id") or session.get("session_id") or "")
    if not session_id:
        raise RuntimeError("Peer did not return a session id for the new Bot Chat")
    return session_id


def _parse_target(target: str) -> tuple[str, str | None]:
    """``<peer>`` or ``<peer>/<profile>`` → (peer, profile|None)."""
    raw = (target or "").strip()
    peer, _, profile = raw.partition("/")
    peer = peer.strip()
    profile = profile.strip() or None
    if not peer:
        raise ValueError("Peer name required (hermes peer dm <peer>[/<agent>] ...)")
    if profile and not _PROFILE_RE.match(profile):
        raise ValueError(f"Invalid agent/profile name: {profile!r}")
    return peer, profile


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
        parsed = json.loads(body)
        message = parsed.get("error", {}).get("message") if isinstance(parsed, dict) else None
        return message or body[:200]
    except Exception:
        return str(exc)


def _resolve_peer_target(target: str) -> tuple[str, str | None, dict, str]:
    """Resolve a registered target to ``(name, profile, config, key)``."""
    peer_name, profile = _parse_target(target)
    peer = _load_peers().get(peer_name)
    if not isinstance(peer, dict) or not peer.get("url"):
        raise LookupError(f"No peer named '{peer_name}'. Run: hermes peer list")
    key = _peer_secret(peer_name)
    if not key:
        raise PermissionError(
            f"No API key for peer '{peer_name}'. Set it: hermes peer add {peer_name} "
            f"--url <url> --key <key> (or add {_peer_key_env(peer_name)}=<key> to ~/.hermes/.env)"
        )
    return peer_name, profile, peer, key


def _message_from_args(args) -> str:
    message = (getattr(args, "message", None) or "").strip()
    if not message and not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    return message


def _peer_run_durability(base: str, key: str) -> bool | None:
    """Return durable support, or None when an older peer cannot advertise it."""
    try:
        capabilities = _request(f"{base}/v1/capabilities", key)
    except Exception:
        return None
    features = capabilities.get("features")
    if not isinstance(features, dict):
        return None
    contract = features.get("runs_idempotency")
    if not isinstance(contract, dict) or not contract.get("supported"):
        return None
    return bool(contract.get("durable"))


def cmd_peer(args) -> int:
    action = getattr(args, "peer_action", None)

    if action in ("add", "set"):
        name = (args.name or "").strip().lower()
        if not _PEER_NAME_RE.match(name):
            print(f"Invalid peer name: {name!r} (lowercase, digits, -, _; max 64)", file=sys.stderr)
            return 2
        url = (args.url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            print("Peer --url must be an http(s) gateway base URL, e.g. http://spark.lan:8377", file=sys.stderr)
            return 2
        peers = _load_peers()
        peers[name] = {"url": url.rstrip("/"), **({"note": args.note.strip()} if getattr(args, "note", "") else {})}
        _save_peers(peers)
        key = (getattr(args, "key", "") or "").strip()
        if key:
            from hermes_cli.config import save_env_value

            save_env_value(_peer_key_env(name), key)
            print(f"Peer '{name}' saved ({url}) — key stored as {_peer_key_env(name)} in ~/.hermes/.env")
        else:
            print(
                f"Peer '{name}' saved ({url}). No key given — set the peer's API_SERVER_KEY with:\n"
                f"  hermes peer add {name} --url {url} --key <key>\n"
                f"  (or add {_peer_key_env(name)}=<key> to ~/.hermes/.env)"
            )
        return 0

    if action in ("remove", "rm"):
        name = (args.name or "").strip().lower()
        peers = _load_peers()
        if name not in peers:
            print(f"No peer named '{name}'.", file=sys.stderr)
            return 1
        peers.pop(name)
        _save_peers(peers)
        print(f"Peer '{name}' removed (its {_peer_key_env(name)} entry in .env is kept; delete it manually if unused).")
        return 0

    if action in ("list", "ls", None):
        peers = _load_peers()
        if not peers:
            print("No peers registered. Add one: hermes peer add <name> --url http://host:port --key <API_SERVER_KEY>")
            return 0
        for name in sorted(peers):
            entry = peers[name] if isinstance(peers[name], dict) else {}
            has_key = "key set" if _peer_secret(name) else f"NO KEY ({_peer_key_env(name)} unset)"
            note = f" — {entry.get('note')}" if entry.get("note") else ""
            print(f"{name}\t{entry.get('url', '?')}\t[{has_key}]{note}")
        return 0

    if action in {"dm", "run", "status", "stop"}:
        try:
            peer_name, profile, peer, key = _resolve_peer_target(args.target)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except (LookupError, PermissionError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        base = _base_url(peer, profile)

        if action in {"status", "stop"}:
            run_id = (getattr(args, "run_id", None) or "").strip()
            if not run_id:
                print("Run ID required.", file=sys.stderr)
                return 2
            try:
                result = _request(
                    f"{base}/v1/runs/{urllib.parse.quote(run_id, safe='')}"
                    + ("/stop" if action == "stop" else ""),
                    key,
                    method="POST" if action == "stop" else "GET",
                    body={} if action == "stop" else None,
                )
            except urllib.error.HTTPError as exc:
                print(
                    f"Peer '{peer_name}' rejected the request (HTTP {exc.code}): {_http_error_detail(exc)}",
                    file=sys.stderr,
                )
                return 1
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                print(f"Could not reach peer '{peer_name}': {exc}", file=sys.stderr)
                return 1

            payload = {"peer": peer_name, "profile": profile, **result}
            if getattr(args, "json", False):
                print(json.dumps(payload))
            else:
                print(f"{run_id}: {result.get('status', 'unknown')}")
                if action == "status" and result.get("output"):
                    print(result["output"])
                elif action == "status" and result.get("error"):
                    print(result["error"], file=sys.stderr)
            return 0

        message = _message_from_args(args)
        if not message:
            print("Message required (argument or stdin).", file=sys.stderr)
            return 2

        if action == "run":
            idempotency_key = (
                getattr(args, "idempotency_key", None) or f"peer-{uuid.uuid4().hex}"
            ).strip()
            if (
                not idempotency_key
                or len(idempotency_key) > 255
                or re.search(r"[\r\n\x00]", idempotency_key)
            ):
                print(
                    "Idempotency key must be 1-255 characters without control newlines.",
                    file=sys.stderr,
                )
                return 2
            try:
                durability = _peer_run_durability(base, key)
                if durability is not True:
                    print(
                        "Warning: this peer does not advertise restart-durable "
                        "run replay; keep the run ID and avoid blind retries "
                        "after a gateway restart.",
                        file=sys.stderr,
                    )
                session_id = _ensure_bot_chat(base, key)
                result = _request(
                    f"{base}/v1/runs",
                    key,
                    method="POST",
                    body={"input": message, "session_id": session_id},
                    headers={"Idempotency-Key": idempotency_key},
                )
            except urllib.error.HTTPError as exc:
                print(
                    f"Peer '{peer_name}' rejected the request (HTTP {exc.code}): {_http_error_detail(exc)}",
                    file=sys.stderr,
                )
                return 1
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                print(f"Could not reach peer '{peer_name}': {exc}", file=sys.stderr)
                return 1

            run_id = str(result.get("run_id") or "")
            if not run_id:
                print(f"Peer '{peer_name}' did not return a run ID.", file=sys.stderr)
                return 1
            payload = {
                "peer": peer_name,
                "profile": profile,
                "session_id": session_id,
                "run_id": run_id,
                "status": result.get("status") or "started",
                "idempotency_key": idempotency_key,
                "replayed": bool(result.get("replayed", False)),
            }
            if getattr(args, "json", False):
                print(json.dumps(payload))
            else:
                replay = " (replayed)" if payload["replayed"] else ""
                print(f"{run_id}: {payload['status']}{replay}")
                print(f"session_id: {session_id}")
                print(f"idempotency_key: {idempotency_key}")
            return 0

        try:
            session_id = _ensure_bot_chat(base, key)
            result = _request(
                f"{base}/api/sessions/{urllib.parse.quote(session_id, safe='')}/chat",
                key,
                method="POST",
                body={"message": message},
                timeout=DM_TIMEOUT_S,
            )
        except urllib.error.HTTPError as exc:
            print(f"Peer '{peer_name}' rejected the request (HTTP {exc.code}): {_http_error_detail(exc)}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"Peer '{peer_name}': {exc}", file=sys.stderr)
            return 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"Could not reach peer '{peer_name}': {exc}", file=sys.stderr)
            return 1

        reply = ""
        msg = result.get("message")
        if isinstance(msg, dict):
            reply = str(msg.get("content") or "")
        if getattr(args, "json", False):
            print(json.dumps({"peer": peer_name, "profile": profile, "session_id": result.get("session_id") or session_id, "reply": reply}))
        else:
            print(reply or "(no reply)")
        return 0

    print("Unknown peer action. See: hermes peer --help", file=sys.stderr)
    return 2


def build_peer_parser(subparsers) -> None:
    """Attach the ``peer`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "peer",
        help="Bot-to-bot DMs across machines (peer Hermes gateways)",
        description=(
            "Register other Hermes gateways as peers and message their agents. "
            "'hermes peer dm <peer>[/<agent>] \"...\"' delivers into the remote "
            "agent's canonical Bot Chat over the peer's API server and prints "
            "the reply — the cross-machine twin of 'hermes -p <bot> chat'. "
            "The peer must run the api_server platform; its API_SERVER_KEY is "
            "stored locally as a credential in ~/.hermes/.env."
        ),
        epilog=(
            "Examples:\n"
            "  hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>\n"
            "  hermes peer list\n"
            '  hermes peer dm spark "Message from 🤖 dixie (@dixie): disk status?"\n'
            '  hermes peer dm spark/researcher "..."   # named profile on a multiplexed peer\n'
            "  hermes peer run spark --idempotency-key ticket-123 < long-task.txt\n"
            "  hermes peer status spark run_abc123\n"
            "  hermes peer stop spark run_abc123\n"
            "  hermes peer remove spark\n"
            "\n"
            "Exit codes: 0 ok, 1 delivery/peer error, 2 usage error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    peer_sub = parser.add_subparsers(dest="peer_action")

    add_p = peer_sub.add_parser("add", aliases=["set"], help="Register (or update) a peer gateway")
    add_p.add_argument("name", help="Peer name (lowercase slug, e.g. spark, homelab)")
    add_p.add_argument("--url", required=True, help="Peer gateway base URL, e.g. http://spark.lan:8377")
    add_p.add_argument("--key", default="", help="The peer's API_SERVER_KEY (stored in ~/.hermes/.env)")
    add_p.add_argument("--note", default="", help="Optional description")

    peer_sub.add_parser("list", aliases=["ls"], help="List registered peers")

    rm_p = peer_sub.add_parser("remove", aliases=["rm"], help="Remove a peer")
    rm_p.add_argument("name", help="Peer name")

    dm_p = peer_sub.add_parser(
        "dm",
        help="Message an agent on a peer gateway and print its reply",
    )
    dm_p.add_argument(
        "target", help="<peer> or <peer>/<agent> (named profile on a multiplexed peer)"
    )
    dm_p.add_argument(
        "message", nargs="?", default=None, help="Message text (or stdin)"
    )
    dm_p.add_argument(
        "--json", action="store_true", default=False, help="Emit a JSON result"
    )

    run_p = peer_sub.add_parser(
        "run",
        help="Start a long peer turn asynchronously and return its run ID",
    )
    run_p.add_argument(
        "target", help="<peer> or <peer>/<agent> (named profile on a multiplexed peer)"
    )
    run_p.add_argument(
        "message", nargs="?", default=None, help="Message text (or stdin)"
    )
    run_p.add_argument(
        "--idempotency-key",
        default=None,
        help="Stable retry key (generated when omitted)",
    )
    run_p.add_argument(
        "--json", action="store_true", default=False, help="Emit a JSON result"
    )

    status_p = peer_sub.add_parser(
        "status",
        help="Read the status and final output of an asynchronous peer run",
    )
    status_p.add_argument(
        "target", help="<peer> or <peer>/<agent> (named profile on a multiplexed peer)"
    )
    status_p.add_argument("run_id", help="Run ID returned by 'hermes peer run'")
    status_p.add_argument(
        "--json", action="store_true", default=False, help="Emit a JSON result"
    )

    stop_p = peer_sub.add_parser(
        "stop",
        help="Stop one asynchronous peer run without affecting another turn",
    )
    stop_p.add_argument(
        "target", help="<peer> or <peer>/<agent> (named profile on a multiplexed peer)"
    )
    stop_p.add_argument("run_id", help="Run ID returned by 'hermes peer run'")
    stop_p.add_argument(
        "--json", action="store_true", default=False, help="Emit a JSON result"
    )

    parser.set_defaults(func=cmd_peer)
