"""``hermes peer`` — bot-to-bot DMs across machines/gateways.

A *peer* is another Hermes gateway running the ``api_server`` platform; its stock
API is the transport (no new server surface). ``dm`` resolves the remote canonical
"Bot Chat" session (creating it when missing) and runs ONE synchronous turn — the
cross-machine twin of ``hermes -p <bot> chat --in ~ -c "Bot Chat"``. ``run``/``status``
/``stop`` do the same turn through the async Runs API. Peer labels/URLs live in
config.yaml (``bot_peers``); the key lives in ``~/.hermes/.env`` as
``HERMES_PEER_<NAME>_KEY``. ``<peer>/<profile>`` targets the ``/p/<profile>/`` mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

BOT_CHAT_TITLE = "Bot Chat"
_PEER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# One synchronous agent turn can legitimately take minutes.
DM_TIMEOUT_S = 600
LIST_TIMEOUT_S = 30

#: Slack added to the caller's wait budget so the gateway's own receipt window
#: (``bot_mode.receipt_after_seconds``) always fits inside the HTTP timeout.
_DM_TIMEOUT_SLACK_S = 30

#: §D3: after a read timeout on a *delivery* POST — the request that carries an
#: idempotency key — the identical request is replayed once with this wait
#: budget. Admission is idempotent on that key, so the replay can never admit a
#: second delivery: it replays the reservation and returns the true envelope.
_DM_REPLAY_WAIT_SECONDS = 1

#: HTTP read ceiling for that replay. A replayed reservation answers
#: immediately (no turn runs), so this stays far below ``DM_TIMEOUT_S``.
_DM_REPLAY_TIMEOUT_S = 30

#: The run/status read's ``object`` tag — the read-path twin of SEND_RESULT_OBJECT.
RUN_OBJECT = "hermes.peer.run"

#: The delivery envelope's ``object`` tag (§1.2). Its presence marks a peer that
#: understands queued receipts; anything else is the legacy reply shape.
SEND_RESULT_OBJECT = "hermes.peer.send_result"

#: ENUM A -> process exit code (§1.5). ``receipt`` is SUCCESS, never a failure.
RESULT_EXIT_CODES = {"delivered": 0, "receipt": 0, "failed": 1, "unknown": 1, "refused": 2}

#: Server error codes meaning "the request itself is defective" -> usage exit 2.
_REFUSAL_ERROR_CODES = frozenset({
    "invalid_idempotency_key", "idempotency_key_conflict", "idempotency_conflict",
    "invalid_request", "invalid_payload"})


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
    url: str, key: str, *, method: str = "GET", body: dict | None = None,
    timeout: float = LIST_TIMEOUT_S, headers: dict[str, str] | None = None) -> dict:
    from hermes_cli.urllib_security import open_credentialed_url
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "hermes-peer-dm"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=request_headers)
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
            f"{base}/api/sessions", key, method="POST",
            body={"title": BOT_CHAT_TITLE, "source": "bot_peer_dm"})
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
                f"session there: PATCH /api/sessions/<id> {{\"hidden\": false}}.") from exc
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
            f"--url <url> --key <key> (or add {_peer_key_env(peer_name)}=<key> to ~/.hermes/.env)")
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


def _peer_failure(peer_name: str, exc: Exception) -> int:
    """Print a peer HTTP rejection or transport failure to stderr; always exit 1."""
    if isinstance(exc, urllib.error.HTTPError):
        detail = _http_error_detail(exc)
        print(f"Peer '{peer_name}' rejected the request (HTTP {exc.code}): {detail}",
              file=sys.stderr)
    else:
        print(f"Could not reach peer '{peer_name}': {exc}", file=sys.stderr)
    return 1


def _is_read_timeout(exc: BaseException) -> bool:
    """True when *exc* is a socket read timeout, never a connection failure.

    ``urlopen`` wraps a timeout raised before the response headers in
    ``URLError``, but the one this lane hit fires mid-body: a synchronous peer
    turn can run for minutes, so the read timeout surfaces from ``resp.read()``
    as a bare ``socket.timeout`` — which *is* ``TimeoutError`` on Python 3.10+.
    Both shapes mean the same thing: the request was sent, the answer was lost.
    """
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return (isinstance(exc, urllib.error.URLError)
            and isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)))


class _DeliveryOutcomeUnknown(Exception):
    """A delivery POST timed out AND its idempotent replay did not answer.

    The first request may still have been admitted, so the caller must report
    ``unknown`` — never "unreachable", and never a blind resend.
    """

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(idempotency_key)
        self.idempotency_key = idempotency_key


def _deliver_with_replay(url: str, key: str, *, body: dict, timeout: float,
                         headers: dict, idempotency_key: str) -> dict:
    """POST one delivery, replaying it once on a read timeout (§D3).

    A read timeout on the *delivery* POST does not mean the peer was
    unreachable: the request may already be admitted, reserved and executing.
    Nothing may claim "unreachable" for that, so the identical request — same
    URL, same body, same idempotency key, wait budget pinned to
    ``_DM_REPLAY_WAIT_SECONDS`` — is re-issued once. Admission is idempotent on
    the key, so this cannot deliver twice: the server replays the reservation
    and returns the true envelope (the ``receipt``, or the settled result if
    the turn has since finished).
    """
    try:
        return _request(url, key, method="POST", body=body, timeout=timeout, headers=headers)
    except urllib.error.HTTPError:
        raise  # an HTTP status is an explicit answer, never a lost response
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if not _is_read_timeout(exc):
            raise
    replay_headers = {**headers, "X-Hermes-Wait-Seconds": str(int(_DM_REPLAY_WAIT_SECONDS))}
    try:
        return _request(url, key, method="POST", body=body,
                        timeout=_DM_REPLAY_TIMEOUT_S, headers=replay_headers)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError,
            RuntimeError) as exc:
        # Deliberately broad: the first POST may have been admitted, so *any*
        # failure to read the replay leaves this delivery's outcome unknown.
        # Reporting the peer as unreachable here is the exact misreport the
        # receipt contract exists to eliminate.
        raise _DeliveryOutcomeUnknown(idempotency_key) from exc


def _unknown_delivery_detail(idempotency_key: str) -> str:
    """``do not resend`` wording for a delivery whose outcome could not be read."""
    try:
        from tools.bot_delivery_queue import unknown_detail

        tail = unknown_detail(idempotency_key)
    except Exception:
        tail = f"outcome unknown; a resend must reuse idempotency_key {idempotency_key}"
    return f"do not resend — {tail}"


def _bot_mode_value(key: str, default):
    """Read one ``bot_mode`` key, falling back to ``default``."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        value = (cfg.get("bot_mode") or {}).get(key, default)
    except Exception:
        value = default
    return default if value is None else value


def _peer_value(key: str, default):
    """Read one ``peer`` key, falling back to ``default``."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        value = (cfg.get("peer") or {}).get(key, default)
    except Exception:
        value = default
    return default if value is None else value


def _resolve_wait_seconds(args) -> float:
    """``--wait`` sent as ``X-Hermes-Wait-Seconds`` (default ``peer.dm_wait_seconds``)."""
    raw = getattr(args, "wait_seconds", None)
    if raw is None:
        raw = _peer_value("dm_wait_seconds", DM_TIMEOUT_S)
    try:
        wait = float(raw)
    except (TypeError, ValueError):
        wait = float(DM_TIMEOUT_S)
    return max(1.0, wait)


def _sender_profile() -> str:
    """This host's profile name, sent as ``X-Hermes-Sender-Profile``."""
    env = (os.environ.get("HERMES_PROFILE") or "").strip()
    if env:
        return env
    try:
        from hermes_cli.profiles import get_active_profile_name

        return (get_active_profile_name() or "").strip() or "default"
    except Exception:
        return "default"


def _validate_idempotency_key(key: str) -> str:
    """1-255 characters, no CR/LF/NUL — the rule already used by ``peer run``."""
    key = (key or "").strip()
    if not key or len(key) > 255 or re.search(r"[\r\n\x00]", key):
        raise ValueError("Idempotency key must be 1-255 characters without control newlines.")
    return key


def _resolve_idempotency_key(args, *, sender_profile: str, target_profile: str,
                             session_id: str, message: str) -> str:
    """Explicit ``--idempotency-key``, else the shared §4.1 derivation.

    Must run *after* the Bot Chat session id is resolved: the derived key is
    bound to ``session_id``, so minting it earlier would make a retry that hit a
    different session a different logical message.
    """
    explicit = (getattr(args, "idempotency_key", None) or "").strip()
    if explicit:
        return _validate_idempotency_key(explicit)
    from hermes_cli.delivery_keys import derive_delivery_key

    try:
        window = int(_bot_mode_value("dedup_window_seconds", 900))
    except (TypeError, ValueError):
        window = 900
    return derive_delivery_key(sender_profile, target_profile, session_id, message,
                               dedup_window_seconds=window)


def _delivery_headers(idempotency_key: str, sender_profile: str, wait_seconds: float) -> dict:
    """The three delivery headers the client owns (§5.1)."""
    return {
        "Idempotency-Key": idempotency_key,
        "X-Hermes-Sender-Profile": sender_profile,
        "X-Hermes-Wait-Seconds": str(int(wait_seconds)),
    }


def _peer_error_body(exc: urllib.error.HTTPError) -> tuple[str, str]:
    """``(code, message)`` from an HTTPError body — a body can be read only once."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:
        return "", str(exc)
    try:
        parsed = json.loads(raw)
    except ValueError:
        return "", (raw[:200] or str(exc))
    if not isinstance(parsed, dict):
        return "", raw[:200]
    error = parsed.get("error")
    if isinstance(error, dict):
        return (str(error.get("code") or error.get("reason") or ""),
                str(error.get("message") or raw[:200]))
    return (str(parsed.get("reason") or parsed.get("code") or ""),
            str(parsed.get("message") or raw[:200]))


def _peer_refusal(peer_name: str, exc: urllib.error.HTTPError) -> int:
    """Report a refused/rejected request. A defect is exit 2, a delivery error 1."""
    code, message = _peer_error_body(exc)
    if code in _REFUSAL_ERROR_CODES:
        print(f"Peer '{peer_name}' refused the request: {message}", file=sys.stderr)
        return 2
    print(f"Peer '{peer_name}' rejected the request (HTTP {exc.code}): {message}",
          file=sys.stderr)
    return 1


def _emit_dm_envelope(args, envelope: dict, *, session_id: str, idempotency_key: str,
                      peer_name: str) -> int:
    """Print one delivery envelope and translate ENUM A into an exit code (§1.5)."""
    result = str(envelope.get("result") or "")
    status = str(envelope.get("status") or "")
    if getattr(args, "json", False):
        print(json.dumps(envelope))
    elif result == "delivered":
        print(str(envelope.get("reply") or "(no reply)"))
    elif result == "receipt":
        print(f"{status}: {envelope.get('detail') or ''}")
        print(f"session_id: {session_id}")
        print(f"idempotency_key: {idempotency_key}")
    else:
        why = envelope.get("error") or envelope.get("detail") or envelope.get("reason") or result
        print(f"Peer '{peer_name}' {status or result}: {why}", file=sys.stderr)
    return RESULT_EXIT_CODES.get(result, 1)


def _emit_unknown_delivery(args, *, peer_name: str, profile: str | None, session_id: str,
                           idempotency_key: str) -> int:
    """Report an accepted-but-unverifiable delivery (§D3) without blaming the peer.

    Both the delivery POST and its idempotent replay went unanswered, so the
    message may still have landed. The wording must never read as "the peer was
    unreachable" (it wasn't) and must warn against a blind resend.
    """
    envelope = {
        "object": SEND_RESULT_OBJECT,
        "result": "unknown",
        "status": "unknown",
        "reason": "unknown",
        "retryable": False,
        "peer": peer_name,
        "profile": profile,
        "session_id": session_id,
        "idempotency_key": idempotency_key,
        "detail": _unknown_delivery_detail(idempotency_key),
    }
    if getattr(args, "json", False):
        print(json.dumps(envelope))
    else:
        print(envelope["detail"], file=sys.stderr)
        print(f"session_id: {session_id}")
        print(f"idempotency_key: {idempotency_key}")
    return RESULT_EXIT_CODES.get("unknown", 1)


def _emit(args, payload: dict, text_lines: list[str]) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload))
    else:
        for line in text_lines:
            print(line)
    return 0


def _peer_add(args) -> int:
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
            f"  (or add {_peer_key_env(name)}=<key> to ~/.hermes/.env)")
    return 0


def _peer_remove(args) -> int:
    name = (args.name or "").strip().lower()
    peers = _load_peers()
    if name not in peers:
        print(f"No peer named '{name}'.", file=sys.stderr)
        return 1
    peers.pop(name)
    _save_peers(peers)
    print(f"Peer '{name}' removed (its {_peer_key_env(name)} entry in .env is kept; delete it manually if unused).")
    return 0


def _peer_list(args) -> int:
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


def _emit_unknown_run(args, peer_name: str, profile: str | None, run_id: str) -> int:
    """A 404 on the READ path is post-retention GC, not a delivery failure.

    Spec T6 / §7: a run's terminal row is deleted at the end of its retention
    window, so 404 means the record aged out — not that the delivery failed.
    Exit status stays 0 so a monitor keying on it cannot report a
    delivered-and-GC'd message as a failure.
    """
    if getattr(args, "json", False):
        print(json.dumps({
            "peer": peer_name, "profile": profile, "object": RUN_OBJECT,
            "result": "unknown", "status": "unknown", "run_id": run_id,
            "reason": "run_not_found", "retryable": False}))
    else:
        print(f"{run_id}: unknown — no such run "
              f"(retention window elapsed; this is not a delivery failure)")
    return 0


def _peer_run_ctl(args, action: str, peer_name: str, profile: str | None, base: str,
                  key: str) -> int:
    """``status`` / ``stop`` on an asynchronous run."""
    run_id = (getattr(args, "run_id", None) or "").strip()
    if not run_id:
        print("Run ID required.", file=sys.stderr)
        return 2
    stop = action == "stop"
    try:
        result = _request(
            f"{base}/v1/runs/{urllib.parse.quote(run_id, safe='')}" + ("/stop" if stop else ""),
            key, method="POST" if stop else "GET", body={} if stop else None)
    except urllib.error.HTTPError as exc:
        # A 404 on the read path is the retention window elapsing, never a
        # delivery signal. Every other status (and ``stop``, which mutates)
        # keeps today's failure semantics exactly.
        if not stop and exc.code == 404:
            return _emit_unknown_run(args, peer_name, profile, run_id)
        return _peer_failure(peer_name, exc)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        return _peer_failure(peer_name, exc)
    if getattr(args, "json", False):
        print(json.dumps({"peer": peer_name, "profile": profile, **result}))
        return 0
    print(f"{run_id}: {result.get('status', 'unknown')}")
    if not stop and result.get("output"):
        print(result["output"])
    elif not stop and result.get("error"):
        print(result["error"], file=sys.stderr)
    return 0


def _turn_body(message: str, *, message_key: str, **extra) -> dict:
    """Request body for one turn. ``author`` is added only when a dispatcher set HERMES_TURN_AUTHOR."""
    from agent.turn_author import turn_author_from_env

    body = {message_key: message, **extra}
    author = turn_author_from_env()
    if author is not None:
        body["author"] = author
    return body


def _peer_run(args, message: str, peer_name: str, profile: str | None, base: str, key: str) -> int:
    sender_profile = _sender_profile()
    wait_seconds = _resolve_wait_seconds(args)
    try:
        if _peer_run_durability(base, key) is not True:
            print(
                "Warning: this peer does not advertise restart-durable "
                "run replay; keep the run ID and avoid blind retries "
                "after a gateway restart.", file=sys.stderr)
        session_id = _ensure_bot_chat(base, key)
        idempotency_key = _resolve_idempotency_key(
            args, sender_profile=sender_profile, target_profile=profile or sender_profile,
            session_id=session_id, message=message)
        result = _request(
            f"{base}/v1/runs", key, method="POST",
            body=_turn_body(message, message_key="input", session_id=session_id),
            headers=_delivery_headers(idempotency_key, sender_profile, wait_seconds))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except urllib.error.HTTPError as exc:
        return _peer_refusal(peer_name, exc)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        return _peer_failure(peer_name, exc)
    if result.get("object") == SEND_RESULT_OBJECT and result.get("result") == "receipt":
        envelope = dict(result)
        envelope.setdefault("peer", peer_name)
        envelope.setdefault("profile", profile)
        return _emit_dm_envelope(args, envelope, session_id=session_id,
                                 idempotency_key=idempotency_key, peer_name=peer_name)
    run_id = str(result.get("run_id") or result.get("delivery_id") or "")
    if not run_id:
        print(f"Peer '{peer_name}' did not return a run ID.", file=sys.stderr)
        return 1
    payload = {
        "peer": peer_name, "profile": profile, "session_id": session_id, "run_id": run_id,
        "status": result.get("status") or "started", "idempotency_key": idempotency_key,
        "replayed": bool(result.get("replayed", False))}
    replay = " (replayed)" if payload["replayed"] else ""
    return _emit(args, payload, [
        f"{run_id}: {payload['status']}{replay}", f"session_id: {session_id}",
        f"idempotency_key: {idempotency_key}"])


def _peer_dm(args, message: str, peer_name: str, profile: str | None, base: str, key: str) -> int:
    wait_seconds = _resolve_wait_seconds(args)
    sender_profile = _sender_profile()
    try:
        session_id = _ensure_bot_chat(base, key)
    except RuntimeError as exc:
        print(f"Peer '{peer_name}': {exc}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _peer_failure(peer_name, exc)
    # The derived key is bound to the resolved session id, so it is minted here
    # rather than before the Bot Chat lookup (§4.1).
    try:
        idempotency_key = _resolve_idempotency_key(
            args, sender_profile=sender_profile, target_profile=profile or sender_profile,
            session_id=session_id, message=message)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    headers = _delivery_headers(idempotency_key, sender_profile, wait_seconds)
    try:
        result = _deliver_with_replay(
            f"{base}/api/sessions/{urllib.parse.quote(session_id, safe='')}/chat", key,
            body=_turn_body(message, message_key="message"), timeout=wait_seconds + _DM_TIMEOUT_SLACK_S,
            headers=headers, idempotency_key=idempotency_key)
    except urllib.error.HTTPError as exc:
        return _peer_refusal(peer_name, exc)
    except _DeliveryOutcomeUnknown:
        return _emit_unknown_delivery(args, peer_name=peer_name, profile=profile,
                                      session_id=session_id, idempotency_key=idempotency_key)
    except RuntimeError as exc:
        print(f"Peer '{peer_name}': {exc}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _peer_failure(peer_name, exc)
    if result.get("object") == SEND_RESULT_OBJECT:
        envelope = dict(result)
        envelope.setdefault("peer", peer_name)
        envelope.setdefault("profile", profile)
        return _emit_dm_envelope(args, envelope, session_id=session_id,
                                 idempotency_key=idempotency_key, peer_name=peer_name)
    # Legacy peer without the delivery envelope: keep the pre-receipt reply shape.
    msg = result.get("message")
    reply = str(msg.get("content") or "") if isinstance(msg, dict) else ""
    payload = {"peer": peer_name, "profile": profile,
               "session_id": result.get("session_id") or session_id, "reply": reply,
               "idempotency_key": idempotency_key, "waited_seconds": float(wait_seconds)}
    return _emit(args, payload, [reply or "(no reply)"])


_REGISTRY_ACTIONS = {
    "add": _peer_add, "set": _peer_add, "remove": _peer_remove, "rm": _peer_remove,
    "list": _peer_list, "ls": _peer_list, None: _peer_list}


def cmd_peer(args) -> int:
    action = getattr(args, "peer_action", None)
    if action in _REGISTRY_ACTIONS:
        return _REGISTRY_ACTIONS[action](args)
    if action not in {"dm", "run", "status", "stop"}:
        print("Unknown peer action. See: hermes peer --help", file=sys.stderr)
        return 2
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
        return _peer_run_ctl(args, action, peer_name, profile, base, key)
    message = _message_from_args(args)
    if not message:
        print("Message required (argument or stdin).", file=sys.stderr)
        return 2
    handler = _peer_run if action == "run" else _peer_dm
    return handler(args, message, peer_name, profile, base, key)


def build_peer_parser(subparsers) -> None:
    """Attach the ``peer`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "peer", help="Bot-to-bot DMs across machines (peer Hermes gateways)",
        description="Register other Hermes gateways as peers and message their agents. "
            "'hermes peer dm <peer>[/<agent>] \"...\"' delivers into the remote "
            "agent's canonical Bot Chat over the peer's API server and prints "
            "the reply — the cross-machine twin of 'hermes -p <bot> chat'. "
            "The peer must run the api_server platform; its API_SERVER_KEY is "
            "stored locally as a credential in ~/.hermes/.env.",
        epilog=(
            "Examples:\n"
            "  hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>\n"
            "  hermes peer list\n"
            '  hermes peer dm spark "Message from 🤖 dixie (@dixie): disk status?"\n'
            '  hermes peer dm spark/researcher "..."   # named profile on a multiplexed peer\n'
            "  hermes peer dm spark --wait 30 \"...\"    # short wait; a busy peer returns a receipt\n"
            "  hermes peer run spark --idempotency-key ticket-123 < long-task.txt\n"
            "  hermes peer status spark run_abc123\n"
            "  hermes peer stop spark run_abc123\n"
            "  hermes peer remove spark\n"
            "\n"
            "Exit codes: 0 delivered or accepted (queued receipt), "
            "1 delivery/peer error, 2 usage or refused request.\n"
            "A receipt means the peer accepted and queued the message — do NOT resend it.\n"
            "A post-retention 'hermes peer status' 404 is reported as unknown, exit 0 — "
            "it is not a delivery failure."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    peer_sub = parser.add_subparsers(dest="peer_action")

    add_p = peer_sub.add_parser("add", aliases=["set"], help="Register (or update) a peer gateway")
    add_p.add_argument("name", help="Peer name (lowercase slug, e.g. spark, homelab)")
    add_p.add_argument("--url", required=True, help="Peer gateway base URL, e.g. http://spark.lan:8377")
    add_p.add_argument("--key", default="", help="The peer's API_SERVER_KEY (stored in ~/.hermes/.env)")
    add_p.add_argument("--note", default="", help="Optional description")

    peer_sub.add_parser("list", aliases=["ls"], help="List registered peers")

    rm_p = peer_sub.add_parser("remove", aliases=["rm"], help="Remove a peer")
    rm_p.add_argument("name", help="Peer name")

    def _remote(name: str, help: str, *, run_id: bool):
        sp = peer_sub.add_parser(name, help=help)
        sp.add_argument("target", help="<peer> or <peer>/<agent> (named profile on a multiplexed peer)")
        if run_id:
            sp.add_argument(
                "run_id",
                help="Run ID or delivery ID returned by 'hermes peer run'/'hermes peer dm'")
        else:
            sp.add_argument("message", nargs="?", default=None, help="Message text (or stdin)")
            sp.add_argument(
                "--idempotency-key", default=None,
                help="Stable retry key; a retry MUST reuse it (derived from the message when omitted)")
            sp.add_argument(
                "--wait", dest="wait_seconds", type=float, default=None,
                help="Seconds to wait for a reply before a queued receipt is returned "
                     f"(default: peer.dm_wait_seconds, {DM_TIMEOUT_S})")
        sp.add_argument("--json", action="store_true", default=False, help="Emit a JSON result")

    _remote("dm", "Message an agent on a peer gateway; returns its reply, else a queued receipt",
            run_id=False)
    _remote("run", "Start a long peer turn asynchronously and return its run ID", run_id=False)
    _remote("status", "Read the status and final output of an asynchronous peer run", run_id=True)
    _remote("stop", "Stop one asynchronous peer run without affecting another turn", run_id=True)

    parser.set_defaults(func=cmd_peer)
