"""``message_agent``'s chain gate (``agent/bot_chain.py``): resolve each target to the identity the
receiving turn's author carries, ask the gate once per call, and dispatch what it admits.

Runs before any transport starts. A reply to the bot whose message started this turn goes through
#144789's notify-only path (``[no reply needed]`` marker + ``wake="none"``); a target that was
definitely not sent hands its ask and hop back, so a retry is neither a duplicate nor a new hop.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any
from urllib.parse import urlparse


def recipient_key(raw_target: str, sender: Any) -> str:
    """The author id a turn started by ``raw_target`` would carry here, resolved in ``_dispatch``'s
    order: a registered peer, then a connection-qualified relay row, then a local bot (folder id,
    ``hermes``, friendly name), then any relay row."""
    from agent.turn_author import bot_author_id
    from hermes_agent.tools.bot_mode_dm import _PEER_TARGET_RE, _resolve_local_name

    name = raw_target.strip().lstrip("@")
    peer = _PEER_TARGET_RE.match(name)
    if peer or name.lower() in sender.peers:
        peer_name, peer_profile = peer.groups() if peer else (name.lower(), None)
        return _peer_key(peer_name, peer_profile or "default", sender)
    row = _relay_row(name, sender) if "@" in name else None
    local = None if row is not None else _resolve_local_name(name, list(sender.roster_homes), sender.root)
    if local is not None and local != sender.me:
        return bot_author_id(local)
    row = row or _relay_row(name, sender)
    if row is not None:
        return bot_author_id(row["profile"], row["connection_id"])
    return f"unresolved:{name.lower()}"


def _relay_row(name: str, sender: Any) -> dict | None:
    from pathlib import Path

    from hermes_agent.tools.bot_relay import read_remote_roster, resolve_remote_target

    roster = read_remote_roster(Path(sender.home))
    match = resolve_remote_target(name, roster) if roster else None
    return match if isinstance(match, dict) else None


def _peer_key(peer_name: str, peer_profile: str, sender: Any) -> str:
    """A peer's dm reaches us as ``bot:<its hostname>/<profile>``: when this turn's author is that bot
    (same profile, hostname matching the registered peer URL's host) the target is that identity;
    otherwise the peer name keeps two peers' same-named agents apart."""
    author = getattr(sender.agent, "_turn_author", None) or {}
    origin, _, profile = str(author.get("id") or "").removeprefix("bot:").rpartition("/")
    if author.get("is_bot") and origin and profile.lower() == peer_profile.lower():
        host = (urlparse(str(_peer_url(peer_name, sender))).hostname or "").lower()
        if host and origin.lower() in {host, host.split(".", 1)[0]}:
            return str(author["id"])
    return f"peer:{peer_name}/{peer_profile}".lower()


def _peer_url(peer_name: str, sender: Any) -> str:
    from hermes_cli.config_backend import read_config_doc_readonly

    doc = read_config_doc_readonly(sender.root / "config.yaml")
    peers = doc.get("bot_peers") if isinstance(doc, dict) else None
    entry = peers.get(peer_name) if isinstance(peers, dict) else None
    return str(entry.get("url") or "") if isinstance(entry, dict) else ""


def _definitely_not_sent(ack: dict) -> bool:
    """A refusal before anything left: an error with no in-flight status (``queued``/``ambiguous``
    keep their reservation, since the other side may already be answering)."""
    return bool(ack.get("error")) and ack.get("status") in (None, "failed")


def send_gated(targets: list[str], reply: bool, body: str, content_for, sender: Any) -> str:
    """Gate one ``message_agent`` call, then dispatch: the shared transports keep their own rules."""
    from agent.bot_chain import admit_send
    from hermes_agent.tools.bot_mode_dm import _dispatch, _err
    from hermes_agent.tools.bot_mode_dm_coordination import fan_out

    keys = [recipient_key(target, sender) for target in targets]
    admission = admit_send(sender.agent, keys, body)
    if admission.refusal:
        return _err(admission.refusal)
    sender = dataclasses.replace(sender, chain=admission.envelope())
    refused = [(t, k) for t, k in zip(targets, keys) if k in admission.refused]
    notify_only = [(t, k) for t, k in zip(targets, keys) if k in admission.notify_only and reply]
    normal = [(t, k) for t, k in zip(targets, keys) if (t, k) not in refused and (t, k) not in notify_only]
    if len(targets) == 1 and refused:
        return _err(admission.refused[keys[0]])
    acks: list[tuple[str, dict]] = []
    if len(targets) == 1:
        target, key = (notify_only or normal)[0]
        wake = "none" if notify_only or not reply else "notify"
        raw = _dispatch(target, content_for(wake != "none"), sender, wake=wake)
        _release_failed(admission, [(key, json.loads(raw))])
        return raw
    for target, key in notify_only:
        acks.append((key, {"to": f"@{target}", **json.loads(
            _dispatch(target, content_for(False), sender, wake="none"))}))
    combined = json.loads(fan_out([t for t, _k in normal], content_for(reply), sender, expects_reply=reply)) \
        if normal else {"status": "queued", "fan_out": []}
    acks += list(zip([k for _t, k in normal], combined.get("fan_out", [])))
    _release_failed(admission, acks)
    combined["fan_out"] = ([{"to": f"@{t}", "error": admission.refused[k]} for t, k in refused]
                           + [ack for _k, ack in acks])
    return json.dumps(combined)


def _release_failed(admission: Any, acks: list[tuple[str, dict]]) -> None:
    for key, ack in acks:
        if _definitely_not_sent(ack):
            admission.release(key)
