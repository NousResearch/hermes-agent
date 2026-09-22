#!/usr/bin/env python3
"""Publish a read-only, idempotent operational projection to a Discord channel.

The JSON passed to this script is a snapshot produced from GitHub, Trello and a
#demandas post.  It is deliberately not a command surface: Discord receives a
rendered view and the local state file stores only Discord message identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DISCORD_API = "https://discord.com/api/v10"
DEFAULT_CHANNEL_ID = "1552030258872586322"
MODELS = frozenset({"index", "card", "transition", "waiting_pedro", "closing"})


class ProjectionError(RuntimeError):
    """A failed Discord projection request."""


@dataclass(frozen=True)
class Projection:
    key: str
    content: str


def _markdown_link(label: str, url: object) -> str:
    text = str(url or "").strip()
    return f"[{label}](<{text}>)" if text else f"{label}: evidência indisponível"


def _links(raw: object) -> str:
    links = raw if isinstance(raw, dict) else {}
    return " · ".join((
        _markdown_link("GitHub", links.get("github")),
        _markdown_link("Trello", links.get("trello")),
        _markdown_link("Demanda", links.get("thread")),
    ))


def _epic_lines(epic: dict[str, Any]) -> list[str]:
    fronts = epic.get("fronts") if isinstance(epic.get("fronts"), list) else []
    prs = epic.get("prs") if isinstance(epic.get("prs"), list) else []
    lines = [
        f"**Épica #{epic['id']} — {epic['title']}**",
        f"Estado: **{epic['state']}**",
        f"Objetivo: {epic['objective']}",
        f"Última transição: {epic['last_transition']}",
        _links(epic.get("links")),
    ]
    if fronts:
        lines.append("Frentes: " + ", ".join(str(front) for front in fronts))
    if prs:
        lines.append("PRs: " + ", ".join(str(pr) for pr in prs))
    return lines


def render_projection(raw: dict[str, Any]) -> Projection:
    """Render one of the five UX-approved read-only models.

    Required epic fields make a missing operational datum explicit instead of
    silently inventing a link or a state.
    """
    model = str(raw.get("model", "")).strip()
    if model not in MODELS:
        raise ProjectionError(f"model must be one of: {', '.join(sorted(MODELS))}")

    if model == "index":
        epics = raw.get("epics")
        if not isinstance(epics, list) or not epics:
            raise ProjectionError("index requires a non-empty epics list")
        lines = ["# Pipeline operacional", "Leitura de GitHub, Trello e #demandas."]
        for epic in epics:
            if not isinstance(epic, dict):
                raise ProjectionError("each index epic must be an object")
            for field in ("id", "title", "state"):
                if not str(epic.get(field, "")).strip():
                    raise ProjectionError(f"index epic requires {field}")
            lines.append(f"· [#{epic['id']}] {epic['title']} — **{epic['state']}**")
        return Projection("index", "\n".join(lines))

    for field in ("id", "title", "state", "objective", "last_transition"):
        if not str(raw.get(field, "")).strip():
            raise ProjectionError(f"{model} requires {field}")
    epic_id = str(raw["id"])
    lines = _epic_lines(raw)
    if model == "transition":
        lines.insert(0, "## Transição operacional")
    elif model == "waiting_pedro":
        lines.insert(0, "## Aguardando Pedro")
        lines.append("Ação de Pedro: " + str(raw.get("pedro_action") or "evidência indisponível"))
    elif model == "closing":
        lines.insert(0, "## Fechamento")
        lines.append("Resultado: " + str(raw.get("result") or "evidência indisponível"))
    else:
        lines.insert(0, "## Cartão de épica")
    return Projection(f"epic:{epic_id}", "\n".join(lines))


def load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProjectionError(f"invalid projection state: {exc}") from exc
    if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise ProjectionError("projection state must be an object of string keys and message IDs")
    return data


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(state, handle, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def discord_request(method: str, route: str, token: str, payload: dict[str, Any] | None, opener: Callable[..., Any] = urlopen) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{DISCORD_API}{route}", body, method=method,
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=20) as response:
            data = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProjectionError(f"Discord {method} {route} failed: HTTP {exc.code} {detail}") from exc
    except OSError as exc:
        raise ProjectionError(f"Discord {method} {route} failed: {exc}") from exc
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProjectionError(f"Discord {method} {route} returned non-JSON") from exc
    if not isinstance(decoded, dict):
        raise ProjectionError(f"Discord {method} {route} returned an unexpected payload")
    return decoded


def publish(channel_id: str, projection: Projection, state_path: Path, token: str, opener: Callable[..., Any] = urlopen) -> tuple[str, str]:
    """Create once then update the same projection message on every later run."""
    state = load_state(state_path)
    message_id = state.get(projection.key)
    if message_id:
        message = discord_request("PATCH", f"/channels/{channel_id}/messages/{message_id}", token, {"content": projection.content}, opener)
        action = "edited"
    else:
        message = discord_request("POST", f"/channels/{channel_id}/messages", token, {"content": projection.content}, opener)
        action = "created"
    resolved_id = str(message.get("id", "")).strip()
    if not resolved_id:
        raise ProjectionError("Discord response did not include a message ID")
    state[projection.key] = resolved_id
    save_state(state_path, state)
    return action, resolved_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="JSON snapshot from GitHub/Trello/#demandas")
    parser.add_argument("--channel-id", default=DEFAULT_CHANNEL_ID, help="Discord #pipeline channel ID")
    parser.add_argument("--state-file", type=Path, required=True, help="Projection message-ID ledger (not a work-state store)")
    parser.add_argument("--token-env", default="DISCORD_BOT_TOKEN", help="environment variable holding the existing bot token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise ProjectionError(f"missing Discord credential in {args.token_env}")
    try:
        raw = json.loads(args.snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"invalid snapshot: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProjectionError("snapshot must be a JSON object")
    action, message_id = publish(str(args.channel_id), render_projection(raw), args.state_file, token)
    print(json.dumps({"action": action, "message_id": message_id, "channel_id": str(args.channel_id)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
