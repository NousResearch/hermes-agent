"""llama.cpp context-window probes (sibling of ``model_metadata``)."""

from __future__ import annotations

from itertools import pairwise
from typing import Optional
from urllib.parse import quote


def _json_object(resp) -> dict:
    """The response body when it is a JSON object, else an empty dict."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def llamacpp_context(client, server_url: str, model: str) -> Optional[int]:
    """llama.cpp's RUNTIME n_ctx: /props for a loaded model, else the ``--ctx-size`` the router will
    launch it with (its /models entry), so a lazily-loaded model doesn't fall to a family catch-all.
    The /props read passes autoload=false: the router loads any model a request names, and a context
    probe must not undo an Eject or an idle unload."""
    import httpx

    from agent.model_metadata import _positive_int

    for props_path in (f"/props?model={quote(model)}&autoload=false", "/props"):
        try:
            resp = client.get(f"{server_url}{props_path}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            continue
        settings = _json_object(resp).get("default_generation_settings")
        n_ctx = _positive_int(settings.get("n_ctx")) if isinstance(settings, dict) else None
        if n_ctx is not None:
            return n_ctx
    return router_launch_ctx(client, server_url, model)


def router_launch_ctx(client, server_url: str, model: str) -> Optional[int]:
    """``--ctx-size`` from the router's launch args for *model* (GET /models), or None."""
    import httpx

    from agent.model_metadata import _model_id_matches

    try:
        resp = client.get(f"{server_url}/models")
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    entries = _json_object(resp).get("data")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not _model_id_matches(str(entry.get("id", "")), model):
            continue
        status = entry.get("status")
        raw_args = status.get("args") if isinstance(status, dict) else None
        args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
        for flag, value in pairwise(args):
            if flag in ("--ctx-size", "-c") and value.isdigit() and int(value) > 0:
                return int(value)
    return None
