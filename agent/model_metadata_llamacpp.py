"""llama.cpp context-window probes (sibling of ``model_metadata``)."""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Optional
from urllib.parse import quote

from agent import model_metadata_http


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


def apply_llamacpp_props(cache: dict[str, dict[str, Any]], request_candidate: str, headers: dict[str, str], verify) -> None:
    """Overwrite ``context_length`` with llama.cpp's allocated ``n_ctx`` from /props (``/v1/props``, then
    ``/props`` for older builds). Callers pass any local or "llamacpp"-branded endpoint: gateways like
    Unsloth Studio speak this shape without carrying the brand. In router mode the bare endpoint 400s,
    so each LOADED child is read via ``/props?model=``; unloaded children are skipped — probing could
    autoload them."""
    base = request_candidate.rstrip("/").replace("/v1", "")
    def _props(params=None):
        resp = model_metadata_http.get(base + "/v1/props", params=params, headers=headers, timeout=5, verify=verify)
        if resp.is_error:
            resp = model_metadata_http.get(base + "/props", params=params, headers=headers, timeout=5, verify=verify)
        return resp
    def _n_ctx(props: dict[str, Any]) -> Any:
        return (props.get("default_generation_settings") or {}).get("n_ctx")
    props_resp = _props()
    if not props_resp.is_error:
        props = props_resp.json()
        n_ctx, model_alias = _n_ctx(props), props.get("model_alias", "")
        if n_ctx and model_alias and model_alias in cache:
            cache[model_alias]["context_length"] = n_ctx
        return
    native = model_metadata_http.get(base + "/models", headers=headers, timeout=5, verify=verify)
    if native.is_error:
        return
    for child in (native.json() or {}).get("data", [])[:16]:
        child_id = child.get("id") if isinstance(child, dict) else None
        if not child_id or child_id not in cache or (child.get("status") or {}).get("value") not in ("loaded", "ready"):
            continue
        pr = _props({"model": child_id, "autoload": "false"})  # it may have unloaded since /models
        child_ctx = _n_ctx(pr.json()) if not pr.is_error else None
        if child_ctx:
            cache[child_id]["context_length"] = child_ctx
