"""llama.cpp context-length probes that never load a model.

A llama.cpp router loads whatever model a request names (``--models-autoload``), so a metadata probe that
names an unloaded child pays for a full model load. Only a child ``/models`` reports loaded is asked directly.
"""
from __future__ import annotations

from itertools import pairwise
from typing import Any, Optional
from urllib.parse import quote


def _launch_ctx(child: dict[str, Any]) -> Optional[int]:
    """The ``--ctx-size`` a router child launches with (its ``status.args`` in ``/models``)."""
    args = (child.get("status") or {}).get("args") or []
    for flag, value in pairwise(args):
        if flag in ("--ctx-size", "-c") and str(value).isdigit():
            return int(value) or None  # 0 means "the model's own length", not a known size
    return None


def _router_child(client, server_url: str, model: str) -> Optional[dict[str, Any]]:
    """``model``'s entry in a router's ``/models`` (router entries carry ``status``), else None."""
    resp = client.get(f"{server_url}/models")
    if resp.status_code != 200:
        return None
    for child in (resp.json() or {}).get("data", []):
        if isinstance(child, dict) and isinstance(child.get("status"), dict) and model in (
                child.get("id"), *(child.get("aliases") or [])):
            return child
    return None


def llamacpp_context(client, server_url: str, model: str) -> Optional[int]:
    """llama.cpp n_ctx. A loaded router child answers ``/props?model=`` with its RUNTIME n_ctx; an unloaded one
    answers with the ``--ctx-size`` it will launch with, so a lazily-loaded model doesn't fall to a family
    catch-all. A single-model server answers the bare ``/props``."""
    import httpx

    from agent.model_metadata import _positive_int

    try:
        child = _router_child(client, server_url, model)
    except (httpx.HTTPError, ValueError):
        child = None
    props_paths = ["/props"]
    if child is not None:
        if (child.get("status") or {}).get("value") != "loaded":  # unloaded, loading, sleeping (weights dropped)
            return _launch_ctx(child)
        # autoload=false: should the child unload between the two requests, the router answers 400 instead of
        # loading it again.
        props_paths.insert(0, f"/props?model={quote(model, safe='')}&autoload=false")
    for props_path in props_paths:
        try:
            resp = client.get(f"{server_url}{props_path}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            continue
        n_ctx = _positive_int((resp.json().get("default_generation_settings") or {}).get("n_ctx"))
        if n_ctx is not None:
            return n_ctx
    return None
