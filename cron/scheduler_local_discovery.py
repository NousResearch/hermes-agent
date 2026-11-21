"""Unpinned-job local-model discovery for cron (issue #20125).

An agent-backed cron job with no model anywhere in its resolution chain
(job pin > ``cron.model`` > main agent model) currently fails fast with
"No model configured". But when the resolved provider is a local inference
server (LM Studio, llama.cpp server, Ollama, LM Studio-compatible
gateways on loopback), the server itself knows which model is loaded —
and the interactive side already asks it: ``_get_model_config`` fills an
empty model from a loopback base_url via ``_auto_detect_local_model``
(runtime_provider.py), and the CLI init and model-switch surfaces call it
directly. Cron was the only surface that never did, so a user who swaps
models on a local server without ever pinning one watches every unpinned
job fail with an actionable-but-avoidable error.

This module gives ``_load_cron_job_config`` the same final step: when the
model chain came up empty, ask the local server. Two probes, native first:

1. LM Studio's native ``/api/v1/models`` — per-model ``loaded_instances`` /
   ``state`` make it the only endpoint that can answer "which model is
   LOADED" when several are downloaded (JIT loading lists every downloaded
   model on the compat endpoint, loaded or not). Exactly one loaded model
   wins; several are ambiguous. The URL root strips known ``/v1`` /
   ``/api`` suffixes so the doubled-``/v1`` URL bug the #20150 review
   caught cannot recur.
2. Upstream's compat ``_auto_detect_local_model`` — unchanged, and still
   the answer for servers without the native endpoint (single-model
   llama.cpp / Ollama setups).

- probe ONLY loopback endpoints (localhost / 127.0.0.1) — a cron
  tick must never make a surprise network call to a remote host;
- accept the answer ONLY when exactly one model is loaded — an ambiguous
  answer (several models listed, e.g. Ollama) returns "" and the job
  fails fast with the existing message instead of guessing;
- any failure (server down, timeout, bad body) is a debug log and "",
  never an exception — the fail-fast raise in the caller stays the
  single error path.

Where the base_url comes from: an explicit ``model.base_url`` when the
model config carries one, otherwise the resolved provider's registry
default (so a bare ``provider: lmstudio`` config with no URL still
probes the LM Studio default endpoint).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCAL_PROVIDER_DEFAULTS = ("lmstudio", "ollama", "llama.cpp", "vllm", "local")


def _loopback_url(url: str) -> bool:
    try:
        from utils import base_url_hostname

        return base_url_hostname(url) in ("localhost", "127.0.0.1")
    except Exception:
        return False


def _provider_registry_base_url(provider: str) -> str:
    """Registry-default endpoint for a *local* provider, else ``""``."""
    if not provider:
        return ""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get(str(provider).strip().lower())
        if pconfig is None:
            return ""
        default_url = str(getattr(pconfig, "inference_base_url", "") or "").strip()
        return default_url if default_url and _loopback_url(default_url) else ""
    except Exception:
        return ""


def _lmstudio_server_root(url: str) -> str:
    """Server root for LM Studio's native ``/api/v1`` endpoints (known suffixes stripped)."""
    root = (url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)].rstrip("/")
    return root


def _lmstudio_loaded_model(base_url: str) -> str:
    """Model id when LM Studio's native endpoint shows EXACTLY ONE loaded instance, else ``""``.

    The OpenAI-compat ``/v1/models`` also lists downloaded-but-unloaded models when JIT
    loading is on, so a non-empty compat answer cannot answer "what is loaded" (observed
    in practice 2026-10-29: a server with models installed but none loaded sailed through
    a compat-shaped probe, and the run died with "HTTP 400: No models loaded").
    The native ``/api/v1/models`` envelope carries per-model ``loaded_instances`` /
    ``state``; exactly one loaded model is an unambiguous answer, several loaded are as
    ambiguous as any multi-model server. Missing endpoint (older LM Studio builds,
    Ollama, llama.cpp), zero or several loaded, and every failure return ``""`` — the
    caller falls through to the compat heuristic. Never raises.
    """
    root = _lmstudio_server_root(base_url)
    if not root:
        return ""
    try:
        import requests

        resp = requests.get(root + "/api/v1/models", timeout=(2, 3))
        if not resp.ok:
            return ""
        models = resp.json().get("models", [])
        if not isinstance(models, list):
            return ""
        loaded = []
        for model in models:
            if not isinstance(model, dict):
                continue
            instances = model.get("loaded_instances")
            is_loaded = model.get("state") == "loaded" or (
                isinstance(instances, list)
                and any(isinstance(i, dict) for i in instances)
            )
            if not is_loaded:
                continue
            model_id = model.get("key") or model.get("id")
            if isinstance(model_id, str) and model_id:
                loaded.append(model_id)
        return loaded[0] if len(loaded) == 1 else ""
    except Exception:
        logger.debug("native LM Studio probe against %s failed", root, exc_info=True)
        return ""


def discover_local_model(
    job_id: str, model_cfg: Any, provider_hint: str = "", explicit_base_url: str = ""
) -> str:
    """Loaded model id from the local inference server, or ``""``.

    Called only when every configured model source is empty; every input is
    therefore best-effort: ``model_cfg`` may be a dict, a string, or junk.
    Base-url precedence: explicit (job-level) > ``model.base_url`` > the
    provider's registry default. The LM Studio native loaded-instance endpoint
    is asked first; upstream's compat single-model heuristic is the fallback
    (and the answer for servers without the native endpoint). Never raises (a
    broken probe must not change the failure the caller is about to produce
    anyway).
    """
    try:
        cfg = model_cfg if isinstance(model_cfg, dict) else {}
        base_url = (explicit_base_url or str(cfg.get("base_url") or "")).strip()
        provider = provider_hint or str(cfg.get("provider") or "").strip().lower()
        if not base_url and provider in _LOCAL_PROVIDER_DEFAULTS:
            base_url = _provider_registry_base_url(provider)
        if not base_url or not _loopback_url(base_url):
            return ""
        discovered = _lmstudio_loaded_model(base_url)
        if not discovered:
            from hermes_cli.runtime_provider import _auto_detect_local_model

            discovered = _auto_detect_local_model(base_url)
        if discovered:
            logger.info(
                "Job '%s': no model configured; auto-discovered '%s' from local server %s "
                "(pin a model with `hermes cron edit %s --model <name>` to stop relying on this)",
                job_id,
                discovered,
                base_url,
                job_id,
            )
        else:
            logger.debug(
                "Job '%s': no model configured; local server %s offered no unambiguous model",
                job_id,
                base_url,
            )
        return discovered
    except Exception:
        logger.debug("Job '%s': local model discovery errored", job_id, exc_info=True)
        return ""
