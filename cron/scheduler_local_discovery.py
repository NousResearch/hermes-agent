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
model chain came up empty, ask the local server. Policy is upstream's own
``_auto_detect_local_model``, unchanged — including its URL join
(``<base>/v1`` or ``<base>/models`` depending on the /v1 suffix), so the
doubled-``/v1`` URL bug the #20150 review caught cannot recur here: this
module never builds model-list URLs itself.

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


def discover_local_model(
    job_id: str, model_cfg: Any, provider_hint: str = "", explicit_base_url: str = ""
) -> str:
    """Loaded model id from the local inference server, or ``""``.

    Called only when every configured model source is empty; every input is
    therefore best-effort: ``model_cfg`` may be a dict, a string, or junk.
    Base-url precedence: explicit (job-level) > ``model.base_url`` > the
    provider's registry default. Never raises (a broken probe must not
    change the failure the caller is about to produce anyway).
    """
    try:
        cfg = model_cfg if isinstance(model_cfg, dict) else {}
        base_url = (explicit_base_url or str(cfg.get("base_url") or "")).strip()
        provider = provider_hint or str(cfg.get("provider") or "").strip().lower()
        if not base_url and provider in _LOCAL_PROVIDER_DEFAULTS:
            base_url = _provider_registry_base_url(provider)
        if not base_url or not _loopback_url(base_url):
            return ""
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
