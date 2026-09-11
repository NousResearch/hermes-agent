"""fleet-models — runtime half of the fleet's single model-settings file (``~/.hermes/fleet/models.yaml``).

``core.py`` compiles models.yaml into each profile's config.yaml. Two things Hermes does not do on its own
are closed here, at the plugin seam, so upstream stays untouched:

1. **Auxiliary calls ignore provider_routing.** Compression, titles, vision, the decomposer… build their
   OpenRouter request from ``auxiliary.<task>.extra_body`` only, and a ``fallback_chain`` entry inherits the
   TASK's extra_body (an entry's own ``extra_body`` is never read). So an aux call falling back to v4.1
   carried none of v4.1's host pins, no reasoning pin (v4.1 runs away at the default), and — on tasks
   without an extra_body — not even ``data_collection: deny``. We wrap ``_build_call_kwargs`` so every
   OpenRouter aux request gets: the destination MODEL's ``provider_routing.models.<id>`` pins, its
   ``agent.reasoning_overrides`` level, and ``data_collection: deny`` always (the no-training rule).
   Host pins written for a task's primary model are dropped when the request goes to a different model.
2. **Cron has no chain of its own.** ``cron.scheduler`` uses the profile's main chain for every job; we
   let ``cron.fallback_providers`` (written by the compiler) override it, inside cron.scheduler only.

Bundled + ``kind: backend``: auto-loads in every profile, worker, cron job and the dashboard, which also
mounts the Models tab from ``dashboard/``. Fail-open everywhere — a bug here must never block a call.
"""
from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import sys

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
_MARK = "_fleet_models_wrapped"
HOST_PIN_KEYS = ("only", "order", "ignore", "sort", "require_parameters", "quantizations")
MODEL_SPECIFIC = ("only", "order", "ignore", "quantizations")
ARK_HINTS = ("bytepluses.com",)


def _is_openrouter(provider, base_url) -> bool:
    return str(provider or "").strip().lower() == "openrouter" or "openrouter.ai" in str(base_url or "")


def _task_primary_model(task):
    if not task:
        return None
    try:
        from agent.auxiliary_client import _get_auxiliary_task_config
        return str(_get_auxiliary_task_config(task).get("model") or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def enforce(kwargs: dict, provider, model, base_url=None, task=None, cfg=None) -> dict:
    """Mutate + return aux request *kwargs* per the rules in the module docstring."""
    if not isinstance(kwargs, dict):
        return kwargs
    eb = kwargs.get("extra_body")
    if not _is_openrouter(provider, base_url):
        # OpenRouter preferences mean nothing to ModelArk (or any direct endpoint): never send them there.
        if isinstance(eb, dict) and "provider" in eb and (
                str(provider or "").lower() in ("modelark", "custom:modelark") or any(h in str(base_url or "") for h in ARK_HINTS)):
            eb = {k: v for k, v in eb.items() if k != "provider"}
            if eb:
                kwargs["extra_body"] = eb
            else:
                kwargs.pop("extra_body", None)
        return kwargs
    if cfg is None:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
    from hermes_constants import resolve_per_model_provider_routing, resolve_per_model_reasoning_effort
    pr = cfg.get("provider_routing") if isinstance(cfg.get("provider_routing"), dict) else {}
    per = resolve_per_model_provider_routing(str(model or ""), pr.get("models"))
    eb = dict(eb) if isinstance(eb, dict) else {}
    pv = dict(eb.get("provider")) if isinstance(eb.get("provider"), dict) else {}
    primary = _task_primary_model(task)
    if primary and model and primary != model:
        for k in MODEL_SPECIFIC:
            pv.pop(k, None)
    for k in HOST_PIN_KEYS:
        if k in per:
            v = per[k]
            if v in (None, [], "", False):
                pv.pop(k, None)
            else:
                pv[k] = v
    pv["data_collection"] = "deny"
    eb["provider"] = pv
    agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    pin = resolve_per_model_reasoning_effort(str(model or ""), agent_cfg.get("reasoning_overrides") or {})
    if pin is not None:
        eb["reasoning"] = {"enabled": False} if pin.get("enabled") is False else {"enabled": True, "effort": pin.get("effort") or "medium"}
    kwargs["extra_body"] = eb
    return kwargs


def _wrap_build_call_kwargs(orig):
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def _build_call_kwargs(*args, **kw):
        out = orig(*args, **kw)
        try:
            b = sig.bind_partial(*args, **kw).arguments
            enforce(out, b.get("provider"), b.get("model"), b.get("base_url"), b.get("task"))
        except Exception:  # noqa: BLE001 — fail open: an unenforced call beats a blocked one
            logger.debug("fleet-models: aux routing enforcement skipped", exc_info=True)
        return out
    setattr(_build_call_kwargs, _MARK, True)
    return _build_call_kwargs


def cron_chain_patch(module) -> bool:
    """Make cron.scheduler honour ``cron.fallback_providers`` (else the profile's main chain)."""
    orig = getattr(module, "get_fallback_chain", None)
    if orig is None or getattr(orig, _MARK, False):
        return False

    @functools.wraps(orig)
    def get_fallback_chain(config):
        c = (config or {}).get("cron") if isinstance(config, dict) else None
        own = c.get("fallback_providers") if isinstance(c, dict) else None
        if own is not None:
            return orig({"fallback_providers": own})
        return orig(config)
    setattr(get_fallback_chain, _MARK, True)
    module.get_fallback_chain = get_fallback_chain
    return True


class _AfterImport(importlib.abc.MetaPathFinder):
    """Run *fn(module)* right after *name* is first imported (no eager import in every worker)."""

    def __init__(self, name, fn):
        self.name, self.fn, self.busy = name, fn, False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name or self.busy:
            return None
        self.busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self.busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader, fn = spec.loader, self.fn

        class _Loader(importlib.abc.Loader):
            def create_module(self, s):
                return loader.create_module(s)

            def exec_module(self, module):
                loader.exec_module(module)
                try:
                    fn(module)
                except Exception:  # noqa: BLE001
                    logger.warning("fleet-models: post-import patch of %s failed", fullname, exc_info=True)
        spec.loader = _Loader()
        return spec


def install() -> list:
    """Idempotent. Returns what was installed this call."""
    done = []
    try:
        from agent import auxiliary_client as ac
        if not getattr(ac._build_call_kwargs, _MARK, False):
            ac._build_call_kwargs = _wrap_build_call_kwargs(ac._build_call_kwargs)
            done.append("aux routing")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fleet-models: aux routing seam not installed (%s)", exc)
    mod = sys.modules.get("cron.scheduler")
    if mod is not None:
        if cron_chain_patch(mod):
            done.append("cron chain")
    elif not any(isinstance(f, _AfterImport) and f.name == "cron.scheduler" for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterImport("cron.scheduler", cron_chain_patch))
        done.append("cron chain (on import)")
    return done


def register(ctx) -> None:  # noqa: ARG001 — plugin loader entry point
    logger.info("fleet-models %s: %s", VERSION, install() or "already installed")
