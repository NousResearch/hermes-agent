"""The ``model:`` section's load/save canon: root-level legacy keys, the ``api_base`` alias, a
dict-valued model id, and the model-id key itself. ``hermes_cli.config`` re-exports it and runs it
on every load and on what every save writes."""

from __future__ import annotations

from typing import Any

from hermes_cli.config_providers import _warn_once_per_provider, coerce_provider_id


def _normalize_root_model_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize the ``model`` section at the single load/save chokepoint.
    Root-level ``provider``/``base_url``/``context_length`` (older layouts) are moved under
    ``model`` only when the corresponding ``model.*`` key is empty — never overriding. ``api_base``
    (the OpenAI-SDK/LiteLLM name users reach for) is an alias for ``base_url``; the runtime reads
    only ``model.base_url``. A dict-valued ``default``/``model``/``name`` is flattened so no reader
    sees a nested dict, and the id is canonicalized to ``default``.

    Also aliases ``api_base`` → ``base_url`` (issue #8919). ``api_base`` is the intuitive name OpenAI-SDK /
    LiteLLM users reach for, and ``hermes config set`` blindly accepts any dotted key — so
    ``model.api_base`` got written, confirmed, and then silently ignored by the runtime resolver (which
    reads only ``model.base_url``), causing requests to fall back to OpenRouter. We migrate the alias to the
    canonical key (fallback-only — never override an explicit ``base_url``) and drop the alias so it can't
    confuse later loads.
    Finally, canonicalizes the model-id key to ``model.default`` (issue #34500). The runtime resolver and
    ~14 other readers select the chat model via ``model.default``; ``model.model`` was already aliased
    inline at some sites but ``model.name`` was not, so a custom-provider config like ``model: {name: <id>,
    provider: <custom>}`` resolved to an empty model and the API request went out with ``model=`` (HTTP 400
    from OpenAI-compatible backends) — while display paths (``hermes status``/``dump``) read ``name`` and
    *showed* the model, making the failure silent. Normalizing here (the single load/save chokepoint) means
    every reader, present and future, sees a populated ``default`` and the stale alias is migrated out of
    config.yaml on the next save. Precedence: ``default`` > ``model`` > ``name`` (never overrides an
    explicit ``default``, so existing configs are unaffected).
    """
    model_in = config.get("model")
    model_provider = model_in.get("provider") if isinstance(model_in, dict) else None
    needs_model_work = (model_provider is not None and not isinstance(model_provider, str)) or (
        isinstance(model_in, dict) and (
            model_in.get("api_base")
            or model_in.get("model") or model_in.get("name")
            or any(isinstance(model_in.get(k), dict) for k in ("default", "model", "name"))))
    has_root = any(config.get(k) for k in ("provider", "base_url", "context_length", "api_base"))
    if not has_root and not needs_model_work:
        return config

    config = dict(config)
    model = config.get("model")
    model = dict(model) if isinstance(model, dict) else {"default": model} if model else {}
    config["model"] = model

    # Flatten ``{provider: <p>, model: <m>}``. The nested provider wins over the merged default
    # ``"auto"`` (which runtime resolution treats as authoritative) but never over a configured one.
    # Extra nested keys were never runtime settings. Warn by path before dropping them (#153296),
    # without promoting them into live routing or exposing potentially secret values.
    for _key in ("default", "model", "name"):
        _val = model.get(_key)
        if isinstance(_val, dict):
            _nested_model = _val.get("model") or _val.get("default")
            _nested_provider = str(_val.get("provider") or "").strip()
            model[_key] = str(_nested_model or "").strip()
            if _nested_provider:
                _outer_provider = str(model.get("provider") or "").strip()
                if not _outer_provider or _outer_provider == "auto":
                    model["provider"] = _nested_provider
            for _extra in _val:
                if _extra in ("model", "default", "provider"):
                    continue
                _warn_once_per_provider(
                    "model-section", f"discarded:{_key}.{_extra}",
                    "config.yaml: unsupported model.%s.%s is discarded when normalizing "
                    "the model ID; move it to model.%s if intended", _key, _extra, _extra)

    for key in ("provider", "base_url", "context_length"):
        root_val = config.get(key)
        if root_val and not model.get(key):
            model[key] = root_val
        config.pop(key, None)

    # Provider identity is a string (#117345): an unquoted YAML scalar (``provider: 2``)
    # loads as int, and downstream readers call ``(provider or "").strip()`` — a gateway
    # turn dies before the agent runs. Normalize at the load/save chokepoint so every
    # reader (and the next save, which rewrites config.yaml) heals the persisted value.
    # Guard on presence: coerce_provider_id(None) is "" — injecting an empty key into
    # provider-less configs would add churn to config.yaml on the next save.
    if model.get("provider") is not None:
        model["provider"] = coerce_provider_id(model.get("provider"))

    for alias_val in (config.get("api_base"), model.get("api_base")):
        if alias_val and not model.get("base_url"):
            model["base_url"] = alias_val
    config.pop("api_base", None)
    model.pop("api_base", None)

    # ``model``/``name`` are last-resort aliases (in that order), then dropped.
    alias = model.get("model") or model.get("name")
    if not model.get("default") and alias:
        model["default"] = alias
    if model.get("default"):
        model.pop("model", None)
        model.pop("name", None)

    return config
