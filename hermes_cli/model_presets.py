"""Static expansion of named model routes in configuration.

Model presets are authoring conveniences, not runtime providers: every consumer receives
an ordinary inline route before provider and credential resolution begins.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_ROUTE_KEYS = frozenset({"provider", "model", "reasoning_effort", "fallbacks"})
_MAIN_FORBIDDEN_REFERENCE_FIELDS = frozenset({
    # Presets select provider/model only. Credentials and endpoints are provider-owned;
    # accepting them here would silently discard a user setting during expansion.
    "base_url", "api_base", "api_key", "api", "key_env", "api_key_env", "api_mode", "transport",
})
_EMPTY_DEFAULT_ROUTE_FIELDS = _MAIN_FORBIDDEN_REFERENCE_FIELDS | frozenset({"extra_body"})


class ModelPresetError(ValueError):
    """A model preset cannot be expanded safely."""


def _error(path: str, message: str) -> ModelPresetError:
    return ModelPresetError(f"model_presets: {path}: {message}")


def _text(value: Any, path: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, f"'{field}' must be a non-empty string")
    return value.strip()


def _validate_reasoning(value: Any, path: str) -> None:
    if value is None:
        return
    from hermes_constants import parse_reasoning_effort
    if parse_reasoning_effort(value) is None:
        raise _error(path, "'reasoning_effort' must be one of none, minimal, low, medium, high, xhigh, max, ultra")


def _validate_route(value: Any, path: str, *, allow_fallbacks: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(path, "must be a mapping")
    unknown = set(value) - _ROUTE_KEYS
    if unknown:
        raise _error(path, "unsupported field(s): " + ", ".join(sorted(map(str, unknown))))
    route: dict[str, Any] = {
        "provider": _text(value.get("provider"), path, "provider"),
        "model": _text(value.get("model"), path, "model"),
    }
    if "reasoning_effort" in value:
        _validate_reasoning(value["reasoning_effort"], path)
        route["reasoning_effort"] = value["reasoning_effort"]
    if "fallbacks" in value:
        if not allow_fallbacks:
            raise _error(path, "fallbacks are not supported in a fallback route")
        fallbacks = value["fallbacks"]
        if not isinstance(fallbacks, list):
            raise _error(path, "'fallbacks' must be an ordered list of inline routes")
        route["fallbacks"] = [
            _validate_route(item, f"{path}.fallbacks[{index}]", allow_fallbacks=False)
            for index, item in enumerate(fallbacks)
        ]
    return route


def _definitions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = config.get("model_presets")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _error("model_presets", "must be a mapping of names to flat route definitions")
    definitions: dict[str, dict[str, Any]] = {}
    for name, route in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise _error("model_presets", "preset names must be non-empty strings")
        clean_name = name.strip()
        if clean_name in definitions:
            raise _error("model_presets", f"duplicate preset name '{clean_name}'")
        definitions[clean_name] = _validate_route(route, f"model_presets.{clean_name}", allow_fallbacks=True)
    return definitions


def _reference(site: Any, path: str, definitions: dict[str, dict[str, Any]], *, conflicts: set[str]) -> dict[str, Any] | None:
    if not isinstance(site, dict) or "model_preset" not in site:
        return None
    name = _text(site["model_preset"], path, "model_preset")
    inline = sorted(str(key) for key in set(site).intersection(conflicts))
    if inline:
        raise _error(path, f"'model_preset: {name}' cannot be combined with inline route field(s): {', '.join(inline)}")
    route = definitions.get(name)
    if route is None:
        available = ", ".join(definitions) or "(none)"
        raise _error(path, f"unknown preset '{name}'. Available presets: {available}")
    return deepcopy(route)


def _fallbacks_disabled(site: Any, path: str) -> bool:
    """The sole reference-site override: ``fallbacks: []`` disables a preset chain."""
    if not isinstance(site, dict) or "model_preset" not in site or "fallbacks" not in site:
        return False
    if site["fallbacks"] == []:
        return True
    raise _error(path, "'fallbacks' may only be [] with model_preset (to disable preset fallbacks)")


def _route_fields(route: dict[str, Any], *, fallback_key: str | None = None) -> dict[str, Any]:
    result = {key: deepcopy(route[key]) for key in ("provider", "model", "reasoning_effort") if key in route}
    if "fallbacks" in route:
        if fallback_key is None:
            raise _error("route", "fallbacks are unsupported at this reference site")
        result[fallback_key] = deepcopy(route["fallbacks"])
    return result


def _expand_model(config: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        return
    disabled_fallbacks = _fallbacks_disabled(model, "model")
    route = _reference(
        model, "model", definitions,
        conflicts={"provider", "default", "model", "reasoning_effort", *_MAIN_FORBIDDEN_REFERENCE_FIELDS},
    )
    if route is None:
        return
    # Preserve non-route model metadata such as context_length.
    config["model"] = {
        **{key: deepcopy(value) for key, value in model.items() if key not in {"model_preset", "fallbacks"}},
        "provider": route["provider"], "default": route["model"],
    }
    agent = config.get("agent")
    if "reasoning_effort" in route:
        if isinstance(agent, dict) and "reasoning_effort" in agent:
            raise _error("model", "preset reasoning_effort cannot be combined with agent.reasoning_effort")
        if not isinstance(agent, dict):
            agent = {}
            config["agent"] = agent
        agent["reasoning_effort"] = route["reasoning_effort"]
    if disabled_fallbacks:
        # Explicit opt-out must survive raw gateway loads and default/overlay inheritance.
        config["fallback_providers"] = []
    elif "fallbacks" in route:
        if "fallback_providers" in config:
            raise _error("model", "preset fallbacks cannot be combined with top-level fallback_providers (including an empty list)")
        config["fallback_providers"] = deepcopy(route["fallbacks"])


def _expand_delegation(config: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> None:
    site = config.get("delegation")
    disabled_fallbacks = _fallbacks_disabled(site, "delegation")
    route = _reference(site, "delegation", definitions, conflicts={"provider", "model", "base_url", "api_key", "reasoning_effort", "fallback_providers", "fallback_chain"})
    if route is not None:
        assert isinstance(site, dict)
        fields = _route_fields({key: value for key, value in route.items() if key != "fallbacks"} if disabled_fallbacks else route, fallback_key="fallback_providers")
        if disabled_fallbacks:
            fields["fallback_providers"] = []
        config["delegation"] = {**{key: value for key, value in site.items() if key not in {"model_preset", "fallbacks"}}, **fields}


def _expand_auxiliary(config: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> None:
    auxiliary = config.get("auxiliary")
    if not isinstance(auxiliary, dict):
        return
    for task, site in list(auxiliary.items()):
        path = f"auxiliary.{task}"
        disabled_fallbacks = _fallbacks_disabled(site, path)
        route = _reference(site, path, definitions, conflicts={"provider", "model", "base_url", "api_key", "api_mode", "reasoning_effort", "fallback_chain", "fallback_providers"})
        if route is not None:
            assert isinstance(site, dict)
            fields = _route_fields({key: value for key, value in route.items() if key != "fallbacks"} if disabled_fallbacks else route, fallback_key="fallback_chain")
            if disabled_fallbacks:
                fields["fallback_chain"] = []
            auxiliary[task] = {**{key: value for key, value in site.items() if key not in {"model_preset", "fallbacks"}}, **fields}


def _expand_fallback_entries(config: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> None:
    for key in ("fallback_providers", "fallback_model"):
        raw = config.get(key)
        entries = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else None
        if entries is None:
            continue
        expanded = []
        for index, entry in enumerate(entries):
            route = _reference(entry, f"{key}[{index}]", definitions, conflicts={"provider", "model", "base_url", "api_key", "key_env", "api_key_env", "reasoning_effort", "fallbacks", "fallback_chain", "fallback_providers"})
            if route is not None:
                if "fallbacks" in route:
                    raise _error(f"{key}[{index}]", "a fallback entry cannot reference a preset that declares fallbacks")
                expanded.append(_route_fields(route))
            else:
                expanded.append(entry)
        config[key] = expanded if isinstance(raw, list) else expanded[0]


def _expand_moa_slot(site: Any, path: str, definitions: dict[str, dict[str, Any]]) -> Any:
    route = _reference(site, path, definitions, conflicts={"provider", "model", "reasoning_effort", "fallbacks"})
    if route is None:
        return site
    if "fallbacks" in route:
        raise _error(path, "a MoA slot cannot reference a preset that declares fallbacks")
    return {**{key: value for key, value in site.items() if key != "model_preset"}, **_route_fields(route)}


def _expand_moa(config: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> None:
    moa = config.get("moa")
    if not isinstance(moa, dict):
        return
    named_presets = isinstance(moa.get("presets"), dict)
    blocks = moa["presets"] if named_presets else {"default": moa}
    for name, block in blocks.items():
        if not isinstance(block, dict):
            continue
        prefix = f"moa.presets.{name}" if named_presets else "moa"
        refs = block.get("reference_models")
        if isinstance(refs, list):
            block["reference_models"] = [_expand_moa_slot(slot, f"{prefix}.reference_models[{index}]", definitions) for index, slot in enumerate(refs)]
        if "aggregator" in block:
            block["aggregator"] = _expand_moa_slot(block["aggregator"], f"{prefix}.aggregator", definitions)


def expand_model_presets(config: Any) -> dict[str, Any]:
    """Return a copied config with every supported ``model_preset`` reference expanded.

    The source mapping is never mutated so callers that save configuration retain the user's
    named references rather than a flattened implementation detail.
    """
    if not isinstance(config, dict):
        return config
    expanded = deepcopy(config)
    definitions = _definitions(expanded)
    _expand_model(expanded, definitions)
    _expand_delegation(expanded, definitions)
    _expand_auxiliary(expanded, definitions)
    _expand_fallback_entries(expanded, definitions)
    _expand_moa(expanded, definitions)
    return expanded


def preserve_model_preset_references(config: dict[str, Any], authored: Any) -> dict[str, Any]:
    """Put unchanged named-route references back before a config write.

    ``load_config`` returns runtime-ready inline routes.  An unrelated config edit must not
    turn those routes into permanently flattened YAML, but an edited route must remain edited.
    """
    if not isinstance(authored, dict) or not authored.get("model_presets"):
        return config
    expanded_authored = expand_model_presets(authored)
    result = deepcopy(config)

    def same_route(actual: Any, expected: Any, fallback_key: str | None = None) -> bool:
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
        keys = {"provider", "model", "reasoning_effort"} | _EMPTY_DEFAULT_ROUTE_FIELDS
        def value(site: dict[str, Any], key: str) -> Any:
            item = site.get(key)
            return None if item in (None, "", {}, []) else item
        if any(value(actual, key) != value(expected, key) for key in keys):
            return False
        return fallback_key is None or value(actual, fallback_key) == value(expected, fallback_key)

    raw_model = authored.get("model")
    expected_model = expanded_authored.get("model")
    if isinstance(raw_model, dict) and "model_preset" in raw_model and isinstance(expected_model, dict):
        actual_model = result.get("model")
        unsafe_model_fields = any(
            key in actual_model and key not in raw_model and actual_model[key] not in (None, "", {}, [])
            for key in _MAIN_FORBIDDEN_REFERENCE_FIELDS
        ) if isinstance(actual_model, dict) else True
        preset = _definitions(authored)[raw_model["model_preset"].strip()]
        reasoning_unchanged = (
            "reasoning_effort" not in preset
            or (result.get("agent") or {}).get("reasoning_effort")
            == (expanded_authored.get("agent") or {}).get("reasoning_effort")
        )
        fallbacks_unchanged = (
            "fallbacks" not in preset and "fallbacks" not in raw_model
            or result.get("fallback_providers") == expanded_authored.get("fallback_providers")
        )
        if (isinstance(actual_model, dict) and not unsafe_model_fields
                and reasoning_unchanged and fallbacks_unchanged
                and actual_model.get("provider") == expected_model.get("provider") and actual_model.get("default") == expected_model.get("default")):
            restored_model = deepcopy(actual_model)
            restored_model.pop("provider", None)
            restored_model.pop("default", None)
            for key in _MAIN_FORBIDDEN_REFERENCE_FIELDS:
                if key not in raw_model and restored_model.get(key) in (None, "", {}, []):
                    restored_model.pop(key, None)
            restored_model["model_preset"] = raw_model["model_preset"]
            if raw_model.get("fallbacks") == []:
                restored_model["fallbacks"] = []
            result["model"] = restored_model
            expected_agent = (expanded_authored.get("agent") or {}).get("reasoning_effort")
            raw_agent = authored.get("agent")
            if (expected_agent is not None and not (isinstance(raw_agent, dict) and "reasoning_effort" in raw_agent)
                    and isinstance(result.get("agent"), dict) and result["agent"].get("reasoning_effort") == expected_agent):
                result["agent"].pop("reasoning_effort", None)
            if (("fallbacks" in preset or "fallbacks" in raw_model)
                    and "fallback_providers" not in authored
                    and result.get("fallback_providers") == expanded_authored.get("fallback_providers")):
                result.pop("fallback_providers", None)

    def restore_site(container: dict[str, Any], raw_container: Any, expected_container: Any, key: str, fallback_key: str | None = None) -> None:
        raw_site = raw_container.get(key) if isinstance(raw_container, dict) else None
        expected_site = expected_container.get(key) if isinstance(expected_container, dict) else None
        actual_site = container.get(key)
        if isinstance(raw_site, dict) and "model_preset" in raw_site and same_route(actual_site, expected_site, fallback_key):
            assert isinstance(actual_site, dict)
            # Keep edits to unrelated site settings (timeouts, concurrency, MoA enabled),
            # but remove only fields injected by expansion before restoring the reference.
            restored = deepcopy(actual_site)
            for route_key in ("provider", "model", "reasoning_effort"):
                if isinstance(expected_site, dict) and route_key in expected_site:
                    restored.pop(route_key, None)
            if fallback_key and isinstance(expected_site, dict) and fallback_key in expected_site:
                restored.pop(fallback_key, None)
            # strip_defaults=False exposes empty provider-owned defaults which would otherwise
            # become inline route fields and make a saved reference fail on its next load.
            for route_key in _EMPTY_DEFAULT_ROUTE_FIELDS:
                if route_key not in raw_site and restored.get(route_key) in (None, "", {}, []):
                    restored.pop(route_key, None)
            restored["model_preset"] = raw_site["model_preset"]
            if raw_site.get("fallbacks") == []:
                restored["fallbacks"] = []
            container[key] = restored

    restore_site(result, authored, expanded_authored, "delegation", "fallback_providers")
    actual = result.get("auxiliary")
    raw = authored.get("auxiliary")
    expected = expanded_authored.get("auxiliary")
    if isinstance(actual, dict) and isinstance(raw, dict):
        for key in raw:
            restore_site(actual, raw, expected, key, "fallback_chain")

    def restore_slot(actual_slot: Any, raw_slot: Any, expected_slot: Any) -> Any:
        if not (isinstance(actual_slot, dict) and isinstance(raw_slot, dict) and "model_preset" in raw_slot):
            return actual_slot
        if not same_route(actual_slot, expected_slot):
            return actual_slot
        restored = deepcopy(actual_slot)
        for route_key in ("provider", "model", "reasoning_effort"):
            if isinstance(expected_slot, dict) and route_key in expected_slot:
                restored.pop(route_key, None)
        restored["model_preset"] = raw_slot["model_preset"]
        return restored

    for fallback_key in ("fallback_providers", "fallback_model"):
        raw_entries = authored.get(fallback_key)
        actual_entries = result.get(fallback_key)
        expected_entries = expanded_authored.get(fallback_key)
        raw_list = raw_entries if isinstance(raw_entries, list) else [raw_entries] if isinstance(raw_entries, dict) else None
        actual_list = actual_entries if isinstance(actual_entries, list) else [actual_entries] if isinstance(actual_entries, dict) else None
        expected_list = expected_entries if isinstance(expected_entries, list) else [expected_entries] if isinstance(expected_entries, dict) else None
        if raw_list is not None and actual_list is not None and expected_list is not None and len(raw_list) == len(actual_list) == len(expected_list):
            restored = [restore_slot(actual, raw, expected) for actual, raw, expected in zip(actual_list, raw_list, expected_list)]
            result[fallback_key] = restored if isinstance(actual_entries, list) else restored[0]

    raw_moa, actual_moa, expected_moa = authored.get("moa"), result.get("moa"), expanded_authored.get("moa")
    if isinstance(raw_moa, dict) and isinstance(actual_moa, dict) and isinstance(expected_moa, dict):
        raw_blocks = raw_moa.get("presets") if isinstance(raw_moa.get("presets"), dict) else {"default": raw_moa}
        actual_blocks = actual_moa.get("presets") if isinstance(actual_moa.get("presets"), dict) else {"default": actual_moa}
        expected_blocks = expected_moa.get("presets") if isinstance(expected_moa.get("presets"), dict) else {"default": expected_moa}
        for name, raw_block in raw_blocks.items():
            actual_block, expected_block = actual_blocks.get(name), expected_blocks.get(name)
            if not (isinstance(raw_block, dict) and isinstance(actual_block, dict) and isinstance(expected_block, dict)):
                continue
            assert isinstance(actual_block, dict) and isinstance(expected_block, dict)
            for slot_key in ("aggregator",):
                actual_block[slot_key] = restore_slot(actual_block.get(slot_key), raw_block.get(slot_key), expected_block.get(slot_key))
            raw_refs, actual_refs, expected_refs = raw_block.get("reference_models"), actual_block.get("reference_models"), expected_block.get("reference_models")
            if isinstance(raw_refs, list) and isinstance(actual_refs, list) and isinstance(expected_refs, list) and len(raw_refs) == len(actual_refs) == len(expected_refs):
                actual_block["reference_models"] = [restore_slot(actual, raw, expected) for actual, raw, expected in zip(actual_refs, raw_refs, expected_refs)]
    return result
