"""Container-shape tables for ``hermes config set``.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); the facade re-exports
both names so existing callers and patch targets keep working.
"""

# Container roots absent from DEFAULT_CONFIG whose shape is nonetheless fixed by their readers,
# so the guardrail holds before anything is on disk (#114471: `model.aliases notamap`).
_KNOWN_CONTAINER_TYPES = {
    "custom_providers": "list",
    "providers": "mapping",
    "model.aliases": "mapping",
    "model_aliases": "mapping",
    # Omitted from DEFAULT_CONFIG on purpose (an empty default would clobber a user allow-list),
    # so without these rows `config set plugins.enabled foo` stored a string every reader ignored.
    "plugins.enabled": "list",
    "plugins.disabled": "list",
    "model_catalog.excluded_providers": "list",
    "model_catalog.excluded_models": "mapping",
}
# List slots whose readers go through ``parse_config_string_list``: a bare name is one entry.
_SCALAR_AS_ONE_ITEM_LIST_KEYS = frozenset({"agent.disabled_toolsets", "skills.disabled"})
