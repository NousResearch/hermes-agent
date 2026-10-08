"""Factories for the OPTIONAL_ENV_VARS entries in config_defaults."""


def _env(description, prompt, **keys):
    """One OPTIONAL_ENV_VARS entry; keyword order is preserved as dict key order."""
    return {"description": description, "prompt": prompt, **keys}


_OMIT = object()


def _category(category, password, advanced):
    """Entry factory for one category with its usual password/advanced defaults.

    ``url``/``help``/``tools`` are only written when passed; ``password=None`` omits the key;
    ``advanced`` is only written when true. Key order matches the plain ``_env`` entries.
    """
    def make(description, prompt, url=_OMIT, *, help=_OMIT, tools=_OMIT, password=password,
             advanced=advanced):
        d = {"description": description, "prompt": prompt}
        d.update((k, v) for k, v in (("help", help), ("url", url), ("tools", tools)) if v is not _OMIT)
        if password is not None:
            d["password"] = password
        d["category"] = category
        if advanced:
            d["advanced"] = True
        return d
    return make


_prov = _category("provider", password=True, advanced=True)
_tool = _category("tool", password=True, advanced=False)
_msg = _category("messaging", password=False, advanced=False)
_skill = _category("skill", password=True, advanced=True)
_setting = _category("setting", password=False, advanced=False)


def _base_url(name, prompt_name=None):
    """Provider ``*_BASE_URL`` override entry (advanced, not a secret)."""
    prompt = f"{prompt_name or name} base URL (leave empty for default)"
    return _prov(f"{name} base URL override", prompt, None, password=False)
