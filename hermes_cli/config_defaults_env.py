"""Entry factories for ``config_defaults.OPTIONAL_ENV_VARS`` (pure-data leaf, stdlib only)."""

import threading

_FILL_LOCK = threading.RLock()  # re-entrant: the fill itself reads the table


class OptionalEnvVars(dict):
    """``OPTIONAL_ENV_VARS``: a dict that runs one armed ``fill(table)`` before it is first read.

    Some entries (provider profiles') are only known after work that must not run at import
    (``hermes_cli.config_provider_env``); writes never trigger the fill. A fill that raises is
    re-armed, so the next read tries again."""

    _fill = None

    def arm(self, fill) -> None:
        self._fill = fill

    def _ready(self) -> None:
        if self._fill is None:
            return
        with _FILL_LOCK:
            fill, self._fill = self._fill, None
            if fill is None:
                return
            try:
                fill(self)
            except BaseException:
                self._fill = fill
                raise

    def __getitem__(self, key):
        self._ready()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._ready()
        return dict.get(self, key, default)

    def __contains__(self, key) -> bool:
        self._ready()
        return dict.__contains__(self, key)

    def __iter__(self):
        self._ready()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._ready()
        return dict.__len__(self)

    def keys(self):
        self._ready()
        return dict.keys(self)

    def items(self):
        self._ready()
        return dict.items(self)

    def values(self):
        self._ready()
        return dict.values(self)

    def copy(self) -> dict:
        self._ready()
        return dict(dict.items(self))


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
