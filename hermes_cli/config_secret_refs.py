"""Config writers keep credentials in ``.env`` and put only a reference into ``config.yaml``.

``config.yaml`` is settings; ``.env`` is secrets (#77471). A writer that receives a credential for a
config field (``hermes config set model.api_key``, the dashboard model / auxiliary pickers, the
``hermes model`` custom-endpoint and auxiliary flows) saves the value to the profile's ``.env`` under
a deterministic name and stores ``${NAME}`` in the field. ``load_config`` expands the reference, so
resolution is unchanged, and ``save_config`` keeps the template (``_preserve_env_ref_templates``).
Inline values a user already has keep working; nothing is migrated.

Names follow the conventions already in the tree:

- ``model.api_key`` -> ``HERMES_MODEL_API_KEY``
- ``auxiliary.<task>.api_key`` -> ``AUXILIARY_<TASK>_API_KEY`` (the legacy aux env-var spelling;
  a task key that is not lowercase ASCII also gets ``_<HASH>``)
- ``delegation.api_key`` -> ``HERMES_DELEGATION_API_KEY``
- ``providers.<name>.api_key`` -> ``provider_name_key_env(name)``
- ``custom_providers[N].api_key`` -> ``endpoint_api_key_env(row base_url)`` (by name for a row with no URL)
- a custom endpoint the dashboard or ``hermes model`` saves -> ``endpoint_api_key_env(base_url)``

The derived names carry a hash of the exact endpoint / name: a readable slug alone folds ``-``, ``.``,
``/``, ``:`` and ``_`` together, so ``my-host.com`` and ``my.host.com`` would share one variable and
one host's key would be sent to the other.

Every name is stripped from terminal / subprocess children (``_is_hermes_internal_secret``).
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from typing import Any, Optional

_ENV_REF_ONLY_RE = re.compile(r"^\$\{[^}]+\}$")
_PLAIN_AUX_TASK_RE = re.compile(r"[a-z0-9_]+")


def _env_slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(text or "").upper()).strip("_")


def model_api_key_env() -> str:
    return "HERMES_MODEL_API_KEY"


def aux_api_key_env(task: str) -> str:
    """``AUXILIARY_<TASK>_API_KEY``; ``delegation`` (stored in its own top-level section) gets
    ``HERMES_DELEGATION_API_KEY``."""
    if task == "delegation":
        return "HERMES_DELEGATION_API_KEY"
    # Injective over task keys: ``upper()`` is one-to-one on lowercase ASCII keys (every built-in,
    # and ``my_task`` / ``my__task`` / ``_my_task`` keep their underscores), so those keep the
    # legacy spelling. Any other key (``My_Task``, a non-ASCII letter) would fold onto another
    # task's variable, so its name also carries a hash of the exact key.
    task = str(task or "")
    slug = re.sub(r"[^A-Z0-9_]+", "_", task.upper())
    if not _PLAIN_AUX_TASK_RE.fullmatch(task):
        slug = f"{slug}_{_digest(task)}"
    return f"AUXILIARY_{slug}_API_KEY"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8].upper()


def _canonical_endpoint(base_url: str) -> str:
    """``scheme://host:port/path`` with the host lowercased, the default port filled in and trailing
    slashes dropped: one string per endpoint, whatever spelling the caller used."""
    parsed = urllib.parse.urlparse(str(base_url or "").strip())
    scheme = parsed.scheme.lower()
    port = parsed.port or {"http": 80, "https": 443}.get(scheme, "")
    return f"{scheme}://{(parsed.hostname or '').lower()}:{port}{parsed.path.rstrip('/')}"


def endpoint_identity(base_url: str) -> str:
    """``host[_port][_path]_<hash>``: the readable part names the endpoint in ``.env``; the hash of
    :func:`_canonical_endpoint` keeps two endpoints whose slugs coincide (``my-host.com`` /
    ``my.host.com``, ``a.com:8443`` / ``a.com/8443``) on separate variables."""
    parsed = urllib.parse.urlparse(str(base_url or "").strip())
    identity = parsed.hostname or ""
    if not identity:
        return ""
    if parsed.port:
        identity = f"{identity}_{parsed.port}"
    path = parsed.path.strip("/")
    if path and path.lower() != "v1":
        identity = f"{identity}_{path}"
    return f"{identity}_{_digest(_canonical_endpoint(base_url))}"


def endpoint_api_key_env(base_url: str) -> str:
    """``HERMES_CUSTOM_<HOST[_PORT][_PATH]>_<HASH>_API_KEY``; ``HERMES_MODEL_API_KEY`` when the URL
    has no host."""
    from hermes_cli.config import custom_endpoint_key_env
    identity = endpoint_identity(base_url)
    return custom_endpoint_key_env(identity) if identity else model_api_key_env()


def provider_name_key_env(name: str) -> str:
    """``HERMES_PROVIDER_<NAME>_<HASH>_API_KEY`` for a credential addressed by provider NAME
    (``providers.<name>``, a ``custom_providers`` row): its own prefix, so a name never lands on an
    endpoint's variable, and a hash of the exact name, so ``my.host`` and ``my-host`` stay apart."""
    return f"HERMES_PROVIDER_{_env_slug(name) or 'CUSTOM'}_{_digest(str(name))}_API_KEY"


def _credential_blocks(cfg: dict) -> list[dict]:
    """Every config block that can hold a credential reference next to its own ``base_url``."""
    blocks: list[Any] = [cfg.get("model"), cfg.get("delegation")]
    for section in ("auxiliary", "providers"):
        value = cfg.get(section)
        blocks.extend(value.values() if isinstance(value, dict) else [])
    for section in ("custom_providers", "fallback_providers", "fallback_model"):
        rows = cfg.get(section)
        blocks.extend(rows if isinstance(rows, list) else [rows])
    return [b for b in blocks if isinstance(b, dict)]


def env_reference_owners(cfg: dict, env_name: str) -> list[dict]:
    """The blocks of a RAW (unexpanded) config that read ``env_name``: ``api_key: ${NAME}`` or
    ``key_env`` / ``api_key_env: NAME``."""
    ref = f"${{{env_name}}}"
    return [b for b in _credential_blocks(cfg)
            if str(b.get("api_key") or "").strip() == ref or env_name in (b.get("key_env"), b.get("api_key_env"))]


def env_still_referenced(cfg: dict, env_name: str) -> bool:
    """True when a block of ``cfg`` (raw or ``load_config``-expanded) still reads ``env_name``; an
    expanded ``${NAME}`` shows up as the variable's value, so a block holding that value counts."""
    from hermes_cli.config import load_env
    if env_reference_owners(cfg, env_name):
        return True
    value = load_env().get(env_name)
    return bool(value) and any(str(b.get("api_key") or "").strip() == value for b in _credential_blocks(cfg))


def _block_base_url(block: dict) -> str:
    return str(block.get("base_url") or block.get("url") or block.get("api") or "").strip()


def other_endpoint_reads(env_name: str, base_url: str, own_provider: Any = None) -> bool:
    """True when a RAW config block for an endpoint other than ``base_url`` reads ``env_name``.
    ``own_provider`` is the ``providers.<key>`` being saved: its on-disk row is the caller itself,
    even when this save changes its ``base_url``."""
    from hermes_cli.config import read_raw_config
    raw = read_raw_config()
    providers = raw.get("providers")
    own = providers.get(own_provider) if own_provider is not None and isinstance(providers, dict) else None
    mine = _canonical_endpoint(base_url)
    return any(b is not own and _block_base_url(b) and _canonical_endpoint(_block_base_url(b)) != mine
               for b in env_reference_owners(raw, env_name))


def _refuse_cross_endpoint_overwrite(env_name: str, value: str, base_url: str, own_provider: Any = None) -> None:
    """Refuse to change ``env_name`` when a config block for ANOTHER endpoint reads it: the new value
    is this endpoint's key, and overwriting would send it to that other host."""
    from hermes_cli.config import load_env
    current = load_env().get(env_name)
    if not current or current == value:
        return
    if other_endpoint_reads(env_name, base_url, own_provider):
        raise ValueError(f"refusing to overwrite {env_name}: another endpoint in config.yaml reads it "
                         "and it holds a different key")


def _endpoint_key_envs(cfg: dict, base_url: str):
    """``(name, shared)`` for each Hermes-written ``key_env`` of a block registered for ``base_url``;
    ``shared`` when another endpoint also reads the name (an older release keyed rows by host:port)."""
    wanted = _canonical_endpoint(base_url)
    for block in _credential_blocks(cfg):
        name = str(block.get("key_env") or "")
        if (name.startswith(("HERMES_CUSTOM_", "HERMES_PROVIDER_")) and _block_base_url(block)
                and _canonical_endpoint(_block_base_url(block)) == wanted):
            yield name, other_endpoint_reads(name, base_url)


def known_endpoint_key_env(cfg: dict, base_url: str) -> str:
    """The Hermes-written ``key_env`` of a ``providers`` / ``custom_providers`` row already registered
    for ``base_url``, so a save for that endpoint updates the row's variable instead of forking a
    second one. User-chosen names (``OPENAI_API_KEY``) are never reused: they may serve elsewhere, and
    neither is a name another endpoint also reads (:func:`shared_endpoint_key_envs`)."""
    return next((name for name, shared in _endpoint_key_envs(cfg, base_url) if not shared), "")


def shared_endpoint_key_envs(cfg: dict, base_url: str) -> list[str]:
    """The legacy host:port ``key_env`` names this endpoint's blocks share with other endpoints (two
    when rows spell the default port both ways). A writer that saves this endpoint's key under a new
    variable moves the blocks onto it (:func:`repoint_endpoint_key_env`), else selecting them later
    brings back the other key."""
    return list(dict.fromkeys(name for name, shared in _endpoint_key_envs(cfg, base_url) if shared))


def repoint_endpoint_key_env(cfg: dict, base_url: str, old_env: str, new_env: str) -> None:
    """Point every block of ``cfg`` for ``base_url`` whose ``key_env`` is ``old_env`` at ``new_env``;
    blocks for other endpoints keep ``old_env`` and the key it holds."""
    wanted = _canonical_endpoint(base_url)
    for block in _credential_blocks(cfg):
        if (block.get("key_env") == old_env and _block_base_url(block)
                and _canonical_endpoint(_block_base_url(block)) == wanted):
            block["key_env"] = new_env


def dashboard_endpoint_key_env(entry: dict, endpoint_id: str) -> str:
    """The ``.env`` name for a dashboard ``providers.<id>`` endpoint's key: the Hermes-written
    ``key_env`` the row already has (an older release's ``HERMES_CUSTOM_<ID>_API_KEY`` keeps
    working), else :func:`provider_name_key_env`, whose hash keeps ``my-host`` / ``my_host`` apart.
    A user-chosen ``key_env`` (``OPENAI_API_KEY``) is never written over: it may serve elsewhere."""
    current = str(entry.get("key_env") or "")
    if current.startswith(("HERMES_CUSTOM_", "HERMES_PROVIDER_")):
        return current
    return provider_name_key_env(endpoint_id)


def is_env_reference(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENV_REF_ONLY_RE.match(value.strip()))


def refs_supported() -> bool:
    """Only a file-backed config shares the machine's ``.env``. Under Remote Config the document is
    served to every host while ``.env`` stays local, so a reference would not resolve elsewhere."""
    from hermes_cli.config_backend import supports_file_tooling
    return supports_file_tooling()


def store_secret(env_name: str, value: str, base_url: str = "", own_provider: Any = None) -> str:
    """Save ``value`` to ``.env`` as ``env_name`` and return the ``${env_name}`` reference.
    Raises ``ValueError`` when the ``.env`` write lock forbids the name, or when ``base_url`` (the
    endpoint the key is for) is given and another endpoint's block reads ``env_name``, so the caller
    refuses the whole write instead of pointing config.yaml at a variable that was never saved."""
    from hermes_cli.config import require_env_writable, save_env_value
    require_env_writable(env_name, "set")
    if base_url:
        _refuse_cross_endpoint_overwrite(env_name, value, base_url, own_provider)
    save_env_value(env_name, value)
    return f"${{{env_name}}}"


def secret_ref_for(env_name: str, value: Any, base_url: str = "") -> Any:
    """The value a writer should persist for a credential field: ``${env_name}`` after saving a
    literal secret to ``.env``; empty values, existing references and non-strings unchanged (also
    under a backend that cannot share ``.env``)."""
    if not isinstance(value, str) or not value.strip() or is_env_reference(value) or not refs_supported():
        return value
    return store_secret(env_name, value.strip(), base_url)


def secret_key_env_for(env_name: str, value: Any, base_url: str = "") -> str:
    """``key_env`` flavour of :func:`secret_ref_for` for blocks that take a variable NAME
    (``custom_providers`` entries): ``env_name`` once the secret is in ``.env``, else ""."""
    return env_name if secret_ref_for(env_name, value, base_url) != value else ""


def _literal_path(parts: list[str], user_config: dict) -> list[str]:
    """``parts`` regrouped the way ``_set_nested`` walks them: an existing literal key containing
    dots (``providers["lab.example"]``) is one segment, so ``providers.lab.example.api_key`` is
    classified as the provider block it writes, not as an unknown 4-part path."""
    from hermes_cli.config import _greedy_literal_match
    out: list[str] = []
    node: Any = user_config
    i = 0
    while i < len(parts):
        match = _greedy_literal_match(node, parts[i:]) if isinstance(node, dict) else None
        if match is None:  # a list index, or a key the write will create: one segment each
            out.append(parts[i])
            node = (node[int(parts[i])] if isinstance(node, list) and parts[i].isdigit()
                    and int(parts[i]) < len(node) else None)
            i += 1
            continue
        key, consumed = match
        out.append(key)
        node = node[key]
        i += consumed
    return out


def _config_block(parts: list[str], user_config: dict) -> dict:
    """The mapping that holds the leaf of ``parts`` (``{}`` when the path does not exist yet)."""
    node: Any = user_config
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)] if part.isdigit() and int(part) < len(node) else None
        else:
            node = node.get(part) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def _custom_provider_entry_key_env(entry: dict, index: str) -> str:
    """A ``custom_providers`` row's variable, named by the row's endpoint as
    :func:`_save_custom_provider` names it: row names need not be unique and may equal a
    ``providers.<name>`` key, so naming by name would let two hosts share one key."""
    if endpoint_identity(_block_base_url(entry)):
        return endpoint_api_key_env(_block_base_url(entry))
    return provider_name_key_env(str(entry.get("name") or "").strip() or f"custom_providers_{index}")


def config_path_api_key_env(parts: list[str], user_config: dict) -> Optional[str]:
    """The ``.env`` name for a credential config path ``hermes config set`` stores by reference, or
    None for paths it writes as given."""
    if not parts or parts[-1] != "api_key":
        return None
    parts = _literal_path(parts, user_config)
    head = parts[:-1]
    if head == ["model"]:
        return model_api_key_env()
    if head == ["delegation"]:
        return aux_api_key_env("delegation")
    if len(head) == 2 and head[0] == "auxiliary":
        return aux_api_key_env(head[1])
    if len(head) == 2 and head[0] == "providers":
        return provider_name_key_env(head[1])
    if len(head) == 2 and head[0] == "custom_providers":
        return _custom_provider_entry_key_env(_config_block(parts, user_config), head[1])
    return None


def config_set_secret_value(key: str, value: Any, user_config: dict) -> Any:
    """``hermes config set``'s credential routing: the value to write at ``key``. A literal secret
    for a path in :func:`config_path_api_key_env` is saved to ``.env`` (a note names the variable)
    and its ``${VAR}`` reference returned; a refused ``.env`` write exits without writing."""
    from hermes_cli.config import _exit_invalid, _split_key_path
    parts = _literal_path(_split_key_path(key), user_config)
    env_name = config_path_api_key_env(parts, user_config)
    if env_name is None:
        return value
    try:  # the block's URL arms store_secret's refusal when another endpoint reads env_name
        stored = secret_ref_for(env_name, value, _block_base_url(_config_block(parts, user_config)))
    except ValueError as exc:
        _exit_invalid(f"✗ {exc}")
    if stored != value:
        print(f"  (secret saved to .env as {env_name}; config.yaml stores the reference {stored})")
    return stored


def masked_display_value(key: str, value: Any) -> Any:
    """The value ``hermes config set`` echoes: masked when the (possibly nested) key is
    credential-shaped, e.g. ``model.api_key`` (lowercase, so it misses the .env routing). A
    ``${VAR}`` reference is not a secret and is shown as is. A mapping or list value gets its nested
    credential keys masked, as ``config get`` does."""
    from hermes_cli.config import _is_secret_config_key
    from hermes_cli.config_secret_display import _maskable_secret_text, redact_config_value
    if not (text := _maskable_secret_text(value)):
        return redact_config_value(value)
    if _is_secret_config_key(key) and not is_env_reference(text):
        from agent.redact import mask_secret
        return mask_secret(text)
    return value
