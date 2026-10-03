"""Dotted config-key path navigation for ``hermes config get/set/unset``.

Split out of ``hermes_cli/config.py``; every name is re-imported there, so
``hermes_cli.config.<name>`` keeps resolving (and monkeypatching) as before.

Dots inside config keys are the norm (model IDs like ``grok-4.6``, Matrix room
IDs), so navigation is escape-aware (``a\\.b``) and prefers an existing literal
dotted key over blind splitting; a write that would shadow such a sibling fails
loudly (see #84064 / #17876).
"""

from typing import Optional, Tuple


def _split_key_path(key: str) -> list[str]:
    """Split a dotted config-key path, honoring backslash-escaped dots (``a\\.b`` -> ``a.b``).
    Backslashes before any other character are preserved verbatim.

    ``hermes config set`` uses ``.`` as the nesting separator, so a key that itself contains a literal dot
    (e.g. provider names like ``qwen3.5-397b-wafer``) was silently split into bogus nested segments
    (#84064).
    """
    parts: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(key):
        ch = key[i]
        if ch == "\\" and key[i + 1:i + 2] == ".":
            current.append(".")
            i += 2
            continue
        if ch == ".":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    parts.append("".join(current))
    return parts


def _greedy_literal_match(container: dict, parts: list) -> Optional[Tuple[str, int]]:
    """Return ``(literal_key, n_consumed)`` for the longest dotted literal key present in
    *container*, or None. With no multi-segment literal this is the historic plain-split walk.

    Dots in config key names are the norm, not the exception — model IDs (``grok-4.6``, ``glm-5.3``), Matrix
    room IDs (``!room:chat.example.cc``), and versioned provider names all embed dots. Users typing
    ``providers.myprov.models.grok-4.6.context_length`` do not know the escape syntax exists, so when
    navigating an EXISTING mapping we prefer an existing literal key equal to the dot-join of the next N
    path segments (longest match wins) over blindly splitting. See #84064 / #80006 / 91095 / #91607 /
    #99124.
    """
    if not isinstance(container, dict) or not parts:
        return None
    return next(
        ((".".join(parts[:n]), n) for n in range(len(parts), 0, -1) if ".".join(parts[:n]) in container),
        None)


def _phantom_sibling(container: dict, part: str) -> Optional[str]:
    """Existing literal dotted key that creating an intermediate mapping ``part`` would shadow
    (``grok-4`` beside ``grok-4.5``) — the write would produce a phantom sibling the runtime never
    reads, so callers fail loudly instead.

    Called when a write is about to CREATE a new intermediate mapping named ``part``. See #84064.
    """
    if not isinstance(container, dict):
        return None
    prefix = part + "."
    return next((k for k in container if isinstance(k, str) and k.startswith(prefix)), None)


def _set_nested(config, dotted_key: str, value):
    """Set a value at a dotted key path, creating intermediate dicts on demand.
    Numeric segments index lists; the index must already exist (lists are never grown).

    Guards against #17876: before this fix the code unconditionally replaced any non-dict value (including
    lists) with ``{}``, silently destroying list-typed config like ``custom_providers`` whenever a caller
    used an indexed path.
    Dotted key names (#84064 family): when navigating an existing mapping, an existing literal key equal to
    the dot-join of the next N segments is preferred over blind splitting (see ``_greedy_literal_match``),
    so ``models.grok-4.6.supports_vision`` lands on the real ``grok-4.6`` entry. And when a write WOULD
    create a new intermediate mapping that shadows an existing dotted sibling (``grok-4`` beside
    ``grok-4.5``), it raises ``ValueError`` instead of silently writing a phantom the runtime never reads.
    """
    parts = _split_key_path(dotted_key)
    current = config
    i = 0
    while i < len(parts):
        remaining = parts[i:]
        at_leaf = len(remaining) == 1
        if isinstance(current, list):
            part = remaining[0]
            if at_leaf:
                current[int(part)] = value
                return
            try:
                current = current[int(part)]
            except (TypeError, ValueError):
                raise TypeError(
                    f"Cannot navigate into list at key {dotted_key!r}: "
                    f"segment {part!r} is not a numeric index")
            i += 1
        elif isinstance(current, dict):
            match = _greedy_literal_match(current, remaining)
            if match is not None:
                key, consumed = match
                if i + consumed == len(parts):
                    current[key] = value
                    return
                # Preserve dicts and lists; replace scalar with a fresh dict.
                if not isinstance(current.get(key), (dict, list)):
                    current[key] = {}
                current = current[key]
                i += consumed
                continue
            part = remaining[0]
            if at_leaf:
                current[part] = value
                return
            shadowed = _phantom_sibling(current, part)
            if shadowed is not None:
                escaped = shadowed.replace(".", "\\.")
                raise ValueError(
                    f"Refusing to create nested key {part!r} in {dotted_key!r}: the mapping "
                    f"already contains a literal key {shadowed!r} that contains a dot. If you "
                    f"meant that key, escape its dots with a backslash (e.g. {escaped}).")
            current = current.setdefault(part, {})
            i += 1
        else:
            raise TypeError(f"Cannot navigate into {type(current).__name__} at key {dotted_key!r}")


_MISSING = object()


def _locate_nested(config, parts: list):
    """Walk *parts* through nested dicts/lists (escape-aware, greedy-literal like ``_set_nested``).
    Returns ``(parents, container, key)`` where ``container[key]`` is the addressed leaf and
    ``parents`` lists the ``(container, key)`` hops above it, or ``None`` when any hop is missing,
    a list index is non-numeric/out of range, or a scalar is hit before the path is consumed."""
    parents = []
    current = config
    i = 0
    while True:
        remaining = parts[i:]
        if isinstance(current, list):
            try:
                key = int(remaining[0])
                current[key]
            except (TypeError, ValueError, IndexError):
                return None
            consumed = 1
        elif isinstance(current, dict):
            match = _greedy_literal_match(current, remaining)
            if match is None:
                return None
            key, consumed = match
        else:
            return None
        i += consumed
        if i == len(parts):
            return parents, current, key
        parents.append((current, key))
        current = current[key]


def _get_nested(config, dotted_key: str):
    """Return a dotted-path value (``_MISSING`` when absent); same navigation as ``_set_nested``
    so ``models.grok-4.6.context_length`` reads the real ``grok-4.6`` entry.

    Mirrors ``_set_nested``'s navigation: honors backslash-escaped dots and prefers an existing literal
    dotted key over blind splitting, so ``config get providers.p.models.grok-4.6.context_length`` reads the
    real ``grok-4.6`` entry instead of reporting the key unset (#84064).
    """
    loc = _locate_nested(config, _split_key_path(dotted_key))
    if loc is None:
        return _MISSING
    _, container, key = loc
    return container[key]


def _unset_nested(config, dotted_key: str) -> bool:
    """Remove a dotted-path value; True if it existed. Empty dict containers left behind are
    dropped, while user-authored empty lists and non-empty sibling branches are preserved.

    Same escape-aware, greedy-literal navigation as ``_set_nested`` / ``_get_nested`` (#84064): unsetting an
    unescaped dotted key removes the real literal entry rather than a phantom sibling.
    """
    loc = _locate_nested(config, _split_key_path(dotted_key))
    if loc is None:
        return False
    parents, current, key = loc
    del current[key]
    # ``parent[part] is current`` for every hop, so each now-empty dict container is dropped.
    for parent, part in reversed(parents):
        if current != {}:
            break
        del parent[part]
        current = parent
    return True
