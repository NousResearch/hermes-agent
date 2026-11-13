"""Config path writes; kept outside the large config facade."""


def _set_nested(config, dotted_key: str, value):
    """Set a value at a dotted config path, creating intermediate dicts on demand.

    Numeric segments index existing list entries. Existing literal dotted keys
    take precedence over split paths; creating a phantom beside a dotted key is
    refused. Reasoning override maps are keyed directly by model ID, so a new
    dotted ID is kept as one literal key on first write.
    """
    # Late-bind these helpers from the facade to preserve its public patch seams
    # and avoid an import cycle during config module initialization.
    from hermes_cli.config import _phantom_sibling, _split_key_path, _greedy_literal_match

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
            # Reasoning override maps are scalar-valued and keyed by model ID.
            # Handle them before greedy matching: a shorter existing model key
            # (``glm-5``) must not absorb the prefix of a new dotted ID
            # (``glm-5.3-flash``) and be replaced with a phantom mapping.
            if (
                i == 2
                and parts[:2] == ["agent", "reasoning_overrides"]
                and len(remaining) > 1
            ):
                current[".".join(remaining)] = value
                return
            match = _greedy_literal_match(current, remaining)
            if match is not None:
                key, consumed = match
                if i + consumed == len(parts):
                    current[key] = value
                    return
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
