"""CLI parsing for the strict ``context.external_files`` path list."""

import hermes_yaml as yaml


def parse_external_context_files(value: str) -> list[str]:
    """Accept a plain/comma-separated path or a YAML list without scalar coercion.

    A malformed structured value must fail closed, like other config containers. Quoted
    YAML strings preserve commas and reserved words inside explicit lists.
    """
    from hermes_cli.config import _exit_invalid, _looks_structured_value

    if not _looks_structured_value(value):
        return [path.strip() for path in value.split(",") if path.strip()]
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        _exit_invalid("✗ context.external_files is not valid YAML/JSON — nothing was written.")
    if not isinstance(parsed, list) or any(not isinstance(path, str) for path in parsed):
        _exit_invalid(
            "✗ context.external_files must be a list of path strings — nothing was written. "
            "Quote paths such as 'off' or '123' inside YAML lists.")
    return [path.strip() for path in parsed if path.strip()]
