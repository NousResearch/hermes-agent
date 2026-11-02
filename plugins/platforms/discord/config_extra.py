"""Authored YAML values for the Discord plugin's configuration bridge."""


def authored_extra(block: dict, keys: tuple[str, ...]) -> dict:
    """Values authored in the effective YAML block, without consumer defaults."""
    return {key: block[key] for key in keys if key in block}
