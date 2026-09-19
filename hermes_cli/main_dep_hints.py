"""Repair guidance for an unavailable declared dependency group."""


def missing_optional_deps_message(surface: str, what: str, extra: str) -> str:
    return (
        f"The {surface} can't start: {what} are missing from this install.\n"
        "Run `hermes pm install` to prepare the declared dependencies.\n"
        "If an installed dependency is damaged, run `hermes pm repair`, then restart Hermes."
    )
