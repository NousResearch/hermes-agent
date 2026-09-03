"""Compatibility indirection for the decomposed updater."""

def facade():
    """Return the historical updater module without creating an import cycle."""
    from hermes_cli import update_cmd

    return update_cmd
