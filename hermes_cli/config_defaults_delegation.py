"""Pure-data defaults for delegation batch execution."""


def _delegation_execution_defaults():
    """Return a fresh execution-policy block for the default configuration."""
    return {
        # Max parallel children per batch AND max concurrent background delegation units; async
        # dispatches beyond it run synchronously. Floor 1, no ceiling.
        "max_concurrent_children": 10,
        # Join the whole batch in the caller, running one child at a time. Useful for a single local model.
        "sequential": False,
        # Background fan-outs return as ONE message when the whole call finishes. true = each task
        # (or `group`) returns on its own as it finishes — more new turns for the orchestrator.
        "independent_completions": False,
    }
