"""Pure-data factory for the in-process agent-run pool defaults (``agent_loop:``).

The same pool serves the ``agent_run`` tool and the plugin ``ctx.agent_loop`` capability: one
daemon thread per run, up to ``max_workers`` at once, then a FIFO wait line of at most
``max_queued``. A run that holds its slot past ``max_run_seconds`` is settled as stalled and its
slot released.
"""


def _agent_loop(max_workers: int = 8, max_queued: int = 64, max_run_seconds: int = 1800) -> dict:
    """One ``agent_loop`` config block (see ``DEFAULT_CONFIG['agent_loop']``)."""
    return {"max_workers": max_workers, "max_queued": max_queued,
            "max_run_seconds": max_run_seconds}
