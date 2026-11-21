"""``hermes sessions`` verbs that act on a gateway-hosted session through its owner.

- ``target``: the shared ``<session>`` argument (id, unique prefix, title, ``-``) and the
  one-connection runner every gateway verb uses.
- ``remote``: ``send``, ``wait`` and ``watch``.
- ``interrupt``: ``interrupt`` / ``stop`` (non-interactive ``/stop``).
- ``log``: ``log``, the stored transcript read from ``state.db``.

No cross-package surface: the parser and the dispatch table import each topic module lazily,
when its subcommand is built or run.
"""

__all__: list[str] = []
