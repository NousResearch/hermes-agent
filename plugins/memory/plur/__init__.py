"""PLUR memory provider plugin for Hermes Agent.

Thin registration shim — all implementation lives in the ``plur-hermes``
PyPI package (``pip install plur-hermes``).

This plugin activates PLUR as a first-class Hermes memory provider so it
appears in ``hermes plugins --memory`` and can be selected via::

    memory.provider: plur

in config.yaml (or via ``hermes memory setup``).

The ``plur-hermes`` package already auto-registers itself through the
``hermes_agent.plugins`` entry-point group when installed, so the standalone
hook path (pre_llm_call / post_llm_call auto-inject + auto-learn) is always
active.  This plugin adds the MemoryProvider ABC path on top, which makes
PLUR a *selectable*, *named* memory provider visible to MemoryManager — the
two paths share a single PlurBridge instance so CLI subprocess spawns are
deduplicated.

Requirements
------------
* ``pip install plur-hermes>=0.18.1``
* ``plur`` CLI reachable on PATH (ships with the npm package ``@plur-ai/cli``
  or via ``npx -y @plur-ai/cli``)
"""

from __future__ import annotations


def register(ctx) -> None:
    """Register PLUR as a Hermes memory provider.

    Called by Hermes on startup when this plugin directory is discovered.
    Delegates entirely to ``PlurMemoryProvider`` from ``plur-hermes``.
    """
    try:
        from plur_hermes.memory_provider import PlurMemoryProvider
    except ImportError as exc:
        import logging
        logging.getLogger(__name__).error(
            "plur-hermes is not installed — run `pip install plur-hermes>=0.18.1`. "
            "Error: %s",
            exc,
        )
        return

    ctx.register_memory_provider(PlurMemoryProvider())
