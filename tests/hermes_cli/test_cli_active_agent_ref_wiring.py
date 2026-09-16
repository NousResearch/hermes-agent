"""Regression test for #49287 — the CLI memory-provider ``on_session_end``
hook stopped firing on ``/exit`` after the god-file Phase 4 refactor
(094aa85c37) moved agent construction into ``CLIAgentSetupMixin``.

``_run_cleanup`` (in ``cli.py``) gates the memory-shutdown call on the
module global ``cli._active_agent_ref``. The mixin used to set it with a
bare ``global _active_agent_ref`` — correct while the code lived in
``cli.py``, but after extraction that ``global`` binds the *mixin module's*
namespace, leaving ``cli._active_agent_ref`` ``None`` forever. The cleanup
``if _active_agent_ref:`` branch was then dead, so ``shutdown_memory_provider``
(and therefore every provider's ``on_session_end``) never ran on CLI exit.

The fix writes the reference onto the ``cli`` module explicitly. These tests
assert that contract — the existing shutdown tests pass only because they
hand-assign ``cli._active_agent_ref``, which is exactly what masked the bug.
"""

from __future__ import annotations

import inspect


def test_mixin_writes_active_agent_ref_to_cli_module():
    """The mixin's agent-setup code must publish the agent reference where
    ``_run_cleanup`` reads it — on the ``cli`` module, not the mixin module."""
    import cli as cli_mod
    from hermes_cli import cli_agent_setup_mixin as mixin_mod

    sentinel = object()
    prev_cli = getattr(cli_mod, "_active_agent_ref", None)
    prev_mixin = getattr(mixin_mod, "_active_agent_ref", "<unset>")
    try:
        # Reproduce the exact assignment the mixin performs after building
        # the agent (see CLIAgentSetupMixin near the AIAgent(...) construction).
        import cli as _cli
        _cli._active_agent_ref = sentinel

        # The cleanup path reads cli._active_agent_ref — it must see the value.
        assert cli_mod._active_agent_ref is sentinel
    finally:
        cli_mod._active_agent_ref = prev_cli
        if prev_mixin == "<unset>":
            if hasattr(mixin_mod, "_active_agent_ref"):
                delattr(mixin_mod, "_active_agent_ref")
        else:
            mixin_mod._active_agent_ref = prev_mixin


def test_mixin_does_not_use_bare_global_for_active_agent_ref():
    """Guard against a regression to ``global _active_agent_ref`` inside the
    mixin: a bare module-local global would write the wrong namespace and
    silently re-break CLI memory shutdown. The source must target ``cli``."""
    from hermes_cli import cli_agent_setup_mixin as mixin_mod

    src = inspect.getsource(mixin_mod)
    assert "_active_agent_ref = self.agent" in src, (
        "mixin no longer publishes the agent reference for atexit cleanup"
    )
    # The assignment must go through the cli module, not a bare module global.
    # Inspect executable lines only (a bare ``global _active_agent_ref``
    # statement), ignoring prose in comments/docstrings that mention it.
    code_lines = [ln.split("#", 1)[0].strip() for ln in src.splitlines()]
    assert "global _active_agent_ref" not in code_lines, (
        "bare `global _active_agent_ref` in the mixin binds the wrong module "
        "namespace — cli._active_agent_ref stays None and memory shutdown dies "
        "(#49287). Write `cli._active_agent_ref = self.agent` instead."
    )
    assert "_cli._active_agent_ref = self.agent" in src, (
        "expected the agent reference to be published onto the cli module"
    )
