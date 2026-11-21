"""Permit launch maintenance only for product entry points, not arbitrary helpers."""
from pathlib import Path
import sys

# Source entry points also cover python -m and editable console wrappers. The
# managed -I -c launcher identifies itself before importing bootstrap.
_ENTRY_MODULES = frozenset({
    'hermes_cli.main', 'agent.legacy_cli', 'agent.managed_worker',
    'agent.transports.hermes_tools_mcp_server', 'acp_adapter.entry',
    'batch_runner', 'cli', 'run_agent', 'gateway.run', 'tui_gateway.entry',
    'tui_gateway.slash_worker', 'tui_gateway.compute_host', 'cron.worker_bootstrap',
    'hermes_cli.kanban_worker_client', 'hermes_cli.plugin_host_child',
    'hermes_cli.update_finish', 'tools.bot_mode_dm',
})


def is_product_launch(root: Path) -> bool:
    """A helper being ``__main__`` is not proof that it owns install maintenance.

    Keep named product importers for editable console wrappers outside the
    checkout. For direct bootstrap imports, require the managed launcher marker
    or an actual source entry-point file, never a word in sys.argv.
    """
    if getattr(sys, '_hermes_pin_default_home', False):
        return True
    main_file = getattr(sys.modules.get('__main__'), '__file__', None)
    if main_file:
        entry = Path(main_file).resolve()
        if any(entry == root / (name.replace('.', '/') + '.py') for name in _ENTRY_MODULES):
            return True
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get('__name__', '')
        if not name.startswith(('importlib', 'runpy', '_bootstrap', 'hermes_bootstrap')):
            return name in _ENTRY_MODULES
        frame = frame.f_back
    return False
