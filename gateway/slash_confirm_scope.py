"""The config home a slash-confirm gate was read from.

A gate that asks "always approve?" reads its flag from the config of the profile that is
serving the chat: the inbound message handler binds that profile's scope
(``run_adapters.py::_make_profile_message_handler``), so ``load_config()`` resolves the
PROFILE's ``config.yaml``. The "always" callback fires later, from the button's interaction
handler - outside that scope - so an unscoped ``save_config_value()`` writes the LAUNCH home's
``config.yaml`` instead. The profile never sees the opt-out, the prompt comes back on every
``/new``, and the root config is rewritten on every click (#142157).

So the home is captured with the prompt and re-entered for the write: the opt-out lands in the
file the gate was read from. This is the deferred-callback rule the gateway profile-scope
section already states - capture the routed home at command time, re-enter it in the callback.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


def current_config_home() -> Path:
    """The home whose ``config.yaml`` a gate is reading right now.

    The context-local override when a profile scope is bound (the routed profile's home),
    else the process home - exactly the file ``load_config()`` resolved a moment ago."""
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


@contextmanager
def config_home_scope(home: Optional[str | Path]) -> Iterator[None]:
    """Bind ``home`` as the config home for the block; a no-op when ``home`` is None.

    Only the home override - the write touches one file, so it needs no secret scope."""
    if home is None:
        yield
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)
