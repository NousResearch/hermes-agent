"""AIS Memory Provider — bundled plugin entrypoint.

Re-exports the implementation from the ``hermes-memory-ais`` PyPI
package. Keeping the bundled wrapper thin means upstream consumers
don't need to vendor our code; ``pip install hermes-memory-ais``
provides everything, and this file just hooks it into the plugin
discovery scanner.

Upstream repo for the implementation:
https://github.com/rudedoggg/hermes-memory-ais
"""

from __future__ import annotations

try:
    from ais import (  # type: ignore[import-not-found]
        AISMemoryProvider,
        __version__,
        register_memory_provider,
    )
except ImportError as exc:  # pragma: no cover — covered when extras installed
    msg = (
        "hermes-memory-ais is not installed. Run `pip install hermes-memory-ais` "
        "or `hermes memory setup` and select 'ais'."
    )
    raise ImportError(msg) from exc


__all__ = ["AISMemoryProvider", "__version__", "register_memory_provider"]
