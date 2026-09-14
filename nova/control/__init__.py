"""The Platform Control API — the only path from a customer surface to the runtime.

The dashboard talks to this and to nothing else. It never reaches a runtime endpoint,
never learns a runtime path, and never sees a runtime's own vocabulary. That boundary is
the reason this package exists; see ``docs/platform/ARCHITECTURE_BOUNDARIES.md`` §5.

Phase 1 was **read-only**: there was no write path to get wrong, and the read models are
what every later feature depends on. Phase 8 added writes as the narrow typed commands
``ARCHITECTURE_BOUNDARIES.md`` §5 rule 3 called for — four verbs that map onto transitions
the runtime already owns, gated on an admin principal, and recorded against the human who
made the decision rather than against the process that served the request.

Reads and writes are dispatched by **separate functions** rather than one that branches on
a method string, because a branch is a thing somebody eventually gets the wrong way round.
"""

from nova.control.api import ControlAPI, Response
from nova.control.server import serve

__all__ = ["ControlAPI", "Response", "serve"]
