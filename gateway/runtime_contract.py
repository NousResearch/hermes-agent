"""Gateway launch decisions independent of operating-system probes."""

from typing import Literal

RuntimeState = Literal[
    "absent", "starting", "ready", "draining", "incompatible", "inaccessible", "conflict"
]
