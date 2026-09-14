"""The Hermes runtime adapter.

Maps NOVA's *agent* onto Hermes's *profile*. This package is the only place in NOVA that
knows Hermes exists, and it reaches Hermes exclusively through documented extension
points — the profiles directory, the skins directory, ``config.yaml`` and ``SOUL.md``.
It imports no Hermes module, so the platform layer stays importable without the runtime
installed and the patch budget stays at zero.
"""

from nova.runtime.hermes.adapter import HermesRuntime
from nova.runtime.hermes.paths import HermesPaths, resolve_home
from nova.runtime.hermes.skin import build_skin, skin_filename

__all__ = [
    "HermesRuntime",
    "HermesPaths",
    "build_skin",
    "resolve_home",
    "skin_filename",
]
