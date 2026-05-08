"""vMLX provider profiles — Apple Silicon local inference.

Registers two profiles against the v0.13.0 ``providers/`` plugin contract:

  * ``vmlx``         — primary, http://localhost:8000/v1, large context.
  * ``vmlx-janitor`` — auxiliary (compression / summarization / memory writes /
                       skill curation), http://localhost:8001/v1, short context.

Both speak the OpenAI chat-completions wire protocol against a local
``vmlx serve`` instance. Apple Silicon only; the ``platform.system()`` guard
below makes the import fail cleanly on Linux/Windows so the discovery loop
in ``providers/__init__.py`` skips the directory.

Sibling of the bundled ``custom``/Ollama profile, but with sane localhost
defaults baked in for vMLX so airgap setups need zero configuration.
"""
from __future__ import annotations

import platform

from providers import register_provider
from providers.base import ProviderProfile

if platform.system() != "Darwin":
    raise ImportError("vmlx provider requires macOS (Apple Silicon)")

vmlx_primary = ProviderProfile(
    name="vmlx",
    aliases=("mlx", "mlx-server", "apple-mlx", "vmlx-primary"),
    display_name="vMLX (Apple Silicon local inference)",
    description=(
        "MLX-format LLMs served locally via `vmlx serve` on Apple Silicon. "
        "Airgap-friendly: no API key, no cloud fallback."
    ),
    env_vars=(),
    base_url="http://localhost:8000/v1",
    fallback_models=(),
)

vmlx_janitor = ProviderProfile(
    name="vmlx-janitor",
    aliases=("vmlx-aux", "mlx-janitor"),
    display_name="vMLX janitor (Apple Silicon, auxiliary tasks)",
    description=(
        "Smaller MLX model on a separate `vmlx serve` instance — drives "
        "compression, summarization, memory writes, and skill curation so the "
        "primary model's context window stays free for the agent loop."
    ),
    env_vars=(),
    base_url="http://localhost:8001/v1",
    fallback_models=(),
)

register_provider(vmlx_primary)
register_provider(vmlx_janitor)
