"""Pet generation — base-draft → hatch pipeline.

Public surface used by the gateway RPCs, the CLI ``hermes pets generate``
command, and tests:

- :func:`generate_base_drafts` / :func:`hatch_pet` — the two-step flow.
- :class:`HatchResult`, :class:`GenerationError`.
- :mod:`atlas` — deterministic frame extraction + atlas composition/validation.

Image generation is delegated to the active reference-capable
:class:`~agent.image_gen_provider.ImageGenProvider` (OpenAI gpt-image-2 or Krea);
atlas assembly is fully deterministic so it's testable without any API calls.
"""

from __future__ import annotations

from agent.pet.generate.imagegen import GenerationError
from agent.pet.generate.orchestrate import (
    HatchResult,
    generate_base_drafts,
    hatch_pet,
)

__all__ = [
    "GenerationError",
    "HatchResult",
    "generate_base_drafts",
    "hatch_pet",
]
