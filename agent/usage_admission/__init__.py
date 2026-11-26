"""Soft usage ceilings for ``openai-codex`` pool credentials.

- ``policy``: the configured ceilings, the policy window chosen from a usage snapshot, the shared
  decision cache and the module-level admission checks used by the runtime paths.
- ``pool_mixin``: the ``CredentialPool`` methods that probe, cache and apply those decisions.

Topic modules import one another; nothing imports this facade.
"""
