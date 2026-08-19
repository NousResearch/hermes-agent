# Hermes Upstream

This repository is derived from NousResearch/hermes-agent.

Upstream:
https://github.com/NousResearch/hermes-agent

Strategy:

1. Reuse upstream capabilities whenever practical.
2. Prefer new Engineering modules over modifying Hermes core.
3. Prefer Hermes Plugin/Hook extension points where appropriate.
4. Patch Hermes core only when required by the Engineering Agent lifecycle.
5. Record every core patch and its reason.
6. Keep upstream synchronization possible.

Do not modify Hermes core merely for convenience.