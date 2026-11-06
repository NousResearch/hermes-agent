"""Model slot for Hermes Desktop's prompt refinement.

The Desktop plugin (``apps/desktop/src/plugins/prompt-refinement``) rewrites a draft through the
stateless ``llm.oneshot`` RPC with ``task="prompt_refinement"``. Registering the task here gives it
an ``auxiliary.prompt_refinement`` config block, so the existing Settings > Model > Auxiliary picker
(provider sign-in, credential pools, fallback chain) chooses the refinement model per profile.

Without this plugin the same call resolves through the automatic auxiliary route, so Desktop keeps
working against a home that never enabled it; only the model choice is lost.
"""

TASK_KEY = "prompt_refinement"


def register(ctx) -> None:
    ctx.register_auxiliary_task(
        TASK_KEY,
        display_name="Prompt refinement",
        description="Rewrites a draft prompt for clarity before you approve sending it (Desktop).",
    )
