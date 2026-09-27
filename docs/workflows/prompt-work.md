# Model-Facing Changes

Read this before changing prompts, tool schemas or results, model-visible
messages, memory injection, or background-run instructions.

1. Trace the actual native builder and caller for every affected surface.
2. State the exact before/after behavior, including platform and session differences.
3. Preserve the cache and message-flow rules in [AGENTS.md](../../AGENTS.md).
4. Update the relevant native documentation and examples in the same change.
   Runtime code remains authoritative; documentation is not a second prompt source.
5. Validate the real assembly path and affected behavior with the owning tests,
   including unchanged warm-session prompts and profile isolation where relevant.

Clear fixes that restore documented behavior can proceed within the task
scope. Surface product choices or material tradeoffs before implementing them;
existing user decisions and authorization remain valid.

Current assembly and caching are documented in
[prompt assembly](../../website/docs/developer-guide/prompt-assembly.md) and
[context compression and caching](../../website/docs/developer-guide/context-compression-and-caching.md).
