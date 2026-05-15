# Research background profile pattern

Use this when Brian asks for research to be spawned/published.

## Profile/session model

- Keep one durable `research` profile for shared research configuration, tools, memory, and publisher assumptions.
- The active Telegram/chat session should enqueue/spawn and return quickly; it should not do long source gathering inline.

## Execution options

Prefer the most durable host-side mechanism available:
1. Host/dashboard `/background` or equivalent background session if available.
2. `hermes -p research chat -q '<contract>'` launched as a background host process.
3. One-shot cron / scheduler run only when the user explicitly wants scheduling or the work must survive chat interruption.
4. Kanban only when queue/status/retry semantics are desired.

Avoid synchronous `delegate_task` for long research because parent interruption cancels the child.

## Default contract for spawned research

- Use the exact Hermes CLI form `hermes -p research chat -q ...` (or `hermes chat -q ...` when already inside the research profile).
- Do not use bare `hermes -q ...` without the `chat` subcommand.
- Publish as an existing-format `research.briankeefe.dev` page.
- Verify the public URL is reachable.
- Final response contains only the public URL or `PUBLISH_FAILED: <brief reason>`.
