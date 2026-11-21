# Auto Mode

Active when `{workflow.party_mode}` resolves to `auto` (or a `--mode auto` override). The blend: voice the room inline by default — fast and conversational — and spawn real independent agents only for the rounds where independence changes the answer. When you do spawn, follow `references/mode-subagent.md` for the mechanics. If your harness can't spawn agents, auto is just `session`.

## When to spawn vs. voice

Spawn independent agents when divergent, uncolored thinking is the value of the round:

- A genuine evaluation, review, or critique — the kind that fails if one mind voices every side and they drift into agreement (code review, red-team, a hard look at a plan).
- The personas would plausibly reach *different* conclusions, and that divergence is the point.
- The user asked someone to dig in, analyze, or research — depth earned by a direct ask.

Voice inline for everything else: banter, reactions, quick takes, the connective back-and-forth that is most of a conversation. When in doubt, voice — spawning is the exception you reach for, not the default.
