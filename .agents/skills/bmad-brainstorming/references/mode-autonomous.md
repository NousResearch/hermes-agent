# Mode: Ideate For Me

The user handed you the topic and wants to see what you come up with on your own, then look at the result. You become the brainstormer — this is the one interactive mode where the ideas are yours to generate.

- **Run a real divergent session yourself.** If the user supplied techniques (e.g. a composed prompt pasted from the selector page), honor those first; otherwise pick and run techniques on your own (use `brain.py` as in `## Choosing Techniques`, but *you* choose — no menu for the user). Capture each idea to the memlog with `--type idea --by coach`, marking each technique switch with a `technique` entry, shifting the creative domain every ~10 ideas, aiming past 100. Push past the obvious.
- **Don't pepper the user with questions** — this is your run. One quick confirm of topic and goal up front is plenty.
- **When it's mined out, synthesize and produce the keepsake.** Go to `## Wrap-Up` (`references/finalize.md`): record the insights, mark the memlog complete, and **auto-generate the imaginative HTML keepsake — don't ask first; the keepsake is the result you promised to show them.** Offer the other artifacts (intent doc, etc.) after.
- **Then, because a human is here, offer to keep going together.** They may want to push an idea further or react to what you found — if so, switch into **Facilitator** or **Creative Partner** (load that frame), **record the switch in the memlog** so a resume restores the new stance — `uv run {project-root}/_bmad/scripts/memlog.py set --workspace {doc_workspace} --key mode --value <facilitator|partner>` — and continue from the same memlog.

This is the interactive sibling of headless mode (`references/headless.md`): the same self-generation, but a person is present to receive the output and may continue. headless is the no-human, returns-JSON runner; this one greets, presents, and hands off.
