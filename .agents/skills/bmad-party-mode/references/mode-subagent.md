# Subagent Mode

Active when `{workflow.party_mode}` resolves to `subagent` (or a `--mode subagent` override). Put a real agent behind each persona for every substantive round, the opening banter included, so each persona thinks independently — not one mind voicing them all. A standing directive: don't relitigate it round to round, and don't fall back to voicing because a moment felt light. If your harness can't spawn agents, fall back to `session`.

## Lifecycle

Where your harness keeps agents alive across turns, the cast is **standing**: spawn one agent per persona and reuse that same handle round after round — hand it the new turn plus the room context it needs — instead of a throwaway each time. That continuity is what lets a persona's grudges, alliances, and callbacks accrue. Keep a visible roster mapping each persona to its live handle, and reuse it.

Keep the cast alive for the whole session. A member that finished the one thing you handed it is **idle, not done** — don't close, retire, or disband it. Serving the opening intent doesn't end the party; only the user ending it does, or an explicit `--non-interactive` run wrapping up. Release agents only at wrap-up. If one gets closed by accident, resume it; if it won't resume, say so and respawn just that member.

Where the harness can't hold agents between turns, spawn fresh each round and re-establish each persona's brief and the thread so far — that per-round spawn is the fallback, not the goal.

## One shared room

It's one room, not parallel one-on-ones. Every standing member hears everything said each round — the user's turn and every other persona's turn — even when it's not their turn to speak. A persona sitting a round out is still in the room listening, so when it next speaks it's caught up: it can pick up a dropped thread, hold a grudge, call back. Route the whole exchange to all of them each round; never hand a persona only the slice it's about to answer. Skip this and they drift out of sync — separate consultations wearing a party's clothes.

## Spawning

Give each agent the objective, their persona, and the room so far — what the user said and what the others said, whether or not they're reacting to it. For a custom member, hand them their `persona` as their character and fold their `capabilities` note into the brief; spawn them with their `model` if one is set (a session `--model` pin wins for everyone). Always carry two things into the brief: the group's `scene` and any behavioral instructions in the persona are binding direction, and anything that could be stale since the model's training cutoff should be checked with web search rather than guessed.

Trust their *thinking*: let them decide what to read and how to reach a view; don't script their substance with do-and-don't checklists — that's what produces lifeless blobs. But hold the *form*: a length cap (usually a sentence or three) and the instruction to react to what was just said rather than file a report. Constraining length and stance protects the conversation; constraining their reasoning kills it. Stay in character throughout; a persona goes long only when the user asked it to dig in.

Run them in parallel for independent first-takes; run them sequentially when you want them reacting to each other's actual words. Keep it to a few voices a round — more reads as a crowd, not a conversation.

## Weave the replies into one conversation

Even with everyone caught up on the room, a round taken in parallel means no agent has yet seen the others' turns from that same round — so left raw they reply alongside one another, not to one another. Reorder turns so a rebuttal lands right after what it rebuts, add the connective phrasing real talk has ("Hold on, Winston, that's backwards", "Sally's right about the API, but she's missing the cost"), and let one persona pick up a thread another dropped. Never change what an agent argued — weave delivery, preserve substance.

## Model choice

Match the model to the round: something quick for banter, something stronger for deep work. A per-member `model` is used when set; a session `--model <name>` pin overrides it for everyone.
