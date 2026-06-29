# Subagent Mode

Active when `{workflow.party_mode}` resolves to `subagent` (or a `--mode subagent` override). Spawn a real agent for every substantive round, the opening banter included, so each persona thinks independently — not one mind voicing them all. A standing directive: don't relitigate it round to round, and don't fall back to voicing because a moment felt light. If your harness can't spawn agents, fall back to `session`.

## Spawning

Give each agent the objective, their persona, the context, and what the others said if they're reacting. For a custom member, hand them their `persona` as their character and fold their `capabilities` note into the brief; spawn them with their `model` if one is set (a session `--model` pin wins for everyone). Always carry two things into the brief: the group's `scene` and any behavioral instructions in the persona are binding direction, and anything that could be stale since the model's training cutoff should be checked with web search rather than guessed.

Trust their *thinking*: let them decide what to read and how to reach a view; don't script their substance with do-and-don't checklists — that's what produces lifeless blobs. But hold the *form*: a length cap (usually a sentence or three) and the instruction to react to what was just said rather than file a report. Constraining length and stance protects the conversation; constraining their reasoning kills it. Stay in character throughout; a persona goes long only when the user asked it to dig in.

Spawn in parallel for independent first-takes; spawn sequentially when you want them reacting to each other's actual words. Keep it to a few voices a round — more reads as a crowd, not a conversation.

## Weave the replies into one conversation

Each agent saw only the user's message and the context you handed it, so left raw they reply in parallel and never to one another. Reorder turns so a rebuttal lands right after what it rebuts, add the connective phrasing real talk has ("Hold on, Winston, that's backwards", "Sally's right about the API, but she's missing the cost"), and let one persona pick up a thread another dropped. Never change what an agent argued — weave delivery, preserve substance.

## Model choice

Match the model to the round: something quick for banter, something stronger for deep work. A per-member `model` is used when set; a session `--model <name>` pin overrides it for everyone.
