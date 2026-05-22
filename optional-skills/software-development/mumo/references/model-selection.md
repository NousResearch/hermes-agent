# Model Selection

Default behavior: omit `models` and let mumo select the panel. mumo's defaults evolve as model quality and pricing change. Hardcoded model strategy in the skill will go stale.

## When to call `list_models`

- The user names specific models.
- The user asks for a cheap gut-check or a premium panel.
- A previous call failed because a model was unavailable.
- You need to confirm model IDs.

## Selection principles

When selecting yourself:

- **Provider diversity over intra-family differences.** A panel with models from three different providers is more valuable than three variants from one.
- **Include your own family only when the user wants it** or the comparison specifically benefits from it. You're already in the conversation as moderator.
- **Avoid nano/fast variants for high-stakes decisions** unless the user explicitly prioritizes cost.
- **Don't present model choice as a leaderboard judgment.** You're picking for diversity, not ranking.

## Cost awareness

`get_credit` is useful when balance is uncertain or the user asks about cost.

Don't make cost preflight a default ritual before every deliberation — it creates friction and can bias agents away from using mumo when the task genuinely warrants a panel.

If cost is part of the user's problem, translate it into problem context:

Good: "Assume this workflow can afford at most one additional model call per important decision."
Weak: "The previous mumo round cost $0.34."

Frontier panels typically cost $0.20-0.60 per round. Mid-tier panels are significantly cheaper.
