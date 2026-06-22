# FalconPM Hermes Fork Setup

FalconPM is shipped as a Hermes fork customized for Growth PM workflows.

This fork adds the FalconPM customization layer:

- agent identity
- Growth PM behavior
- The Pickle Romance context
- D2C growth planning skill
- demo prompt and sample output

## V1 Customization Strategy

Do not rewrite Hermes core for V1.

Customize the fork through:

- context files
- skills
- examples
- product-specific memory
- README/project positioning

This keeps FalconPM shippable and easy to understand while still satisfying the assignment goal of building from a Hermes fork.

## Files Added For FalconPM

```text
SOUL.md
MEMORY.md
docs/falconpm/hermes-fork-setup.md
docs/the-pickle-romance/brand-context.md
examples/falconpm/the-pickle-romance-demo.md
skills/productivity/d2c-growth-experiment-planner/SKILL.md
```

## V1 Demo

Run the demo prompt from:

```text
examples/falconpm/the-pickle-romance-demo.md
```

The demo shows FalconPM planning a 30-day campaign for The Pickle Romance to sell Rs. 50,000 of the Aam Romantics Combo.

## Later Runtime Changes

Only change Hermes internals if FalconPM needs:

- custom memory behavior
- custom skill loading
- custom D2C templates in the agent loop
- native integrations for Instagram, Shopify, WhatsApp, or analytics
- multi-brand workspace switching
