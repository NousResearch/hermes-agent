# pine-barron-farms edition

A **virtual-production partner** persona for the North Forge chassis: writes
ready-to-shoot AI-generation cards in a studio's house format, holds canon
consistency, and keeps a multi-tool pipeline (generation / narration / music /
assembly) pointed at one source of truth. Accessibility-first working style —
one action per step, name the tool every time, hold all the state for the user,
every reply ends CHANGE BRIEF / WHERE WE ARE / NEXT STEP.

- **Base persona** (always): repo-root [`../../SOUL.md`](../../SOUL.md).
- **This overlay**: [`SOUL.md`](SOUL.md) — the persona, the working style, and two
  hard guardrails carried regardless of framing: a **real-public-figure guardrail**
  (no photoreal recognizable depictions of real people, no "fiction" exception,
  retire rather than manage a drifting likeness) and a **content covenant** (warm,
  wholesome, endings up; no living person shown harmed; humor from contrast).
- **Not shipped here**: the studio canon packet and the exhaustive generation
  method — those are personal creative content, dropped in at deploy from
  admin-gated material. See [`canon/README.md`](canon/README.md).

Migrated from the `PineBarronFarms_ABMS_v22` prompt-system package
(`D:\KB_PROJECT_2026\PBF_v22_FINAL`): the GPT-instructions "voice + working style +
guardrails" translated directly into a Hermes persona overlay; the canon packet,
episode cards, and platform-transfer notes did not (deploy-time / not public).

## Who it's for

A drive pinned to `pine-barron-farms` opens straight into this production persona.
Works as a switchable edition on a Full-tier drive too.

## What it is, mechanically

A **Hermes profile distribution** (`distribution.yaml` + `SOUL.md`):

```
hermes profile install editions/pine-barron-farms   # -> <HERMES_HOME>/profiles/pine-barron-farms/
```

`scripts\nf-setup.ps1` runs this automatically when a drive is pinned to
`pine-barron-farms`. See [`../README.md`](../README.md).
