# penny-pincher edition

A **household budgeting / expense-tracking persona** for the North Forge chassis.
Warm, plain-language, one-step-at-a-time. Built to be somebody's pinned front door
— not a power-user finance tool.

- **Base persona** (always): repo-root [`../../SOUL.md`](../../SOUL.md) — the generic
  North Forge chassis voice.
- **This overlay**: [`SOUL.md`](SOUL.md) — first-person, plainer register: "I help you
  keep an eye on the household money." Tracks bills and due dates, answers
  "what's left this month?" / "can we afford it?", flags spending changes calmly.
  No jargon, no nagging, no investment advice.

## Who it's for

Basic-tier drives whose recipient wants a friendly money helper and nothing more —
e.g. a drive pinned to `penny-pincher` so it opens straight into this persona with
no path to anything else. Fine on a Full-tier drive too, as one switchable edition.

## What it is, mechanically

A **Hermes profile distribution** — `distribution.yaml` + `SOUL.md` at the root of
this folder. It becomes a live profile the ordinary way:

```
hermes profile install editions/penny-pincher      # -> <HERMES_HOME>/profiles/penny-pincher/
```

`scripts\nf-setup.ps1` does this for you: pinning a drive to `penny-pincher`
installs the edition first, then writes the signed provisioning record. See
[`../README.md`](../README.md).

## What it does not carry

No skills, no MCP servers, no proprietary content, no external credentials. It is
persona only — the chassis's own tools (memory, files, web, image generation, …)
do the work.
