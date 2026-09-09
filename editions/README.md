# editions/

Optional **profile overlays** for North Forge. Each subfolder tailors the generic
chassis to a particular audience or deployment — a persona variant and, optionally,
pinned defaults.

An overlay is **additive and optional**. The base identity is always the repo-root
`SOUL.md` (the generic chassis voice). An overlay does not replace it — it is
applied *on top* when a deployment explicitly opts in. Nothing here is loaded
automatically, and nothing here changes how the engine runs.

## What's here

| Edition | What it adds | Who it's for |
| --- | --- | --- |
| [`field-service/`](field-service/) | Field-service-flavoured persona: plainer register, written for technicians and reps. | Deployments aimed at field techs / sales reps. |
| [`penny-pincher/`](penny-pincher/) | Warm, plain-language household-budgeting persona — bills, due dates, "what's left this month?". | A Basic-tier drive whose recipient wants a simple money helper as its front door. |
| [`pine-barron-farms/`](pine-barron-farms/) | Virtual-production partner: writes AI-generation cards in a studio format, holds canon, accessibility-first. Persona + guardrails only. | A drive pinned to the studio workflow. Canon packet is deploy-time, not in this repo. |

## How an edition becomes a live profile

Mechanically, **an edition is a Hermes profile distribution**: a folder with a
`distribution.yaml` manifest and a `SOUL.md` at its root (see
[`../website/docs/reference/profile-commands.md`](../website/docs/reference/profile-commands.md)).
The engine's own installer clones it into `<HERMES_HOME>/profiles/<name>/`:

```
hermes profile install editions/penny-pincher
```

`scripts\nf-setup.ps1` does this for you: pinning a drive to an edition installs
it from `editions/<name>/` first (if it isn't already a profile), then writes the
signed tier / pinned-edition provisioning record that
[`hermes_cli/nf_tier.py`](../hermes_cli/nf_tier.py) reads on every launch. On a
**Basic**-tier drive the pin is the only reachable edition; on **Full** it's just
the default landing profile and the switcher stays open.

```
scripts\nf-setup.ps1 -NonInteractive -Tier basic -Pin penny-pincher
scripts\nf-setup.ps1 -NonInteractive -Tier full  -Pin pine-barron-farms -Installed field-service,penny-pincher,pine-barron-farms
```

## What stays out of the public chassis

`editions/` carries **persona overlays and lightweight config** — voice, working
style, guardrails. It does **not** carry proprietary or personal vertical *content*:

- Named vertical skill-sets (Kyocera, Sales Edition, …) live in separate
  admin-gated repos/content, never here.
- `pine-barron-farms/` ships its persona and method, but its **studio canon
  packet** (characters, locations, episode history — specific personal creative
  content) is dropped in at deploy time under `profiles/pine-barron-farms/canon/`
  and is deliberately **not** `distribution_owned`. See
  [`pine-barron-farms/canon/README.md`](pine-barron-farms/canon/README.md).

See [`../BRANDING.md`](../BRANDING.md) for which surfaces North Forge owns.
