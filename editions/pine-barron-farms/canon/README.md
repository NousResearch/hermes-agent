# canon/ — deploy-time, not in the public chassis

This edition's persona and method ship in [`../SOUL.md`](../SOUL.md). The **studio
canon packet** — the single source of truth for characters, locations, props,
plates, episode history, open patches, and the full generation method — is **not**
in this public repository, for the same reason the other verticals aren't
([`../../README.md`](../../README.md)): it is specific personal creative content.

## How the real canon gets here

At deploy time, an admin drops the current canon packet into this folder as:

```
canon/PINE_BARRON_FARMS_CANON.md      (or .txt — the studio's master packet)
```

on the provisioned drive, under `<HERMES_HOME>/profiles/pine-barron-farms/canon/`.
`SOUL.md` already instructs the agent to *"follow the loaded canon packet exactly."*
`canon/` is intentionally **not** `distribution_owned` in `distribution.yaml`, so
`hermes profile update` refreshes the persona without ever touching the packet.

## How the packet reaches the agent's context

`SOUL.md` used to *say* "I follow the loaded canon packet" while nothing loaded
it. Now `load_canon.py` (this folder) does, via existing Hermes mechanisms — no
engine change:

- **a session-start command** — `python canon/load_canon.py`. `SOUL.md` instructs
  the agent to run this before its first substantive reply and treat the output as
  canon.
- **the `pbf-canon` skill** (`../skills/pbf-canon/SKILL.md`) — when preloaded
  (`hermes -s pbf-canon`) or invoked (`/pbf-canon`), its body — which tells the
  agent to run the loader and how to read the result — is assembled into the
  system prompt.

Either way the output is a `canon loaded: <path>, sha256=<hash>, <n> bytes`
receipt followed by the packet between `--- BEGIN PINE BARRON FARMS CANON ---`
markers — or, when nothing was dropped in, a `WARNING: canon packet NOT FOUND`
message naming every path checked.

## If nothing is dropped in

The edition still loads and runs on the persona and method in `SOUL.md` alone —
`load_canon.py` prints the not-found warning, and `SOUL.md` tells the agent to
say so plainly and not guess at characters, plates, episodes, or patches.
