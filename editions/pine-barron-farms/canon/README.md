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

## If nothing is dropped in

The edition still loads and runs on the persona and method in `SOUL.md` alone —
it just has no studio-specific facts to enforce, and will say so plainly when
asked about canon.
