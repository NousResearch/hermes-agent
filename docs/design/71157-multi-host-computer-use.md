# Design: Multi-host `computer_use` via paired Cua Driver hosts

> **Status:** design proposal (not implemented)  
> **Tracking:** https://github.com/NousResearch/hermes-agent/issues/71157  
> **Upstream protocol:** https://github.com/trycua/cua/issues/2562 · RFC PR https://github.com/trycua/cua/pull/2563  
> **Blocked on:** Cua Driver paired-host attach API (pair/revoke, host list, authenticated remote tools/MCP/SDK)

## Intent

Allow Hermes on **comp1** to drive desktops on **comp2** / **comp3** through **Cua’s** paired multi-host Driver connection, without requiring a full Hermes install on every target.

```text
comp1  Hermes  ── computer_use ──► local cua-driver
                 └─ host=comp2 ──► paired cua-driver agent on comp2
                 └─ host=comp3 ──► paired cua-driver agent on comp3
```

## Layering (non-negotiable)

| Layer | Owns |
| --- | --- |
| **trycua/cua** | Pairing, auth, host registry, remote Driver framing, OS session correctness |
| **Hermes** | `computer_use` host routing, approvals text, `hermes computer-use doctor --host`, skill/docs UX |

Hermes must **not** ship a parallel remote click/AX stack (SSH+pyautogui, raw TCP automation, unauthenticated HTTP desktop APIs).

## Defaults

- Unconfigured Hermes remains **local-only** (today’s behavior).
- Remote hosts require explicit opt-in (`computer_use.allow_remote` or equivalent in `config.yaml`) **and** successful Cua pairing.
- No new non-secret `HERMES_*` env vars for behavior.

## Proposed surface (when unblocked)

### Config

```yaml
computer_use:
  default_host: local   # local | <paired name>
  allow_remote: false   # opt-in
```

### Tool

Extend existing `computer_use` (avoid a second core tool):

- optional `host` argument, defaulting to `default_host`
- `action=list_hosts` → Cua registry snapshot
- SOM/element tokens scoped **per host**

### CLI

```text
hermes computer-use hosts
hermes computer-use doctor --host <name>
```

Thin wrappers over Cua CLI/SDK once stable (`cua-driver hosts list --json`, etc.).

### Approvals

Destructive GUI approvals must include **host identity** (“Allow click on **comp2**?”). Consider stricter default approval mode for remote hosts.

### Backend attach modes

Build on local-daemon attach ideas ([#65259](https://github.com/NousResearch/hermes-agent/issues/65259)):

| Mode | Meaning |
| --- | --- |
| `spawn` | Current local `cua-driver mcp` child |
| `local_daemon` | Existing local daemon (#65259) |
| `remote_host` | Paired host via Cua |

## Non-goals

- Implementing remote UIA/AX inside Hermes
- Hermes-on-every-PC as the feature (documented **workaround** only)
- Peer Hermes messaging ([#69147](https://github.com/NousResearch/hermes-agent/issues/69147)) — complementary, different problem
- RDP/VNC-as-backend

## Phases

0. **Coordinate** on Cua RFC #2562 / PR #2563 attach contracts.  
1. **Hermes plumbing** after Cua Phase A: host resolve, remote backend, list_hosts, host-aware approvals, tests with mocks.  
2. **UX**: Desktop host picker, slash command, doctor `--host`, skill updates.

## Acceptance (implementation PR later)

- Local-only regression-free without remote config
- Paired remote capture+click path works when Cua supports it
- Cross-host SOM index reuse impossible
- Revoked/offline host → clear error
- Docs + skill updated; default-off remote

## Workaround today

Install Hermes (and computer-use) on each machine, or use messaging gateways to talk to per-machine agents. That is multi-**agent**, not multi-host `computer_use`.

## Decision ask

If Cua accepts paired multi-host Driver, will Hermes accept a follow-up **consumer** implementation PR that is default-off, host-aware, and protocol-thin?
