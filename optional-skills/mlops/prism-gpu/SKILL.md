---
name: prism-gpu
description: Rent an NVIDIA GPU by the second from an agent wallet.
version: 0.1.0
author: Mika Winter (winterstacks)
license: MIT
dependencies: ["prismnetwork>=0.4.0,<0.6"]
platforms: [linux, macos, windows]
required_environment_variables:
  - name: PRISM_AGENT_KEY
    prompt: Prism agent wallet private key
    help: A wallet on Robinhood Chain (id 4663) holding USDG for the deposit and a little native ETH for gas. See https://prismnetwork.tech
    required_for: every SDK call; scripts/pick_gpu.py and scripts/verify_receipt.py read the public feeds without it
metadata:
  hermes:
    tags: [GPU, NVIDIA, CUDA, Compute Rental, USDG, Prism]
    related_skills: [lambda-labs, modal]
---

# Prism GPU Skill

Rents a dedicated NVIDIA GPU on Prism Network for a fixed window, runs commands on it
over SSH, and releases it. Payment settles on-chain in USDG on Robinhood Chain (id 4663)
from a wallet this agent controls, so there is no account to create and no API key to hold.
It does not manage clusters, persistent volumes, or long-running services; a lease is a
single machine for a bounded window and nothing survives its release.

## When to Use

- A job needs CUDA and the local machine has no NVIDIA card, or has one with too little
  VRAM for the model or kernel under test.
- Another skill in this tree needs a card the machine does not have. `flash-attention`
  states it has no CPU path. `slime` starts its container with `--gpus all`. `peft` sizes
  every example in GPU memory. `nemo-curator` has a CPU install that gives up the speedups
  it quotes. Lease a box, run that skill's own commands on it, release. Published capacity
  reports CUDA 12, so a skill that requires CUDA 13 or newer is not served by it: read
  `gpu.cuda_major` in the offer before planning the run.
- Several steps depend on each other and need the same box: build, then run, then collect.
- A result has to be citable by someone who does not trust the caller. Every settled lease
  publishes a receipt whose hash is committed on chain.
- Don't use for a single `nvidia-smi` or a five-second sanity check. Provisioning takes one
  to four minutes and the deposit books the whole window against the day's budget.
- Don't use for a whole interactive session on a GPU shell. The separate `hermes-plugin-prism`
  package (PyPI) makes the agent's own terminal a rented GPU, which is a different shape of
  purchase from the per-job leasing here.
- Don't use for a multi-GPU example. A lease is one machine with one card, so `--tp_size 4`
  and an eight-worker cluster have nothing to run on.
- Don't use before the job has run once at reduced size, on CPU where the code has a CPU
  path. A crash loop on a rented box bills for the crash loop.

## Prerequisites

Python 3.10 or newer: every published `prismnetwork` release requires it, and below that pip
answers "no matching distribution", which reads as a missing package rather than a version
floor. Check, then install, through the `terminal` tool:

```bash
python3 --version
pip install "prismnetwork>=0.4.0,<0.6"
```

If `python3` is older, run the scripts with a 3.10+ interpreter by name, such as `python3.12`.
The three SDK-backed scripts refuse on an older one and say why.

`PRISM_AGENT_KEY` holds the private key of the paying wallet. It needs USDG for the deposit
and a small amount of native ETH for gas, both on Robinhood Chain (id 4663). The wallet is
the credential: anything holding that key can spend the balance, so keep it out of a rented
box and out of anything the workload can read.

| Variable | Default | Purpose |
|---|---|---|
| `PRISM_AGENT_KEY` | none | The paying wallet. Every SDK call signs a session with it. |
| `PRISM_MAX_USDG` | `1` | Ceiling on one lease's deposit. |
| `PRISM_DAILY_BUDGET_USDG` | `5` | Ceiling on a rolling 24 hours. `0` removes it. |
| `PRISM_ESCROW` | `0xfD4228eEEfC49e4b76A0CD40af9fdd546220B2FD` | Escrow contract. |
| `PRISM_LEDGER_PATH` | `~/.prism/spend.json` | Where spend is recorded before money moves. |
| `PRISM_API_BASE` | `https://prismnetwork.tech` | Control plane for SDK calls. Point it elsewhere and the lease is not Prism's. The public offer and receipt feeds are read from `api.prismnetwork.tech` and do not follow it. |
| `PRISM_RPC_URL` | `https://rpc.mainnet.chain.robinhood.com` | Robinhood Chain RPC used to sign and confirm. |

`ssh` and `ssh-keygen` from OpenSSH must be on `PATH`; the SDK generates a throwaway key per
lease and connects with it. macOS and mainstream Linux ship both. On Windows they come from
the optional OpenSSH Client feature.

## How to Run

Every script below runs through the `terminal` tool from the skill directory, on Python 3.10
or newer. `python3` on a stock macOS is 3.9 and the SDK-backed scripts refuse on it, so use
the interpreter the environment installed `prismnetwork` into.

`lease_run_release.py` is the only script that spends. The other four need no wallet:
`pick_gpu.py` and `verify_receipt.py` read the public feeds, and `budget_check.py` reads
local limits.

```bash
python3 scripts/pick_gpu.py --min-vram-gb 24 --duration 900
python3 scripts/budget_check.py --duration 900 --rate-per-second 222
python3 scripts/lease_run_release.py "nvidia-smi" --duration 900 --min-vram-gb 24 --dry-run
python3 scripts/lease_run_release.py "nvidia-smi" --duration 900 --min-vram-gb 24
python3 scripts/verify_receipt.py --lease 34
python3 scripts/release_lease.py --list          # and --lease <id> to stop one
```

`--dry-run` prices the window, checks it against both ceilings and prints the plan without a
wallet and without signing anything. Run it first on any command that has not been leased
before.

`--min-vram-gb` takes the number on the card. Nodes publish a little under it, so the floor
converts at 1000 MiB per GB and `24` selects a card that reports 24564 MiB. Published
capacity is a handful of machines and it changes, so set the floor from what the job needs
and let `pick_gpu.py` say whether anything answers it.

To fund the exact offer that was priced instead of whatever is cheapest a minute later,
carry the node id and rate from `pick_gpu.py` into the lease:

```bash
python3 scripts/lease_run_release.py "nvidia-smi" --duration 900 \
    --node <node_id from pick_gpu.py> --rate-per-second <rate_per_second from pick_gpu.py>
```

## Quick Reference

| Call | Effect |
|---|---|
| `PrismAgent(private_key, escrow)` | Client. Both arguments are required; pass `DEFAULT_ESCROW`. |
| `.authenticate()` | Wallet-signature session. Every other call does it if needed. |
| `.offers()` | Live capacity. Free and reserves nothing, but signs a session like the rest. |
| `.balances()` | USDG and gas held by the wallet. |
| `.quote(image, duration_seconds, min_vram_mib=...)` | Price and machine, no payment. |
| `.fund_quote(quote)` | Pays that quote. Returns a `Lease`. |
| `.lease(image, duration_seconds, preferred_node_id=..., max_deposit=...)` | Quote and fund in one call. |
| `.run(lease, command, timeout=...)` | One command over SSH. Repeatable. |
| `.release(lease_id)` / `.end_lease(lease)` | Stops the meter. |
| `.leases()` | Every lease this wallet holds, open ones included. |
| `.infer(...)` / `.confidential_infer(...)` | Per-generation inference, no GPU to size. |
| `PrismToolset` | Framework adapter with the budget wired in. |

The wallet-free way to read capacity is the public endpoint
`https://api.prismnetwork.tech/v1/offers`, which is what `pick_gpu.py` uses.

Full surface in [references/python-sdk.md](references/python-sdk.md). Billing arithmetic and a
worked settlement in [references/billing.md](references/billing.md).

## Procedure

1. **Read capacity before planning around it.** Run `pick_gpu.py` with the VRAM floor the job
   actually needs. It reports the chosen offer, why it won, what the others were, and the
   deposit for the window. Supply is a handful of machines at a time, so treat a plan that
   assumes a specific card as unfinished until the offer list shows one. Done when a node id
   and a rate per second are in hand.

2. **Check the deposit against the limits.** Run `budget_check.py --duration N
   --rate-per-second R` with the rate from step 1. It spends nothing and exits `1` naming the
   limit that blocks the plan. Done when it prints `fits inside both limits`, or when the
   window has been shortened until it does.

3. **Pin the image.** The control plane matches on the digest, and the SDK rejects an image
   without one. Use `image@sha256:...`, never a tag. Omitted, the SDK's default is upstream
   Ollama at a pinned digest, which carries no training stack. Done when the image string
   contains `@sha256:`.

4. **Lease, run, release.** Run `lease_run_release.py "<command>" --node <node id>
   --rate-per-second <rate>` with the pair from step 1, so the machine that gets funded is
   the machine that was priced and checked. The script reprints the planned deposit, then
   caps the escrow at that figure and books the same figure against the day, so step 2 and
   this step cannot disagree. The release sits in a `finally` block, so a failing command
   still stops the meter, and both output streams are printed because the machine is gone
   a moment later. Read the exit code as well as the output: `0` ran and released, `4`
   funded and released but the command never ran, `3` the deposit is in escrow and the
   lease is still open. Done when the script prints `released`.

5. **Collect the receipt.** Settlement lands shortly after the release and publishes a
   receipt. Run `verify_receipt.py --lease <id>` with the id from step 4. It recomputes the
   receipt hash offline, then names the escrow event that commits the same figures on chain,
   which is what makes the charge checkable independently of anything Prism publishes. Lease
   ids restart at 1 with every escrow deployment, so the lookup is scoped to the escrow in
   `PRISM_ESCROW` and refuses to return a same-numbered lease from a retired one. Done when
   it prints `receipt_hash ... ok` for the escrow the lease was funded on.

For a multi-step job, hold the lease instead: `lease()` once, `run()` repeatedly, then
`end_lease()` in a `finally` block of your own. Write that driver with `write_file` and run
it through `terminal` instead of issuing the steps by hand, because an unreleased lease
keeps billing after the session moves on. If that driver dies mid-flight,
`scripts/release_lease.py --list` names what the wallet is still paying for and
`--lease <id>` stops it.

## Pitfalls

- **The deposit is the whole window, not the time used.** Funding a 900-second lease moves
  900 seconds of USDG into escrow up front. Releasing is what stops the meter. Settlement
  then charges the seconds between access opening and the release, and returns the rest. A
  lease nobody releases bills to the end of its window, so budget against the deposit.
- **A crash after funding leaves a paid machine running.** `PrismError` carries `.broadcast`:
  `False` before the funding transaction is signed, and the transaction hash after. On a hash,
  run `scripts/release_lease.py --list` and then `--lease <id>`. `lease_run_release.py` exits
  `3` only when the lease is still open, and prints those two commands with the funding hash,
  because a failure inside `lease()` often happens before any lease id comes back. A command
  that fails on a lease it did release exits `4`, and there is nothing to rescue.
- **A node id names a slot, and the machine behind it changes.** The same id can publish an
  RTX 5090 in one poll and an A6000 in the next, at a different rate. `--node` funds that id
  and the deposit is capped at the rate that was checked, so a machine that has moved is
  refused rather than funded quietly. Price the window again when that happens.
- **A rate read from the offer list can move before the lease is funded.** The deposit is
  capped at the figure that was checked, so a dearer quote is refused before anything is
  signed. Re-run `pick_gpu.py` and price the window again.
- **A lease id alone does not name a run.** Ids are numbered per escrow deployment and
  restart at 1 with each one, so the same number exists on retired escrows with different
  durations and charges. Cite a `receipt_id`, or a lease id together with its escrow.
- **Trust class `open` means the host operator can read anything the workload touches.**
  Treat a rented box as a public machine. Higher classes (`isolated`, `attested`,
  `confidential`) exist and are far rarer in the offer list; ask for one with
  `min_trust_class` and expect to find no capacity.
- **`staker_only` offers are usually the cheapest and usually unrentable.** They are reserved
  for bonded stakers. `pick_gpu.py` drops them by default and says so under `passed over`.
- **The spend ledger is shared.** `~/.prism/spend.json` is one file for every Prism client on
  the machine, and spend is written before the money moves. A crash between funding and the
  answer still counts against the day, which is the conservative direction.
- **`max_usdg` only lowers a ceiling.** A value above `PRISM_MAX_USDG` is clamped back down.
  Raise the environment variable instead of routing around it.
- **Nothing persists.** Files, caches, and processes die with the lease. Copy results off the
  box in the same command that produces them.

Error codes and what to do about each are in [references/troubleshooting.md](references/troubleshooting.md).

## Verification

Two checks, neither of which spends anything:

```bash
python3 scripts/verify_receipt.py --self-test
python3 scripts/pick_gpu.py --min-vram-gb 24
```

The first verifies two pinned settlement receipts offline, rejects an edited one, and checks
that a lease id published on three escrows resolves to the run on the escrow named rather
than to whichever entry the feed lists first. The second prints a live offer with a node id,
an hourly rate, and the deposit for the window, or exits `1` naming what each published offer
failed on. The node id and rate it prints are what step 4 takes.
