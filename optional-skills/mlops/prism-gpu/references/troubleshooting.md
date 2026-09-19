# Troubleshooting

## The install

`pip install "prismnetwork>=0.4.0,<0.6"` ending in "Could not find a version that satisfies
the requirement" is a Python version floor, not a missing package. Every published release
requires 3.10 or newer. On an older interpreter pip says `(from versions: none)`, and older
pips print nothing else, so there is no ignored-versions list to look for. Check
`python3 --version` first: a stock macOS `python3` is 3.9.

## When a lease fails

Read `.broadcast` on the `PrismError` first. It decides whether this is a retry or a rescue.

| `.broadcast` | What happened | What to do |
|---|---|---|
| `False` | Nothing was signed. | Fix the cause and retry. Nothing was spent. |
| `"0x..."` | The deposit is in escrow. | Release the lease before anything else. |
| `None` | Undetermined. | Treat it as spent: `release_lease.py --list`, then release what is billing. |

## Rescuing a funded lease

Two commands through the `terminal` tool, from the skill directory:

```bash
python3 scripts/release_lease.py --list         # every lease, and which are still billing
python3 scripts/release_lease.py --lease 77     # stops the meter; settlement follows
```

The script calls `agent.leases()` and `agent.release(lease_id)`. `release()` takes the id
alone, so it works when the `Lease` object and the process that held it are both gone. The
error's `.body` carries `funding_hash`, `lease_id`, and `key_path` after a broadcast. The
private key stays on disk deliberately: it is the only way into a machine the wallet is
paying for.

`scripts/lease_run_release.py` exits `3` only when the lease is still open, and prints both
commands. A failure on a lease it did manage to release exits `4`: the meter is stopped and
there is nothing to rescue.

## Codes before any money moves

| Code | Cause | Fix |
|---|---|---|
| `wallet_unfunded` | No USDG, or no native ETH for gas. | Fund the wallet on Robinhood Chain (id 4663). Both tokens are required. |
| `image_must_be_digest_pinned` | The image string has no `@sha256:`. | Resolve the tag to a digest and pass that. |
| `cost_exceeds_max` | The quote is dearer than `max_deposit`. | Shorten the window, choose a cheaper offer, or raise the ceiling. |
| `invalid_trust_class` | Not one of `open`, `isolated`, `attested`, `confidential`. | Correct the spelling. |
| `invalid_command` / `request_too_large` | The batch command is empty or over 8 KiB. | Fetch the payload on the box instead of inlining it. |
| `control_plane_unreachable` | Network or a control-plane blip. | Retry. Nothing was reserved. |
| `pre_broadcast_failure` | Anything else before signing. | Read `.body["cause"]`. |

An empty `offers()` list is not an error. It means no machine at that trust class is online.
Supply is a handful of machines, so widen the VRAM floor, drop to `open`, or wait.

At quote time the control plane says the same thing one step later, as `no_capacity`,
`no_offer`, or `no_matching_offer`: the constraint matched nothing bookable. Re-read
`offers()` rather than retrying the same quote.

## Codes after the deposit is in escrow

| Code | Meaning | What to do |
|---|---|---|
| `tx_reverted` | The funding transaction failed on chain. | Nothing is leased. Check the USDG allowance and the gas balance. |
| `confirmation_timeout` | Broadcast, then the RPC stopped answering. | Do not resend. Look the hash up on the explorer, then run `release_lease.py --list`. |
| `malformed_lease_record` | Funded, but the control plane returned no lease id. | Take the id from `release_lease.py --list` and release it. |
| `access_timeout` | Funded, and the box never opened access. | Release it. The receipt settles as `provisioning_timeout` and refunds. |
| `lease_failed_after_funding` | Any other post-funding failure. | Release the lease named in `.body`. |

## Codes while running

| Code | Meaning | What to do |
|---|---|---|
| `ssh_access_unavailable` | The node granted gateway access, which has no SSH endpoint. | Use a batch lease (`command=`) for this node, or pick another offer. |
| `ssh_keygen_failed` | `ssh-keygen` is missing from `PATH`. | Install OpenSSH. On Windows it is an optional feature. |
| `host_key_unpublished` / `host_key_unavailable` | `require_host_key=True` and the node published none. | Accept the weaker guarantee, or wait for capacity that publishes one. |
| `host_key_mismatch` | The host key differs from the one pinned earlier. | Stop. Do not release the lease. Report the node id. |

`agent.run()` already retries a connection to a box that is still booting, for up to four
minutes by default. An exit code of `255` with `Connection refused` that survives that window
is a real failure rather than a slow boot.

A command that exits non-zero is not an SDK error. `run()` returns `{"code", "stdout",
"stderr"}` and the caller reads `code`.

## Budget refusals

`BudgetError` never touches the network. It reads as one of:

- the deposit is over `PRISM_MAX_USDG`, so shorten the window or raise the variable;
- the deposit is over what is left of `PRISM_DAILY_BUDGET_USDG` in the rolling 24 hours;
- `PRISM_MAX_USDG` is above `PRISM_DAILY_BUDGET_USDG`, which is refused at startup;
- the ledger file is unreadable, in which case nothing may spend until it is fixed.

`scripts/budget_check.py` prints the numbers behind whichever it is.

## Receipts

A lease that has not settled yet has no receipt, and `verify_receipt.py --lease <id>` says so
rather than inventing one. Settlement follows the release; give it a minute.

Lease ids are numbered per escrow deployment and restart at 1 with each one, so the published
feed holds several receipts for most low ids. `--lease` resolves against one escrow only,
`PRISM_ESCROW` or the live default, and reports the escrows that do hold the id rather than
returning one of them. Pass `--escrow` to check a lease funded on a retired deployment.

A `receipt_hash MISMATCH` on a receipt fetched from the feed means the payload was edited
between the feed and the checker. Do not cite it. Re-fetch from
`https://api.prismnetwork.tech/proof/receipts/<receipt_id>.json` and check again.
