# prismnetwork 0.4.0 surface

Install through the `terminal` tool: `pip install "prismnetwork>=0.4.0,<0.6"`. Python 3.10 or
newer; older interpreters get "no matching distribution" rather than a version message.

```python
from prismnetwork import PrismAgent, PrismError, DEFAULT_ESCROW, DEFAULT_IMAGE
import os

agent = PrismAgent(os.environ["PRISM_AGENT_KEY"], DEFAULT_ESCROW)
```

`PrismAgent(private_key, escrow, api_base=..., rpc_url=..., require_host_key=False)`. The
key is 32 bytes of hex with or without `0x`. `require_host_key=True` refuses any node whose
grant does not name its SSH host key, which most published capacity does not, so it trades
supply for the guarantee.

## Reading, which costs nothing

Every call on the agent authenticates first, `offers()` included, so all of these need a
private key even though none of them spend.

| Call | Returns |
|---|---|
| `agent.authenticate()` | Session dict. A wallet signature over a server challenge. |
| `agent.offers(min_trust="open")` | List of live offers. |
| `agent.balances()` | `{address, usdg, eth}` in base units and wei. |
| `agent.leases()` | Every lease this wallet holds. |
| `agent.access(lease_id)` | SSH endpoint of an open lease. |

An offer carries `node_id`, `gpu.{model, vram_mib, cuda_major}`, `rate_per_second` in USDG
base units, `trust_class`, `benchmark_score`, `staker_only`, `online`, and `bonded`.
`scripts/pick_gpu.py` reads the same list without a wallet through the public endpoint
`https://api.prismnetwork.tech/v1/offers`.

## Paying

```python
lease = agent.lease(
    image=DEFAULT_IMAGE,          # must contain @sha256:
    duration_seconds=900,
    min_vram_mib=24000,           # MiB, and nodes report under nominal: a 24 GB card
                                  # publishes 24564, so 24576 matches nothing
    preferred_node_id=node_id,    # the offer that was priced, not the next one
    max_deposit=199_800,          # rate_per_second × duration; a dearer quote is refused
    min_trust_class="open",
)
try:
    print(agent.run(lease, "nvidia-smi", timeout=120))
finally:
    agent.end_lease(lease)
```

`lease()` authenticates, refuses an unfunded wallet, quotes, compares against `max_deposit`,
and funds. Everything before funding costs nothing and raises with `broadcast=False`.

Split it when a human approves the price: `quote = agent.quote(...)` shows `quote_id`,
`node_id`, `maximum_escrow`, and `duration_seconds`, then `agent.fund_quote(quote)` pays that
exact quote. An expired quote is refused rather than silently replaced.

A `Lease` carries `lease_id`, `access`, `key_path`, `funding_hash`, `quote`, and
`deposit_micros` with `deposit_source` saying whether that figure was read from the funding
receipt or fell back to the quote's ceiling.

Pass `command=` to `lease()` or `quote()` for a batch lease. The node runs the command and
reports its output, there is no SSH access to wait for, and the call returns a `BatchLease`
with `result` instead of `access`.

## Releasing

`agent.end_lease(lease)` releases and removes the local key. `agent.release(lease_id)`
releases by id, which is what to call after a crash when only the id survives. A refused
release raises, because the wallet is still paying for the machine.
`scripts/release_lease.py` is both calls behind `--list` and `--lease <id>`, so a rescue is
a shell command rather than a driver to write.

## Errors

`PrismError` has `.status`, `.code`, `.body`, and `.broadcast`. Only `.broadcast` says
whether money moved:

| `.broadcast` | Meaning |
|---|---|
| `False` | Nothing was signed onto the wire. The wallet is untouched. |
| `"0x..."` | The funding transaction hash. The deposit is in escrow. |
| `None` | Undetermined. Treat it as spent and check `agent.leases()`. |

After a broadcast, `.body` also carries `funding_hash`, `lease_id` when one was assigned, and
`key_path`. The key stays on disk in that case because it is the only way into a machine the
wallet is paying for.

## Budget

```python
from prismnetwork import read_budget, SpendLedger, call_ceiling, record_spend

budget = read_budget()                       # reads PRISM_MAX_USDG, PRISM_DAILY_BUDGET_USDG
ledger = SpendLedger(budget.ledger_path, budget.daily_micros, budget.max_per_call_micros)
ceiling = call_ceiling(0.2, budget.max_per_call_micros)   # lowers only, never raises
deposit = rate_per_second * duration_seconds              # what this lease will actually pull
lease = record_spend(ledger, "my_tool", min(deposit, ceiling), fund)
```

Reserve the deposit, not the ceiling. Reserving the ceiling books the full per-lease cap
against the day until settlement corrects it, which turns a plan that cleared
`budget_check.py` into a lease the ledger refuses.

`record_spend` writes the reservation, runs `fund`, then settles the entry to what the escrow
pulled. It hands the reservation back only when the failure proves nothing reached the chain.
`ledger.status()` and `ledger.remaining()` report the day without spending anything;
`scripts/budget_check.py` prints both.

## Inference, no GPU to size

`agent.infer(prompt=..., model=..., max_tokens=..., max_usdg=0.05)` buys one generation from
the open tier and returns `text`, `usage`, `price_usdg`, `tx`, and `receipt_id`. The supplier
running that GPU can read the prompt and the answer.

`agent.confidential_infer(...)` serves the request from a model inside a GPU TEE, encrypted
to a key the enclave's attestation quote commits to, established before anything is sent. A
check that fails raises `ConfidentialError` and no prompt leaves the process. Verifying the
quote needs the extra `confidential`: `pip install "prismnetwork[confidential]>=0.4.0,<0.6"`.

## Framework adapter

`PrismToolset` exposes `wallet`, `budget_status`, `list_gpus`, `lease_and_run`, `run`, and
`end_lease` as string-returning tools with the budget already wired in, for LangChain-style
agents. It registers an `atexit` hook that releases what it opened. A process killed outright
skips that hook and leaves the lease billing until its window ends, which is why
`scripts/lease_run_release.py` releases in a `finally` block of its own.

## The terminal backend

`hermes-plugin-prism` on PyPI is a separate package that makes the agent's whole shell a
rented GPU: one lease per session, released on cleanup or after five idle minutes. Use it
when the work is a session rather than a job. This skill covers per-job leasing, where the
lease is opened and closed inside one script.
