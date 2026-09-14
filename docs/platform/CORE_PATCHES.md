# Core Patch Budget

Every change to an **upstream-owned** file, with the reason and the seam that was missing.

**The target is near zero.** This ledger is reviewed at every upstream merge. Growth is the signal
that a contributor reached for a core edit where an extension point existed
(`ARCHITECTURE_BOUNDARIES.md` §3).

Before adding a row, answer: *which extension point did I fail to find?* If you cannot name one,
the change probably belongs in `nova/`.

| # | File | Change | Why | Seam that was missing | Merge risk |
|---|---|---|---|---|---|
| 1 | `AGENTS.md` | One row added to the routing table, pointing `nova/`, `deploy/` and `docs/platform/` at `nova/AGENTS.md` | Without it the boundary rules are invisible to future contributors and AI assistants, who are instructed to read the routing table before editing an area | None — this *is* the repo's mechanism for area guidance; it simply has to be registered | **Low.** A single row appended to a stable table |

## Not counted here

Paths that do not exist upstream and therefore cannot conflict:

- `nova/**`, `deploy/**`, `docs/nova/**`, `customer/**`
- `tests/nova/**`
- `scripts/check_protected_identifiers.py`

Brand-owned files replaced wholesale and resolved by the `merge=brandours` driver
(`README.md`, `SECURITY.md`, `CONTRIBUTING.md`, `locales/*.yaml`, `assets/**`) are tracked by that
driver, not by this ledger — they are replacements, not patches.

## Anticipated future entries

Recorded here so their cost is visible before they are spent. **None of these are approved yet;
each needs a decision at the time.**

| Candidate | Purpose | Phase | Estimated size |
|---|---|---|---|
| `pyproject.toml` `[project.scripts]` | Add the `nova` console script beside `hermes` | 2+ | 1 line |
| `hermes_constants.py` `get_hermes_home()` | `$NOVA_HOME` → `~/.nova` → `$HERMES_HOME` → `~/.hermes` | 2+ | 1 function |
| Environment accessor | `NOVA_*` resolved before `HERMES_*` | 2+ | 1 helper |
| `providers/__init__.py` | Discover `nova.plugins` alongside `hermes_agent.plugins` | 2+ | 1 line |
| `toolsets.py` | `nova-*` aliases to existing definitions | 2+ | alias map |
| `.gitattributes` | `merge=brandours` for brand-owned paths | 2 | ~6 lines |

Six small additive hook points is the whole budgeted cost of an independent product identity.
