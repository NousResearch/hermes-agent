# First Principles

Aristotle's *protai archai*. Decompose a problem to assumptions you trust, then rebuild without inheriting anything by default. Often paired with "5 Whys" excavation of why each assumption is in place.

## When to use

- A domain has accreted practice that may no longer be load-bearing
- You're in an unfamiliar domain and bootstrapping understanding
- You suspect the standard framing is wrong
- Trying to reduce cost or complexity (accumulated overhead is often the main cost)
- Teaching the domain (first-principles reconstruction surfaces what beginners actually need)

## Don't use when

- You don't know the domain well enough — first principles applied by an outsider produces confidently wrong answers
- Transaction costs of replacement exceed the gains
- Problem is irreducible (aesthetic, social, gestalt — decomposition destroys what makes it coherent)
- You're trying to seem original — performance of first-principles thinking is slop

## Procedure

1. **State the problem precisely.**
2. **List assumptions in the conventional solution.** What does the standard approach take for granted? List 5–10, including ones that "go without saying."
3. **Categorize each:**
   - **Physical** — law of nature; can't be relaxed.
   - **Informational** — logical / mathematical / information-theoretic; can't be relaxed without contradiction.
   - **Conventional** — could be different; matters for compatibility.
   - **Historical** — was necessary at some point; may not be now.
   - **Pedagogical** — simplification used for teaching; may not be how experts actually do it.
4. **For each non-physical / non-informational assumption:** still load-bearing? Conventional and historical assumptions are where the gains live.
5. **Rebuild.** Construct a candidate respecting only physical and informational constraints, plus your specific context.
6. **Apply Chesterton's fence.** For each element you've removed, find the original reason it was added. If you can't find a reason, *don't conclude there isn't one* — assume you haven't looked hard enough.
7. **Decide whether to switch.** Even when the rebuild is technically better, consider transaction cost, ecosystem compatibility, team familiarity.

## Worked example

**Problem**: typical CRUD web app — login, dashboard, few CRUD entities. Conventional stack: React + Node/Express + PostgreSQL + REST API + managed platform. ~12,000 LOC, monthly hosting ~$100.

**Assumptions**:
- React: conventional, was historical (SPA promise ~2014), pedagogical (taught everywhere).
- Backend separate from frontend: conventional; informational *if* multi-client, otherwise historical.
- PostgreSQL: physical *if* concurrency/ACID required; otherwise conventional.
- REST API between frontend and backend: was informational (network boundary), now historical for single-client apps.
- Managed platform: conventional; was historical (datacenter complexity); pedagogical.

**Context**: 100 users, ~10 MB data, no real-time, single client (web), no HA constraint.

**Rebuild**:
- Server-rendered HTML + small JS islands. (No SPA. No build pipeline. No API layer.)
- SQLite single file. (No PG server. Backup = copy a file.)
- Single small VM. (No managed platform. Deploy = `rsync` + `systemctl restart`.)
- Single Go/Python/Ruby binary.

**Result**: ~1,500 LOC vs 12,000. ~$5/month vs $100. Tradeoffs: less impressive on resume, fewer contractors familiar with this style, no immediate path to 1M users.

**Chesterton's fence**: the conventional choices are load-bearing for *some* applications. The rebuild is correct *only* for this app's constraints. A different app — high concurrency, multiple clients, large data — needs different choices.

## Anti-slop notes

- The biggest slop is the *performance* of first-principles thinking. "I'm going to think from first principles" followed by a slightly-rearranged conventional answer is slop. Output should look measurably different.
- Don't claim first principles when you're applying common sense.
- Avoid the engineer-hero archetype. Real first principles often reveals what the field already knows.
- Don't recommend removing structure you don't understand. Chesterton's fence applies hard.
