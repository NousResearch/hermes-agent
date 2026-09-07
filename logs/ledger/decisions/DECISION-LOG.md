# North-Forge Decision Register

Append-only. One row per **open judgment call** — a choice between defensible
options, not a fault. "Rebrand vs thin downstream", "keep or delete the attic
clone", "rename the `hermes` command" are decisions; a broken build or a leaked
secret is an error (`errors/ERROR-LOG.md`).

Never delete a row — close it by moving the block under `## Resolved` with a
`DECIDED` / `DEFERRED` / `DROPPED` status, a one-line rationale, and the
implementing `CHG-` id(s). A superseded or migrated item keeps its old id and
gains a `Superseded-by:` / `Supersedes:` link.

`DECISION-` id scheme, the `Confidence` vocabulary, and `Run:` ids are defined in
[`README.md`](../README.md).

---

## Open

### DECISION-2026-09-06-002 — Attic clone — keep or delete it?

- **Opened:** 2026-09-06 · **Base:** hermes@693641aa8b (0 behind upstream/main)
- **Run:** RUN-2026-09-06-001
- **Source:** `AUDIT-2026-09-06-001` F-01 (`CHG-2026-09-06-002`) and its open-item #5;
  restated in `AUDIT-2026-09-06-002` open-item #5.
- **Confidence:** Confirmed Fact — `D:\north-forge-agent-attic\nested-clone-2026-09-06\`
  is a pristine second clone (~869 MB, `.git` ≈ 713 MB), zero unique commits,
  same `origin`/`upstream` remotes; it was moved there out of the checkout, never deleted.
- **Supersedes:** — (this was never an `ERR-` row; it lived only as an audit
  finding + recommendation).
- **The call:** now that `origin/main` carries the real work (`e6c97b43ef`, pushed
  and verified), is the local safety copy still worth ~869 MB?
- **Options:**
  - **A — Delete** `D:\north-forge-agent-attic\nested-clone-2026-09-06\` — reclaim
    ~869 MB. `origin/main` + a fresh `git clone` fully reconstruct it.
  - **B — Keep** it as a cold offline backup (useful only if GitHub is
    unreachable *and* the working `.git` is also lost — a narrow scenario).
- **Leaning:** **A (delete)** — the pushed, verified `origin/main` removes the
  reason it was kept. Low urgency (disk is 76% free).
- **Blocking:** nothing — disk space only.
- **Owner:** Kenneth C. Walker Jr.
- **Status:** OPEN

---

## Resolved

### DECISION-2026-09-06-001 — Fork identity — rebrand vs thin downstream?

- **Opened:** 2026-09-06 · **Base:** hermes@693641aa8b (0 behind upstream/main)
- **Run:** RUN-2026-09-06-001 (opened) · RUN-2026-09-06-002 (implemented)
- **Source:** `AUDIT-2026-09-06-001` §6 (full README review) and F-06 / F-08
- **Confidence:** Confirmed Fact — the branding state is verified (every `README*.md`,
  `SOUL.md`, `LICENSE`, `package.json`, `pyproject.toml` field is still upstream's);
  the *choice* between the two shapes is what is open.
- **Supersedes:** the identity half of `ERR-2026-09-06-002` (migrated here — the
  version-drift half of that ERR was a real fault and stays in `ERROR-LOG.md`,
  RESOLVED by `CHG-2026-09-06-014`).
- **The call:** does north-forge present as its own product, or stay a
  lightly-marked fork kept rebased on `NousResearch/hermes-agent`?
- **Options:**
  - **A — Thin downstream:** add a short north-forge note atop `README.md` + a
    root `CLAUDE.md` (provenance + what north-forge adds + the ledger workflow);
    leave `SOUL.md` / `package.json` / `pyproject.toml` as upstream's. Cheapest
    upstream merges forever; north-forge stays visibly a downstream user.
  - **B — Full rebrand:** rewrite the top of `README.md`, `SOUL.md` persona, the
    `name` / `repository` / `homepage` fields in `package.json` and
    `pyproject.toml`, add the maintainer's copyright line alongside Nous's in
    `LICENSE` (MIT — Nous's line stays). More friction on every future
    `upstream/main` merge; north-forge reads as its own project.
- **Leaning:** **B (full rebrand)** — selected by the maintainer on 2026-09-06.
- **Blocking:** `NF-v0.2.0` (the changelog reserves `NF-v0.2.0` for the first
  identity commit). Nothing else.
- **Owner:** Kenneth C. Walker Jr.
- **Decided:** 2026-09-06 — chose **B (full rebrand)**. Implemented in one
  branding-only commit under `RUN-2026-09-06-002`: `README.md` top-level identity
  rewritten, `SOUL.md` re-voiced to North Forge's established persona,
  `package.json` `name` / `repository` / `homepage` / `bugs` repointed to
  `kwalker7631/north-forge-agent` (+ `package-lock.json` root-name sync),
  `pyproject.toml` gained `[project.urls]`, `LICENSE` gained the maintainer's
  copyright line alongside Nous Research's. Per the carve-out in this entry, the
  `pyproject.toml` **distribution name `hermes-agent` was left unchanged** (never
  published, ~19× internal refs, pinned in `uv.lock`). No application code, no
  `.env`, no attic-clone change — agent behaviour identical before and after.
  Cut `NF-v0.2.0` (MAJOR). Committed locally, **not pushed** (held for review).
  Implementing changes: `CHG-2026-09-06-020`, `CHG-2026-09-06-021`,
  `CHG-2026-09-06-022`, `CHG-2026-09-06-023`, `CHG-2026-09-06-024`.
- **Status:** DECIDED

---

## Register (quick scan)

| ID | Date | Area | Question | Status | Decided |
| --- | --- | --- | --- | --- | --- |
| DECISION-2026-09-06-001 | 2026-09-06 | Fork identity | Rebrand vs thin downstream? | DECIDED — B (full rebrand), landed `NF-v0.2.0` (CHG-2026-09-06-020..024) | 2026-09-06 |
| DECISION-2026-09-06-002 | 2026-09-06 | Repo hygiene | Keep or delete the attic clone? | OPEN — leaning A (delete) | — |
