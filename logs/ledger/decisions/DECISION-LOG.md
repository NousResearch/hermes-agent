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

### DECISION-2026-09-07-002 — Rebrand-claim wording — "engine used unmodified" / "full rebrand"

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main)
- **Run:** RUN-2026-09-07-006
- **Source:** Codex audit **F-07** (see `logs/CODEX-AUDIT-2026-09-07.md`).
- **Confidence:** Confirmed Fact — the phrases and the counter-examples are
  verified in the tree.
- **Supersedes:** —
- **The call:** two `README.md` / `BRANDING.md` claims are broader than the code:
  1. *"engine used unmodified"* — fork commits do modify application files
     (`agent/prompt_builder.py`, `cli.py`, `hermes_cli/**`). Accurate would be
     *"core behavior retained; identity entry points patched."*
  2. *"full rebrand"* — true only under `BRANDING.md`'s narrow ownership taxonomy;
     it is not a full operator journey while some visible CLI strings (interactive
     welcome, chat subparser description) still say Hermes and (pre-`CHG-011`)
     Quick Install installed upstream. Those may be deliberate compatibility
     surfaces, but the boundary should be stated plainly.
- **Options:**
  - **A — Tighten the wording.** Replace the two phrases with the precise form
    above; add one sentence in `BRANDING.md` naming the deliberately-Hermes
    visible surfaces. No identifier churn.
  - **B — Leave as-is.** Defensible under `BRANDING.md`'s stated taxonomy; the
    detailed category tables already qualify it. Cost: an operator reading only
    the headline claims is mildly misled.
- **Leaning:** A (cheap, honest, no code risk) — **not yet done**; logged for
  scheduling.
- **Blocking:** nothing.
- **Owner:** Kenneth C. Walker Jr.
- **Status:** OPEN

### DECISION-2026-09-06-003 — Install model — drive-native run-in-place vs machine-local managed install?

- **Opened:** 2026-09-07 · **Base:** hermes@693641aa8b (0 behind upstream/main)
- **Run:** RUN-2026-09-07-001
- **Id note:** the `2026-09-06` date in the id is kept at the owner's request, for
  continuity with the `-001` / `-002` rebrand batch; the entry was actually opened
  2026-09-07. (The daily-reset id rule in `README.md` is relaxed here by owner call.)
- **Source:** the consolidated pass of 2026-09-07 (step 4); the "portable-first"
  principle in [[north-forge-architecture]]; the minimal bootstrap shipped this
  pass (`CHG-2026-09-07-005`).
- **Confidence:** Field-Reasoned — the two shapes and their trade-offs are as
  stated; nothing here is implemented or ratified beyond the *minimal* bootstrap.
- **Supersedes:** —
- **The call:** when North Forge is deployed on a drive, does it **run in place
  from the drive** (portable, self-contained, nothing installed on the host) or
  does it perform a **managed install into a machine-local location**
  (`~/.hermes`, `%LOCALAPPDATA%\hermes`) the way upstream's installer does?
- **Options:**
  - **A — Drive-native run-in-place:** the venv and `HERMES_HOME` data folder live
    as siblings of the checkout on the same drive; nothing is written to the host;
    unplug and move to another machine. Matches portable-first. Cost: slower cold
    start, venv rebuild if the drive path changes, no host PATH integration.
  - **B — Machine-local managed install:** bootstrap installs into `~/.hermes` /
    `%LOCALAPPDATA%\hermes` like upstream — faster, host PATH integration,
    survives drive re-lettering. Cost: leaves state on every host it touches; not
    portable; contradicts portable-first.
  - **C — Hybrid:** drive-native by default, an opt-in flag for a machine install.
    Most flexible; more surface to build and document.
- **Leaning:** **A (drive-native)**, provisionally — consistent with the
  portable-first principle already established. **Explicitly not ratified.** The
  *hardened* form of A — sealing the drive, the dual-volume
  `NORTHFORGE` / `NORTHFORGE-DATA` split, and the certify / verify / audit
  machinery — is **out of scope** for the current pass and waits for real
  ratification of this decision. What ships now (`CHG-2026-09-07-005`,
  `scripts/bootstrap-north-forge.ps1` + `north-forge.cmd`) is only the *minimal*
  single-drive, single-folder-tree bootstrap + launcher: enough to make a fresh
  clone launch, deliberately **not** built on the unratified hardened design.
- **Blocking:** the hardened install / seal / dual-volume / certification work —
  do not start it until this is `DECIDED`. **Not** blocking: the minimal
  bootstrap, which ships now.
- **Owner:** Kenneth C. Walker Jr.
- **Status:** OPEN

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

### DECISION-2026-09-07-001 — Splash art — keep the stock Hermes launch mark, or swap it?

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main) — committed on `c4e88d2ab6`
- **Run:** RUN-2026-09-07-004 (opened and decided the same run)
- **Id note:** first `DECISION-` dated 2026-09-07, so `-001` per the daily-reset
  rule in [`README.md`](../README.md) § Naming. Unrelated to the `2026-09-06`-dated
  `-002` / `-003` (whose dates were pinned by owner call for continuity with the
  rebrand batch).
- **Source:** carried since `RUN-2026-09-07-003` — the `INDEX.md` "Latest run" row
  recorded the `⚕ NOUS HERMES` launch splash as **"left open for the owner (not
  defaulted)"**. Raised again by the owner this run alongside the drive-native
  onboarding fixes, now that final brand art exists under `assets/icons/`.
- **Confidence:** Confirmed Fact — verified in the checkout that
  `hermes_cli/banner.py` `HERMES_CADUCEUS` (the `⚕` braille hero) and
  `HERMES_AGENT_LOGO` (the `HERMES-AGENT` wordmark) are **byte-identical to
  `upstream/main`**; NF's only diff to that file is the one-line
  `format_banner_version_label()` label. `banner.py` already prefers
  `skin.banner_hero` / `skin.banner_logo` over those constants
  (`banner.py:855`, `:916`).
- **Supersedes:** —
- **The call:** the CLI shows upstream's `⚕` caduceus + `HERMES-AGENT` wordmark at
  every `hermes` / `north-forge.cmd` launch. Leave it, or replace it with North
  Forge's own mark?
- **Options:**
  - **A — Leave as-is.** Consistent with "engine used unmodified"; the ASCII
    startup art is not in `BRANDING.md`'s North-Forge-owned list;
    `assets/icons/splash-alt.png` is already earmarked for a future *graphical*
    loading screen. Zero change.
  - **B — Edit `banner.py`.** Replace the `HERMES_AGENT_LOGO` constant (and maybe
    the caduceus) with North Forge art. Small text swap, but a permanent
    multi-line diff on a hot upstream file — merge-conflict risk on every rebase.
  - **C — Swap via a North Forge skin.** Ship `skins/north-forge.yaml` with
    `banner_logo` / `banner_hero`; `banner.py` stays byte-identical to upstream
    except its existing one-liner. Slightly more than a "text swap" (new skin
    file + activation wiring) but zero added engine drift. The repo's brand PNGs
    can't render in a terminal, so the art is new Rich-markup ASCII either way.
- **Leaning at open:** C.
- **Decided:** 2026-09-07 (`RUN-2026-09-07-004`) — chose **C (swap via a North
  Forge skin)**, owner-selected. Implemented by **`CHG-2026-09-07-012`**: new
  tracked `skins/north-forge.yaml` (`banner_logo` "NORTH FORGE" ANSI-Shadow +
  `banner_hero` anvil/sparks + full North-Forge `branding`; colors/spinner
  inherited from the built-in `default` skin). `scripts/bootstrap-north-forge.ps1`
  seeds it into `HERMES_HOME/skins/` and sets `display.skin=north-forge` on a
  fresh bootstrap (never overriding an operator's own skin choice);
  `north-forge.cmd` re-copies it every launch. `hermes_cli/banner.py` and
  `hermes_cli/skin_engine.py` are **unchanged**. Verified end-to-end against the
  drive venv — the activated skin drives the banner and the `⚕`/`HERMES`
  constants are no longer reached on that path.
- **Reaffirmed:** 2026-09-07 (`RUN-2026-09-07-006`). A mid-stream instruction in
  the `RUN-2026-09-07-005` batch floated deferring this and reverting the skin
  swap; that reversal was **never executed** (RUN-005 shipped only the F-04 fix),
  and the owner's follow-up confirmed **keep the shipped swap — stays DECIDED
  (C), not DEFERRED**. No code change; the `RUN-2026-09-07-005` INDEX hedge notes
  about a "pending deferral" were corrected.
- **Owner:** Kenneth C. Walker Jr.
- **Status:** DECIDED

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
- **Follow-up:** 2026-09-06 (`RUN-2026-09-06-003`, `NF-v0.2.1`, PATCH) — the two
  loose ends the rebrand commit left: `assets/banner.png` re-branded to North
  Forge and re-referenced in `README.md` (**placeholder art**, flagged in the
  PNG `tEXt` chunks — final art still owed), and two drive-facing README sections
  added ("Drive class", "Customizing your agent"). No code / `.env` / dist-name
  change. `CHG-2026-09-06-025`, `CHG-2026-09-06-026`. Local only, not pushed.
- **Status:** DECIDED

---

## Register (quick scan)

| ID | Date | Area | Question | Status | Decided |
| --- | --- | --- | --- | --- | --- |
| DECISION-2026-09-06-001 | 2026-09-06 | Fork identity | Rebrand vs thin downstream? | DECIDED — B (full rebrand), landed `NF-v0.2.0` (CHG-2026-09-06-020..024) | 2026-09-06 |
| DECISION-2026-09-06-002 | 2026-09-06 | Repo hygiene | Keep or delete the attic clone? | OPEN — leaning A (delete) | — |
| DECISION-2026-09-06-003 | 2026-09-07 | Install model | Drive-native run-in-place vs machine-local managed install? | OPEN — leaning A (drive-native); hardened form (seal / dual-volume / certify) awaits ratification | — |
| DECISION-2026-09-07-001 | 2026-09-07 | Branding | Keep the stock Hermes launch splash, or swap it? | DECIDED — C (swap via a North Forge skin), landed `CHG-2026-09-07-012`; reaffirmed `RUN-2026-09-07-006` (deferral floated then withdrawn, never executed) | 2026-09-07 |
| DECISION-2026-09-07-002 | 2026-09-07 | Branding wording | "engine used unmodified" / "full rebrand" broader than the code (Codex F-07) | OPEN — leaning A (tighten wording, no identifier churn); logged only | — |
