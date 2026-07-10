
================================================================
16. PHASE 4 DESIGN — MEMORY API ABSTRACTION (L6 interface)
================================================================
Status: APPROVED + IMPLEMENTED (2026-07-09). Refinement vs below: writes raise typed CapabilityError/UnsupportedCapability — NEVER silent no-ops (see §17.5/§17.8 for the corrected trust stance carried into Phase 5).
Author: Hermes (design), Joe (directive + ordering)
Date: 2026-07-09

Purpose (from Joe's directive):
  Create the STABLE interface between the Memory Router and concrete memory
  providers. Callers (agents, skills, CLI) must never know storage details.
  The API must allow current + future backends — L1 identity, L2 project
  memory, L3 archive, L4 ADR, L5 search, Graphiti, Holographic — to plug in
  behind one contract.

Hard constraints (non-negotiable):
  - Do NOT change existing behavior.
  - Do NOT add new memory storage.
  - Do NOT activate Holographic / Mem0 / Graphiti / embeddings.
  - Keep Markdown + raw transcripts as source of truth.
  - Keep SQLite as a derived, rebuildable cache.
  - This is an INTERFACE/ABSTRACTION milestone, not a new backend.

----------------------------------------------------------------
16.1 Phase ordering (Joe's decision, 2026-07-09)
----------------------------------------------------------------
  Phase 3  L3 archive lifecycle        — DONE + verified.
  Phase 4  L6 Memory API abstraction   — THIS design (next).
  Phase 5  L4 ADR system               — next real build.
  Phase 6  L2 project memory           — after ADR.
  (L5 already exists; Graphiti/Holographic remain optional, dormant.)

  Rationale (Joe): ADRs before project memory because ADRs are HIGH SIGNAL.
  A conversation archive is lots of information; an ADR is the important
  decision distilled. A system that remembers decisions becomes dramatically
  more useful. So we harden the API seam FIRST (Phase 4), then build ADR
  (Phase 5) against that seam, then L2.

----------------------------------------------------------------
16.2 Proposed architecture
----------------------------------------------------------------
Two-layer seam, both already present in spirit, made explicit:

  Caller (agent / skill / CLI)
        │  speaks ONLY the Memory API (§16.3)
        ▼
  MemoryAPI  (abstract base / Protocol)   <-- NEW stable contract
        │  dispatch by operation + capability
        ▼
  MemoryRouter  (existing, thin: classify/select/dispatch/log)
        │  routes to registered Capabilities
        ▼
  Capability handlers  (existing registry; L1 + L5 registered today)
        │
        ▼
  Concrete providers:  markdown files, SQLite index, (future) ADR store,
                       Graphiti, Holographic — all hidden behind handlers.

Key idea: the Memory API is the FRONT DOOR. The Router is the internal switch.
Today callers reach the Router directly (router.search, router.*). Phase 4
introduces a thin `MemoryAPI` facade that wraps the Router and exposes the
operation-oriented vocabulary (search/recent/project/decision/archive/context/
remember) WITHOUT changing what the Router or indexer do underneath.

This is a FACADE + formal interface, not a rewrite. The Router keeps the same
classify/select/dispatch/log responsibilities. The API is the documented,
versioned surface that future backends implement against.

----------------------------------------------------------------
16.3 Operation contracts (the stable interface)
----------------------------------------------------------------
All operations are synchronous-or-awaitable; concrete signatures TBD at
implementation. Each defines: input, output, provenance, errors, capability.

  search(query, limit=N, scope=None)
    in:   free-text query; optional scope filter (layer/intent)
    out:  list[SearchResult]  (content + provenance)
    prov: every result carries source_file, memory_layer, retrieval_method,
          event_ts/session_id where applicable
    err:  empty list on no match; never raises on missing backend
    cap:  L5 (FTS5/LIKE) primary; L1 for identity scope

  remember(content, layer, **meta)
    in:   content blob + target layer + metadata (project, tags, decision_id)
    out:  SearchResult / handle (source_file, layer)
    prov: source_file + layer + timestamp + retrieval_method="write"
    err:  capability-unavailable -> logged no-op (graceful)
    cap:  L1 (MEMORY.md/USER.md), L2 (project md), L4 (ADR md) — writers
    note: Phase 4 defines the contract only; actual writers land in Phases 5/6.

  recent(n=N, layer=None)
    in:   count + optional layer
    out:  list[SearchResult] ordered by recency (event_ts / mtime)
    prov: source_file + layer + event_ts
    err:  empty list if no data
    cap:  L3 archive (recent sessions), L1/L2/L4 by mtime

  project(name)
    in:   project identifier / name substring
    out:  project record (status, path, notes) or None
    prov: source_file (HERMES_PROJECTS.md or project md)
    err:  None on miss
    cap:  L2 (Phase 6); today answered by HERMES_PROJECTS.md read

  decision(id=None, topic=None)
    in:   decision id and/or topic query
    out:  ADR record(s): id, title, status, context, decision, consequences
    prov: source_file (adr/<id>.md) + layer="L4"
    err:  empty list on miss
    cap:  L4 (Phase 5) — contract defined now, impl later

  archive(session_id=None, since=None)
    in:   optional session_id or time window
    out:  archived conversation chunks / session refs
    prov: source_file (sessions/<id>.jsonl) + session_id + role + event_ts
    err:  empty on miss; never touches raw file
    cap:  L3 archive (already indexed via on_session_end lifecycle)

  context(query, budget)
    in:   query + token/result budget for prompt injection
    out:  ranked, de-duplicated context block (fan-out across layers)
    prov: each included result keeps its provenance
    err:  graceful empty / partial on backend failure
    cap:  fan-out: L1 + L5 (+ L2/L4 when built); merge/rank policy deferred

Every operation returns provenance-bearing results or empty — never partial
silence, never an un-attributed fact. This matches the Phase 1 provenance
mandate (source_file, memory_layer, timestamp, retrieval_method).

----------------------------------------------------------------
16.4 How current Router + indexer integrate (no behavior change)
----------------------------------------------------------------
- MemoryRouter (hermes_cli/memory_router/router.py): unchanged. Its
  classify/dispatch/registry/log stay. The new MemoryAPI calls
  router.search / router.* internally — it is a wrapper, not a replacement.
- MemoryIndex (hermes_cli/memory_index/indexer.py): unchanged. It remains the
  L5 implementation. search()/archive_stats()/refresh_pending() keep working;
  MemoryAPI.search and MemoryAPI.archive delegate to it.
- archive_lifecycle listener (on_session_end): unchanged. It owns L3 ingestion.
  MemoryAPI.archive just reads what the listener already indexed.
- CLI `hermes memory search/status`: keeps working. Can be re-pointed at
  MemoryAPI later without behavior change.
- No new storage, no new config keys, no new dependencies.

The seam is additive: MemoryAPI is a new module that imports the existing
Router and delegates. Nothing downstream is forced to migrate.

----------------------------------------------------------------
16.5 Migration risks
----------------------------------------------------------------
R1. Double surface. If both Router and MemoryAPI are public, callers may pick
    inconsistently. Mitigation: MemoryAPI becomes THE documented front door;
    Router stays internal (still callable, but docs point to API). No code
    breakage — Router signature unchanged.

R2. Capability gaps. A caller asks decision() before L4 exists.
    Mitigation: missing backends are explicit and catchable — decision()
    raises UnsupportedCapability (NOT a silent []), so callers can branch on
    the typed error. This is the approved correction to the original 'return
    []' wording: no fake success, and no crash either. Phase 5 (§17) fills
    the L4 seam; until then the typed error IS the contract.

R3. Contract drift between API and Router. If Router internals change, API must
    follow. Mitigation: API has its own contract tests (input/output/provenance)
    independent of Router internals; tests assert the facade forwards correctly.

R4. Over-abstraction (AGENTS.md explicitly rejects speculative infrastructure).
    Mitigation: Phase 4 implements ONLY the facade + contracts with REAL
    consumers (agents/skills/CLI already call router.search etc.). It is not a
    hook with no consumer — the consumers exist today. No new extension point
    without a caller.

R5. Prompt-cache safety (AGENTS.md: sacred). search/context must not mutate
    past context or rebuild system prompt. Mitigation: read-only operations
    only; MemoryAPI adds no writes in Phase 4 (writers defined as contract,
    implemented in Phases 5/6).

----------------------------------------------------------------
16.6 Smallest implementation milestone (proposed)
----------------------------------------------------------------
M1 (the whole of Phase 4 — design-then-implement after approval):
  a. Add hermes_cli/memory_api/__init__.py with a `MemoryAPI` class (facade).
  b. Implement the read operations that already have backends:
       search()      -> router.search -> L5 (+L1 identity scope)
       archive()     -> indexer (L3) read
       recent()      -> indexer recent (L3) + L1 mtime
       context()     -> fan-out over registered capabilities (merge/rank TBD,
                        keep simple: concatenate provenance-bearing results)
  c. Not-yet-built write/query ops raise typed errors — NEVER silent no-op
     success (this is the approved correction to the original M1c stub):
       remember()        -> CapabilityError (read-only derived index)
       decision()/project()-> UnsupportedCapability (L4/L2 not built yet)
     A write that cannot persist must fail loudly, not pretend success.
  d. Contract tests: for each op, assert input->output shape + provenance keys,
     and that a missing backend yields the empty contract (not an exception).
  e. Keep `hermes memory search/status` working unchanged (re-point optional,
     not required for M1).

Out of scope for M1 (explicit): L2/L4 storage, any writer that mutates Markdown,
Graphiti/Holographic/Mem0/embeddings, new config, new deps.

This milestone is KISS: a thin facade over what already works, plus typed
contract stubs for what comes next. It de-risks Phase 5 (ADR) by giving ADR a
real `decision()` seam to implement against, and proves the "callers never know
storage" property before we add more layers.

----------------------------------------------------------------
16.7 Open questions for approval
----------------------------------------------------------------
Q1. Facade vs Protocol: implement MemoryAPI as a concrete class (simplest,
    matches thin-waist style) or a typing.Protocol that backends conform to?
    Recommendation: concrete facade class now; Protocol only if a second
    independent implementation appears. Avoid speculative ABC.
Q2. Should context() merge/rank now or just concatenate? Recommendation:
    simple ordered concatenation in M1; real rank policy deferred (was already
    an open item from Phase 1 §15).
Q3. Naming: keep `hermes memory ...` CLI, add `MemoryAPI` as the code front
    door. No new CLI verbs in M1.

================================================================
17. PHASE 5 DESIGN — L4 ADR (DECISION MEMORY) SYSTEM
================================================================
Status: APPROVED + IMPLEMENTED (2026-07-09). Verified: 11 Phase 5
tests pass (trust boundary + lifecycle), full memory suite 521 passed.
AdrProvider is the sole owner of ADR path resolution; decision() routes
through the Router's DECISION intent; only ACCEPTED ADRs surface as
decisions. See §17.9 for the implemented report.
Author: Hermes (design), Joe (directive: ADRs high-signal, before L2)
Date: 2026-07-09 (follows verified Phase 4)

Purpose:
  Add durable architectural-decision memory (L4) behind the EXISTING
  MemoryAPI.decision() contract defined in Phase 4 (§16.3). ADRs are
  HIGH-SIGNAL: a distilled decision beats a transcript. This makes the
  system "remember what we decided," which Joe rated the biggest
  usefulness jump.

Hard constraints (carry from Phase 4 + this directive):
  - Do NOT implement yet (design only).
  - Markdown remains source of truth.
  - SQLite remains a derived index (no ADR-specific tables; ADRs are
    markdown under memory/, so the EXISTING indexer already walks them).
  - No LLM extraction. No automatic decision creation.
  - No semantic embeddings. No Graphiti/Holographic activation.
  - No changes to existing layers (L1/L3/L5 untouched).

----------------------------------------------------------------
17.1 ADR storage model
----------------------------------------------------------------
Location (source of truth = Markdown):
  Global/system ADRs : ~/.hermes/memory/adr/_system/NNN-title.md
  Per-project ADRs   : ~/.hermes/memory/adr/<project-key>/NNN-title.md
  where <project-key> is a lowercase slug of the project name
  (e.g. "hermes-aios", "sb-sorter"). Derivable from HERMES_PROJECTS.md
  or the cwd repo name.

  Why here: memory/ is already scanned by MemoryIndex._discover_sources
  (mem_dir.rglob("*.md")), so ADR markdown is automatically indexed as
  L5-notes with ZERO indexer changes. ADRs thus become searchable
  through the existing search() for free, while the AdrProvider adds a
  structured L4 lens on the same files. Two lenses, one source file.

File format:
  Markdown with YAML frontmatter. Frontmatter = machine-readable
  metadata; body = human-readable long form. (Frontmatter stripping
  already handled by the indexer.)

Naming:
  NNN-short-kebab-title.md  (e.g. 001-use-protocol-interfaces.md).
  Lowercase, zero-padded 3-digit number, kebab-case title.

Numbering:
  PER-PROJECT monotonic counter. The next number = (max numeric prefix
  in that project's adr/ dir) + 1. No central registry/counter file
  (KISS; avoids a state file that can drift from reality). Provider
  computes it at draft time. Global uniqueness is achieved by prefixing
  the id with <project-key>/ (see §17.3), not by a shared sequence.

Metadata (frontmatter keys):
  id                 : "<project-key>/NNN"  (globally unique handle)
  title              : short human title
  status             : proposed|accepted|deprecated|superseded
  date               : ISO-8601 (decision/acceptance date)
  project            : <project-key>
  decision_maker     : "joe"  (human who accepted)
  proposed_by        : "hermes" | "joe"  (who drafted)
  related_components : [list of system areas / files]
  supersedes         : [ids]  (ADRs this one replaces)
  superseded_by      : [ids]  (ADRs that replace this one)
  tags               : [free-form]

Lifecycle (state machine, human-gated):
  proposed  --(human accept)-->  accepted
  accepted  --(still valid, not recommended)-->  deprecated
  accepted  --(replaced by newer ADR)-->  superseded   (superseded_by set)
  A new accepted ADR that replaces an old one sets its own `supersedes`
  and the old ADR's `superseded_by` (back-link). Back-links are written
  at the same human-approved event, not computed on read.
  No auto-transition. Hermes may DRAFT (status=proposed); only a human
  approval flips to accepted/deprecated/superseded (§17.5).

----------------------------------------------------------------
17.2 Decision API integration
----------------------------------------------------------------
Phase 4 left decision() raising UnsupportedCapability (honest, not a
silent empty — per the approved refinement). Phase 5 wires it:

  MemoryAPI.decision(id=None, topic=None, project=None)
     |  (facade normalizes; routes DECISION intent)
     v
  MemoryRouter  --Intent.DECISION-->  L4 capability ("adr")
     |  delegates to
     v
  AdrProvider  (structural MemoryProvider; owns ADR markdown I/O)

Mapping (call -> provider method -> router L4 method):
  decision(id="hermes-aios/001")
        -> AdrProvider.get(id)              -> L4 "decision" (id=)
  decision(topic="protocol")
        -> AdrProvider.search(topic)        -> L4 "decision" (query=)
  decision(project="hermes-aios")
        -> AdrProvider.by_project(proj)     -> L4 "project_decisions"
  decision()  (recent)
        -> AdrProvider.recent()             -> L4 "recent_decisions"

Provenance results: each ADR becomes a DecisionRecord (defined in
Phase 4 protocols.py: id/title/status/context/decision/consequences/
source) and is also exposable as a MemoryResult:
  source="memory/adr/<project-key>/NNN-title.md"
  provider="adr", layer="L4", retrieval_method="adr"|"adr-search"|"adr-recent"
  timestamp=frontmatter date
  extra={id,title,status,project,supersedes,superseded_by}
Every decision result keeps full provenance (no unattributed decision).

Note on facade routing: Phase 4's decision() currently calls a provider
directly. Phase 5 aligns it to route through the Router's DECISION
intent (architecture: "Router decides routing; providers own storage"),
so all reads centralize in the Router. The AdrProvider instance is
registered in BOTH the facade provider registry and behind the router's
L4 capability (same instance) — no duplication of logic.

----------------------------------------------------------------
17.3 ADR schema (per decision)
----------------------------------------------------------------
Frontmatter (machine):
  id, title, status, date, project, decision_maker, proposed_by,
  related_components, supersedes, superseded_by, tags   (as §17.1)

Markdown body (human, fixed section order):
  ## Context
      Forces/problem that necessitated a decision.
  ## Decision
      The decision itself, stated plainly.
  ## Alternatives considered
      Each option with pros/cons (brief table or bullets).
  ## Reasoning
      Why this option over the others.
  ## Consequences
      Positive + negative; intended + unintended.
  ## Related components
      Files/systems/subsystems affected (mirrors frontmatter).
  ## Supersession
      Links to superseded/superseding ADRs + one-line note.

Required fields (from directive): ID, date, status, context, decision,
alternatives considered, reasoning, consequences, related components,
superseded/superseding decisions — all present above.

----------------------------------------------------------------
17.4 Retrieval behavior
----------------------------------------------------------------
Exact decision lookup:
  By global id "hermes-aios/001" (split on "/", locate dir + NNN file),
  OR by (project=, id="001"). Returns one DecisionRecord or None.
  No fuzzy match — exact handle only (predictable, scriptable).

Decision search:
  Free-text over title + body + frontmatter across the adr/ tree
  (or scoped to a project). Tokenize (reuse indexer's stopword-aware
  tokenizer), match via LIKE/FTS5 over the markdown — NO embeddings.
  Bonus: because ADRs live under memory/, they are ALSO returned by the
  existing search() (L5-notes). decision search is the structured,
  metadata-aware variant, not a replacement for general search.

Recent decisions:
  Sort by frontmatter `date` desc. Global, or scoped to a project.
  Powers "what decisions did we make lately?"

Project-scoped decisions:
  Restrict to <project-key>/ dir. decision(project=...) returns all
  ADRs for that project (any status), newest first.

All four return provenance-bearing DecisionRecord / MemoryResult.

----------------------------------------------------------------
17.5 Trust model
----------------------------------------------------------------
Who creates ADRs:
  Joe is the decision authority. ADRs record ACCEPTED decisions; only a
  human-accepted ADR is injected as a decision into context.

May Hermes suggest ADRs:
  YES — Hermes MAY DRAFT. When it detects a durable decision in
  conversation (or Joe makes a call worth remembering), Hermes can write
  a draft markdown file with status="proposed" and proposed_by="hermes".
  A proposed draft is a SUGGESTION ARTIFACT, never presented as an
  accepted decision.

Must users approve writes:
  YES, for any state that confers authority:
    - Transition proposed -> accepted|deprecated|superseded REQUIRES
      explicit human approval (e.g. "accept ADR 001").
    - Supersession back-links are written in the same approved event.
  The draft write (proposed) is the only autonomous write Hermes may
  make, and it is clearly marked and non-authoritative. Recommendation
  (per directive): Hermes may draft, humans approve persistence.

This keeps the "markdown is source of truth" + "no automatic decision
creation" constraints: automation proposes; humans persist authority.

----------------------------------------------------------------
17.6 Migration (existing layers unchanged)
----------------------------------------------------------------
Integration path (additive):
  MemoryAPI.decision()  ->  Router DECISION intent  ->  L4 AdrCapability
                       ->  AdrProvider (markdown I/O, structured reader)

Concrete steps (post-approval, for reference — NOT done now):
  1. intents.py: add Intent.DECISION (+ metadata row). No change to
     existing intents.
  2. router.py: register an L4 capability named "L4-adr" serving
     DECISION intent with methods decision/search/project_decisions/
     recent_decisions, delegating to AdrProvider.
  3. memory_api/providers.py: add AdrProvider conforming to the
     MemoryProvider Protocol (structural; no base class) — implements
     decision()/search()/by_project()/recent(); remember() writes a
     proposed draft (status=proposed), and a separate accept() that
     requires human approval to flip status.
  4. memory_api/facade.py: point decision() at the Router DECISION
     intent (was raising UnsupportedCapability). Add project=None param
     (additive).
  5. ADR markdown dropped under memory/adr/<project-key>/ — EXISTING
     indexer already indexes them; no indexer change.

No changes to: L1 identity, L3 archive, L5 index/search, the
on_session_end listener, or any CLI verb. The seam decision() already
existed; Phase 5 fills it.

----------------------------------------------------------------
17.7 Revisit: ARCHIVE intent (added in Phase 4)
----------------------------------------------------------------
Question: is ARCHIVE a true user-intent distinction, or should it stay
internal to HISTORICAL?

Two views:
  (A) KEEP ARCHIVE as first-class Intent.
      - Semantic distinction is real: archive() = "what did we SAY in
        past conversations" (session replay, L3); historical search() =
        "what do I KNOW across all indexed memory" (distilled recall).
        Different consumer need, different vocabulary.
      - It is the designated seam for a FUTURE dedicated conversational
        backend (Graphiti/Holographic) to take over WITHOUT disturbing
        search(). That optionality is why Phase 4 separated it.
      - Cost: one Intent + one capability registration — cheap, additive.
  (B) FOLD into HISTORICAL with scope="L3-archive".
      - Simpler surface (fewer intents). Today both query the SAME
        SQLite index (L3 rows are a subset), so functionally it's just a
        scope filter.

Recommendation: KEEP ARCHIVE as a distinct Intent (view A).
  Rationale: the distinction is user-meaningful (replay vs. recall),
  it costs almost nothing, and it preserves the future-backend seam. We
  are honest that, today, ARCHIVE is implemented as a scoped subset of
  the same index — that's an implementation detail, not a reason to
  collapse the intent. If, after Phase 5/6, ARCHIVE still has no
  distinct backend and adds cognitive load, we can revisit and fold it.

----------------------------------------------------------------
17.8 Open questions for approval
----------------------------------------------------------------
Q1. Draft autonomy: allow Hermes to write `proposed` drafts
    autonomously (clearly marked, non-authoritative), or require
    approval even for the draft? Recommendation: autonomous proposed
    drafts allowed; only authority transitions need approval.
Q2. ADR location: memory/adr/<project-key>/ (Hermes-owned, always
    available) vs. inside each project's own repo (version-controlled
    with code, but requires repo access). Recommendation: Hermes-owned
    memory/adr/ for Phase 5; repo-embedding can be a later option.
Q3. Numbering width: 3-digit (001..999) per project sufficient?
    RESOLVED at implementation: yes, 3-digit per-project is used; the
    provider computes the next number from the max existing prefix + 1
    (no central counter).

Open questions resolved at implementation (2026-07-09):
  Q1 -> Hermes may write `proposed` drafts autonomously (clearly marked,
        non-authoritative). Only authority transitions (proposed ->
        accepted|deprecated|superseded) require explicit human approval.
  Q2 -> ADRs live under Hermes-owned memory/adr/<project-key>/ for Phase 5
        (repo-embedding left as a later option).
  Q3 -> 3-digit per-project numbering adopted.

----------------------------------------------------------------
17.9 Implementation report (2026-07-09)
----------------------------------------------------------------
Files changed / created:
  NEW  hermes_cli/memory_api/adr.py
        AdrProvider — SOLE owner of all ADR path resolution
        (memory/adr/<project-key>/NNN-kebab-title.md). Implements draft /
        accept / get / search / recent / by_project / validate. Write
        provenance metadata: created_by, created_at, approved_by,
        approved_at (+ status, date, decision_maker, proposed_by).
        Supersession back-links written at the approved event (not on
        read). No LLM extraction, no embeddings, no Graphiti/Holographic.
  EDIT hermes_cli/memory_api/protocols.py
        DecisionRecord extended with: date, decision_maker, proposed_by,
        related_components, tags, created_by, created_at, approved_by,
        approved_at, supersedes, superseded_by. (Structural Protocol;
        no base class — AdrProvider is isinstance-compatible but not in
        its MRO.)
  EDIT hermes_cli/memory_api/facade.py
        decision(id, topic, project) now routes through the Router's
        DECISION intent (architecture: Router decides, providers own
        storage). No-arg decision() returns recent (newest-first). Added
        draft_decision() / accept_decision() human-gated wrappers.
        remember()/project() still raise typed CapabilityError /
        UnsupportedCapability (no silent no-op).
  EDIT hermes_cli/memory_router/router.py
        Registered L4-adr capability delegating DECISION intent to the
        AdrProvider (methods decision/search/project_decisions/
        recent_decisions). _wire_defaults seeds the AdrProvider. No
        hardcoded ADR paths inside the router (path logic lives in
        AdrProvider only).
  EDIT hermes_cli/memory_router/intents.py
        Intent.DECISION metadata corrected to "Phase 5".
  EDIT hermes_cli/memory_api/__init__.py
        Exports AdrProvider.
  EDIT tests/plugins/memory/test_phase4_api.py
        Replaced obsolete "decision raises before Phase 5" test with
        test_decision_is_wired_after_phase5 (graceful [] when no ADRs).
  NEW  tests/plugins/memory/test_phase5_adr.py
        11 contract + trust-boundary tests.

Tests:
  - 11 new Phase 5 tests (test_phase5_adr.py): trust boundary
    (proposed NOT returned by decision(); accepted IS), draft->accept
    lifecycle, supersession back-links both directions, search + recent
    ordering (microsecond-precision timestamps for determinism),
    AdrProvider sole path-owner (no hardcoded paths outside it),
    no Graphiti/embeddings activation guard.
  - Full memory suite: 521 passed (was 510; +11), ZERO regressions.
  - AST clean on all changed files; AdrProvider verified structurally
    Protocol-conformant; router has zero "memory/adr" string literals.

Example ADR lifecycle (verified end-to-end):
  1. Hermes drafts:  api.draft_decision("Use Protocol interfaces", ...)
        -> file memory/adr/hermes-aios/001-use-protocol-interfaces.md
           status=proposed, proposed_by=hermes
        -> decision(id="hermes-aios/001") returns []  (INVISIBLE)
  2. Joe accepts:    api.accept_decision("hermes-aios/001", approved_by="joe")
        -> status=accepted, decision_maker=joe, approved_by=approved_at set
        -> decision(id="hermes-aios/001") now returns the record
  3. Supersession:    api.accept_decision("hermes-aios/003", supersedes=[".../002"])
        -> new ADR supersedes=[.../002]; old ADR back-linked
           superseded_by=[.../003] (written in same event, not computed on read)

Architecture state:
  Caller
    -> MemoryAPI facade (decision/search/archive/recent/context/
       remember/project)  [L6, Phase 4+5]
    -> MemoryRouter (classify / route by Intent)  [thin]
    -> Capability providers:
         L5-index (sqlite)            : search / archive / recent
         L3-archive (sqlite)          : ARCHIVE intent (session replay)
         L1-identity                  : identity pointers
         L4-adr  (AdrProvider)        : DECISION intent (ADRs)  [NEW]
  Markdown remains source of truth; SQLite stays a derived index.
  ARCHIVE intent kept distinct from search() (replay vs. recall seam).
  Trust boundary enforced at the read path: only ACCEPTED ADRs surface
  as decisions; proposed drafts are suggestion artifacts only.

----------------------------------------------------------------
17.10 Milestone — decisions wired into context() + CLI (2026-07-09)
----------------------------------------------------------------
Small bridge between Phase 5 (storage) and real use, before Phase 6.

  - facade.context(query) now routes decisions through the facade's own
    decision() (Router DECISION intent) instead of poking providers
    directly. Only ACCEPTED ADRs appear in bundle.decision, each with
    provenance (provider="adr", layer="L4", retrieval_method="adr",
    extra id/title/status/project). Proposed drafts stay invisible.
  - New CLI: `hermes memory decision [list|get|search|project|accept|
    draft]`. Reads surface accepted ADRs only; `accept` is the human-gated
    authority transition (requires --by); `draft` creates a proposed
    (non-authoritative) suggestion. --json supported for machine output.
    Lives in hermes_cli/subcommands/memory.py (parser) + main.py
    (_cmd_memory_decision / _print_decisions); uses the MemoryAPI facade
    (narrow-waist extension, no new core tool).
  - DecisionRecord gained a `project` field (was missing); AdrProvider.
    draft() and _parse now preserve it through the draft->accept cycle.
  - Tests: tests/plugins/memory/test_phase5b_decision_cli.py (8 tests:
    context integration + trust boundary + CLI list/get/accept/json).
    Full memory suite: 529 passed (was 521; +8), zero regressions.

================================================================
18. PHASE 6 — LAYER 2: PROJECT MEMORY ("where are we / what next")
================================================================
Status: APPROVED + IMPLEMENTED (2026-07-09). Verified: 14 Phase 6
tests pass, full memory suite 543 passed. ProjectProvider is the sole
owner of L2 path resolution; project() routes through the Router's
PROJECT_STATE intent; context() surfaces the single active L2 with
provenance. Authority model (B): Hermes MAY propose, NEVER writes STATUS.md
without explicit human acceptance. See §18.10 for the implementation report.

----------------------------------------------------------------
18.1 The one principle that decides everything
----------------------------------------------------------------
  L2 describes the PRESENT, not the PAST.

  History already has three excellent homes:
    - Archive  (conversational history — what was said/done)
    - ADRs     (decisions — what was decided, and why)
    - Search   (retrieval over all of the above)

  L2 must NOT become a fourth history store. Its single job is to answer
  the question asked every time a project is reopened:

      "Where were we, and what should we do next?"

  That question is forward-looking. So L2 holds CURRENT truth:
    - what is true now (status, owners, environment facts still valid)
    - what is in the way (blockers, risks, open questions)
    - what to do next (next actions, with owners)
  and it REFERENCES history by link rather than copying it.

  The failure mode to avoid (explicitly forbidden): an auto-generated
  "project summary" scraped from recent sessions. That is just Archive
  re-skinned — it preserves the past and fabricates "current" truth. L2
  is deliberately curated, never auto-accumulated.

----------------------------------------------------------------
18.2 Trust / authority model (APPROVED OPTION B, 2026-07-09)
----------------------------------------------------------------
  (B) HUMAN-CURATED + Hermes-SUGGESTS — APPROVED. L2 source of truth
      stays the markdown STATUS.md. Hermes MAY PROPOSE an update (e.g. at
      session end: "next actions appear to be X, Y") but a proposal is
      presented for acceptance — NEVER written to L2 unless the human
      accepts. No autonomous write. Mirrors the ADR trust boundary:
      Hermes drafts, human authorizes. This keeps L2 curated (present-
      state) while letting Hermes reduce the maintenance burden.

  Approved refinements (option B):
    - Suggested updates MUST be clearly marked as proposals until accepted.
    - The accepted STATUS.md remains the authoritative source.
    - A write that is not actually persisted raises (no silent success),
      consistent with Phase 4's hard rule. Hermes MUST NOT derive L2 from
      Archive/ADR/Search content.

----------------------------------------------------------------
18.3 Storage model
----------------------------------------------------------------
  Source of truth: markdown + YAML frontmatter, one file per project:

    $HERMES_HOME/memory/projects/<project>/STATUS.md

  Mirrors the ADR convention (markdown truth, git-friendly, frontmatter
  for machine fields, body for narrative). Per-project directory, like
  ADRs, so multiple projects stay isolated.

  Frontmatter (machine-readable, consumed by context()):
    project:      <key, e.g. hermes-aios>        # required
    title:        <human name>
    status:       active | paused | blocked | done | archived
    updated_at:   ISO timestamp
    updated_by:   <who last curated>
    owners:       [<handle>, ...]
    next_actions:                                 # forward-looking
      - what:     <imperative next step>
        owner:    <handle | unassigned>
        blocked_by: [<adr-id or action ref>]      # links, not prose
    goals:        [<current objective>, ...]      # what "done" means now
    blockers:     [<what is in the way>, ...]
    open_questions: [<unresolved question>, ...]
    links:                                        # REFERENCES to history
      adrs:       [<adr-id>, ...]                 # why we decided X
      archive:    [<session ref>, ...]            # background context
      search:     [<topic>, ...]                  # retrieval pointers

  Body: a short human narrative of CURRENT state (1-2 paragraphs). Not a
  log. If it grows a timeline, that's a smell — move it to Archive.

  L2 does NOT store: decision rationales (→ ADR), transcripts (→ Archive),
  or retrievable facts (→ Search). It LINKS to them by id/topic.

----------------------------------------------------------------
18.4 Provider + Router wiring
----------------------------------------------------------------
  New Protocol-conformant provider: ProjectProvider
    - get(project) -> ProjectState | None
    - set(project, state) -> persisted ProjectState   (explicit write)
    - status() -> CapabilityStatus (built when dir exists or writable)
    - structural compat stubs: search_files/recent_files/archive raise
      CapabilityError (L2 is not a search/archive backend)

  New facade method:
    MemoryAPI.project(project=None) -> ProjectState | None
      routes via Router Intent.PROJECT -> L2-project capability.
      project=None resolves the CURRENT project (see 18.9 Q2).

  Router:
    - register L2-project capability for Intent.PROJECT (like L4-adr for
      DECISION). No new Intent needed if PROJECT is added; otherwise reuse
      a project intent.
    - ProjectProvider is the SOLE owner of L2 path resolution (mirror
      AdrProvider: no hardcoded paths in router).

----------------------------------------------------------------
18.5 context() integration (the payoff)
----------------------------------------------------------------
  context() already returns bundle.project (currently always empty).
  Phase 6 fills it:

    if project := self._resolve_current_project():
        ps = self.project(project)
        if ps:
            bundle.project.append(MemoryResult(
                source=ps.source, provider="project", layer="L2",
                retrieval_method="project",
                content=ps.narrative or ps.status,
                extra={"project": ps.project, "status": ps.status,
                       "next_actions": ps.next_actions,
                       "blockers": ps.blockers, "owners": ps.owners,
                       "updated_at": ps.updated_at},
            ))

  So every reopen: identity + project (L2) + decision (ADRs) + recent
  (Archive) — the present state and the forward pointer, with history
  one link away. L2 is the only mutable, curated layer in context().

----------------------------------------------------------------
18.6 CLI
----------------------------------------------------------------
  hermes memory project [show|set|status|next]   (additive; narrow waist)
    show  <project?>      -> print current L2 (human + --json)
    set   <project> ...   -> explicit write of fields (human-gated)
    status <project?>     -> just the lifecycle status line
    next  <project?>      -> list next_actions with owners/blocked_by
  Reads surface only EXISTING L2; if none, prints "No L2 for <project>
  — create with: hermes memory project set <project> --status active".
  No command scrapes history to fabricate L2.

----------------------------------------------------------------
18.7 Tests (contract)
----------------------------------------------------------------
  - L2 markdown parses to ProjectState with provenance (provider=project,
    layer=L2) and correct next_actions/blockers/status.
  - context().project populated when L2 exists; EMPTY when not.
  - NEGATIVE trust test: a project with only Archive/ADRs/Search does NOT
    cause context() to fabricate an L2 entry. (Guards the core principle.)
  - write persistence: set() raises if not actually written (no silent).
  - ProjectProvider structural Protocol conformance (no base class).
  - CLI: show prints existing L2; set then show round-trips; --json shape.

----------------------------------------------------------------
18.8 Forbidden (scope guard)
----------------------------------------------------------------
  - Auto-deriving L2 from Archive / ADR / Search content.
  - Embeddings, Graphiti, Holographic, semantic memory.
  - L2 as a log / timeline / daily notes.
  - Duplicating ADR or Archive content inside L2 (link by id only).
  - Silent writes (no persisted == raise).
  - Autonomous Hermes writes to L2 without acceptance (per 18.2).

----------------------------------------------------------------
18.9 Open questions — RESOLVED (approved refinements, 2026-07-09)
----------------------------------------------------------------
  Q1 (Authority) -> OPTION B APPROVED: human-curated + Hermes-suggests.
      Proposals MUST be marked until accepted; accepted STATUS.md is the
      authoritative source; no autonomous write; no silent success.
  Q2 (Current-project resolution) -> APPROVED precedence:
      1. explicit --project (or project=) parameter
      2. configured current project (config.yaml: memory.current_project)
      3. current git repository detection (cwd inside a repo)
      4. None  (preferred over guessing; never infer/fabricate)
  Q3 (next_actions) -> flat list APPROVED; blocked_by for light deps; no
      graph/workflow engine.
  Q4 (multiple projects) -> single active project APPROVED; context()
      surfaces only one active L2 unless a caller explicitly requests
      another; no merging of multiple projects into normal context.

----------------------------------------------------------------
18.10 Implementation report (2026-07-09)
----------------------------------------------------------------
Files changed / created:
  NEW  hermes_cli/memory_api/project.py
        ProjectProvider — SOLE owner of all L2 path resolution
        (memory/projects/<project-key>/STATUS.md). Implements get() /
        set() / status() + propose_update() (writes NOTHING — in-memory
        suggestion only) + structural stubs (search_files / recent_files /
        archive / decision / remember raise CapabilityError; L2 is not a
        search/archive backend). Markdown frontmatter parser is line-based
        and robust; next_actions encoded as 'what::owner::blk1,blk2'.
        set() verifies the write actually landed (raises CapabilityError on
        any failure — no silent success). No LLM extraction, no embeddings,
        no Graphiti/Holographic.
  EDIT hermes_cli/memory_api/protocols.py
        ProjectRecord placeholder replaced by richer ProjectState
        (project/title/status/owners/next_actions[NextAction]/goals/
        blockers/open_questions/links/adrs/last_verified/verified_by/
        narrative/source) + NextAction dataclass. MemoryProvider.project()
        annotation updated to Optional[ProjectState].
  EDIT hermes_cli/memory_router/router.py
        Registered L2-project capability delegating PROJECT_STATE intent to
        ProjectProvider (methods get / set). _wire_defaults seeds the
        ProjectProvider. No hardcoded L2 paths inside the router (path logic
        lives in ProjectProvider only). router.project() now takes
        method="get"|"set" (was hardcoding "search", which the handle does
        not serve).
  EDIT hermes_cli/memory_router/intents.py
        Intent.PROJECT_STATE metadata corrected to "Phase 6".
  EDIT hermes_cli/memory_api/facade.py
        project(project=None) -> ProjectState | None (no longer raises on
        miss — never fabricates). Routes via Router PROJECT_STATE intent.
        Added set_project() (human-gated write; the ONLY L2 persistence
        path), propose_project() (writes nothing), and
        _resolve_current_project() with the approved precedence. context()
        gained an optional project= param and now fills bundle.project with
        the single active L2 (provider="project", layer="L2",
        retrieval_method="project", full provenance) — or empty if no L2
        exists (NEVER fabricated from Archive/ADR/Search).
  EDIT hermes_cli/memory_api/__init__.py
        Exports ProjectProvider, ProjectState, NextAction.
  EDIT hermes_cli/subcommands/memory.py
        Added `hermes memory project [show|set|status|next]` subparser
        (additive; narrow waist). Reads surface only existing L2.
  EDIT hermes_cli/main.py
        Dispatches `project` -> _cmd_memory_project (+ _print_project,
        _slug_safe helpers). set requires --by (human authority); nothing
        fabricates project state.
  EDIT tests/plugins/memory/test_phase4_api.py
        Replaced obsolete "project raises before Phase 6" with
        test_project_absent_returns_none_gracefully.
  EDIT tests/plugins/memory/test_phase1_router.py
        Updated two stale Phase-1 assumptions: project_state now HAS a
        registered capability (L2-project); an unavailable-intent test
        switched to RELATIONSHIP (no backend).
  NEW  tests/plugins/memory/test_phase6_project.py
        14 contract + trust-boundary tests.

Tests:
  - 14 new Phase 6 tests (test_phase6_project.py): parse round-trip,
    invalid-status raises, structural stubs raise, propose writes nothing,
    sole-path-owner, facade project() returns state / None, context()
    includes L2 when present, context() EMPTY when no L2, NEGATIVE trust
    test (ADRs+archive but no L2 => project bucket stays empty), set()
    raises if not persisted (no silent), resolve precedence (explicit wins;
    undeterminable => None).
  - Updated 3 stale pre-Phase-6 tests (2 Phase-1, 1 Phase-4) to the new
    reality (no behavior regressions; they encoded the pre-implementation
    absence).
  - Full memory suite: 543 passed (was 529; +14), ZERO regressions.
  - AST clean on all changed files; ProjectProvider verified structurally
    Protocol-conformant; router has zero "memory/projects" string literals.

Example STATUS.md lifecycle (verified end-to-end via CLI):
  1. Joe curates:  hermes memory project set hermes-aios \
        --title "Hermes AIOS" --status active --owners joe \
        --goals "ship phase6" \
        --next "wire context|hermes|hermes-aios/001" \
        --links-adr hermes-aios/001 --narrative "Integrating L2." \
        --by joe --last-verified 2026-07-09 --verified-by joe
        -> writes memory/projects/hermes-aios/STATUS.md
           (status=active, updated_by=joe; source of truth)
  2. Reopen read:   hermes memory project show hermes-aios
        -> prints Project/Status/Owners/Goals/Next actions (with owner +
           blocked_by)/Links/Last verified/Narrative
                     hermes memory project next hermes-aios
        -> lists next actions with owners
  3. Proposal (no write):  api.propose_project(...) returns an in-memory
        ProjectState with source="" (never touches disk); only
        set_project() (human-gated) persists.
  4. context() integration:  api.context(query, project="hermes-aios")
        -> bundle.project[0] carries provider="project", layer="L2",
           retrieval_method="project", source=path to STATUS.md, and
           extra {status, owners, next_actions, blockers, goals, links,
           last_verified, verified_by}. History (ADRs/Archive) stays one
           link away in the other buckets.

Architecture state:
  Caller
    -> MemoryAPI facade (decision/search/archive/recent/context/
       project/set_project/propose_project)  [L6, Phase 4+5+6]
    -> MemoryRouter (classify / route by Intent)  [thin]
    -> Capability providers:
         L5-index (sqlite)            : search / archive / recent
         L3-archive (sqlite)          : ARCHIVE intent (session replay)
         L1-identity                  : identity pointers
         L4-adr  (AdrProvider)        : DECISION intent (ADRs)
         L2-project (ProjectProvider) : PROJECT_STATE intent (L2)  [NEW]
  Markdown remains source of truth; SQLite stays a derived index.
  L2 = present-state, curated; history homes (Archive/ADR/Search) untouched.
  Authority model (B): Hermes MAY propose; human accepts; never silent.

================================================================

