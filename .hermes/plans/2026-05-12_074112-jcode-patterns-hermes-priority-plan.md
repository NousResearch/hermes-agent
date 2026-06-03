# Jcode → Hermes Priority Execution Plan

> **Za Hermes:** ovo je plan-only artefakt. Ne implementira ništa; služi kao redoslijed izvedbe za Hermes repo.

**Goal:** Pretvoriti korisne jcode pattern-e u konkretan, prioritiziran Hermes execution plan bez diranja Hermes identity/memory jezgre.

**Architecture:** Uzimamo samo pattern-e s visokim ROI-jem za postojeći Hermes runtime: gateway cache observability, async-memory observability, browser provider contract, tool-schema cache audit i delegation report disciplinu. Ne radimo runtime rewrite ni memory graph rewrite. Sve ide incrementalno kroz postojeće entrypointove i postojeće test laneove.

**Repo koji je auditiran:** `/mnt/d/HermesAgent/app`

**Live files pregledani prije planiranja:**
- `run_agent.py`
- `agent/memory_manager.py`
- `agent/memory_provider.py`
- `gateway/run.py`
- `cron/scheduler.py`
- `model_tools.py`
- `tools/browser_tool.py`
- `tests/tools/test_browser_cdp_tool.py`
- relevantni testovi pronađeni searchom u `tests/`

---

## Sažeta presuda po prioritetima

### P1 — odraditi odmah
1. Gateway session/agent cache telemetry
2. Async memory telemetry
3. Browser provider contract

### P2 — odmah nakon toga
4. Memory provider scorecard
5. Tool schema / cache invalidation audit
6. Delegation / worker report standard

### P3 — kasnije / ne sada
7. Ambient/autonomous lane hardening tek nakon P1/P2
8. Ne raditi memory graph rewrite

---

# P1 detaljni execution plan

## Task P1.1 — Gateway session/agent cache telemetry

**Objective:** Dodati stvarni signal o ponašanju `_agent_cache` u gatewayu bez promjene funkcionalne semantike.

**Zašto ovo prvo:** Hermes već ima LRU + TTL cache logiku za AIAgent instance, ali nema dovoljno operativne vidljivosti. Ovo je najbrži ROI za long-lived gateway.

**Files:**
- Modify: `/mnt/d/HermesAgent/app/gateway/run.py`
- Likely add tests in: `/mnt/d/HermesAgent/app/tests/tools/test_zombie_process_cleanup.py`
- Likely add new tests: `/mnt/d/HermesAgent/app/tests/gateway/test_agent_cache_telemetry.py`

**Current-state grounded notes:**
- `_agent_cache` i lock već postoje u `gateway/run.py:1045-1057`
- expiry watcher već finalizira i evicta cache entryje u `gateway/run.py:2996-3068`
- postoji `_evict_cached_agent()` behavior pokriven testovima u `tests/tools/test_zombie_process_cleanup.py`

**Step-by-step:**
1. U `GatewayRunner` dodaj mali telemetry state dict/counter skup:
   - `cache_hits`
   - `cache_misses`
   - `cache_evictions_ttl`
   - `cache_evictions_cap`
   - `cache_evictions_refresh`
   - `cache_current_size`
2. Na svim cache entry točkama označi reason code za evikciju:
   - `ttl_expiry`
   - `lru_cap`
   - `config_signature_changed`
   - `model_switch_or_refresh`
3. Dodaj helper za snapshot telemetryja umjesto copy-paste increment logike.
4. Emitiraj samo structured `logger.debug/info` signal — bez user-facing šuma u Telegramu.
5. Ako postoji runtime-status lane koji to može sigurno izložiti bez UX šuma, dodaj summary hook; ako ne, ostavi samo log + testove u prvom rezu.

**Verification:**
- unit test da cache hit/miss counteri rastu kako treba
- unit test da TTL finalize path označi `ttl_expiry`
- unit test da manual refresh/evict path ne zove `close()` gdje ne smije
- grep/log assertion da se ne emitira user-facing noise

**Suggested test commands:**
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/tools/test_zombie_process_cleanup.py -n 0`
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/gateway/test_agent_cache_telemetry.py -n 0`

**Exit criteria:**
- možeš dokazati hit/miss/eviction razloge kroz testove
- nema promjene u vanjskom ponašanju gatewaya osim boljih logova/telemetryja

---

## Task P1.2 — Async memory telemetry

**Objective:** Instrumentirati postojeći memory lane (`prefetch_all`, `sync_all`, `queue_prefetch_all`) tako da konačno imamo signal o kvaliteti i kašnjenju memory rada.

**Files:**
- Modify: `/mnt/d/HermesAgent/app/agent/memory_manager.py`
- Inspect / possible touch: `/mnt/d/HermesAgent/app/run_agent.py`
- Likely tests:
  - `/mnt/d/HermesAgent/app/tests/run_agent/test_run_agent.py`
  - `/mnt/d/HermesAgent/app/tests/run_agent/test_memory_sync_interrupted.py`
- Likely add new tests:
  - `/mnt/d/HermesAgent/app/tests/agent/test_memory_manager_telemetry.py`

**Current-state grounded notes:**
- `prefetch_all`, `queue_prefetch_all`, `sync_all` postoje u `agent/memory_manager.py:287-324`
- provider contract za turn hooks i session hooks postoji u `agent/memory_provider.py`
- `run_agent` već testira ordering i interrupt behavior kroz memory-related testove pronađene u `tests/run_agent/`

**Step-by-step:**
1. U `MemoryManager` dodaj minimalni telemetry container:
   - per-provider `prefetch_calls`
   - `prefetch_nonempty_hits`
   - `prefetch_failures`
   - `sync_calls`
   - `sync_failures`
   - `queue_prefetch_calls`
   - latency bucket ili zadnji duration za `prefetch_all`
2. Mjeri latenciju na boundaryju `MemoryManager`, ne duboko po providerima u prvom rezu.
3. Dodaj helper tipa `get_telemetry_snapshot()` za testiranje i eventualni debug dump.
4. Ne uvoditi novi persistence layer; prvo samo runtime telemetry + testovi.
5. Provjeri da interrupted-turn guard i dalje ne radi `sync_all`/`queue_prefetch_all` kad ne smije.

**Verification:**
- test da `prefetch_all` broji non-empty hit vs empty response
- test da provider exception broji failure bez rušenja drugog providera
- test da `sync_all` i `queue_prefetch_all` i dalje skipaju interrupted path gdje je već pokriveno
- test da telemetry snapshot ne curi memory-context sadržaj

**Suggested test commands:**
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/run_agent/test_memory_sync_interrupted.py -n 0`
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/agent/test_memory_manager_telemetry.py -n 0`
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/run_agent/test_run_agent.py -k 'prefetch_all or on_turn_start or memory' -n 0`

**Exit criteria:**
- postoji mjerljiv signal za memory lane kvalitetu
- nema promjene u provider API ugovoru koja bi razbila postojeće pluginove

---

## Task P1.3 — Browser provider contract

**Objective:** Formalizirati browser backend contract tako da Hermes browser lane ne ostane skup ad-hoc backend grananja.

**Files:**
- Modify: `/mnt/d/HermesAgent/app/tools/browser_tool.py`
- Inspect/likely modify:
  - `/mnt/d/HermesAgent/app/tools/browser_cdp_tool.py`
  - `/mnt/d/HermesAgent/app/tools/browser_camofox.py`
  - `/mnt/d/HermesAgent/app/tools/browser_providers/browser_use.py`
  - `/mnt/d/HermesAgent/app/tools/browser_providers/firecrawl.py`
- Likely add: `/mnt/d/HermesAgent/app/tools/browser_providers/base.py` expansion or new contract helpers
- Tests:
  - `/mnt/d/HermesAgent/app/tests/tools/test_browser_cdp_tool.py`
  - `/mnt/d/HermesAgent/app/tests/tools/test_browser_homebrew_paths.py`
  - likely new `/mnt/d/HermesAgent/app/tests/tools/test_browser_provider_contract.py`

**Current-state grounded notes:**
- `browser_tool.py` već eksplicitno navodi više backendova i auto-detection
- browser lane već ima više provider modula, ali contract nije još dovoljno čist kao zaseban capability layer
- postoje testovi za CDP i path discovery, što daje dobru bazu za contract-first refactor

**Step-by-step:**
1. Zapiši canonical capability set za browser lane:
   - `open/navigate`
   - `snapshot`
   - `click`
   - `type`
   - `wait`
   - `screenshot`
   - optional: `eval`, `cdp_raw`, `record`, `downloads`
2. Uvedi normalized provider interface ili capability descriptor koji backend vraća.
3. Premjesti backend-specific grananje iz call-siteova prema provider boundaryju gdje je moguće.
4. Jasno odvoji:
   - agent-facing browser tool behavior
   - provider capability detection
   - provider-specific extensions
5. U prvom rezu ne pokušavati unified super-refactor za sve browser module odjednom; krenuti s contract + 1-2 backend adaptera.

**Verification:**
- postojeći CDP testovi i dalje prolaze
- novi testovi potvrđuju da unsupported capability vraća koristan structured error umjesto ad-hoc exceptiona
- browser tool zadržava isti agent-facing API

**Suggested test commands:**
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/tools/test_browser_cdp_tool.py -n 0`
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/tools/test_browser_homebrew_paths.py -n 0`
- `cd /mnt/d/HermesAgent/app && ./venv/bin/python -m pytest -q tests/tools/test_browser_provider_contract.py -n 0`

**Exit criteria:**
- capability matrix postoji i testirana je
- backend-specific errori više nisu razbacani po browser call flowu

---

# P2 detaljni execution plan

## Task P2.1 — Memory provider scorecard

**Objective:** Dodati način da se provideri uspoređuju po ponašanju, ne po dojmu.

**Files:**
- Modify: `/mnt/d/HermesAgent/app/agent/memory_manager.py`
- Modify or inspect: `/mnt/d/HermesAgent/app/agent/memory_provider.py`
- Optional new helper: `/mnt/d/HermesAgent/app/agent/memory_scorecard.py`
- Tests: nova datoteka tipa `/mnt/d/HermesAgent/app/tests/agent/test_memory_provider_scorecard.py`

**Metrics za scorecard:**
- write reliability
- prefetch usefulness rate
- session-boundary correctness
- compression hook participation
- delegation observation support

**Note:** Ovo se radi tek nakon P1.2 jer koristi telemetry koji tamo nastaje.

---

## Task P2.2 — Tool schema / cache invalidation audit

**Objective:** Ojačati `get_tool_definitions()` invalidation i zaštitu od stale/duplicate schema stanja.

**Files:**
- Modify: `/mnt/d/HermesAgent/app/model_tools.py`
- Inspect/possible modify: `/mnt/d/HermesAgent/app/tools/registry.py`
- Tests:
  - `/mnt/d/HermesAgent/app/tests/test_get_tool_definitions_cache_isolation.py`
  - `/mnt/d/HermesAgent/app/tests/test_model_tools.py`
  - moguće nova datoteka `/mnt/d/HermesAgent/app/tests/test_tool_definition_cache_invalidation.py`

**Current-state grounded notes:**
- memoization već postoji u `model_tools.py:250-331`
- već postoji regression test lane za cache isolation i duplicate-tool problem

**Step-by-step:**
1. Auditirati sve izvore invalidacije:
   - config mtime
   - registry generation
   - dynamic schema dependencies
   - plugin/MCP refresh
2. Dopuniti testove za scenario-based invalidation umjesto samo list-copy zaštite.
3. Provjeriti postoji li još path gdje caller može zagaditi shared schema state.

**Verification:**
- svi postojeći cache-isolation testovi prolaze
- novi test pokriva barem jedan config-change invalidation case

---

## Task P2.3 — Delegation / worker report standard

**Objective:** Standardizirati završni output subagenata i background worker laneova.

**Files:**
- Inspect/likely modify: `/mnt/d/HermesAgent/app/tools/delegate_tool.py`
- Inspect/likely modify: `/mnt/d/HermesAgent/app/run_agent.py`
- Optional helper/shared formatter: `/mnt/d/HermesAgent/app/agent/reporting.py`
- Tests:
  - `/mnt/d/HermesAgent/app/tests/tools/test_delegate.py`
  - `/mnt/d/HermesAgent/app/tests/run_agent/test_interrupt_propagation.py`
  - `/mnt/d/HermesAgent/app/tests/run_agent/test_real_interrupt_subagent.py`

**Target report shape:**
- `STATUS`
- `REZULTAT`
- `VERIFIKACIJA`
- `SLJEDEĆI_KORAK`
- `BLOCKER`

**Note:** Ovo nije UX šminka nego orchestration disciplina.

---

# P3 — svjesno odgođeno

## Task P3.1 — Ambient/autonomous lane

**Presuda:** ne raditi prije P1/P2. Prvo instrumentacija, onda autonomija.

## Task P3.2 — Memory graph rewrite

**Presuda:** ne raditi. Uzimamo pattern-e, ne prepisujemo Hermes core memory arhitekturu u jcode smjer.

---

# Preporučeni redoslijed izvedbe

## Sprint 1
1. P1.1 Gateway cache telemetry
2. P1.2 Async memory telemetry
3. P2.2 Tool schema/cache invalidation audit

## Sprint 2
4. P1.3 Browser provider contract
5. P2.1 Memory provider scorecard
6. P2.3 Delegation report standard

## Zašto ovaj redoslijed
- Sprint 1 daje **operativni signal** i smanjuje slijepo debugiranje.
- Sprint 2 radi refactor i policy slojeve tek kad već imamo mjerenje i stabilniju osnovu.

---

# Minimalni acceptance kriteriji po sprintu

## Sprint 1 accepted kad:
- gateway cache ima mjerljive hit/miss/eviction signale
- memory lane ima mjerljiv prefetch/sync signal
- tool schema cache audit ima barem jedan novi regression test za invalidation

## Sprint 2 accepted kad:
- browser backend capability contract postoji i testiran je
- memory provider scorecard može usporediti barem 2 providera ili 2 provider stanja
- delegation completion output je standardiziran bez rušenja postojećih interrupt/report flowova

---

# Rizici i tradeoffovi

1. **Previše telemetryja u user-facing outputu**
   - Mitigacija: držati sve u log/debug/runtime snapshot sloju, ne u Telegram replyju.

2. **Browser contract refactor ode preširoko**
   - Mitigacija: prvo contract + capability map; ne rewrite svih backendova odjednom.

3. **Memory telemetry postane plugin-breaking**
   - Mitigacija: mjeriti na `MemoryManager` boundaryju, ne mijenjati provider API u prvom rezu.

4. **Cache invalidation audit otvori previše sporednih bugova**
   - Mitigacija: prvo regression tests, pa minimalni fix.

---

# Preporučeni prvi execution slice

Ako krećeš odmah u implementaciju, kreni ovim redom:
1. `gateway/run.py` telemetry counters + tests
2. `agent/memory_manager.py` telemetry snapshot + tests
3. `model_tools.py` invalidation regression audit

To je najčišći početni trokut: **gateway state + memory state + tool schema state**.
