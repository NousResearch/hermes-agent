# HERMES//HUB — ROADMAP 3: the AI Lab tab + comprehensiveness pass

A one-stop shop for everything **AI · coding · Claude**, plus an additive
"make every tab richer" programme. Same rules as always: zero-dependency stdlib
server, zero-build vanilla-JS frontend, free/no-key upstreams proxied through
`server.py` with offline sample fixtures, `SOURCES` registry, and the ship
standard (full unit suite green + 3 consecutive green e2e). **Additive only —
nothing is ever removed.**

---

## A. AI Lab tab

### Shipped (this batch)
- ✅ **New "AI Lab" page** (deep-linkable, backfilled for existing users via a
  `pagesSeed` bump).
- ✅ **Code Lab** — an in-browser coding box. Write a JS `solution`; **Run** executes
  it against test cases inside a **sandboxed Web Worker** (with a 2.5s timeout so
  infinite loops can't hang the page); **progressive hints** and a reveal
  **Solution**; 9 problems across arrays/strings/logic/recursion/DP/algorithms.
  Fully offline.
- ✅ **Claude & AI Hub** — a curated, categorised directory (Courses · Claude Code ·
  Skills/Plugins/Connectors · Build & API · Models & updates · Community) with a
  rotating **"today's pick"**. Seeded with Anthropic Academy (Skilljar), the
  prompt-eng tutorial, Building Effective Agents, DeepLearning.AI, Hugging Face,
  fast.ai, MCP, cookbook, release notes, and more. Opens in the in-app viewer.
- ✅ **Prompt Library** — save/tag/search/copy your own reusable prompts, skills
  and snippets; add/edit/delete; seeded with high-leverage examples.

### Planned — live-data widgets (free/no-key, via SOURCES + samples)
1. **Repo Radar** — trending repositories via the **GitHub search API**
   (`sort=stars`, recent windows), general focus. Daily pick + list; opens repo.
2. **arXiv Papers** — daily cs.AI / cs.CL / cs.LG papers (arXiv Atom API), with a
   "summarize with the agent" button.
3. **AI Radar (news)** — a dedicated feed spanning arXiv, Hacker News (AI/LLM),
   Google News "Anthropic/Claude", and Product Hunt AI — reusing the RSS engine.
4. **Claude Changelog watcher** — surface release-notes items; optional per-repo
   GitHub release watchers.
5. **AI Daily Brief (hero)** — one highlighted item per category each day
   (model update · repo · paper · skill/plugin · course lesson · showcase),
   deterministic-by-date over the live pool with curated offline fallback.
6. **Learning tracker** — turn the course directory into a personal study list
   with to-do/doing/done status and an agent-powered "suggest my next lesson".
7. **Code Lab v2** — more problems + languages notes, a Python track via the
   agent (write → agent reviews), and "explain this failing test with Claude".

---

## B. Comprehensiveness pass — richer data on every existing tab (additive)

Only **adding** widgets/sources; existing behaviour untouched.

- **Main:** on-this-day facts, a quote/word-of-the-day, a unit/timezone quick
  convert, a system-status quick glance.
- **Markets:** commodities & metals (gold/oil), bonds/rates, an economic-calendar
  feed, more crypto (funding/derivs where a free source exists), a portfolio
  performance sparkline.
- **Feeds:** more source packs per topic, a "long reads" pack, YouTube/RSS video
  feeds, a subreddit multi-pick, Product Hunt.
- **Sports:** more leagues (rugby/cricket — high SA relevance), a fixtures
  calendar, live commentary where free.
- **Intel:** marine/aviation weather, tides, pollen detail, currency-strength
  heatmap, a world-events map, moon/sun times.
- **Health:** more calculators on request (MELD/Child-Pugh shipped; NIHSS,
  Glasgow-Blatchford, Alvarado, CIWA, opioid conversion, Naegele EDD…), a
  guideline-links directory, an interactions helper via the MedBot.

Each item follows the registry + sample-fixture convention so offline and e2e
stay green.

---

## C. Cross-cutting refinements (recommended)
- **Global command palette entries** for every new widget/page ("Go to AI Lab",
  "New snippet", "Random OSCE station").
- **Per-widget settings** where useful (Repo Radar language filter, arXiv
  categories, AI Radar source toggles).
- **Export/share** a snippet or a solved Code Lab solution.
- **Onboarding hints** for the new tabs (first-run tooltips).

Sequencing: ship the AI Lab live-data widgets (B-list #1–#5) first, then the
comprehensiveness pass tab-by-tab, then cross-cutting polish — each as its own
green PR.
