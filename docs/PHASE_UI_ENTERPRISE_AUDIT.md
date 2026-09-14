# Control Center — enterprise UI audit

Written **before** any redesign code. Baseline recorded: **680 platform tests passing**
(`.venv/bin/python -m pytest tests/platform -q`).

---

## 1. Current UI structure

One route, one screen, one scroll. `nova/control/ui/src/app.tsx` (464 lines) renders a header,
a five-tile stat row, then nine panels in a two-column grid, then a footer.

```
header    identity · runtime badge · theme toggle
stats     Agents · Work in flight · Needs a human · Channels · Corpora
grid      Health | Agents
          Channels (full width)
          Work | Objectives
          Knowledge | Governance
          Recent decisions | Usage
footer    company · support email · refresh note
```

There is **no navigation**, no detail view, and no way to focus on one agent. Everything the
deployment knows is on one page at one level of emphasis.

## 2. Component inventory

| Component | File | Role |
|---|---|---|
| `Card` family | `components/ui/card.tsx` | shadcn card primitives |
| `Badge` | `components/ui/badge.tsx` | cva variants + `good`/`warn` status tones |
| `Tooltip` | `components/ui/tooltip.tsx` | Radix, portalled |
| `Skeleton`, `Separator` | `components/ui/` | loading, rules |
| `Hint`, `InfoDot` | `components/bits.tsx` | hover explanations |
| `Stat` | `components/bits.tsx` | count-up metric tile |
| `Panel` | `components/bits.tsx` | card + the four states (loading / forbidden / error / empty) |
| `Empty`, `Row` | `components/bits.tsx` | empty state, staggered row |
| `usePanel`, `useCountUp`, `useTheme` | `lib/hooks.ts` | polling, animation, theme |
| `load`, `plural` | `lib/api.ts` | the only fetch path |

**Functional vs decorative.** Functional: every panel's data, the theme toggle, tooltips,
the 15s refresh. Purely decorative: the aurora blobs behind the header, the row stagger, the
count-up. Nothing decorative carries meaning, so all of it is safe to replace.

## 3. Current navigation

None. This is the single largest weakness: an operations manager cannot answer *"what is the
sales agent doing"* without reading a shared table.

## 4. API and data dependencies — what is REAL

Every field below was captured from a live deployment. **The redesign may use only these.**

| Route | Role | Fields that exist |
|---|---|---|
| `/health` | viewer | `platform{version,tenant_id}`, `runtime{runtime,capabilities,reachable,detail,work_store_present,agent_count}`, `bundle{digest,agents}` |
| `/identity` | viewer | `product_name`, `company_name`, `logo`, `favicon`, `theme`, `support`, `welcome`, `tenant_id` |
| `/agents` | viewer | `id`, `display_name`, `role`, `description`, `enabled`, `model`, `materialized`, `in_sync`, `declared_digest`, `applied_digest`, `limits{…,delegation{…}}`, `approval_required_for[]`, `knowledge_sources[]` |
| `/tasks` | viewer | `task_id`, `title`, `state`, `runtime_status`, `agent_id`, `created_at`, `started_at`, `completed_at`, `priority`, `consecutive_failures`, `last_error`, `needs_attention`, plus `counts{}` |
| `/objectives` | viewer | `id`, `title`, `owner`, `owner_display_name`, `description`, `acceptance`, `enabled`, `routing_allowed`, `refusals[]`, `ungoverned_steps[]`, `warnings[]`, `state`, `done`, `total`, `steps[]`, `blocking` |
| `/knowledge` | viewer | `retrieval_enabled`, `document_extraction`, `index_detail`, `sources[{id,title,description,classification,root,readable_by,indexed,documents,chunks,bytes}]`, `undeclared_in_index[]` |
| `/channels` | viewer | `id`, `provider`, `provider_label`, `display_name`, `enabled`, `transport`, `needs_public_endpoint`, `verification`, `caveat`, `allowed_agents[]`, `routes[]`, `approval_required_for[]`, `derived_agents[]`, `required_env[]`, `missing_by_agent{}`, `status` |
| `/policy` | **admin** | `declared`, `enforced`, `actions{name:{tools,description,requires_approval}}`, `permissions{}`, `baseline_tools[]`, `agents[{id,display_name,allow,deny,approval_actions,unlisted_tool,has_allowlist,warnings}]` |
| `/decisions` | **admin** | `decisions[]`, `counts{}`, `total` |
| `/budget` | **admin** | `controls[]`, `advisory[]`, `recorded[]`, `observed[{agent_id,available,detail,enforcement,caveats,api_calls,total_tokens,estimated_cost_usd,models}]`, `observed_caveat`, `enforcement_classes[]` |

Writes (admin): `POST /work/{id}/decide`, `/objectives/{id}/submit`, `/channels/apply`.

### What the brief asks for that the backend does NOT have

Recorded here so the redesign designs around reality rather than inventing it:

- **No approval *queue*.** There is no endpoint returning pending approval requests. What
  exists is *declared* requirements (`approval_required_for`), *escalations already made*
  (`/decisions` with an escalate outcome), and *work needing a human* (`needs_attention`).
  An Approvals screen must be built from those three and must not imply a live inbox.
- **No trends, no history, no percentages.** Every figure is a current reading. No sparkline,
  no "+12% this week".
- **No workspace selector, notifications feed, user menu, or org settings.** No backend.
- **No integrations surface** beyond channels.
- **No per-task timeline.** `/tasks` has three timestamps; `/decisions` has its own events.
  An activity timeline can merge those two real sources and nothing else.

Navigation will therefore offer: **Overview, Agents, Objectives, Work, Approvals, Activity,
Knowledge, Channels, Policies, Usage** — and *not* Integrations or Organization.

## 5. Current responsive behaviour

`grid gap-6 lg:grid-cols-2` and `grid-cols-2 sm:grid-cols-3 lg:grid-cols-5` for stats;
`max-w-7xl` container. Works down to tablet; below that the stat row becomes two columns and
panels stack. No layout is *designed* for narrow desktop — it is only a collapse.

## 6. Current accessibility behaviour

**Holding up:** Radix tooltips are keyboard- and screen-reader-accessible; the theme toggle
has an `aria-label` and a visible focus ring; `prefers-reduced-motion` is respected in
`index.css` and in `useCountUp`; text is real text with `textContent` semantics (React
escapes; `dangerouslySetInnerHTML` is absent and tested for).

**Gaps:** no skip link; no landmark structure beyond `header`/`main`/`footer`; panels are
`div`s rather than `section`s with accessible names; status is conveyed by a coloured badge
whose *text* carries the meaning (good) but tone alone distinguishes `good` from `warn` at a
glance; no live region announces a refresh.

## 7. Visual weaknesses (the honest list)

1. **No hierarchy.** Nine panels of equal weight. Health sits beside Agents at the same size.
2. **No navigation or drill-down.** Cannot enter an agent.
3. **Looks like a shadcn starter.** Default card, default border, default radius, one flat
   surface. The brief's "does anything look like a generic template?" — yes, this does.
4. **Flat.** No depth, no material. Two aurora blobs behind the header and nothing else.
5. **Tables of rows where a timeline or a profile would read better** (work, decisions).
6. **Status colour is under-used.** `good`/`warn` only; blocked/failed/escalated all read the
   same.
7. **Numbers are decoration.** Five stats animate on load, then never change meaningfully.
8. **Empty states are honest but plain** — a dashed box.
9. **Long names untested.** No truncation strategy beyond `truncate` on one line.
10. **Light mode is an afterthought** — tokens exist, but the design was composed dark.

## 8. Proposed design system

**Four surface levels**, so glass means something rather than being applied everywhere:

| Level | Use | Treatment |
|---|---|---|
| 0 canvas | page | very dark, two slow radial fields, no blur |
| 1 glass | panels, cards | translucent fill, `backdrop-blur`, hairline border, inner top highlight |
| 2 elevated | active card, command palette, agent header | stronger fill + border luminosity + larger shadow |
| 3 solid | inputs, primary buttons, anything read at length | near-opaque for contrast |

**Semantic colour** — one token per state, used *only* for state: `running` (emerald),
`waiting` (amber), `blocked` (rose), `info` (blue), `neutral`. Plus `glass-border`,
`glass-highlight`, text ink levels.

**Motion**: entrance 260–400ms `cubic-bezier(.22,1,.36,1)`; hover 150ms; pointer-tracked
highlight via two CSS custom properties written on `mousemove` (CSS does the paint, JS only
sets `--x`/`--y`); pulsing only for genuinely live states. All of it off under
`prefers-reduced-motion`.

**Primitives to build**: `GlassPanel`, `GlassCard` (pointer-aware), `StatusDot`, `MetricCard`,
`AgentCard`, `ActivityTimeline`, `ApprovalCard`, `CommandBar`, `SectionHeader`, `EmptyState`,
`Skeleton` (glass shimmer), `Sidebar`, `Tabs`.

## 9. Files that will change

- `nova/control/ui/src/**` — rewritten (app shell, screens, primitives, tokens)
- `nova/control/ui/package.json` — possibly one small dependency for the command palette
- `nova/control/static/**` — rebuilt output (committed, as the stdlib server cannot build)
- `tests/platform/test_control_server.py` — only if an assertion names a file that moves

## 10. Files that must NOT change

- `nova/control/server.py` — **the CSP stays exactly as it is** (`script-src 'self'`,
  `style-src 'self'`, no `unsafe-inline`). Verified achievable: Radix positions through the
  CSSOM, which CSP does not govern.
- `nova/control/api.py`, `auth.py` — no new routes, no relaxed roles, no fabricated fields.
- Anything under `agent/`, `gateway/`, `hermes_cli/`, `tools/` — Hermes core.
- The workspace boundary: the app stays at `nova/control/ui/`, never `apps/*`.
- `nova/policy/**`, `nova/channels/**` — backend behaviour is settled.

## 11. Acceptance, restated as checks

Visual quality and functional validation are **separate claims**. This audit commits to
reporting them separately: the redesign can be beautiful and still have connected nothing to
a real provider. The completion document will say so.
