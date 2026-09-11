/**
 * Hermes fleet — Models tab.
 *
 * The control surface for ~/.hermes/fleet/models.yaml: every agent's waterfalls, the model registry with
 * OpenRouter-level host control, live prices/uptime, real usage and billed cost, decisions and history.
 * Edits are drafted locally, previewed (validation + the exact config diff), then applied through
 * plugins/fleet-models/core.py — which writes models.yaml, compiles the nine configs, verifies them and
 * keeps a pre-image for one-click revert. Built from src/index.jsx with esbuild (IIFE, host React).
 */
const SDK = window.__HERMES_PLUGIN_SDK__;
if (SDK) {
const { React } = SDK;
const h = React.createElement;
const Fragment = React.Fragment;
const { useState, useEffect, useCallback, useMemo, useRef } = SDK.hooks;
const API = "/api/plugins/fleet-models";
const fetchJSON = SDK.fetchJSON;

const AUX_TASKS = ["vision", "compression", "title_generation", "background_review", "goal_judge",
  "kanban_decomposer", "web_extract", "session_search", "skills_hub", "approval", "flush_memories"];
const REASONING = ["", "none", "minimal", "low", "medium", "high", "xhigh", "max"];
const SLOT_LABEL = { main: "Main loop", subagents: "Subagents", cron: "Cron jobs" };
const TASK_LABEL = { vision: "Vision", compression: "Compression", title_generation: "Titles",
  background_review: "Background review", goal_judge: "Goal judge", kanban_decomposer: "Decomposer",
  web_extract: "Web extract", session_search: "Session search", skills_hub: "Skills hub", approval: "Approval",
  flush_memories: "Memory flush" };

// ── helpers ──────────────────────────────────────────────────────────────────────────────
const clone = (o) => JSON.parse(JSON.stringify(o));
const chainOf = (spec) => (spec == null ? [] : Array.isArray(spec) ? spec : spec.chain || []);
const reasoningOf = (spec) => (spec && !Array.isArray(spec) ? spec.reasoning || "" : "");
const withChain = (spec, chain) => (spec && !Array.isArray(spec) ? { ...spec, chain } : chain);
const money = (v, d = 4) => (v == null ? "—" : v === 0 ? "$0" : v < 0.0001 ? "<$0.0001" : "$" + Number(v).toFixed(v >= 10 ? 2 : d));
const num = (v) => (v == null ? "—" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(v));
const pct = (v) => (v == null ? "—" : v.toFixed(v >= 99.95 ? 2 : 1) + "%");
const price = (v) => (v == null ? "—" : "$" + (v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4)));
const ago = (ts) => {
  if (!ts) return "—";
  const s = Math.max(0, Date.now() / 1000 - (typeof ts === "string" ? Date.parse(ts) / 1000 : ts));
  return s < 60 ? "just now" : s < 3600 ? Math.round(s / 60) + " min ago" : s < 86400 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " d ago";
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const hostSlug = (t) => String(t || "").split("/")[0];

function cls(...xs) { return xs.filter(Boolean).join(" "); }

// ── small UI atoms ───────────────────────────────────────────────────────────────────────
function Pill({ doc, alias, onRemove, onLeft, onRight, first, last, dim, compact }) {
  const m = (doc.models || {})[alias];
  const prov = m ? m.provider : "missing";
  return (
    <span className={cls("fm-pill", "fm-pill--" + prov, dim && "fm-pill--dim")} title={m ? `${m.id}${m.notes ? "\n\n" + m.notes : ""}` : "not in the registry"}>
      {onLeft && !first ? <button className="fm-pill-btn" onClick={onLeft} title="move earlier">‹</button> : null}
      <span className="fm-pill-dot" />
      <span className="fm-pill-name">{m ? m.short || alias : alias}</span>
      {m && m.billing === "subscription" ? <span className="fm-tag fm-tag--sub">SUB</span> : null}
      {m && m.reasoning && !compact ? <span className="fm-tag" title="reasoning pin">{m.reasoning}</span> : null}
      {onRight && !last ? <button className="fm-pill-btn" onClick={onRight} title="move later">›</button> : null}
      {onRemove ? <button className="fm-pill-btn fm-pill-x" onClick={onRemove} title="remove this rung">×</button> : null}
    </span>
  );
}

function Chain({ doc, chain, empty }) {
  if (!chain || !chain.length) return <span className="fm-muted">{empty || "—"}</span>;
  // arrow + rung wrap together, so a line never ends on a dangling arrow
  return (
    <span className="fm-chain">
      {chain.map((a, i) => (
        <span key={a + i} className="fm-link-step">
          {i ? <span className="fm-arrow">→</span> : null}
          <Pill doc={doc} alias={a} dim={i > 0} compact />
        </span>
      ))}
    </span>
  );
}

function ChainEditor({ doc, chain, onChange, filter }) {
  const opts = Object.keys(doc.models || {}).filter((a) => !chain.includes(a) && (!filter || filter(doc.models[a])));
  const move = (i, d) => { const c = chain.slice(); const [x] = c.splice(i, 1); c.splice(i + d, 0, x); onChange(c); };
  return (
    <span className="fm-chain fm-chain--edit">
      {chain.map((a, i) => (
        <Fragment key={a + i}>
          {i ? <span className="fm-arrow">→</span> : null}
          <Pill doc={doc} alias={a} first={i === 0} last={i === chain.length - 1}
            onLeft={() => move(i, -1)} onRight={() => move(i, 1)}
            onRemove={chain.length > 1 ? () => onChange(chain.filter((_, j) => j !== i)) : null} />
        </Fragment>
      ))}
      {opts.length ? (
        <select className="fm-add" value="" onChange={(e) => e.target.value && onChange(chain.concat([e.target.value]))}>
          <option value="">+ rung</option>
          {opts.map((a) => <option key={a} value={a}>{doc.models[a].short || a}</option>)}
        </select>
      ) : null}
    </span>
  );
}

function Stat({ label, value, sub, tone }) {
  return (
    <div className={cls("fm-stat", tone && "fm-stat--" + tone)}>
      <div className="fm-stat-label">{label}</div>
      <div className="fm-stat-value">{value}</div>
      {sub ? <div className="fm-stat-sub">{sub}</div> : null}
    </div>
  );
}

function Badge({ tone, children, title }) {
  return <span className={cls("fm-badge", tone && "fm-badge--" + tone)} title={title}>{children}</span>;
}

function Section({ title, right, children, className }) {
  return (
    <section className={cls("fm-section", className)}>
      {title ? <header className="fm-section-head"><h3>{title}</h3>{right}</header> : null}
      {children}
    </section>
  );
}

// ── Fleet view ───────────────────────────────────────────────────────────────────────────
function helperGroups(aux) {
  const groups = {};
  Object.entries(aux || {}).forEach(([t, s]) => {
    if (t === "vision") return;
    const k = JSON.stringify(chainOf(s)) + "|" + reasoningOf(s);
    (groups[k] = groups[k] || { chain: chainOf(s), reasoning: reasoningOf(s), tasks: [] }).tasks.push(t);
  });
  return Object.values(groups);
}

function AgentCard({ doc, p, drift, use, onOpen }) {
  const a = doc.agents[p];
  const aux = a.aux || {};
  const top = use.hosts.sort((x, y) => y[1] - x[1])[0];
  return (
    <article className={cls("fm-card", drift && "fm-card--drift")} onClick={onOpen} role="button" tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onOpen()}>
      <header className="fm-card-head">
        <div>
          <div className="fm-card-name">{a.name || p} {a.locked ? <span className="fm-lock" title="Locked — changes need an explicit unlock">🔒</span> : null}</div>
          <div className="fm-card-role">{a.role || p}</div>
        </div>
        <div className="fm-card-badges">
          {drift ? <Badge tone="warn" title={drift.join("\n")}>drift</Badge> : <Badge tone="ok">in sync</Badge>}
          {a.reasoning ? <Badge title="agent default reasoning">{a.reasoning}</Badge> : null}
        </div>
      </header>
      <dl className="fm-slots">
        <dt>Main</dt><dd><Chain doc={doc} chain={chainOf(a.main)} /></dd>
        <dt>Subagents</dt><dd><Chain doc={doc} chain={chainOf(a.subagents)} empty="inherit main" /></dd>
        {a.cron ? <Fragment><dt>Cron</dt><dd><Chain doc={doc} chain={chainOf(a.cron)} /></dd></Fragment> : null}
        <dt>Vision</dt><dd><Chain doc={doc} chain={chainOf(aux.vision)} empty="main model" /></dd>
        {helperGroups(aux).map((g) => (
          <Fragment key={g.tasks.join()}>
            <dt title={g.tasks.join(", ")}>{g.tasks.length > 1 ? "Helpers ×" + g.tasks.length : TASK_LABEL[g.tasks[0]] || g.tasks[0]}</dt>
            <dd><Chain doc={doc} chain={g.chain} />{g.reasoning ? <span className="fm-tag fm-tag--r">{g.reasoning}</span> : null}</dd>
          </Fragment>
        ))}
      </dl>
      <footer className="fm-card-foot">
        <span><b>{num(use.calls)}</b> calls</span>
        <span><b>{money(use.billed, 3)}</b> billed</span>
        <span><b>{num(use.ma)}</b> on subscription</span>
        {top ? <span className="fm-muted" title="most calls served by">via {top[0]}</span> : null}
      </footer>
    </article>
  );
}

function usageByProfile(usage) {
  const out = {};
  ((usage && usage.rows) || []).forEach((r) => {
    const u = (out[r.profile] = out[r.profile] || { calls: 0, billed: 0, ma: 0, hostMap: {} });
    u.calls += r.calls; u.billed += r.billed_usd; if (r.modelark) u.ma += r.calls;
    if (r.host) u.hostMap[r.host] = (u.hostMap[r.host] || 0) + r.calls;
  });
  Object.values(out).forEach((u) => { u.hosts = Object.entries(u.hostMap); });
  return out;
}

function FleetView({ state, doc, usage, onOpen }) {
  const byP = usageByProfile(usage);
  const empty = { calls: 0, billed: 0, ma: 0, hosts: [] };
  const tot = Object.values(byP).reduce((a, u) => ({ calls: a.calls + u.calls, billed: a.billed + u.billed, ma: a.ma + u.ma }), { calls: 0, billed: 0, ma: 0 });
  const models = doc.models || {};
  return (
    <Fragment>
      <div className="fm-stats">
        <Stat label={`Calls · ${usage ? usage.days : 7}d`} value={num(tot.calls)} />
        <Stat label="Billed (OpenRouter)" value={money(tot.billed, 2)} sub="real invoices" tone="money" />
        <Stat label="ModelArk subscription" value={num(tot.ma) + " calls"} sub={tot.calls ? Math.round((100 * tot.ma) / tot.calls) + "% of all calls · $0" : "$0"} tone="sub" />
        <Stat label="Registry" value={Object.keys(models).length + " models"} sub={Object.values(models).filter((m) => m.provider === "modelark").length + " subscription · " + Object.values(models).filter((m) => m.provider === "openrouter").length + " OpenRouter"} />
        <Stat label="Sync" value={Object.keys(state.drift || {}).length ? Object.keys(state.drift).length + " drifted" : "all 9 in sync"} tone={Object.keys(state.drift || {}).length ? "warn" : "ok"} sub={"revision " + state.revision} />
      </div>
      <div className="fm-grid">
        {state.profiles.map((p) => (
          <AgentCard key={p} doc={doc} p={p} drift={(state.drift || {})[p]} use={byP[p] || empty} onOpen={() => onOpen(p)} />
        ))}
      </div>
    </Fragment>
  );
}

// ── Agent editor ─────────────────────────────────────────────────────────────────────────
function LiveChain({ doc, live }) {
  if (!live) return <span className="fm-muted">not set</span>;
  const byId = {};
  Object.entries(doc.models || {}).forEach(([a, m]) => { byId[m.provider + "|" + m.id] = a; });
  return (
    <span className="fm-chain">
      {live.map(([prov, id], i) => (
        <Fragment key={i}>
          {i ? <span className="fm-arrow">→</span> : null}
          {byId[prov + "|" + id] ? <Pill doc={doc} alias={byId[prov + "|" + id]} dim /> : <code className="fm-code">{prov}:{id}</code>}
        </Fragment>
      ))}
    </span>
  );
}

function SlotRow({ doc, label, spec, live, onChange, nullable, nullLabel, withReasoning, filter, onDelete, hint }) {
  const chain = chainOf(spec);
  const isNull = spec == null;
  const primary = Object.keys(doc.models || {}).find((a) => !filter || filter(doc.models[a]));
  return (
    <div className="fm-slot">
      <div className="fm-slot-label">
        <div>{label}</div>
        {hint ? <div className="fm-slot-hint">{hint}</div> : null}
      </div>
      <div className="fm-slot-body">
        {isNull ? (
          <span className="fm-muted">{nullLabel} <button className="fm-link" onClick={() => onChange([primary])}>set a chain</button></span>
        ) : (
          <ChainEditor doc={doc} chain={chain} filter={filter} onChange={(c) => onChange(withChain(spec, c))} />
        )}
        <div className="fm-slot-live">live: <LiveChain doc={doc} live={live} /></div>
      </div>
      <div className="fm-slot-side">
        {withReasoning && !isNull ? (
          <select value={reasoningOf(spec)} title="reasoning effort for this task"
            onChange={(e) => onChange(e.target.value ? { chain, reasoning: e.target.value } : chain)}>
            {REASONING.map((r) => <option key={r} value={r}>{r ? "reasoning " + r : "reasoning —"}</option>)}
          </select>
        ) : null}
        {nullable && !isNull ? <button className="fm-link" onClick={() => onChange(null)} title="remove this chain">→ {nullLabel || "clear"}</button> : null}
        {onDelete ? <button className="fm-link fm-link--danger" onClick={onDelete}>remove</button> : null}
      </div>
    </div>
  );
}

function listInput(v) { return (v || []).join(", "); }
function parseList(s) { return s.split(",").map((x) => x.trim()).filter(Boolean); }

function AgentView({ state, draft, setDraft, p, setP }) {
  const a = draft.agents[p];
  const live = state.live[p] || {};
  const set = (fn) => setDraft((d) => { const n = clone(d); fn(n.agents[p], n); return n; });
  const aux = a.aux || {};
  const unusedTasks = AUX_TASKS.filter((t) => !(t in aux));
  const drift = (state.drift || {})[p];
  const visionOk = (m) => !!m.vision;
  const toolsOk = (m) => m.tools !== false;
  return (
    <div className="fm-agent">
      <nav className="fm-agent-nav">
        {state.profiles.map((q) => (
          <button key={q} className={cls("fm-agent-tab", q === p && "is-active", (state.drift || {})[q] && "has-drift")} onClick={() => setP(q)}>
            {draft.agents[q].name || q}{draft.agents[q].locked ? " 🔒" : ""}
          </button>
        ))}
      </nav>
      <Section title={`${a.name || p} — ${a.role || ""}`} right={
        <span className="fm-row">
          {a.locked ? <Badge tone="warn">locked</Badge> : null}
          {drift ? <Badge tone="warn" title={drift.join("\n")}>config drifted from models.yaml</Badge> : <Badge tone="ok">config matches</Badge>}
        </span>}>
        {drift ? <div className="fm-note fm-note--warn">{drift.map((d, i) => <div key={i}>{d}</div>)}<div>Applying any change rewrites this profile from models.yaml.</div></div> : null}
        <div className="fm-slots-edit">
          <SlotRow doc={draft} label="Main loop" hint="primary → fallbacks" spec={a.main} live={live.main} filter={toolsOk}
            onChange={(c) => set((x) => { x.main = c; })} />
          <SlotRow doc={draft} label="Subagents" hint="delegated children — their own chain" spec={a.subagents} live={live.subagents}
            nullable nullLabel="inherit main" filter={toolsOk} onChange={(c) => set((x) => { x.subagents = c; })} />
          <SlotRow doc={draft} label="Cron jobs" hint="scheduled jobs on this profile" spec={a.cron} live={live.cron}
            nullable nullLabel="Hermes default" filter={toolsOk} onChange={(c) => set((x) => { x.cron = c; })} />
          {Object.keys(aux).sort((x, y) => (x === "vision" ? -1 : y === "vision" ? 1 : x.localeCompare(y))).map((t) => (
            <SlotRow key={t} doc={draft} label={TASK_LABEL[t] || t} hint={t === "vision" ? "images — every rung must accept them" : "auxiliary task"}
              spec={aux[t]} live={(live.aux || {})[t] ? live.aux[t].chain : null} withReasoning filter={t === "vision" ? visionOk : null}
              onChange={(c) => set((x) => { x.aux[t] = c; })} onDelete={() => set((x) => { delete x.aux[t]; })} />
          ))}
          {unusedTasks.length ? (
            <div className="fm-slot fm-slot--add">
              <div className="fm-slot-label">Add helper</div>
              <div className="fm-slot-body">
                <select value="" onChange={(e) => e.target.value && set((x) => { x.aux = x.aux || {}; x.aux[e.target.value] = [e.target.value === "vision" ? "glm" in draft.models ? "glm" : Object.keys(draft.models)[0] : Object.keys(draft.models)[0]]; })}>
                  <option value="">+ auxiliary task…</option>
                  {unusedTasks.map((t) => <option key={t} value={t}>{TASK_LABEL[t] || t}</option>)}
                </select>
                <span className="fm-muted"> tasks left unset use Hermes' automatic routing</span>
              </div>
            </div>
          ) : null}
        </div>
      </Section>
      <div className="fm-two">
        <Section title="Agent defaults">
          <label className="fm-field">
            <span>Default reasoning effort</span>
            <select value={a.reasoning || ""} onChange={(e) => set((x) => { x.reasoning = e.target.value || null; })}>
              {REASONING.map((r) => <option key={r} value={r}>{r || "Hermes default"}</option>)}
            </select>
            <small>Per-model pins (Models tab) win — e.g. v4.1 always runs high.</small>
          </label>
          <label className="fm-field">
            <span>Why this setup</span>
            <input value={a.why || ""} onChange={(e) => set((x) => { x.why = e.target.value; })} />
            <small>Shown in Smith's SOUL model table.</small>
          </label>
          {p === "root" ? (
            <label className="fm-check"><input type="checkbox" checked={!!a.locked} onChange={(e) => set((x) => { x.locked = e.target.checked; })} /> Locked (Smith is the overwatch — changes need an explicit unlock)</label>
          ) : null}
        </Section>
        <Section title="Profile default OpenRouter routing">
          <p className="fm-muted fm-small">Applies to this agent's OpenRouter calls for models without their own host pins. Models with pins (Models tab) override it wherever they run. <b>data_collection: deny</b> is always on.</p>
          <label className="fm-field"><span>Prefer hosts (order)</span>
            <input defaultValue={listInput((a.routing || {}).order)} key={"o" + p + draft.revision}
              onBlur={(e) => set((x) => { x.routing = { ...(x.routing || {}), order: parseList(e.target.value) }; })} /></label>
          <label className="fm-field"><span>Never use (ignore)</span>
            <input defaultValue={listInput((a.routing || {}).ignore)} key={"i" + p + draft.revision}
              onBlur={(e) => set((x) => { x.routing = { ...(x.routing || {}), ignore: parseList(e.target.value) }; })} /></label>
          <label className="fm-check"><input type="checkbox" checked={!!(a.routing || {}).require_parameters}
            onChange={(e) => set((x) => { x.routing = { ...(x.routing || {}), require_parameters: e.target.checked }; })} /> Only hosts that support every request parameter (require_parameters)</label>
        </Section>
      </div>
    </div>
  );
}

// ── Models view ──────────────────────────────────────────────────────────────────────────
function usedBy(doc, alias) {
  const out = [];
  Object.entries(doc.agents || {}).forEach(([p, a]) => {
    ["main", "subagents", "cron"].forEach((s) => { const c = chainOf(a[s]); const i = c.indexOf(alias); if (i >= 0) out.push({ p, slot: s, pos: i }); });
    Object.entries(a.aux || {}).forEach(([t, s]) => { const i = chainOf(s).indexOf(alias); if (i >= 0) out.push({ p, slot: t, pos: i }); });
  });
  return out;
}

function HostTable({ model, market, onChange, onProbe, probes, minUptime }) {
  const hosts = model.hosts || {};
  const pinned = hosts.order || hosts.only || [];
  const restricted = !!(hosts.only && hosts.only.length);
  const eps = (market && market.endpoints) || [];
  const byTag = {};
  eps.forEach((e) => { byTag[e.tag] = e; });
  const find = (t) => byTag[t] || eps.find((e) => hostSlug(e.tag) === t);
  const rows = pinned.map((t) => ({ tag: t, ep: find(t), pinned: true }))
    .concat(eps.filter((e) => !pinned.some((t) => t === e.tag || t === hostSlug(e.tag) && !byTag[t])).map((e) => ({ tag: e.tag, ep: e, pinned: false })));
  const write = (list, only) => onChange({ ...hosts, order: list.length ? list : undefined, only: only && list.length ? list : undefined });
  const move = (i, d) => { const l = pinned.slice(); const [x] = l.splice(i, 1); l.splice(i + d, 0, x); write(l, restricted); };
  const why = (ep) => {
    if (!ep) return "not listed right now — a pin that matches nothing fails SILENTLY";
    if (ep.tools === false) return "no tool calling";
    if (minUptime && ep.uptime_1d != null && ep.uptime_1d < minUptime) return `uptime ${pct(ep.uptime_1d)} < ${minUptime}%`;
    if (ep.status != null && ep.status < 0) return "degraded right now";
    return null;
  };
  return (
    <div className="fm-hosts">
      <div className="fm-row fm-hosts-bar">
        <label className="fm-check"><input type="checkbox" checked={restricted} onChange={(e) => write(pinned, e.target.checked)} /> Only these hosts — unticked, the order is a preference and other hosts may serve</label>
        <span className="fm-muted fm-small">{market ? (market.source === "live" ? "live from OpenRouter · " + ago(market.fetched_at) : market.source === "snapshot" ? "nightly snapshot · " + ago(market.fetched_at) : "market data unavailable") : "loading…"}</span>
      </div>
      <div className="fm-table-wrap">
        <table className="fm-table">
          <thead><tr>
            <th>#</th><th>Host</th><th>Quant</th><th className="r">In $/M</th><th className="r">Out $/M</th><th className="r">Cache $/M</th>
            <th className="r">Uptime 1d</th><th className="r">30m</th><th>Tools</th><th className="r">p50 ms</th><th className="r">tok/s</th><th></th>
          </tr></thead>
          <tbody>
            {rows.map((r, i) => {
              const ep = r.ep || {};
              const warn = why(r.ep);
              const pr = probes[r.tag];
              return (
                <tr key={r.tag} className={cls(r.pinned ? "is-pinned" : "is-other", warn && r.pinned && "is-warn")}>
                  <td className="fm-order">
                    {r.pinned ? (
                      <span className="fm-row">
                        <b>{i + 1}</b>
                        <button className="fm-mini" disabled={i === 0} onClick={() => move(i, -1)}>▲</button>
                        <button className="fm-mini" disabled={i === pinned.length - 1} onClick={() => move(i, 1)}>▼</button>
                        <button className="fm-mini" title="unpin" onClick={() => write(pinned.filter((t) => t !== r.tag), restricted)}>×</button>
                      </span>
                    ) : <button className="fm-mini fm-mini--add" onClick={() => write(pinned.concat([r.tag]), restricted)}>pin</button>}
                  </td>
                  <td><div className="fm-host">{ep.provider || hostSlug(r.tag)}</div><code className="fm-code">{r.tag}</code>
                    {warn ? <div className="fm-warn-line">{warn}</div> : null}</td>
                  <td>{ep.quant && ep.quant !== "unknown" ? ep.quant : "—"}</td>
                  <td className="r">{price(ep.in)}</td><td className="r">{price(ep.out)}</td><td className="r">{price(ep.cache_read)}</td>
                  <td className={cls("r", ep.uptime_1d != null && ep.uptime_1d < (minUptime || 95) && "fm-bad")}>{pct(ep.uptime_1d)}</td>
                  <td className="r">{pct(ep.uptime_30m)}</td>
                  <td>{ep.tools == null ? "—" : ep.tools ? "✓" : "✗"}</td>
                  <td className="r">{ep.latency_ms ? Math.round(ep.latency_ms) : "—"}</td>
                  <td className="r">{ep.tps ? Math.round(ep.tps) : "—"}</td>
                  <td>
                    <button className="fm-mini" onClick={() => onProbe(r.tag)} disabled={pr === "…"} title="one tiny call pinned to this host with fallbacks off — proves it routes">probe</button>
                    {pr && pr !== "…" ? <div className={cls("fm-small", pr.rate_limited ? "fm-warn-line" : pr.routable ? "fm-good" : "fm-bad")} title={pr.error || ""}>{pr.rate_limited ? "busy now · pin matches" : pr.routable ? `served by ${pr.served_by} · ${pr.latency_ms}ms` : (pr.status || "") + " " + (pr.error || "not routable").slice(0, 60)}</div> : pr === "…" ? <div className="fm-small fm-muted">probing…</div> : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ModelDetail({ draft, alias, setDraft, usage }) {
  const m = draft.models[alias];
  const [mk, setMk] = useState(null);
  const [probes, setProbes] = useState({});
  useEffect(() => {
    setMk(null); setProbes({});
    if (m && m.provider === "openrouter") fetchJSON(`${API}/market?model=${encodeURIComponent(m.id)}`).then(setMk).catch(() => setMk({ endpoints: [], source: "unavailable" }));
  }, [alias]);
  if (!m) return null;
  const set = (fn) => setDraft((d) => { const n = clone(d); fn(n.models[alias]); return n; });
  const probe = (tag) => {
    setProbes((p) => ({ ...p, [tag]: "…" }));
    fetchJSON(`${API}/probe`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, host: tag }) })
      .then((r) => setProbes((p) => ({ ...p, [tag]: r }))).catch((e) => setProbes((p) => ({ ...p, [tag]: { routable: false, error: String(e.message || e) } })));
  };
  const uses = usedBy(draft, alias);
  const rows = ((usage && usage.rows) || []).filter((r) => r.model === m.id || (m.served_as || []).includes(r.model));
  const hostUse = {};
  rows.forEach((r) => { const k = r.host || "?"; hostUse[k] = hostUse[k] || { calls: 0, billed: 0 }; hostUse[k].calls += r.calls; hostUse[k].billed += r.billed_usd; });
  const totalCalls = rows.reduce((a, r) => a + r.calls, 0);
  const ce = m.cap_equivalent || {};
  return (
    <div className="fm-model">
      <header className="fm-model-head">
        <div>
          <h2>{m.short || alias} <span className={cls("fm-prov", "fm-prov--" + m.provider)}>{m.provider === "modelark" ? "ModelArk · subscription" : "OpenRouter · metered"}</span></h2>
          <code className="fm-code">{m.id}</code>
        </div>
        <div className="fm-row">
          {m.vision ? <Badge>vision</Badge> : <Badge tone="dim">text-only</Badge>}
          {m.tools !== false ? <Badge>tools</Badge> : <Badge tone="warn">no tools</Badge>}
          {m.context ? <Badge>{num(m.context)} ctx</Badge> : null}
          <Badge>{m.vendor}</Badge>
        </div>
      </header>
      {m.notes ? <p className="fm-notes">{m.notes}</p> : null}
      <div className="fm-two">
        <Section title="Settings">
          <label className="fm-field"><span>Display name</span><input value={m.short || ""} onChange={(e) => set((x) => { x.short = e.target.value; })} /></label>
          <label className="fm-field"><span>Reasoning pin</span>
            <select value={m.reasoning || ""} onChange={(e) => set((x) => { if (e.target.value) x.reasoning = e.target.value; else delete x.reasoning; })}>
              {REASONING.map((r) => <option key={r} value={r}>{r || "none — agent default applies"}</option>)}
            </select>
            <small>Wins over every agent's default, on every surface this model runs (main, fallback, subagents, helpers).</small>
          </label>
          <div className="fm-row">
            <label className="fm-check"><input type="checkbox" checked={!!m.vision} onChange={(e) => set((x) => { x.vision = e.target.checked; })} /> accepts images</label>
            <label className="fm-check"><input type="checkbox" checked={m.tools !== false} onChange={(e) => set((x) => { x.tools = e.target.checked; })} /> tool calling</label>
          </div>
          <label className="fm-field"><span>Notes</span><textarea rows={3} value={m.notes || ""} onChange={(e) => set((x) => { x.notes = e.target.value; })} /></label>
        </Section>
        <Section title={m.provider === "modelark" ? "Pricing — cap-equivalent" : "Usage · " + ((usage && usage.days) || 7) + "d"}>
          {m.provider === "modelark" ? (
            <Fragment>
              <p className="fm-muted fm-small">The Coding Plan reports no cost, so calls record <b>$0 "modelark subscription"</b>. The $1 card cap still counts these rates per million tokens, so a runaway worker trips it. Changes reach the cap on the next call — no deploy.</p>
              <div className="fm-row">
                {["input", "output", "cache_read"].map((k) => (
                  <label key={k} className="fm-field fm-field--num"><span>{k.replace("_", " ")} $/M</span>
                    <input type="number" step="0.001" min="0" value={ce[k] == null ? "" : ce[k]}
                      onChange={(e) => set((x) => { x.cap_equivalent = { ...(x.cap_equivalent || {}), [k]: e.target.value === "" ? null : Number(e.target.value) }; })} /></label>
                ))}
              </div>
            </Fragment>
          ) : null}
          <div className="fm-hostuse">
            {Object.entries(hostUse).sort((a, b) => b[1].calls - a[1].calls).map(([host, u]) => (
              <div key={host} className="fm-bar-row">
                <span className="fm-bar-label">{host}</span>
                <span className="fm-bar"><span style={{ width: (totalCalls ? (100 * u.calls) / totalCalls : 0) + "%" }} /></span>
                <span className="fm-bar-val">{num(u.calls)} · {money(u.billed, 3)}</span>
              </div>
            ))}
            {!totalCalls ? <div className="fm-muted fm-small">No calls in this window.</div> : null}
          </div>
        </Section>
      </div>
      {m.provider === "openrouter" ? (
        <Section title="Hosts" right={<span className="fm-muted fm-small">{(m.rules || {}).first_host ? `rule: ${m.rules.first_host} first` : ""}{(m.rules || {}).min_uptime ? ` · later hosts ≥ ${m.rules.min_uptime}% uptime` : ""}</span>}>
          <HostTable model={m} market={mk} probes={probes} onProbe={probe} minUptime={(m.rules || {}).min_uptime || (draft.policy || {}).min_host_uptime}
            onChange={(hosts) => set((x) => { const hh = { ...hosts }; Object.keys(hh).forEach((k) => hh[k] === undefined && delete hh[k]); x.hosts = hh; })} />
        </Section>
      ) : null}
      <Section title={`Used by ${uses.length} slot${uses.length === 1 ? "" : "s"}`}>
        <div className="fm-uses">
          {uses.map((u, i) => <span key={i} className="fm-use"><b>{draft.agents[u.p].name || u.p}</b> {SLOT_LABEL[u.slot] || TASK_LABEL[u.slot] || u.slot} <span className="fm-muted">#{u.pos + 1}</span></span>)}
          {!uses.length ? <span className="fm-muted">Not in any waterfall. <button className="fm-link fm-link--danger" onClick={() => setDraft((d) => { const n = clone(d); delete n.models[alias]; return n; })}>Remove from registry</button></span> : null}
        </div>
      </Section>
    </div>
  );
}

function AddModel({ draft, setDraft, onAdded }) {
  const [id, setId] = useState("");
  const [info, setInfo] = useState(null);
  const [busy, setBusy] = useState(false);
  const look = () => {
    if (!id.trim()) return;
    setBusy(true);
    fetchJSON(`${API}/market?model=${encodeURIComponent(id.trim())}&fresh=true`).then((r) => { setInfo(r); setBusy(false); }).catch((e) => { setInfo({ error: String(e.message || e) }); setBusy(false); });
  };
  const add = () => {
    const mid = id.trim();
    const alias = mid.split("/").pop().toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const eps = (info && info.endpoints) || [];
    setDraft((d) => {
      const n = clone(d);
      n.models[alias in n.models ? alias + "-2" : alias] = {
        id: mid, short: mid.split("/").pop(), provider: "openrouter", vendor: mid.split("/")[0], billing: "metered",
        tools: eps.some((e) => e.tools), vision: /image/.test((info && info.modality) || ""),
        context: Math.max(0, ...eps.map((e) => e.ctx || 0)) || undefined, notes: "",
      };
      return n;
    });
    onAdded(alias); setId(""); setInfo(null);
  };
  return (
    <div className="fm-addmodel">
      <input placeholder="OpenRouter model id, e.g. qwen/qwen3.8-27b" value={id} onChange={(e) => { setId(e.target.value); setInfo(null); }} onKeyDown={(e) => e.key === "Enter" && look()} />
      <button className="fm-btn" onClick={look} disabled={busy || !id.trim()}>{busy ? "checking…" : "Look up"}</button>
      {info ? (info.endpoints && info.endpoints.length ? (
        <span className="fm-row"><span className="fm-good fm-small">{info.endpoints.length} hosts · {info.modality}</span><button className="fm-btn fm-btn--primary" onClick={add}>Add to registry</button></span>
      ) : <span className="fm-bad fm-small">{info.error || "no endpoints for that id"}</span>) : null}
    </div>
  );
}

function ModelsView({ draft, setDraft, usage, sel, setSel }) {
  const aliases = Object.keys(draft.models || {});
  const cur = sel && draft.models[sel] ? sel : aliases[0];
  return (
    <div className="fm-models">
      <aside className="fm-model-list">
        {aliases.map((a) => {
          const m = draft.models[a];
          return (
            <button key={a} className={cls("fm-model-item", a === cur && "is-active")} onClick={() => setSel(a)}>
              <span className={cls("fm-pill-dot", "fm-dot--" + m.provider)} />
              <span className="fm-model-item-name">{m.short || a}</span>
              <span className="fm-muted fm-small">{usedBy(draft, a).length} slots</span>
            </button>
          );
        })}
        <div className="fm-model-list-foot"><AddModel draft={draft} setDraft={setDraft} onAdded={setSel} /></div>
      </aside>
      <div className="fm-model-main">{cur ? <ModelDetail key={cur} draft={draft} alias={cur} setDraft={setDraft} usage={usage} /> : null}</div>
    </div>
  );
}

// ── Costs view ───────────────────────────────────────────────────────────────────────────
function CostsView({ doc, usage, days, setDays }) {
  if (!usage) return <div className="fm-muted">Loading usage…</div>;
  const rows = usage.rows || [];
  const byAgent = {};
  rows.forEach((r) => { (byAgent[r.profile] = byAgent[r.profile] || []).push(r); });
  const tot = rows.reduce((a, r) => ({ billed: a.billed + r.billed_usd, calls: a.calls + r.calls, ma: a.ma + (r.modelark ? r.calls : 0), cap: a.cap + r.cap_equivalent_usd, tin: a.tin + r.input, tout: a.tout + r.output }), { billed: 0, calls: 0, ma: 0, cap: 0, tin: 0, tout: 0 });
  const maxDay = Math.max(0.000001, ...usage.daily.map((d) => d.billed_usd));
  const maxCalls = Math.max(1, ...usage.daily.map((d) => d.calls));
  const idToShort = {};
  Object.values(doc.models || {}).forEach((m) => { idToShort[m.id] = m.short; (m.served_as || []).forEach((s) => { idToShort[s] = m.short; }); });
  return (
    <Fragment>
      <div className="fm-row fm-costs-bar">
        {[1, 7, 30].map((d) => <button key={d} className={cls("fm-chip", d === days && "is-active")} onClick={() => setDays(d)}>{d === 1 ? "24 h" : d + " days"}</button>)}
        <span className="fm-muted fm-small">from all nine state.db ledgers · {ago(usage.generated_at)}</span>
      </div>
      <div className="fm-stats">
        <Stat label="Billed (OpenRouter)" value={money(tot.billed, 2)} tone="money" sub="what actually gets invoiced" />
        <Stat label="ModelArk subscription" value={num(tot.ma) + " calls"} tone="sub" sub={"$0 · cap-equivalent " + money(tot.cap, 2)} />
        <Stat label="All calls" value={num(tot.calls)} sub={num(tot.tin) + " in · " + num(tot.tout) + " out tokens"} />
        <Stat label="Subscription share" value={tot.calls ? Math.round((100 * tot.ma) / tot.calls) + "%" : "—"} sub="of calls served on the flat plan" />
      </div>
      <Section title="Per day">
        <div className="fm-days">
          {usage.daily.map((d) => (
            <div key={d.day} className="fm-day" title={`${new Date(d.day * 86400000).toDateString()}\nbilled ${money(d.billed_usd)}\n${d.calls} calls (${d.modelark_calls} on subscription)`}>
              <div className="fm-day-bars">
                <span className="fm-day-money" style={{ height: (100 * d.billed_usd) / maxDay + "%" }} />
                <span className="fm-day-calls" style={{ height: (100 * d.calls) / maxCalls + "%" }}><span style={{ height: (d.calls ? (100 * d.modelark_calls) / d.calls : 0) + "%" }} /></span>
              </div>
              <div className="fm-day-label">{new Date(d.day * 86400000).toLocaleDateString(undefined, { day: "numeric", month: "short" })}</div>
            </div>
          ))}
        </div>
        <div className="fm-legend"><span className="fm-lg fm-lg--money" /> billed $ <span className="fm-lg fm-lg--calls" /> calls <span className="fm-lg fm-lg--sub" /> of which subscription</div>
      </Section>
      <div className="fm-two">
        <Section title={`Spend by agent · ${usage.days === 1 ? "24 h" : usage.days + " days"}`} right={<span className="fm-muted fm-small">bar = billed $ · teal = subscription calls</span>}>
          <BarList rows={Object.entries(byAgent).map(([p, rs]) => ({ label: (doc.agents[p] || {}).name || p,
            billed: rs.reduce((a, r) => a + r.billed_usd, 0), calls: rs.reduce((a, r) => a + r.calls, 0), ma: rs.reduce((a, r) => a + (r.modelark ? r.calls : 0), 0) }))} />
        </Section>
        <Section title="Spend by model" right={<span className="fm-muted fm-small">old OpenRouter DeepSeek ids are history from before 09-11</span>}>
          <BarList rows={Object.values(rows.reduce((acc, r) => { const k = idToShort[r.model] || r.model; const a = (acc[k] = acc[k] || { label: k, billed: 0, calls: 0, ma: 0 }); a.billed += r.billed_usd; a.calls += r.calls; if (r.modelark) a.ma += r.calls; return acc; }, {}))} />
        </Section>
      </div>
      <Section title="By agent, model and host">
        <div className="fm-table-wrap">
          <table className="fm-table">
            <thead><tr><th>Agent</th><th>Model</th><th>Served by</th><th>Where</th><th className="r">Calls</th><th className="r">In</th><th className="r">Out</th><th className="r">Cache read</th><th className="r">Billed</th><th className="r">Cap-equiv.</th></tr></thead>
            <tbody>
              {Object.entries(byAgent).map(([p, rs]) => rs.map((r, i) => (
                <tr key={p + i}>
                  <td>{i === 0 ? <b>{(doc.agents[p] || {}).name || p}</b> : null}</td>
                  <td>{idToShort[r.model] || <code className="fm-code">{r.model}</code>}</td>
                  <td>{r.modelark ? <span className="fm-prov fm-prov--modelark">modelark subscription</span> : r.host || "—"}</td>
                  <td className="fm-muted">{r.task === "main" ? "main" : TASK_LABEL[r.task] || r.task}</td>
                  <td className="r">{num(r.calls)}</td><td className="r">{num(r.input)}</td><td className="r">{num(r.output)}</td><td className="r">{num(r.cache_read)}</td>
                  <td className="r">{r.modelark ? "$0" : money(r.billed_usd)}</td>
                  <td className="r fm-muted">{r.modelark ? money(r.cap_equivalent_usd) : ""}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </Section>
    </Fragment>
  );
}

function BarList({ rows }) {
  const sorted = rows.slice().sort((a, b) => b.billed - a.billed || b.calls - a.calls);
  const max = Math.max(0.000001, ...sorted.map((r) => r.billed));
  return (
    <div className="fm-barlist">
      {sorted.map((r) => (
        <div key={r.label} className="fm-bl-row" title={`${r.label}: ${money(r.billed)} billed · ${r.calls} calls (${r.ma} on subscription)`}>
          <span className="fm-bl-label">{r.label}</span>
          <span className="fm-bl-track">
            <span className="fm-bl-money" style={{ width: (100 * r.billed) / max + "%" }} />
          </span>
          <span className="fm-bl-val"><b>{money(r.billed, 2)}</b> <span className="fm-muted">· {num(r.calls)} calls{r.ma ? <Fragment> · <span className="fm-sub-txt">{num(r.ma)} sub</span></Fragment> : null}</span></span>
        </div>
      ))}
      {!sorted.length ? <div className="fm-muted fm-small">No calls in this window.</div> : null}
    </div>
  );
}

// ── Decisions & history ──────────────────────────────────────────────────────────────────
function DecisionsView({ state, draft, setDraft, onRevert }) {
  const pol = draft.policy || {};
  const [what, setWhat] = useState(""); const [why, setWhy] = useState("");
  return (
    <div className="fm-two fm-two--wide">
      <div>
        <Section title="Standing rules">
          <div className="fm-rule"><Badge tone="ok">locked</Badge><div><b>data_collection: deny on every OpenRouter call</b><div className="fm-muted fm-small">The no-training rule. Enforced by the compiler on every config and by the plugin on every helper call — not editable here.</div></div></div>
          <div className="fm-rule"><Badge tone="ok">read-only</Badge><div><b>Cost caps</b> — {Object.entries(state.caps || {}).map(([k, v]) => `${k.replace(/_/g, " ")} $${v}`).join(" · ") || "see config"}<div className="fm-muted fm-small">Richie's alone. Shown, never edited, from this tab.</div></div></div>
          <div className="fm-rule"><Badge>policy</Badge><div><b>Hosts after the rule-pinned first host need ≥ </b>
            <input className="fm-inline-num" type="number" min="50" max="100" value={pol.min_host_uptime || 95}
              onChange={(e) => setDraft((d) => { const n = clone(d); n.policy = { ...(n.policy || {}), min_host_uptime: Number(e.target.value) }; return n; })} /><b>% uptime</b>
            <div className="fm-muted fm-small">Binding where a model carries the rule (v4.1); advice elsewhere.</div></div></div>
        </Section>
        <Section title="Decisions">
          <ul className="fm-decisions">
            {(draft.decisions || []).map((d, i) => (
              <li key={i}><span className="fm-date">{String(d.date)}</span><div><b>{d.what}</b><div className="fm-muted">{d.why}</div></div>
                <button className="fm-mini" title="remove" onClick={() => setDraft((x) => { const n = clone(x); n.decisions.splice(i, 1); return n; })}>×</button></li>
            ))}
          </ul>
          <div className="fm-row fm-adddec">
            <input placeholder="Decision" value={what} onChange={(e) => setWhat(e.target.value)} />
            <input placeholder="Why" value={why} onChange={(e) => setWhy(e.target.value)} />
            <button className="fm-btn" disabled={!what.trim()} onClick={() => { setDraft((x) => { const n = clone(x); n.decisions = [{ date: new Date().toISOString().slice(0, 10), what: what.trim(), why: why.trim() }].concat(n.decisions || []); return n; }); setWhat(""); setWhy(""); }}>Add</button>
          </div>
        </Section>
      </div>
      <Section title="History" right={<span className="fm-muted fm-small">every apply keeps a pre-image of models.yaml, the 9 configs and the SOULs</span>}>
        <ul className="fm-history">
          {(state.history || []).map((hh) => (
            <li key={hh.id}>
              <div className="fm-row fm-between">
                <span><b>{hh.summary}</b> <span className="fm-muted">· {hh.by} · {ago(hh.ts)}</span></span>
                <button className="fm-mini" onClick={() => onRevert(hh)}>revert</button>
              </div>
              <div className="fm-muted fm-small">{hh.revision ? "revision " + hh.revision + " · " : ""}{(hh.changed || []).length} config(s){hh.reverts ? " · reverts " + hh.reverts : ""} · <code>{hh.id}</code></div>
              {hh.changes ? (
                <details><summary className="fm-small">what changed</summary>
                  {Object.entries(hh.changes).map(([p, ch]) => <div key={p} className="fm-changes"><b>{p}</b>{ch.map((c, i) => <div key={i}><code>{c}</code></div>)}</div>)}
                </details>
              ) : null}
            </li>
          ))}
          {!(state.history || []).length ? <li className="fm-muted">No applies yet.</li> : null}
        </ul>
      </Section>
    </div>
  );
}

// ── plan / apply panel ───────────────────────────────────────────────────────────────────
function PlanPanel({ plan, onClose, onApply, applying, needsUnlock, unlock, setUnlock, summary, setSummary }) {
  const changed = plan.changed || [];
  const [open, setOpen] = useState(null);
  return (
    <div className="fm-overlay" onClick={onClose}>
      <div className="fm-panel" onClick={(e) => e.stopPropagation()}>
        <header className="fm-panel-head"><h3>Preview</h3><button className="fm-mini" onClick={onClose}>close</button></header>
        {plan.errors && plan.errors.length ? (
          <div className="fm-note fm-note--bad"><b>Can't apply:</b>{plan.errors.map((e, i) => <div key={i}>{e}</div>)}</div>
        ) : (
          <div className="fm-note fm-note--ok">{changed.length ? `${changed.length} config file(s) will change: ${changed.join(", ")}.` : "Nothing in the configs changes (registry notes/decisions only)."} Running workers, gateways and cron pick it up on their next call — no restart.</div>
        )}
        {needsUnlock && !(plan.errors && plan.errors.length) ? <div className="fm-note fm-note--warn"><b>Smith is locked.</b> This change touches the overwatch agent — tick <b>Unlock Smith for this apply</b> below to allow it.</div> : null}
        {plan.warnings && plan.warnings.length ? <details className="fm-note fm-note--warn"><summary>{plan.warnings.length} warning(s)</summary>{plan.warnings.map((w, i) => <div key={i}>{w}</div>)}</details> : null}
        <div className="fm-plan-list">
          {Object.entries(plan.plan || {}).filter(([, v]) => v.changes.length).map(([p, v]) => (
            <div key={p} className="fm-plan-item">
              <button className="fm-plan-toggle" onClick={() => setOpen(open === p ? null : p)}><b>{p}</b> · {v.changes.length} change(s) {open === p ? "▾" : "▸"}</button>
              {open === p ? (
                <Fragment>
                  <div className="fm-changes">{v.changes.map((c, i) => <div key={i}><code>{c}</code></div>)}</div>
                  <pre className="fm-diff">{v.diff.split("\n").map((l, i) => <span key={i} className={l.startsWith("+") && !l.startsWith("+++") ? "fm-add-l" : l.startsWith("-") && !l.startsWith("---") ? "fm-del-l" : ""}>{l + "\n"}</span>)}</pre>
                </Fragment>
              ) : null}
            </div>
          ))}
        </div>
        {!(plan.errors && plan.errors.length) ? (
          <footer className="fm-panel-foot">
            <input className="fm-summary" placeholder="What is this change? (goes in the history)" value={summary} onChange={(e) => setSummary(e.target.value)} />
            {needsUnlock ? <label className="fm-check fm-unlock"><input type="checkbox" checked={unlock} onChange={(e) => setUnlock(e.target.checked)} /> Unlock Smith for this apply</label> : null}
            <button className="fm-btn fm-btn--primary" disabled={applying || (needsUnlock && !unlock)} onClick={onApply}>{applying ? "Applying…" : "Apply to the fleet"}</button>
          </footer>
        ) : null}
      </div>
    </div>
  );
}

// ── the page ─────────────────────────────────────────────────────────────────────────────
function ModelsPage() {
  const [state, setState] = useState(null);
  const [err, setErr] = useState(null);
  const [draft, setDraft] = useState(null);
  const [usage, setUsage] = useState(null);
  const [days, setDays] = useState(7);
  const [tab, setTab] = useState("fleet");
  const [agent, setAgent] = useState("root");
  const [modelSel, setModelSel] = useState(null);
  const [plan, setPlan] = useState(null);
  const [busy, setBusy] = useState(false);
  const [unlock, setUnlock] = useState(false);
  const [summary, setSummary] = useState("");
  const [flash, setFlash] = useState(null);
  const baseRef = useRef(null);

  const load = useCallback(() => fetchJSON(`${API}/state`).then((s) => {
    setState(s); setErr(null);
    // keep an in-progress edit; otherwise adopt the live document
    setDraft((d) => {
      if (!d || !baseRef.current || same(d, baseRef.current)) { baseRef.current = clone(s.doc); return clone(s.doc); }
      return d;
    });
    return s;
  }).catch((e) => setErr(String(e.message || e))), []);
  const loadUsage = useCallback(() => fetchJSON(`${API}/usage?days=${days}`).then(setUsage).catch(() => {}), [days]);

  useEffect(() => { load(); }, []);
  useEffect(() => { loadUsage(); const t = setInterval(loadUsage, 60000); return () => clearInterval(t); }, [days]);
  useEffect(() => { const t = setInterval(() => { if (!document.hidden) load(); }, 15000); return () => clearInterval(t); }, [load]);

  const base = baseRef.current;
  const dirty = !!(state && draft && base && !same(draft, base));
  const movedUnder = dirty && state.revision !== Number(base.revision || 0);
  const needsUnlock = !!(dirty && base.agents && base.agents.root && base.agents.root.locked && !same(base.agents.root, draft.agents.root));

  const say = (msg, tone) => { setFlash({ msg, tone }); setTimeout(() => setFlash(null), 6000); };
  // Preview always dry-runs a Smith change WITH the unlock so you can see the diff; Apply sends the unlock only
  // when the "Unlock Smith for this apply" box is ticked.
  const body = (isPreview) => JSON.stringify({ doc: draft, base_revision: Number((baseRef.current || {}).revision || 0),
    unlock: (isPreview ? needsUnlock : unlock) ? ["root"] : [], summary });
  const preview = () => { setBusy(true); fetchJSON(`${API}/plan`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(true) })
    .then((r) => { setPlan(r); setBusy(false); }).catch((e) => { setBusy(false); say(String(e.message || e), "bad"); }); };
  const apply = () => { setBusy(true); fetchJSON(`${API}/apply`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(false) })
    .then((r) => {
      setBusy(false);
      if (r.ok) { setPlan(null); setSummary(""); setUnlock(false); baseRef.current = null; setDraft(null);
        load().then(() => {}); loadUsage();
        say(`Applied — ${r.changed.length} config(s) rewritten and verified${r.souls && r.souls.length ? ", SOULs refreshed" : ""}. Revert from History if needed.`, "ok");
      } else { setPlan({ ...(plan || {}), errors: r.errors, warnings: r.warnings }); }
    }).catch((e) => { setBusy(false); say(String(e.message || e), "bad"); }); };
  const revert = (hh) => {
    if (!window.confirm(`Put models.yaml, the configs and SOULs back as they were before “${hh.summary}”?`)) return;
    fetchJSON(`${API}/revert`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: hh.id }) })
      .then((r) => { if (r.ok) { baseRef.current = null; setDraft(null); load(); say("Reverted — recorded as " + r.id, "ok"); } else say((r.errors || []).join("; "), "bad"); });
  };

  if (err && !state) return <div className="fm-root"><div className="fm-note fm-note--bad">Models tab can't load: {err}</div></div>;
  if (!state || !draft) return <div className="fm-root"><div className="fm-muted fm-loading">Loading the fleet's model settings…</div></div>;
  const doc = draft;
  const TABS = [["fleet", "Fleet"], ["agent", "Agents"], ["models", "Models & hosts"], ["costs", "Costs"], ["decisions", "Rules & history"]];
  return (
    <div className="fm-root">
      <header className="fm-head">
        <div>
          <h1>Fleet Models</h1>
          <div className="fm-sub">Every agent's waterfall from one file — <code>~/.hermes/fleet/models.yaml</code> · revision {state.revision} · {state.doc.updated_by || "—"} {ago(state.doc.updated_at)}</div>
        </div>
        <div className="fm-row">
          <Badge tone="ok" title="data_collection: deny on every OpenRouter call">no-training · deny</Badge>
          <Badge tone={state.runtime.aux_routing_seam ? "ok" : "warn"} title="helper calls follow each model's pins (fleet-models plugin)">{state.runtime.aux_routing_seam ? "helper routing on" : "helper routing off"}</Badge>
          {state.validation.errors.length ? <Badge tone="bad" title={state.validation.errors.join("\n")}>{state.validation.errors.length} rule breach</Badge> : null}
          <button className="fm-btn" onClick={() => { load(); loadUsage(); }}>Refresh</button>
        </div>
      </header>
      <nav className="fm-tabs">
        {TABS.map(([k, l]) => <button key={k} className={cls("fm-tab", tab === k && "is-active")} onClick={() => setTab(k)}>{l}</button>)}
      </nav>
      {movedUnder ? <div className="fm-note fm-note--warn">Someone applied a change while you were editing (now revision {state.revision}). Preview will refuse a stale edit — discard and redo it.</div> : null}
      <main className="fm-main">
        {tab === "fleet" ? <FleetView state={state} doc={doc} usage={usage} onOpen={(p) => { setAgent(p); setTab("agent"); }} /> : null}
        {tab === "agent" ? <AgentView state={state} draft={draft} setDraft={setDraft} p={agent} setP={setAgent} /> : null}
        {tab === "models" ? <ModelsView draft={draft} setDraft={setDraft} usage={usage} sel={modelSel} setSel={setModelSel} /> : null}
        {tab === "costs" ? <CostsView doc={doc} usage={usage} days={days} setDays={setDays} /> : null}
        {tab === "decisions" ? <DecisionsView state={state} draft={draft} setDraft={setDraft} onRevert={revert} /> : null}
      </main>
      {dirty ? (
        <div className="fm-dock">
          <span><b>Unsaved changes</b> <span className="fm-muted">— nothing reaches the fleet until you apply</span></span>
          <span className="fm-row">
            <button className="fm-btn" onClick={() => { setDraft(clone(state.doc)); baseRef.current = clone(state.doc); }}>Discard</button>
            <button className="fm-btn fm-btn--primary" disabled={busy} onClick={preview}>{busy ? "Checking…" : "Preview & apply"}</button>
          </span>
        </div>
      ) : null}
      {plan ? <PlanPanel plan={plan} onClose={() => setPlan(null)} onApply={apply} applying={busy} needsUnlock={needsUnlock}
        unlock={unlock} setUnlock={setUnlock} summary={summary} setSummary={setSummary} /> : null}
      {flash ? <div className={cls("fm-flash", "fm-flash--" + flash.tone)}>{flash.msg}</div> : null}
    </div>
  );
}

if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
  window.__HERMES_PLUGINS__.register("fleet-models", ModelsPage);
}
}
