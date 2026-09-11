/**
 * Hermes fleet — Fleet Models page.
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

// ── time windows ─────────────────────────────────────────────────────────────────────────
// One window drives every usage figure on the page. The chart's bucket comes from the window AND the width it
// has: each window has a natural bucket (24 h → hourly) that coarsens until every bar gets MIN_PITCH px; tick
// labels are then thinned to clean local-clock steps (every 3rd hour on a phone). The server spreads each ledger
// row across its active span, so sub-hour figures are close estimates.
const PERIODS = [["15m", 900, "15 minutes"], ["30m", 1800, "30 minutes"], ["1h", 3600, "hour"], ["2h", 7200, "2 hours"],
  ["3h", 10800, "3 hours"], ["6h", 21600, "6 hours"], ["12h", 43200, "12 hours"], ["24h", 86400, "24 hours"],
  ["7d", 604800, "7 days"], ["30d", 2592000, "30 days"]];
const NICE = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800];
const LABEL_STEPS = NICE.concat([1209600]);
const MIN_PITCH = 12;      // px per bar, at least
const MIN_LABEL_GAP = 36;  // px between tick labels, at least
const TZ = (() => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch (e) { return ""; } })();
const naturalBucket = (w) => {
  const m = [[900, 60], [1800, 60], [3600, 120], [7200, 300], [10800, 300], [21600, 900], [43200, 1800], [86400, 3600], [604800, 21600]];
  const hit = m.find(([x]) => w <= x); return hit ? hit[1] : 86400;
};
const pickBucket = (w, width) => {
  const maxBars = Math.max(6, Math.floor(width / MIN_PITCH));
  let b = naturalBucket(w);
  while (w / b > maxBars) { const nx = NICE.find((x) => x > b); if (!nx) break; b = nx; }
  return b;
};
const periodName = (w) => { const p = PERIODS.find((x) => x[1] === w); return p ? p[2] : Math.round(w / 3600) + " hours"; };
const periodShort = (w) => { const p = PERIODS.find((x) => x[1] === w); return p ? p[0] : Math.round(w / 3600) + "h"; };
const lastLabel = (w) => (w === 3600 ? "Last hour" : "Last " + periodName(w));
const bucketName = (b) => (b < 3600 ? b / 60 + "-minute" : b < 86400 ? (b / 3600 === 1 ? "hourly" : b / 3600 + "-hour") : b === 86400 ? "daily" : b / 86400 + "-day");
const WIN_KEY = "fleet-models.window";
const readWin = () => { try { const v = Number(window.localStorage.getItem(WIN_KEY)); return PERIODS.some((p) => p[1] === v) ? v : 604800; } catch (e) { return 604800; } };
const saveWin = (v) => { try { window.localStorage.setItem(WIN_KEY, String(v)); } catch (e) { /* private mode */ } };
const hhmm = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
const dayMon = (d) => d.toLocaleDateString([], { day: "numeric", month: "short" });
const wday = (d) => d.toLocaleDateString([], { weekday: "short" });
function tickLabel(t, b, step) {
  const d = new Date(t * 1000);
  if (step >= 86400 || b >= 86400) return dayMon(d);
  if (d.getHours() === 0 && d.getMinutes() === 0) return wday(d);
  return hhmm(d);
}
function isTick(t, step) {
  const d = new Date(t * 1000);
  if (step >= 86400) {
    if (d.getHours() !== 0 || d.getMinutes() !== 0) return false;
    const ord = Math.round(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / 86400000);
    return ord % (step / 86400) === 0;
  }
  const sec = d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds();
  return sec % step === 0;
}
function rangeLabel(s, b, since, now) {
  const a = new Date(Math.max(s.t, since) * 1000), z = new Date(Math.min(s.end, now) * 1000);
  if (b >= 86400) return wday(a) + " " + dayMon(a) + (b > 86400 ? " – " + dayMon(new Date((s.end - 1) * 1000)) : "") + (s.end > now ? " (so far)" : "");
  const pre = new Date(now * 1000).toDateString() === a.toDateString() ? "" : wday(a) + " ";
  return pre + hhmm(a) + "–" + (s.end > now ? "now" : hhmm(z));
}

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

// ── time charts ──────────────────────────────────────────────────────────────────────────
function PeriodBar({ win, setWin, usage, loading }) {
  const ref = useRef(null);
  // keep the active chip in view on a phone — scroll the chip row only, never the page
  useEffect(() => {
    const box = ref.current, el = box && box.querySelector(".is-active");
    if (!el) return;
    const l = el.offsetLeft - box.offsetLeft, r = l + el.offsetWidth;
    if (l < box.scrollLeft) box.scrollLeft = l - 8; else if (r > box.scrollLeft + box.clientWidth) box.scrollLeft = r - box.clientWidth + 8;
  }, [win]);
  return (
    <div className="fm-period">
      <div className="fm-period-chips" role="group" aria-label="Time window" ref={ref}>
        {PERIODS.map(([k, w, name]) => (
          <button key={k} className={cls("fm-chip fm-chip--sm", w === win && "is-active")} aria-pressed={w === win}
            title={lastLabel(w)} onClick={() => setWin(w)}>{k}</button>
        ))}
      </div>
      <span className="fm-muted fm-small fm-period-note">
        {loading ? "updating…" : usage ? `${bucketName(usage.bucket)} bars · ${ago(usage.generated_at)}` : ""}
      </span>
    </div>
  );
}

function Spark({ values, label, height }) {
  const v = values || [];
  const max = Math.max(0, ...v);
  if (!v.length) return null;
  return (
    <span className="fm-spark" style={{ height: (height || 22) + "px" }} aria-label={label} title={label}>
      {v.map((x, i) => <span key={i} style={{ height: max ? Math.max(x > 0 ? 8 : 0, (100 * x) / max) + "%" : "0%" }} />)}
    </span>
  );
}

function TimeChart({ usage, width }) {
  const s = (usage && usage.series) || [];
  const [sel, setSel] = useState(null);
  useEffect(() => { setSel(null); }, [usage && usage.window, usage && usage.bucket]);
  if (!s.length) return null;
  const b = usage.bucket, n = s.length;
  const pitch = Math.max(1, (width || 600) / n);
  const need = Math.ceil(MIN_LABEL_GAP / pitch) * b;
  const step = LABEL_STEPS.find((x) => x >= need && x % b === 0) || need;
  const maxB = Math.max(0, ...s.map((x) => x.billed_usd));
  const maxC = Math.max(0, ...s.map((x) => x.calls));
  const tot = s.reduce((a, x) => ({ b: a.b + x.billed_usd, c: a.c + x.calls, m: a.m + x.modelark_calls }), { b: 0, c: 0, m: 0 });
  const cur = sel != null && s[sel] ? s[sel] : null;
  const readout = cur
    ? { when: rangeLabel(cur, b, usage.since, usage.now), b: cur.billed_usd, c: cur.calls, m: cur.modelark_calls }
    : { when: lastLabel(usage.window), b: tot.b, c: tot.c, m: tot.m };
  const gap = pitch < 7 ? 1 : pitch < 14 ? 2 : 3;
  return (
    <div className="fm-tc" onPointerLeave={(e) => { if (e.pointerType === "mouse") setSel(null); }}>
      <div className="fm-tc-readout" aria-live="polite">
        <b>{readout.when}</b>
        <span><span className="fm-lg fm-lg--money" /> {money(readout.b, 3)} billed</span>
        <span><span className="fm-lg fm-lg--calls" /> {num(Math.round(readout.c))} calls</span>
        <span><span className="fm-lg fm-lg--sub" /> {num(Math.round(readout.m))} on subscription</span>
      </div>
      <div className="fm-tc-plot" style={{ gap: gap + "px" }}>
        <span className="fm-tc-max fm-tc-max--money">{maxB ? money(maxB, 3) : "$0"}</span>
        <span className="fm-tc-max fm-tc-max--calls">{maxC ? num(Math.round(maxC)) + " calls" : "0 calls"}</span>
        {s.map((x, i) => {
          const tick = isTick(x.t, step) && !(i === 0 && x.t < usage.since && n > 3);
          return (
            <button key={x.t} type="button" className={cls("fm-tc-col", x.partial && "is-partial", sel === i && "is-sel")}
              onPointerEnter={(e) => { if (e.pointerType === "mouse") setSel(i); }} onFocus={() => setSel(i)} onClick={() => setSel(i)}
              aria-label={`${rangeLabel(x, b, usage.since, usage.now)}: ${money(x.billed_usd, 3)} billed, ${Math.round(x.calls)} calls`}>
              <span className="fm-tc-m"><span style={{ height: maxB ? (100 * x.billed_usd) / maxB + "%" : "0%" }} /></span>
              <span className="fm-tc-c"><span style={{ height: maxC ? (100 * x.calls) / maxC + "%" : "0%" }}>
                <span style={{ height: x.calls ? (100 * x.modelark_calls) / x.calls + "%" : "0%" }} /></span></span>
              <span className="fm-tc-lbl">{tick ? tickLabel(x.t, b, step) : ""}</span>
            </button>
          );
        })}
      </div>
      {!tot.c ? <div className="fm-muted fm-small fm-tc-empty">No calls in this window.</div> : null}
    </div>
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

function AgentCard({ doc, p, drift, use, spark, win, onOpen }) {
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
        {spark ? <Spark values={spark} label={`${a.name || p}: calls over the last ${periodName(win)}`} /> : null}
        <span className="fm-card-win">{periodShort(win)}</span>
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

function FleetView({ state, doc, usage, win, onOpen }) {
  const byP = usageByProfile(usage);
  const empty = { calls: 0, billed: 0, ma: 0, hosts: [] };
  const tot = Object.values(byP).reduce((a, u) => ({ calls: a.calls + u.calls, billed: a.billed + u.billed, ma: a.ma + u.ma }), { calls: 0, billed: 0, ma: 0 });
  const models = doc.models || {};
  return (
    <Fragment>
      <div className="fm-stats">
        <Stat label={`Calls · ${periodShort(win)}`} value={num(tot.calls)} sub={lastLabel(win).toLowerCase()} />
        <Stat label={`Billed · ${periodShort(win)}`} value={money(tot.billed, 2)} sub="OpenRouter, real invoices" tone="money" />
        <Stat label="ModelArk subscription" value={num(tot.ma) + " calls"} sub={tot.calls ? Math.round((100 * tot.ma) / tot.calls) + "% of all calls · $0" : "$0"} tone="sub" />
        <Stat label="Registry" value={Object.keys(models).length + " models"} sub={Object.values(models).filter((m) => m.provider === "modelark").length + " subscription · " + Object.values(models).filter((m) => m.provider === "openrouter").length + " OpenRouter"} />
        <Stat label="Sync" value={Object.keys(state.drift || {}).length ? Object.keys(state.drift).length + " drifted" : "all 9 in sync"} tone={Object.keys(state.drift || {}).length ? "warn" : "ok"} sub={"revision " + state.revision} />
      </div>
      <div className="fm-grid">
        {state.profiles.map((p) => (
          <AgentCard key={p} doc={doc} p={p} drift={(state.drift || {})[p]} use={byP[p] || empty} win={win}
            spark={usage && usage.by_profile ? ((usage.by_profile[p] || {}).calls || (usage.series || []).map(() => 0)) : null} onOpen={() => onOpen(p)} />
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
            <small>Per-model pins (Models & hosts) win — e.g. v4.1 always runs high.</small>
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
          <p className="fm-muted fm-small">Applies to this agent's OpenRouter calls for models without their own host pins. Models with pins (Models & hosts) override it wherever they run. <b>data_collection: deny</b> is always on.</p>
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

function ModelDetail({ draft, alias, setDraft, usage, win }) {
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
  const ids = [m.id].concat(m.served_as || []);
  const bm = (usage && usage.by_model) || {};
  const spark = usage && usage.series ? usage.series.map((_, i) => ids.reduce((a, id) => a + (((bm[id] || {}).calls || [])[i] || 0), 0)) : null;
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
        <Section title={m.provider === "modelark" ? "Pricing — cap-equivalent" : "Usage · " + periodShort(win)}
          right={spark && totalCalls ? <span className="fm-muted fm-small">{num(totalCalls)} calls · {lastLabel(win).toLowerCase()}</span> : null}>
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
          {m.provider === "modelark" ? <div className="fm-subhead">Usage · {periodShort(win)}{totalCalls ? " · " + num(totalCalls) + " calls" : ""}</div> : null}
          {spark && totalCalls ? <Spark values={spark} height={30} label={`${m.short || alias}: calls over the last ${periodName(win)}`} /> : null}
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

function ModelsView({ draft, setDraft, usage, win, sel, setSel }) {
  const aliases = Object.keys(draft.models || {});
  const cur = sel && draft.models[sel] ? sel : aliases[0];
  return (
    <div className="fm-models">
      <aside className="fm-model-list">
        <div className="fm-model-items" role="tablist" aria-label="Models">
        {aliases.map((a) => {
          const m = draft.models[a];
          return (
            <button key={a} role="tab" aria-selected={a === cur} className={cls("fm-model-item", a === cur && "is-active")} onClick={() => setSel(a)}>
              <span className={cls("fm-pill-dot", "fm-dot--" + m.provider)} />
              <span className="fm-model-item-name">{m.short || a}</span>
              <span className="fm-muted fm-small">{usedBy(draft, a).length} slots</span>
            </button>
          );
        })}
        </div>
        <details className="fm-model-list-foot"><summary>+ Add a model</summary><AddModel draft={draft} setDraft={setDraft} onAdded={setSel} /></details>
      </aside>
      <div className="fm-model-main">{cur ? <ModelDetail key={cur} draft={draft} alias={cur} setDraft={setDraft} usage={usage} win={win} /> : null}</div>
    </div>
  );
}

// ── Costs view ───────────────────────────────────────────────────────────────────────────
function CostsView({ doc, usage, win, width }) {
  if (!usage) return <div className="fm-muted">Loading usage…</div>;
  const rows = usage.rows || [];
  const byAgent = {};
  rows.forEach((r) => { (byAgent[r.profile] = byAgent[r.profile] || []).push(r); });
  const tot = rows.reduce((a, r) => ({ billed: a.billed + r.billed_usd, calls: a.calls + r.calls, ma: a.ma + (r.modelark ? r.calls : 0), cap: a.cap + r.cap_equivalent_usd, tin: a.tin + r.input, tout: a.tout + r.output }), { billed: 0, calls: 0, ma: 0, cap: 0, tin: 0, tout: 0 });
  const idToShort = {};
  Object.values(doc.models || {}).forEach((m) => { idToShort[m.id] = m.short; (m.served_as || []).forEach((s) => { idToShort[s] = m.short; }); });
  return (
    <Fragment>
      <div className="fm-stats">
        <Stat label={`Billed · ${periodShort(win)}`} value={money(tot.billed, 2)} tone="money" sub="OpenRouter — what actually gets invoiced" />
        <Stat label="ModelArk subscription" value={num(tot.ma) + " calls"} tone="sub" sub={"$0 · cap-equivalent " + money(tot.cap, 2)} />
        <Stat label="All calls" value={num(tot.calls)} sub={num(tot.tin) + " in · " + num(tot.tout) + " out tokens"} />
        <Stat label="Subscription share" value={tot.calls ? Math.round((100 * tot.ma) / tot.calls) + "%" : "—"} sub="of calls served on the flat plan" />
      </div>
      <Section title="Over time" right={<span className="fm-muted fm-small">{bucketName(usage.bucket)} bars · tap or hover a bar</span>}>
        <TimeChart usage={usage} width={width} />
        <p className="fm-muted fm-small fm-tc-foot">From all nine ledgers. Each session's usage is spread evenly between its first and last call, so short windows are close estimates. Faded bars are part-way through.</p>
      </Section>
      <div className="fm-two">
        <Section title={`Spend by agent · ${periodShort(win)}`} right={<span className="fm-muted fm-small">bar = billed $ · teal = subscription calls</span>}>
          <BarList rows={Object.entries(byAgent).map(([p, rs]) => ({ label: (doc.agents[p] || {}).name || p,
            billed: rs.reduce((a, r) => a + r.billed_usd, 0), calls: rs.reduce((a, r) => a + r.calls, 0), ma: rs.reduce((a, r) => a + (r.modelark ? r.calls : 0), 0) }))} />
        </Section>
        <Section title={`Spend by model · ${periodShort(win)}`} right={<span className="fm-muted fm-small">old OpenRouter DeepSeek ids are history from before 09-11</span>}>
          <BarList rows={Object.values(rows.reduce((acc, r) => { const k = idToShort[r.model] || r.model; const a = (acc[k] = acc[k] || { label: k, billed: 0, calls: 0, ma: 0 }); a.billed += r.billed_usd; a.calls += r.calls; if (r.modelark) a.ma += r.calls; return acc; }, {}))} />
        </Section>
      </div>
      <Section title={`By agent, model and host · ${periodShort(win)}`}>
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
  const [win, setWinState] = useState(readWin);
  const [rootEl, setRootEl] = useState(null);
  const [width, setWidth] = useState(900);
  const [uLoading, setULoading] = useState(false);
  const useq = useRef(0);
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
  const setWin = (w) => { saveWin(w); setWinState(w); };
  // the chart's drawing width ≈ the page width less a section's padding and borders
  useEffect(() => {
    if (!rootEl || typeof ResizeObserver === "undefined") return undefined;
    let t = null;
    const ro = new ResizeObserver((es) => { const w = Math.round(es[0].contentRect.width); clearTimeout(t); t = setTimeout(() => setWidth(w), 150); });
    ro.observe(rootEl); setWidth(Math.round(rootEl.getBoundingClientRect().width));
    return () => { ro.disconnect(); clearTimeout(t); };
  }, [rootEl]);
  const chartW = Math.max(240, width - 36);
  const bucket = pickBucket(win, chartW);
  const loadUsage = useCallback(() => {
    const q = ++useq.current; setULoading(true);
    return fetchJSON(`${API}/usage?window=${win}&bucket=${bucket}&tz=${encodeURIComponent(TZ)}`)
      .then((u) => { if (q === useq.current) { setUsage(u); setULoading(false); } })
      .catch(() => { if (q === useq.current) setULoading(false); });
  }, [win, bucket]);

  useEffect(() => { load(); }, []);
  useEffect(() => { loadUsage(); const t = setInterval(() => { if (!document.hidden) loadUsage(); }, win <= 10800 ? 30000 : 60000); return () => clearInterval(t); }, [loadUsage]);
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

  if (err && !state) return <div className="fm-root" ref={setRootEl}><div className="fm-note fm-note--bad">Fleet Models can't load: {err}</div></div>;
  if (!state || !draft) return <div className="fm-root" ref={setRootEl}><div className="fm-muted fm-loading">Loading the fleet's model settings…</div></div>;
  const stale = !!(usage && usage.window !== win);
  const shown = usage;
  const doc = draft;
  const TABS = [["fleet", "Fleet"], ["agent", "Agents"], ["models", "Models & hosts"], ["costs", "Costs"], ["decisions", "Rules & history"]];
  return (
    <div className="fm-root" ref={setRootEl}>
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
      {tab === "fleet" || tab === "models" || tab === "costs" ? <PeriodBar win={win} setWin={setWin} usage={shown} loading={uLoading} /> : null}
      <main className={cls("fm-main", stale && "is-stale")}>
        {tab === "fleet" ? <FleetView state={state} doc={doc} usage={shown} win={shown ? shown.window : win} onOpen={(p) => { setAgent(p); setTab("agent"); }} /> : null}
        {tab === "agent" ? <AgentView state={state} draft={draft} setDraft={setDraft} p={agent} setP={setAgent} /> : null}
        {tab === "models" ? <ModelsView draft={draft} setDraft={setDraft} usage={shown} win={shown ? shown.window : win} sel={modelSel} setSel={setModelSel} /> : null}
        {tab === "costs" ? <CostsView doc={doc} usage={shown} win={shown ? shown.window : win} width={chartW} /> : null}
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
