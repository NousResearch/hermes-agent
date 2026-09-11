(() => {
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (SDK) {
    let tickLabel = function(t, b, step) {
      const d = new Date(t * 1e3);
      if (step >= 86400 || b >= 86400) return dayMon(d);
      if (d.getHours() === 0 && d.getMinutes() === 0) return wday(d);
      return hhmm(d);
    }, isTick = function(t, step) {
      const d = new Date(t * 1e3);
      if (step >= 86400) {
        if (d.getHours() !== 0 || d.getMinutes() !== 0) return false;
        const ord = Math.round(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / 864e5);
        return ord % (step / 86400) === 0;
      }
      const sec = d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds();
      return sec % step === 0;
    }, rangeLabel = function(s, b, since, now) {
      const a = new Date(Math.max(s.t, since) * 1e3), z = new Date(Math.min(s.end, now) * 1e3);
      if (b >= 86400) return wday(a) + " " + dayMon(a) + (b > 86400 ? " \u2013 " + dayMon(new Date((s.end - 1) * 1e3)) : "") + (s.end > now ? " (so far)" : "");
      const pre = new Date(now * 1e3).toDateString() === a.toDateString() ? "" : wday(a) + " ";
      return pre + hhmm(a) + "\u2013" + (s.end > now ? "now" : hhmm(z));
    }, cls = function(...xs) {
      return xs.filter(Boolean).join(" ");
    }, Pill = function({ doc, alias, onRemove, onLeft, onRight, first, last, dim, compact }) {
      const m = (doc.models || {})[alias];
      const prov = m ? m.provider : "missing";
      return /* @__PURE__ */ h("span", { className: cls("fm-pill", "fm-pill--" + prov, dim && "fm-pill--dim"), title: m ? `${m.id}${m.notes ? "\n\n" + m.notes : ""}` : "not in the registry" }, onLeft && !first ? /* @__PURE__ */ h("button", { className: "fm-pill-btn", onClick: onLeft, title: "move earlier" }, "\u2039") : null, /* @__PURE__ */ h("span", { className: "fm-pill-dot" }), /* @__PURE__ */ h("span", { className: "fm-pill-name" }, m ? m.short || alias : alias), m && m.billing === "subscription" ? /* @__PURE__ */ h("span", { className: "fm-tag fm-tag--sub" }, "SUB") : null, m && m.reasoning && !compact ? /* @__PURE__ */ h("span", { className: "fm-tag", title: "reasoning pin" }, m.reasoning) : null, onRight && !last ? /* @__PURE__ */ h("button", { className: "fm-pill-btn", onClick: onRight, title: "move later" }, "\u203A") : null, onRemove ? /* @__PURE__ */ h("button", { className: "fm-pill-btn fm-pill-x", onClick: onRemove, title: "remove this rung" }, "\xD7") : null);
    }, Chain = function({ doc, chain, empty }) {
      if (!chain || !chain.length) return /* @__PURE__ */ h("span", { className: "fm-muted" }, empty || "\u2014");
      return /* @__PURE__ */ h("span", { className: "fm-chain" }, chain.map((a, i) => /* @__PURE__ */ h("span", { key: a + i, className: "fm-link-step" }, i ? /* @__PURE__ */ h("span", { className: "fm-arrow" }, "\u2192") : null, /* @__PURE__ */ h(Pill, { doc, alias: a, dim: i > 0, compact: true }))));
    }, ChainEditor = function({ doc, chain, onChange, filter }) {
      const opts = Object.keys(doc.models || {}).filter((a) => !chain.includes(a) && (!filter || filter(doc.models[a])));
      const move = (i, d) => {
        const c = chain.slice();
        const [x] = c.splice(i, 1);
        c.splice(i + d, 0, x);
        onChange(c);
      };
      return /* @__PURE__ */ h("span", { className: "fm-chain fm-chain--edit" }, chain.map((a, i) => /* @__PURE__ */ h(Fragment, { key: a + i }, i ? /* @__PURE__ */ h("span", { className: "fm-arrow" }, "\u2192") : null, /* @__PURE__ */ h(
        Pill,
        {
          doc,
          alias: a,
          first: i === 0,
          last: i === chain.length - 1,
          onLeft: () => move(i, -1),
          onRight: () => move(i, 1),
          onRemove: chain.length > 1 ? () => onChange(chain.filter((_, j) => j !== i)) : null
        }
      ))), opts.length ? /* @__PURE__ */ h("select", { className: "fm-add", value: "", onChange: (e) => e.target.value && onChange(chain.concat([e.target.value])) }, /* @__PURE__ */ h("option", { value: "" }, "+ rung"), opts.map((a) => /* @__PURE__ */ h("option", { key: a, value: a }, doc.models[a].short || a))) : null);
    }, Stat = function({ label, value, sub, tone }) {
      return /* @__PURE__ */ h("div", { className: cls("fm-stat", tone && "fm-stat--" + tone) }, /* @__PURE__ */ h("div", { className: "fm-stat-label" }, label), /* @__PURE__ */ h("div", { className: "fm-stat-value" }, value), sub ? /* @__PURE__ */ h("div", { className: "fm-stat-sub" }, sub) : null);
    }, Badge = function({ tone, children, title }) {
      return /* @__PURE__ */ h("span", { className: cls("fm-badge", tone && "fm-badge--" + tone), title }, children);
    }, Section = function({ title, right, children, className }) {
      return /* @__PURE__ */ h("section", { className: cls("fm-section", className) }, title ? /* @__PURE__ */ h("header", { className: "fm-section-head" }, /* @__PURE__ */ h("h3", null, title), right) : null, children);
    }, PeriodBar = function({ win, setWin, usage, loading }) {
      const ref = useRef(null);
      useEffect(() => {
        const box = ref.current, el = box && box.querySelector(".is-active");
        if (!el) return;
        const l = el.offsetLeft - box.offsetLeft, r = l + el.offsetWidth;
        if (l < box.scrollLeft) box.scrollLeft = l - 8;
        else if (r > box.scrollLeft + box.clientWidth) box.scrollLeft = r - box.clientWidth + 8;
      }, [win]);
      return /* @__PURE__ */ h("div", { className: "fm-period" }, /* @__PURE__ */ h("div", { className: "fm-period-chips", role: "group", "aria-label": "Time window", ref }, PERIODS.map(([k, w, name]) => /* @__PURE__ */ h(
        "button",
        {
          key: k,
          className: cls("fm-chip fm-chip--sm", w === win && "is-active"),
          "aria-pressed": w === win,
          title: lastLabel(w),
          onClick: () => setWin(w)
        },
        k
      ))), /* @__PURE__ */ h("span", { className: "fm-muted fm-small fm-period-note" }, loading ? "updating\u2026" : usage ? `${bucketName(usage.bucket)} bars \xB7 ${ago(usage.generated_at)}` : ""));
    }, Spark = function({ values, label, height }) {
      const v = values || [];
      const max = Math.max(0, ...v);
      if (!v.length) return null;
      return /* @__PURE__ */ h("span", { className: "fm-spark", style: { height: (height || 22) + "px" }, "aria-label": label, title: label }, v.map((x, i) => /* @__PURE__ */ h("span", { key: i, style: { height: max ? Math.max(x > 0 ? 8 : 0, 100 * x / max) + "%" : "0%" } })));
    }, TimeChart = function({ usage, width }) {
      const s = usage && usage.series || [];
      const [sel, setSel] = useState(null);
      useEffect(() => {
        setSel(null);
      }, [usage && usage.window, usage && usage.bucket]);
      if (!s.length) return null;
      const b = usage.bucket, n = s.length;
      const pitch = Math.max(1, (width || 600) / n);
      const need = Math.ceil(MIN_LABEL_GAP / pitch) * b;
      const step = LABEL_STEPS.find((x) => x >= need && x % b === 0) || need;
      const maxB = Math.max(0, ...s.map((x) => x.billed_usd));
      const maxC = Math.max(0, ...s.map((x) => x.calls));
      const tot = s.reduce((a, x) => ({ b: a.b + x.billed_usd, c: a.c + x.calls, m: a.m + x.modelark_calls }), { b: 0, c: 0, m: 0 });
      const cur = sel != null && s[sel] ? s[sel] : null;
      const readout = cur ? { when: rangeLabel(cur, b, usage.since, usage.now), b: cur.billed_usd, c: cur.calls, m: cur.modelark_calls } : { when: lastLabel(usage.window), b: tot.b, c: tot.c, m: tot.m };
      const gap = pitch < 7 ? 1 : pitch < 14 ? 2 : 3;
      return /* @__PURE__ */ h("div", { className: "fm-tc", onPointerLeave: (e) => {
        if (e.pointerType === "mouse") setSel(null);
      } }, /* @__PURE__ */ h("div", { className: "fm-tc-readout", "aria-live": "polite" }, /* @__PURE__ */ h("b", null, readout.when), /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("span", { className: "fm-lg fm-lg--money" }), " ", money(readout.b, 3), " billed"), /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("span", { className: "fm-lg fm-lg--calls" }), " ", num(Math.round(readout.c)), " calls"), /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("span", { className: "fm-lg fm-lg--sub" }), " ", num(Math.round(readout.m)), " on subscription")), /* @__PURE__ */ h("div", { className: "fm-tc-plot", style: { gap: gap + "px" } }, /* @__PURE__ */ h("span", { className: "fm-tc-max fm-tc-max--money" }, maxB ? money(maxB, 3) : "$0"), /* @__PURE__ */ h("span", { className: "fm-tc-max fm-tc-max--calls" }, maxC ? num(Math.round(maxC)) + " calls" : "0 calls"), s.map((x, i) => {
        const tick = isTick(x.t, step) && !(i === 0 && x.t < usage.since && n > 3);
        return /* @__PURE__ */ h(
          "button",
          {
            key: x.t,
            type: "button",
            className: cls("fm-tc-col", x.partial && "is-partial", sel === i && "is-sel"),
            onPointerEnter: (e) => {
              if (e.pointerType === "mouse") setSel(i);
            },
            onFocus: () => setSel(i),
            onClick: () => setSel(i),
            "aria-label": `${rangeLabel(x, b, usage.since, usage.now)}: ${money(x.billed_usd, 3)} billed, ${Math.round(x.calls)} calls`
          },
          /* @__PURE__ */ h("span", { className: "fm-tc-m" }, /* @__PURE__ */ h("span", { style: { height: maxB ? 100 * x.billed_usd / maxB + "%" : "0%" } })),
          /* @__PURE__ */ h("span", { className: "fm-tc-c" }, /* @__PURE__ */ h("span", { style: { height: maxC ? 100 * x.calls / maxC + "%" : "0%" } }, /* @__PURE__ */ h("span", { style: { height: x.calls ? 100 * x.modelark_calls / x.calls + "%" : "0%" } }))),
          /* @__PURE__ */ h("span", { className: "fm-tc-lbl" }, tick ? tickLabel(x.t, b, step) : "")
        );
      })), !tot.c ? /* @__PURE__ */ h("div", { className: "fm-muted fm-small fm-tc-empty" }, "No calls in this window.") : null);
    }, helperGroups = function(aux) {
      const groups = {};
      Object.entries(aux || {}).forEach(([t, s]) => {
        if (t === "vision") return;
        const k = JSON.stringify(chainOf(s)) + "|" + reasoningOf(s);
        (groups[k] = groups[k] || { chain: chainOf(s), reasoning: reasoningOf(s), tasks: [] }).tasks.push(t);
      });
      return Object.values(groups);
    }, AgentCard = function({ doc, p, drift, use, spark, win, onOpen }) {
      const a = doc.agents[p];
      const aux = a.aux || {};
      const top = use.hosts.sort((x, y) => y[1] - x[1])[0];
      return /* @__PURE__ */ h(
        "article",
        {
          className: cls("fm-card", drift && "fm-card--drift"),
          onClick: onOpen,
          role: "button",
          tabIndex: 0,
          onKeyDown: (e) => e.key === "Enter" && onOpen()
        },
        /* @__PURE__ */ h("header", { className: "fm-card-head" }, /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("div", { className: "fm-card-name" }, a.name || p, " ", a.locked ? /* @__PURE__ */ h("span", { className: "fm-lock", title: "Locked \u2014 changes need an explicit unlock" }, "\u{1F512}") : null), /* @__PURE__ */ h("div", { className: "fm-card-role" }, a.role || p)), /* @__PURE__ */ h("div", { className: "fm-card-badges" }, drift ? /* @__PURE__ */ h(Badge, { tone: "warn", title: drift.join("\n") }, "drift") : /* @__PURE__ */ h(Badge, { tone: "ok" }, "in sync"), a.reasoning ? /* @__PURE__ */ h(Badge, { title: "agent default reasoning" }, a.reasoning) : null)),
        /* @__PURE__ */ h("dl", { className: "fm-slots" }, /* @__PURE__ */ h("dt", null, "Main"), /* @__PURE__ */ h("dd", null, /* @__PURE__ */ h(Chain, { doc, chain: chainOf(a.main) })), /* @__PURE__ */ h("dt", null, "Subagents"), /* @__PURE__ */ h("dd", null, /* @__PURE__ */ h(Chain, { doc, chain: chainOf(a.subagents), empty: "inherit main" })), a.cron ? /* @__PURE__ */ h(Fragment, null, /* @__PURE__ */ h("dt", null, "Cron"), /* @__PURE__ */ h("dd", null, /* @__PURE__ */ h(Chain, { doc, chain: chainOf(a.cron) }))) : null, /* @__PURE__ */ h("dt", null, "Vision"), /* @__PURE__ */ h("dd", null, /* @__PURE__ */ h(Chain, { doc, chain: chainOf(aux.vision), empty: "main model" })), helperGroups(aux).map((g) => /* @__PURE__ */ h(Fragment, { key: g.tasks.join() }, /* @__PURE__ */ h("dt", { title: g.tasks.join(", ") }, g.tasks.length > 1 ? "Helpers \xD7" + g.tasks.length : TASK_LABEL[g.tasks[0]] || g.tasks[0]), /* @__PURE__ */ h("dd", null, /* @__PURE__ */ h(Chain, { doc, chain: g.chain }), g.reasoning ? /* @__PURE__ */ h("span", { className: "fm-tag fm-tag--r" }, g.reasoning) : null)))),
        /* @__PURE__ */ h("footer", { className: "fm-card-foot" }, spark ? /* @__PURE__ */ h(Spark, { values: spark, label: `${a.name || p}: calls over the last ${periodName(win)}` }) : null, /* @__PURE__ */ h("span", { className: "fm-card-win" }, periodShort(win)), /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("b", null, num(use.calls)), " calls"), /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("b", null, money(use.billed, 3)), " billed"), /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("b", null, num(use.ma)), " on subscription"), top ? /* @__PURE__ */ h("span", { className: "fm-muted", title: "most calls served by" }, "via ", top[0]) : null)
      );
    }, usageByProfile = function(usage) {
      const out = {};
      (usage && usage.rows || []).forEach((r) => {
        const u = out[r.profile] = out[r.profile] || { calls: 0, billed: 0, ma: 0, hostMap: {} };
        u.calls += r.calls;
        u.billed += r.billed_usd;
        if (r.modelark) u.ma += r.calls;
        if (r.host) u.hostMap[r.host] = (u.hostMap[r.host] || 0) + r.calls;
      });
      Object.values(out).forEach((u) => {
        u.hosts = Object.entries(u.hostMap);
      });
      return out;
    }, FleetView = function({ state, doc, usage, win, onOpen }) {
      const byP = usageByProfile(usage);
      const empty = { calls: 0, billed: 0, ma: 0, hosts: [] };
      const tot = Object.values(byP).reduce((a, u) => ({ calls: a.calls + u.calls, billed: a.billed + u.billed, ma: a.ma + u.ma }), { calls: 0, billed: 0, ma: 0 });
      const models = doc.models || {};
      return /* @__PURE__ */ h(Fragment, null, /* @__PURE__ */ h("div", { className: "fm-stats" }, /* @__PURE__ */ h(Stat, { label: `Calls \xB7 ${periodShort(win)}`, value: num(tot.calls), sub: lastLabel(win).toLowerCase() }), /* @__PURE__ */ h(Stat, { label: `Billed \xB7 ${periodShort(win)}`, value: money(tot.billed, 2), sub: "OpenRouter, real invoices", tone: "money" }), /* @__PURE__ */ h(Stat, { label: "ModelArk subscription", value: num(tot.ma) + " calls", sub: tot.calls ? Math.round(100 * tot.ma / tot.calls) + "% of all calls \xB7 $0" : "$0", tone: "sub" }), /* @__PURE__ */ h(Stat, { label: "Registry", value: Object.keys(models).length + " models", sub: Object.values(models).filter((m) => m.provider === "modelark").length + " subscription \xB7 " + Object.values(models).filter((m) => m.provider === "openrouter").length + " OpenRouter" }), /* @__PURE__ */ h(Stat, { label: "Sync", value: Object.keys(state.drift || {}).length ? Object.keys(state.drift).length + " drifted" : "all 9 in sync", tone: Object.keys(state.drift || {}).length ? "warn" : "ok", sub: "revision " + state.revision })), /* @__PURE__ */ h("div", { className: "fm-grid" }, state.profiles.map((p) => /* @__PURE__ */ h(
        AgentCard,
        {
          key: p,
          doc,
          p,
          drift: (state.drift || {})[p],
          use: byP[p] || empty,
          win,
          spark: usage && usage.by_profile ? (usage.by_profile[p] || {}).calls || (usage.series || []).map(() => 0) : null,
          onOpen: () => onOpen(p)
        }
      ))));
    }, LiveChain = function({ doc, live }) {
      if (!live) return /* @__PURE__ */ h("span", { className: "fm-muted" }, "not set");
      const byId = {};
      Object.entries(doc.models || {}).forEach(([a, m]) => {
        byId[m.provider + "|" + m.id] = a;
      });
      return /* @__PURE__ */ h("span", { className: "fm-chain" }, live.map(([prov, id], i) => /* @__PURE__ */ h(Fragment, { key: i }, i ? /* @__PURE__ */ h("span", { className: "fm-arrow" }, "\u2192") : null, byId[prov + "|" + id] ? /* @__PURE__ */ h(Pill, { doc, alias: byId[prov + "|" + id], dim: true }) : /* @__PURE__ */ h("code", { className: "fm-code" }, prov, ":", id))));
    }, SlotRow = function({ doc, label, spec, live, onChange, nullable, nullLabel, withReasoning, filter, onDelete, hint }) {
      const chain = chainOf(spec);
      const isNull = spec == null;
      const primary = Object.keys(doc.models || {}).find((a) => !filter || filter(doc.models[a]));
      return /* @__PURE__ */ h("div", { className: "fm-slot" }, /* @__PURE__ */ h("div", { className: "fm-slot-label" }, /* @__PURE__ */ h("div", null, label), hint ? /* @__PURE__ */ h("div", { className: "fm-slot-hint" }, hint) : null), /* @__PURE__ */ h("div", { className: "fm-slot-body" }, isNull ? /* @__PURE__ */ h("span", { className: "fm-muted" }, nullLabel, " ", /* @__PURE__ */ h("button", { className: "fm-link", onClick: () => onChange([primary]) }, "set a chain")) : /* @__PURE__ */ h(ChainEditor, { doc, chain, filter, onChange: (c) => onChange(withChain(spec, c)) }), /* @__PURE__ */ h("div", { className: "fm-slot-live" }, "live: ", /* @__PURE__ */ h(LiveChain, { doc, live }))), /* @__PURE__ */ h("div", { className: "fm-slot-side" }, withReasoning && !isNull ? /* @__PURE__ */ h(
        "select",
        {
          value: reasoningOf(spec),
          title: "reasoning effort for this task",
          onChange: (e) => onChange(e.target.value ? { chain, reasoning: e.target.value } : chain)
        },
        REASONING.map((r) => /* @__PURE__ */ h("option", { key: r, value: r }, r ? "reasoning " + r : "reasoning \u2014"))
      ) : null, nullable && !isNull ? /* @__PURE__ */ h("button", { className: "fm-link", onClick: () => onChange(null), title: "remove this chain" }, "\u2192 ", nullLabel || "clear") : null, onDelete ? /* @__PURE__ */ h("button", { className: "fm-link fm-link--danger", onClick: onDelete }, "remove") : null));
    }, listInput = function(v) {
      return (v || []).join(", ");
    }, parseList = function(s) {
      return s.split(",").map((x) => x.trim()).filter(Boolean);
    }, AgentView = function({ state, draft, setDraft, p, setP }) {
      const a = draft.agents[p];
      const live = state.live[p] || {};
      const set = (fn) => setDraft((d) => {
        const n = clone(d);
        fn(n.agents[p], n);
        return n;
      });
      const aux = a.aux || {};
      const unusedTasks = AUX_TASKS.filter((t) => !(t in aux));
      const drift = (state.drift || {})[p];
      const visionOk = (m) => !!m.vision;
      const toolsOk = (m) => m.tools !== false;
      return /* @__PURE__ */ h("div", { className: "fm-agent" }, /* @__PURE__ */ h("nav", { className: "fm-agent-nav" }, state.profiles.map((q) => /* @__PURE__ */ h("button", { key: q, className: cls("fm-agent-tab", q === p && "is-active", (state.drift || {})[q] && "has-drift"), onClick: () => setP(q) }, draft.agents[q].name || q, draft.agents[q].locked ? " \u{1F512}" : ""))), /* @__PURE__ */ h(Section, { title: `${a.name || p} \u2014 ${a.role || ""}`, right: /* @__PURE__ */ h("span", { className: "fm-row" }, a.locked ? /* @__PURE__ */ h(Badge, { tone: "warn" }, "locked") : null, drift ? /* @__PURE__ */ h(Badge, { tone: "warn", title: drift.join("\n") }, "config drifted from models.yaml") : /* @__PURE__ */ h(Badge, { tone: "ok" }, "config matches")) }, drift ? /* @__PURE__ */ h("div", { className: "fm-note fm-note--warn" }, drift.map((d, i) => /* @__PURE__ */ h("div", { key: i }, d)), /* @__PURE__ */ h("div", null, "Applying any change rewrites this profile from models.yaml.")) : null, /* @__PURE__ */ h("div", { className: "fm-slots-edit" }, /* @__PURE__ */ h(
        SlotRow,
        {
          doc: draft,
          label: "Main loop",
          hint: "primary \u2192 fallbacks",
          spec: a.main,
          live: live.main,
          filter: toolsOk,
          onChange: (c) => set((x) => {
            x.main = c;
          })
        }
      ), /* @__PURE__ */ h(
        SlotRow,
        {
          doc: draft,
          label: "Subagents",
          hint: "delegated children \u2014 their own chain",
          spec: a.subagents,
          live: live.subagents,
          nullable: true,
          nullLabel: "inherit main",
          filter: toolsOk,
          onChange: (c) => set((x) => {
            x.subagents = c;
          })
        }
      ), /* @__PURE__ */ h(
        SlotRow,
        {
          doc: draft,
          label: "Cron jobs",
          hint: "scheduled jobs on this profile",
          spec: a.cron,
          live: live.cron,
          nullable: true,
          nullLabel: "Hermes default",
          filter: toolsOk,
          onChange: (c) => set((x) => {
            x.cron = c;
          })
        }
      ), Object.keys(aux).sort((x, y) => x === "vision" ? -1 : y === "vision" ? 1 : x.localeCompare(y)).map((t) => /* @__PURE__ */ h(
        SlotRow,
        {
          key: t,
          doc: draft,
          label: TASK_LABEL[t] || t,
          hint: t === "vision" ? "images \u2014 every rung must accept them" : "auxiliary task",
          spec: aux[t],
          live: (live.aux || {})[t] ? live.aux[t].chain : null,
          withReasoning: true,
          filter: t === "vision" ? visionOk : null,
          onChange: (c) => set((x) => {
            x.aux[t] = c;
          }),
          onDelete: () => set((x) => {
            delete x.aux[t];
          })
        }
      )), unusedTasks.length ? /* @__PURE__ */ h("div", { className: "fm-slot fm-slot--add" }, /* @__PURE__ */ h("div", { className: "fm-slot-label" }, "Add helper"), /* @__PURE__ */ h("div", { className: "fm-slot-body" }, /* @__PURE__ */ h("select", { value: "", onChange: (e) => e.target.value && set((x) => {
        x.aux = x.aux || {};
        x.aux[e.target.value] = [e.target.value === "vision" ? "glm" in draft.models ? "glm" : Object.keys(draft.models)[0] : Object.keys(draft.models)[0]];
      }) }, /* @__PURE__ */ h("option", { value: "" }, "+ auxiliary task\u2026"), unusedTasks.map((t) => /* @__PURE__ */ h("option", { key: t, value: t }, TASK_LABEL[t] || t))), /* @__PURE__ */ h("span", { className: "fm-muted" }, " tasks left unset use Hermes' automatic routing"))) : null)), /* @__PURE__ */ h("div", { className: "fm-two" }, /* @__PURE__ */ h(Section, { title: "Agent defaults" }, /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Default reasoning effort"), /* @__PURE__ */ h("select", { value: a.reasoning || "", onChange: (e) => set((x) => {
        x.reasoning = e.target.value || null;
      }) }, REASONING.map((r) => /* @__PURE__ */ h("option", { key: r, value: r }, r || "Hermes default"))), /* @__PURE__ */ h("small", null, "Per-model pins (Models & hosts) win \u2014 e.g. v4.1 always runs high.")), /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Why this setup"), /* @__PURE__ */ h("input", { value: a.why || "", onChange: (e) => set((x) => {
        x.why = e.target.value;
      }) }), /* @__PURE__ */ h("small", null, "Shown in Smith's SOUL model table.")), p === "root" ? /* @__PURE__ */ h("label", { className: "fm-check" }, /* @__PURE__ */ h("input", { type: "checkbox", checked: !!a.locked, onChange: (e) => set((x) => {
        x.locked = e.target.checked;
      }) }), " Locked (Smith is the overwatch \u2014 changes need an explicit unlock)") : null), /* @__PURE__ */ h(Section, { title: "Profile default OpenRouter routing" }, /* @__PURE__ */ h("p", { className: "fm-muted fm-small" }, "Applies to this agent's OpenRouter calls for models without their own host pins. Models with pins (Models & hosts) override it wherever they run. ", /* @__PURE__ */ h("b", null, "data_collection: deny"), " is always on."), /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Prefer hosts (order)"), /* @__PURE__ */ h(
        "input",
        {
          defaultValue: listInput((a.routing || {}).order),
          key: "o" + p + draft.revision,
          onBlur: (e) => set((x) => {
            x.routing = { ...x.routing || {}, order: parseList(e.target.value) };
          })
        }
      )), /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Never use (ignore)"), /* @__PURE__ */ h(
        "input",
        {
          defaultValue: listInput((a.routing || {}).ignore),
          key: "i" + p + draft.revision,
          onBlur: (e) => set((x) => {
            x.routing = { ...x.routing || {}, ignore: parseList(e.target.value) };
          })
        }
      )), /* @__PURE__ */ h("label", { className: "fm-check" }, /* @__PURE__ */ h(
        "input",
        {
          type: "checkbox",
          checked: !!(a.routing || {}).require_parameters,
          onChange: (e) => set((x) => {
            x.routing = { ...x.routing || {}, require_parameters: e.target.checked };
          })
        }
      ), " Only hosts that support every request parameter (require_parameters)"))));
    }, usedBy = function(doc, alias) {
      const out = [];
      Object.entries(doc.agents || {}).forEach(([p, a]) => {
        ["main", "subagents", "cron"].forEach((s) => {
          const c = chainOf(a[s]);
          const i = c.indexOf(alias);
          if (i >= 0) out.push({ p, slot: s, pos: i });
        });
        Object.entries(a.aux || {}).forEach(([t, s]) => {
          const i = chainOf(s).indexOf(alias);
          if (i >= 0) out.push({ p, slot: t, pos: i });
        });
      });
      return out;
    }, HostTable = function({ model, market, onChange, onProbe, probes, minUptime }) {
      const hosts = model.hosts || {};
      const pinned = hosts.order || hosts.only || [];
      const restricted = !!(hosts.only && hosts.only.length);
      const eps = market && market.endpoints || [];
      const byTag = {};
      eps.forEach((e) => {
        byTag[e.tag] = e;
      });
      const find = (t) => byTag[t] || eps.find((e) => hostSlug(e.tag) === t);
      const rows = pinned.map((t) => ({ tag: t, ep: find(t), pinned: true })).concat(eps.filter((e) => !pinned.some((t) => t === e.tag || t === hostSlug(e.tag) && !byTag[t])).map((e) => ({ tag: e.tag, ep: e, pinned: false })));
      const write = (list, only) => onChange({ ...hosts, order: list.length ? list : void 0, only: only && list.length ? list : void 0 });
      const move = (i, d) => {
        const l = pinned.slice();
        const [x] = l.splice(i, 1);
        l.splice(i + d, 0, x);
        write(l, restricted);
      };
      const why = (ep) => {
        if (!ep) return "not listed right now \u2014 a pin that matches nothing fails SILENTLY";
        if (ep.tools === false) return "no tool calling";
        if (minUptime && ep.uptime_1d != null && ep.uptime_1d < minUptime) return `uptime ${pct(ep.uptime_1d)} < ${minUptime}%`;
        if (ep.status != null && ep.status < 0) return "degraded right now";
        return null;
      };
      return /* @__PURE__ */ h("div", { className: "fm-hosts" }, /* @__PURE__ */ h("div", { className: "fm-row fm-hosts-bar" }, /* @__PURE__ */ h("label", { className: "fm-check" }, /* @__PURE__ */ h("input", { type: "checkbox", checked: restricted, onChange: (e) => write(pinned, e.target.checked) }), " Only these hosts \u2014 unticked, the order is a preference and other hosts may serve"), /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, market ? market.source === "live" ? "live from OpenRouter \xB7 " + ago(market.fetched_at) : market.source === "snapshot" ? "nightly snapshot \xB7 " + ago(market.fetched_at) : "market data unavailable" : "loading\u2026")), /* @__PURE__ */ h("div", { className: "fm-table-wrap" }, /* @__PURE__ */ h("table", { className: "fm-table" }, /* @__PURE__ */ h("thead", null, /* @__PURE__ */ h("tr", null, /* @__PURE__ */ h("th", null, "#"), /* @__PURE__ */ h("th", null, "Host"), /* @__PURE__ */ h("th", null, "Quant"), /* @__PURE__ */ h("th", { className: "r" }, "In $/M"), /* @__PURE__ */ h("th", { className: "r" }, "Out $/M"), /* @__PURE__ */ h("th", { className: "r" }, "Cache $/M"), /* @__PURE__ */ h("th", { className: "r" }, "Uptime 1d"), /* @__PURE__ */ h("th", { className: "r" }, "30m"), /* @__PURE__ */ h("th", null, "Tools"), /* @__PURE__ */ h("th", { className: "r" }, "p50 ms"), /* @__PURE__ */ h("th", { className: "r" }, "tok/s"), /* @__PURE__ */ h("th", null))), /* @__PURE__ */ h("tbody", null, rows.map((r, i) => {
        const ep = r.ep || {};
        const warn = why(r.ep);
        const pr = probes[r.tag];
        return /* @__PURE__ */ h("tr", { key: r.tag, className: cls(r.pinned ? "is-pinned" : "is-other", warn && r.pinned && "is-warn") }, /* @__PURE__ */ h("td", { className: "fm-order" }, r.pinned ? /* @__PURE__ */ h("span", { className: "fm-row" }, /* @__PURE__ */ h("b", null, i + 1), /* @__PURE__ */ h("button", { className: "fm-mini", disabled: i === 0, onClick: () => move(i, -1) }, "\u25B2"), /* @__PURE__ */ h("button", { className: "fm-mini", disabled: i === pinned.length - 1, onClick: () => move(i, 1) }, "\u25BC"), /* @__PURE__ */ h("button", { className: "fm-mini", title: "unpin", onClick: () => write(pinned.filter((t) => t !== r.tag), restricted) }, "\xD7")) : /* @__PURE__ */ h("button", { className: "fm-mini fm-mini--add", onClick: () => write(pinned.concat([r.tag]), restricted) }, "pin")), /* @__PURE__ */ h("td", null, /* @__PURE__ */ h("div", { className: "fm-host" }, ep.provider || hostSlug(r.tag)), /* @__PURE__ */ h("code", { className: "fm-code" }, r.tag), warn ? /* @__PURE__ */ h("div", { className: "fm-warn-line" }, warn) : null), /* @__PURE__ */ h("td", null, ep.quant && ep.quant !== "unknown" ? ep.quant : "\u2014"), /* @__PURE__ */ h("td", { className: "r" }, price(ep.in)), /* @__PURE__ */ h("td", { className: "r" }, price(ep.out)), /* @__PURE__ */ h("td", { className: "r" }, price(ep.cache_read)), /* @__PURE__ */ h("td", { className: cls("r", ep.uptime_1d != null && ep.uptime_1d < (minUptime || 95) && "fm-bad") }, pct(ep.uptime_1d)), /* @__PURE__ */ h("td", { className: "r" }, pct(ep.uptime_30m)), /* @__PURE__ */ h("td", null, ep.tools == null ? "\u2014" : ep.tools ? "\u2713" : "\u2717"), /* @__PURE__ */ h("td", { className: "r" }, ep.latency_ms ? Math.round(ep.latency_ms) : "\u2014"), /* @__PURE__ */ h("td", { className: "r" }, ep.tps ? Math.round(ep.tps) : "\u2014"), /* @__PURE__ */ h("td", null, /* @__PURE__ */ h("button", { className: "fm-mini", onClick: () => onProbe(r.tag), disabled: pr === "\u2026", title: "one tiny call pinned to this host with fallbacks off \u2014 proves it routes" }, "probe"), pr && pr !== "\u2026" ? /* @__PURE__ */ h("div", { className: cls("fm-small", pr.rate_limited ? "fm-warn-line" : pr.routable ? "fm-good" : "fm-bad"), title: pr.error || "" }, pr.rate_limited ? "busy now \xB7 pin matches" : pr.routable ? `served by ${pr.served_by} \xB7 ${pr.latency_ms}ms` : (pr.status || "") + " " + (pr.error || "not routable").slice(0, 60)) : pr === "\u2026" ? /* @__PURE__ */ h("div", { className: "fm-small fm-muted" }, "probing\u2026") : null));
      })))));
    }, ModelDetail = function({ draft, alias, setDraft, usage, win }) {
      const m = draft.models[alias];
      const [mk, setMk] = useState(null);
      const [probes, setProbes] = useState({});
      useEffect(() => {
        setMk(null);
        setProbes({});
        if (m && m.provider === "openrouter") fetchJSON(`${API}/market?model=${encodeURIComponent(m.id)}`).then(setMk).catch(() => setMk({ endpoints: [], source: "unavailable" }));
      }, [alias]);
      if (!m) return null;
      const set = (fn) => setDraft((d) => {
        const n = clone(d);
        fn(n.models[alias]);
        return n;
      });
      const probe = (tag) => {
        setProbes((p) => ({ ...p, [tag]: "\u2026" }));
        fetchJSON(`${API}/probe`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, host: tag }) }).then((r) => setProbes((p) => ({ ...p, [tag]: r }))).catch((e) => setProbes((p) => ({ ...p, [tag]: { routable: false, error: String(e.message || e) } })));
      };
      const uses = usedBy(draft, alias);
      const rows = (usage && usage.rows || []).filter((r) => r.model === m.id || (m.served_as || []).includes(r.model));
      const hostUse = {};
      rows.forEach((r) => {
        const k = r.host || "?";
        hostUse[k] = hostUse[k] || { calls: 0, billed: 0 };
        hostUse[k].calls += r.calls;
        hostUse[k].billed += r.billed_usd;
      });
      const totalCalls = rows.reduce((a, r) => a + r.calls, 0);
      const ids = [m.id].concat(m.served_as || []);
      const bm = usage && usage.by_model || {};
      const spark = usage && usage.series ? usage.series.map((_, i) => ids.reduce((a, id) => a + (((bm[id] || {}).calls || [])[i] || 0), 0)) : null;
      const ce = m.cap_equivalent || {};
      return /* @__PURE__ */ h("div", { className: "fm-model" }, /* @__PURE__ */ h("header", { className: "fm-model-head" }, /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("h2", null, m.short || alias, " ", /* @__PURE__ */ h("span", { className: cls("fm-prov", "fm-prov--" + m.provider) }, m.provider === "modelark" ? "ModelArk \xB7 subscription" : "OpenRouter \xB7 metered")), /* @__PURE__ */ h("code", { className: "fm-code" }, m.id)), /* @__PURE__ */ h("div", { className: "fm-row" }, m.vision ? /* @__PURE__ */ h(Badge, null, "vision") : /* @__PURE__ */ h(Badge, { tone: "dim" }, "text-only"), m.tools !== false ? /* @__PURE__ */ h(Badge, null, "tools") : /* @__PURE__ */ h(Badge, { tone: "warn" }, "no tools"), m.context ? /* @__PURE__ */ h(Badge, null, num(m.context), " ctx") : null, /* @__PURE__ */ h(Badge, null, m.vendor))), m.notes ? /* @__PURE__ */ h("p", { className: "fm-notes" }, m.notes) : null, /* @__PURE__ */ h("div", { className: "fm-two" }, /* @__PURE__ */ h(Section, { title: "Settings" }, /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Display name"), /* @__PURE__ */ h("input", { value: m.short || "", onChange: (e) => set((x) => {
        x.short = e.target.value;
      }) })), /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Reasoning pin"), /* @__PURE__ */ h("select", { value: m.reasoning || "", onChange: (e) => set((x) => {
        if (e.target.value) x.reasoning = e.target.value;
        else delete x.reasoning;
      }) }, REASONING.map((r) => /* @__PURE__ */ h("option", { key: r, value: r }, r || "none \u2014 agent default applies"))), /* @__PURE__ */ h("small", null, "Wins over every agent's default, on every surface this model runs (main, fallback, subagents, helpers).")), /* @__PURE__ */ h("div", { className: "fm-row" }, /* @__PURE__ */ h("label", { className: "fm-check" }, /* @__PURE__ */ h("input", { type: "checkbox", checked: !!m.vision, onChange: (e) => set((x) => {
        x.vision = e.target.checked;
      }) }), " accepts images"), /* @__PURE__ */ h("label", { className: "fm-check" }, /* @__PURE__ */ h("input", { type: "checkbox", checked: m.tools !== false, onChange: (e) => set((x) => {
        x.tools = e.target.checked;
      }) }), " tool calling")), /* @__PURE__ */ h("label", { className: "fm-field" }, /* @__PURE__ */ h("span", null, "Notes"), /* @__PURE__ */ h("textarea", { rows: 3, value: m.notes || "", onChange: (e) => set((x) => {
        x.notes = e.target.value;
      }) }))), /* @__PURE__ */ h(
        Section,
        {
          title: m.provider === "modelark" ? "Pricing \u2014 cap-equivalent" : "Usage \xB7 " + periodShort(win),
          right: spark && totalCalls ? /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, num(totalCalls), " calls \xB7 ", lastLabel(win).toLowerCase()) : null
        },
        m.provider === "modelark" ? /* @__PURE__ */ h(Fragment, null, /* @__PURE__ */ h("p", { className: "fm-muted fm-small" }, "The Coding Plan reports no cost, so calls record ", /* @__PURE__ */ h("b", null, '$0 "modelark subscription"'), ". The $1 card cap still counts these rates per million tokens, so a runaway worker trips it. Changes reach the cap on the next call \u2014 no deploy."), /* @__PURE__ */ h("div", { className: "fm-row" }, ["input", "output", "cache_read"].map((k) => /* @__PURE__ */ h("label", { key: k, className: "fm-field fm-field--num" }, /* @__PURE__ */ h("span", null, k.replace("_", " "), " $/M"), /* @__PURE__ */ h(
          "input",
          {
            type: "number",
            step: "0.001",
            min: "0",
            value: ce[k] == null ? "" : ce[k],
            onChange: (e) => set((x) => {
              x.cap_equivalent = { ...x.cap_equivalent || {}, [k]: e.target.value === "" ? null : Number(e.target.value) };
            })
          }
        ))))) : null,
        m.provider === "modelark" ? /* @__PURE__ */ h("div", { className: "fm-subhead" }, "Usage \xB7 ", periodShort(win), totalCalls ? " \xB7 " + num(totalCalls) + " calls" : "") : null,
        spark && totalCalls ? /* @__PURE__ */ h(Spark, { values: spark, height: 30, label: `${m.short || alias}: calls over the last ${periodName(win)}` }) : null,
        /* @__PURE__ */ h("div", { className: "fm-hostuse" }, Object.entries(hostUse).sort((a, b) => b[1].calls - a[1].calls).map(([host, u]) => /* @__PURE__ */ h("div", { key: host, className: "fm-bar-row" }, /* @__PURE__ */ h("span", { className: "fm-bar-label" }, host), /* @__PURE__ */ h("span", { className: "fm-bar" }, /* @__PURE__ */ h("span", { style: { width: (totalCalls ? 100 * u.calls / totalCalls : 0) + "%" } })), /* @__PURE__ */ h("span", { className: "fm-bar-val" }, num(u.calls), " \xB7 ", money(u.billed, 3)))), !totalCalls ? /* @__PURE__ */ h("div", { className: "fm-muted fm-small" }, "No calls in this window.") : null)
      )), m.provider === "openrouter" ? /* @__PURE__ */ h(Section, { title: "Hosts", right: /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, (m.rules || {}).first_host ? `rule: ${m.rules.first_host} first` : "", (m.rules || {}).min_uptime ? ` \xB7 later hosts \u2265 ${m.rules.min_uptime}% uptime` : "") }, /* @__PURE__ */ h(
        HostTable,
        {
          model: m,
          market: mk,
          probes,
          onProbe: probe,
          minUptime: (m.rules || {}).min_uptime || (draft.policy || {}).min_host_uptime,
          onChange: (hosts) => set((x) => {
            const hh = { ...hosts };
            Object.keys(hh).forEach((k) => hh[k] === void 0 && delete hh[k]);
            x.hosts = hh;
          })
        }
      )) : null, /* @__PURE__ */ h(Section, { title: `Used by ${uses.length} slot${uses.length === 1 ? "" : "s"}` }, /* @__PURE__ */ h("div", { className: "fm-uses" }, uses.map((u, i) => /* @__PURE__ */ h("span", { key: i, className: "fm-use" }, /* @__PURE__ */ h("b", null, draft.agents[u.p].name || u.p), " ", SLOT_LABEL[u.slot] || TASK_LABEL[u.slot] || u.slot, " ", /* @__PURE__ */ h("span", { className: "fm-muted" }, "#", u.pos + 1))), !uses.length ? /* @__PURE__ */ h("span", { className: "fm-muted" }, "Not in any waterfall. ", /* @__PURE__ */ h("button", { className: "fm-link fm-link--danger", onClick: () => setDraft((d) => {
        const n = clone(d);
        delete n.models[alias];
        return n;
      }) }, "Remove from registry")) : null)));
    }, AddModel = function({ draft, setDraft, onAdded }) {
      const [id, setId] = useState("");
      const [info, setInfo] = useState(null);
      const [busy, setBusy] = useState(false);
      const look = () => {
        if (!id.trim()) return;
        setBusy(true);
        fetchJSON(`${API}/market?model=${encodeURIComponent(id.trim())}&fresh=true`).then((r) => {
          setInfo(r);
          setBusy(false);
        }).catch((e) => {
          setInfo({ error: String(e.message || e) });
          setBusy(false);
        });
      };
      const add = () => {
        const mid = id.trim();
        const alias = mid.split("/").pop().toLowerCase().replace(/[^a-z0-9]+/g, "-");
        const eps = info && info.endpoints || [];
        setDraft((d) => {
          const n = clone(d);
          n.models[alias in n.models ? alias + "-2" : alias] = {
            id: mid,
            short: mid.split("/").pop(),
            provider: "openrouter",
            vendor: mid.split("/")[0],
            billing: "metered",
            tools: eps.some((e) => e.tools),
            vision: /image/.test(info && info.modality || ""),
            context: Math.max(0, ...eps.map((e) => e.ctx || 0)) || void 0,
            notes: ""
          };
          return n;
        });
        onAdded(alias);
        setId("");
        setInfo(null);
      };
      return /* @__PURE__ */ h("div", { className: "fm-addmodel" }, /* @__PURE__ */ h("input", { placeholder: "OpenRouter model id, e.g. qwen/qwen3.8-27b", value: id, onChange: (e) => {
        setId(e.target.value);
        setInfo(null);
      }, onKeyDown: (e) => e.key === "Enter" && look() }), /* @__PURE__ */ h("button", { className: "fm-btn", onClick: look, disabled: busy || !id.trim() }, busy ? "checking\u2026" : "Look up"), info ? info.endpoints && info.endpoints.length ? /* @__PURE__ */ h("span", { className: "fm-row" }, /* @__PURE__ */ h("span", { className: "fm-good fm-small" }, info.endpoints.length, " hosts \xB7 ", info.modality), /* @__PURE__ */ h("button", { className: "fm-btn fm-btn--primary", onClick: add }, "Add to registry")) : /* @__PURE__ */ h("span", { className: "fm-bad fm-small" }, info.error || "no endpoints for that id") : null);
    }, ModelsView = function({ draft, setDraft, usage, win, sel, setSel }) {
      const aliases = Object.keys(draft.models || {});
      const cur = sel && draft.models[sel] ? sel : aliases[0];
      return /* @__PURE__ */ h("div", { className: "fm-models" }, /* @__PURE__ */ h("aside", { className: "fm-model-list" }, /* @__PURE__ */ h("div", { className: "fm-model-items", role: "tablist", "aria-label": "Models" }, aliases.map((a) => {
        const m = draft.models[a];
        return /* @__PURE__ */ h("button", { key: a, role: "tab", "aria-selected": a === cur, className: cls("fm-model-item", a === cur && "is-active"), onClick: () => setSel(a) }, /* @__PURE__ */ h("span", { className: cls("fm-pill-dot", "fm-dot--" + m.provider) }), /* @__PURE__ */ h("span", { className: "fm-model-item-name" }, m.short || a), /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, usedBy(draft, a).length, " slots"));
      })), /* @__PURE__ */ h("details", { className: "fm-model-list-foot" }, /* @__PURE__ */ h("summary", null, "+ Add a model"), /* @__PURE__ */ h(AddModel, { draft, setDraft, onAdded: setSel }))), /* @__PURE__ */ h("div", { className: "fm-model-main" }, cur ? /* @__PURE__ */ h(ModelDetail, { key: cur, draft, alias: cur, setDraft, usage, win }) : null));
    }, CostsView = function({ doc, usage, win, width }) {
      if (!usage) return /* @__PURE__ */ h("div", { className: "fm-muted" }, "Loading usage\u2026");
      const rows = usage.rows || [];
      const byAgent = {};
      rows.forEach((r) => {
        (byAgent[r.profile] = byAgent[r.profile] || []).push(r);
      });
      const tot = rows.reduce((a, r) => ({ billed: a.billed + r.billed_usd, calls: a.calls + r.calls, ma: a.ma + (r.modelark ? r.calls : 0), cap: a.cap + r.cap_equivalent_usd, tin: a.tin + r.input, tout: a.tout + r.output }), { billed: 0, calls: 0, ma: 0, cap: 0, tin: 0, tout: 0 });
      const idToShort = {};
      Object.values(doc.models || {}).forEach((m) => {
        idToShort[m.id] = m.short;
        (m.served_as || []).forEach((s) => {
          idToShort[s] = m.short;
        });
      });
      return /* @__PURE__ */ h(Fragment, null, /* @__PURE__ */ h("div", { className: "fm-stats" }, /* @__PURE__ */ h(Stat, { label: `Billed \xB7 ${periodShort(win)}`, value: money(tot.billed, 2), tone: "money", sub: "OpenRouter \u2014 what actually gets invoiced" }), /* @__PURE__ */ h(Stat, { label: "ModelArk subscription", value: num(tot.ma) + " calls", tone: "sub", sub: "$0 \xB7 cap-equivalent " + money(tot.cap, 2) }), /* @__PURE__ */ h(Stat, { label: "All calls", value: num(tot.calls), sub: num(tot.tin) + " in \xB7 " + num(tot.tout) + " out tokens" }), /* @__PURE__ */ h(Stat, { label: "Subscription share", value: tot.calls ? Math.round(100 * tot.ma / tot.calls) + "%" : "\u2014", sub: "of calls served on the flat plan" })), /* @__PURE__ */ h(Section, { title: "Over time", right: /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, bucketName(usage.bucket), " bars \xB7 tap or hover a bar") }, /* @__PURE__ */ h(TimeChart, { usage, width }), /* @__PURE__ */ h("p", { className: "fm-muted fm-small fm-tc-foot" }, "From all nine ledgers. Each session's usage is spread evenly between its first and last call, so short windows are close estimates. Faded bars are part-way through.")), /* @__PURE__ */ h("div", { className: "fm-two" }, /* @__PURE__ */ h(Section, { title: `Spend by agent \xB7 ${periodShort(win)}`, right: /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, "bar = billed $ \xB7 teal = subscription calls") }, /* @__PURE__ */ h(BarList, { rows: Object.entries(byAgent).map(([p, rs]) => ({
        label: (doc.agents[p] || {}).name || p,
        billed: rs.reduce((a, r) => a + r.billed_usd, 0),
        calls: rs.reduce((a, r) => a + r.calls, 0),
        ma: rs.reduce((a, r) => a + (r.modelark ? r.calls : 0), 0)
      })) })), /* @__PURE__ */ h(Section, { title: `Spend by model \xB7 ${periodShort(win)}`, right: /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, "old OpenRouter DeepSeek ids are history from before 09-11") }, /* @__PURE__ */ h(BarList, { rows: Object.values(rows.reduce((acc, r) => {
        const k = idToShort[r.model] || r.model;
        const a = acc[k] = acc[k] || { label: k, billed: 0, calls: 0, ma: 0 };
        a.billed += r.billed_usd;
        a.calls += r.calls;
        if (r.modelark) a.ma += r.calls;
        return acc;
      }, {})) }))), /* @__PURE__ */ h(Section, { title: `By agent, model and host \xB7 ${periodShort(win)}` }, /* @__PURE__ */ h("div", { className: "fm-table-wrap" }, /* @__PURE__ */ h("table", { className: "fm-table" }, /* @__PURE__ */ h("thead", null, /* @__PURE__ */ h("tr", null, /* @__PURE__ */ h("th", null, "Agent"), /* @__PURE__ */ h("th", null, "Model"), /* @__PURE__ */ h("th", null, "Served by"), /* @__PURE__ */ h("th", null, "Where"), /* @__PURE__ */ h("th", { className: "r" }, "Calls"), /* @__PURE__ */ h("th", { className: "r" }, "In"), /* @__PURE__ */ h("th", { className: "r" }, "Out"), /* @__PURE__ */ h("th", { className: "r" }, "Cache read"), /* @__PURE__ */ h("th", { className: "r" }, "Billed"), /* @__PURE__ */ h("th", { className: "r" }, "Cap-equiv."))), /* @__PURE__ */ h("tbody", null, Object.entries(byAgent).map(([p, rs]) => rs.map((r, i) => /* @__PURE__ */ h("tr", { key: p + i }, /* @__PURE__ */ h("td", null, i === 0 ? /* @__PURE__ */ h("b", null, (doc.agents[p] || {}).name || p) : null), /* @__PURE__ */ h("td", null, idToShort[r.model] || /* @__PURE__ */ h("code", { className: "fm-code" }, r.model)), /* @__PURE__ */ h("td", null, r.modelark ? /* @__PURE__ */ h("span", { className: "fm-prov fm-prov--modelark" }, "modelark subscription") : r.host || "\u2014"), /* @__PURE__ */ h("td", { className: "fm-muted" }, r.task === "main" ? "main" : TASK_LABEL[r.task] || r.task), /* @__PURE__ */ h("td", { className: "r" }, num(r.calls)), /* @__PURE__ */ h("td", { className: "r" }, num(r.input)), /* @__PURE__ */ h("td", { className: "r" }, num(r.output)), /* @__PURE__ */ h("td", { className: "r" }, num(r.cache_read)), /* @__PURE__ */ h("td", { className: "r" }, r.modelark ? "$0" : money(r.billed_usd)), /* @__PURE__ */ h("td", { className: "r fm-muted" }, r.modelark ? money(r.cap_equivalent_usd) : "")))))))));
    }, BarList = function({ rows }) {
      const sorted = rows.slice().sort((a, b) => b.billed - a.billed || b.calls - a.calls);
      const max = Math.max(1e-6, ...sorted.map((r) => r.billed));
      return /* @__PURE__ */ h("div", { className: "fm-barlist" }, sorted.map((r) => /* @__PURE__ */ h("div", { key: r.label, className: "fm-bl-row", title: `${r.label}: ${money(r.billed)} billed \xB7 ${r.calls} calls (${r.ma} on subscription)` }, /* @__PURE__ */ h("span", { className: "fm-bl-label" }, r.label), /* @__PURE__ */ h("span", { className: "fm-bl-track" }, /* @__PURE__ */ h("span", { className: "fm-bl-money", style: { width: 100 * r.billed / max + "%" } })), /* @__PURE__ */ h("span", { className: "fm-bl-val" }, /* @__PURE__ */ h("b", null, money(r.billed, 2)), " ", /* @__PURE__ */ h("span", { className: "fm-muted" }, "\xB7 ", num(r.calls), " calls", r.ma ? /* @__PURE__ */ h(Fragment, null, " \xB7 ", /* @__PURE__ */ h("span", { className: "fm-sub-txt" }, num(r.ma), " sub")) : null)))), !sorted.length ? /* @__PURE__ */ h("div", { className: "fm-muted fm-small" }, "No calls in this window.") : null);
    }, DecisionsView = function({ state, draft, setDraft, onRevert }) {
      const pol = draft.policy || {};
      const [what, setWhat] = useState("");
      const [why, setWhy] = useState("");
      return /* @__PURE__ */ h("div", { className: "fm-two fm-two--wide" }, /* @__PURE__ */ h("div", null, /* @__PURE__ */ h(Section, { title: "Standing rules" }, /* @__PURE__ */ h("div", { className: "fm-rule" }, /* @__PURE__ */ h(Badge, { tone: "ok" }, "locked"), /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("b", null, "data_collection: deny on every OpenRouter call"), /* @__PURE__ */ h("div", { className: "fm-muted fm-small" }, "The no-training rule. Enforced by the compiler on every config and by the plugin on every helper call \u2014 not editable here."))), /* @__PURE__ */ h("div", { className: "fm-rule" }, /* @__PURE__ */ h(Badge, { tone: "ok" }, "read-only"), /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("b", null, "Cost caps"), " \u2014 ", Object.entries(state.caps || {}).map(([k, v]) => `${k.replace(/_/g, " ")} $${v}`).join(" \xB7 ") || "see config", /* @__PURE__ */ h("div", { className: "fm-muted fm-small" }, "Richie's alone. Shown, never edited, from this tab."))), /* @__PURE__ */ h("div", { className: "fm-rule" }, /* @__PURE__ */ h(Badge, null, "policy"), /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("b", null, "Hosts after the rule-pinned first host need \u2265 "), /* @__PURE__ */ h(
        "input",
        {
          className: "fm-inline-num",
          type: "number",
          min: "50",
          max: "100",
          value: pol.min_host_uptime || 95,
          onChange: (e) => setDraft((d) => {
            const n = clone(d);
            n.policy = { ...n.policy || {}, min_host_uptime: Number(e.target.value) };
            return n;
          })
        }
      ), /* @__PURE__ */ h("b", null, "% uptime"), /* @__PURE__ */ h("div", { className: "fm-muted fm-small" }, "Binding where a model carries the rule (v4.1); advice elsewhere.")))), /* @__PURE__ */ h(Section, { title: "Decisions" }, /* @__PURE__ */ h("ul", { className: "fm-decisions" }, (draft.decisions || []).map((d, i) => /* @__PURE__ */ h("li", { key: i }, /* @__PURE__ */ h("span", { className: "fm-date" }, String(d.date)), /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("b", null, d.what), /* @__PURE__ */ h("div", { className: "fm-muted" }, d.why)), /* @__PURE__ */ h("button", { className: "fm-mini", title: "remove", onClick: () => setDraft((x) => {
        const n = clone(x);
        n.decisions.splice(i, 1);
        return n;
      }) }, "\xD7")))), /* @__PURE__ */ h("div", { className: "fm-row fm-adddec" }, /* @__PURE__ */ h("input", { placeholder: "Decision", value: what, onChange: (e) => setWhat(e.target.value) }), /* @__PURE__ */ h("input", { placeholder: "Why", value: why, onChange: (e) => setWhy(e.target.value) }), /* @__PURE__ */ h("button", { className: "fm-btn", disabled: !what.trim(), onClick: () => {
        setDraft((x) => {
          const n = clone(x);
          n.decisions = [{ date: (/* @__PURE__ */ new Date()).toISOString().slice(0, 10), what: what.trim(), why: why.trim() }].concat(n.decisions || []);
          return n;
        });
        setWhat("");
        setWhy("");
      } }, "Add")))), /* @__PURE__ */ h(Section, { title: "History", right: /* @__PURE__ */ h("span", { className: "fm-muted fm-small" }, "every apply keeps a pre-image of models.yaml, the 9 configs and the SOULs") }, /* @__PURE__ */ h("ul", { className: "fm-history" }, (state.history || []).map((hh) => /* @__PURE__ */ h("li", { key: hh.id }, /* @__PURE__ */ h("div", { className: "fm-row fm-between" }, /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("b", null, hh.summary), " ", /* @__PURE__ */ h("span", { className: "fm-muted" }, "\xB7 ", hh.by, " \xB7 ", ago(hh.ts))), /* @__PURE__ */ h("button", { className: "fm-mini", onClick: () => onRevert(hh) }, "revert")), /* @__PURE__ */ h("div", { className: "fm-muted fm-small" }, hh.revision ? "revision " + hh.revision + " \xB7 " : "", (hh.changed || []).length, " config(s)", hh.reverts ? " \xB7 reverts " + hh.reverts : "", " \xB7 ", /* @__PURE__ */ h("code", null, hh.id)), hh.changes ? /* @__PURE__ */ h("details", null, /* @__PURE__ */ h("summary", { className: "fm-small" }, "what changed"), Object.entries(hh.changes).map(([p, ch]) => /* @__PURE__ */ h("div", { key: p, className: "fm-changes" }, /* @__PURE__ */ h("b", null, p), ch.map((c, i) => /* @__PURE__ */ h("div", { key: i }, /* @__PURE__ */ h("code", null, c)))))) : null)), !(state.history || []).length ? /* @__PURE__ */ h("li", { className: "fm-muted" }, "No applies yet.") : null)));
    }, PlanPanel = function({ plan, onClose, onApply, applying, needsUnlock, unlock, setUnlock, summary, setSummary }) {
      const changed = plan.changed || [];
      const [open, setOpen] = useState(null);
      return /* @__PURE__ */ h("div", { className: "fm-overlay", onClick: onClose }, /* @__PURE__ */ h("div", { className: "fm-panel", onClick: (e) => e.stopPropagation() }, /* @__PURE__ */ h("header", { className: "fm-panel-head" }, /* @__PURE__ */ h("h3", null, "Preview"), /* @__PURE__ */ h("button", { className: "fm-mini", onClick: onClose }, "close")), plan.errors && plan.errors.length ? /* @__PURE__ */ h("div", { className: "fm-note fm-note--bad" }, /* @__PURE__ */ h("b", null, "Can't apply:"), plan.errors.map((e, i) => /* @__PURE__ */ h("div", { key: i }, e))) : /* @__PURE__ */ h("div", { className: "fm-note fm-note--ok" }, changed.length ? `${changed.length} config file(s) will change: ${changed.join(", ")}.` : "Nothing in the configs changes (registry notes/decisions only).", " Running workers, gateways and cron pick it up on their next call \u2014 no restart."), needsUnlock && !(plan.errors && plan.errors.length) ? /* @__PURE__ */ h("div", { className: "fm-note fm-note--warn" }, /* @__PURE__ */ h("b", null, "Smith is locked."), " This change touches the overwatch agent \u2014 tick ", /* @__PURE__ */ h("b", null, "Unlock Smith for this apply"), " below to allow it.") : null, plan.warnings && plan.warnings.length ? /* @__PURE__ */ h("details", { className: "fm-note fm-note--warn" }, /* @__PURE__ */ h("summary", null, plan.warnings.length, " warning(s)"), plan.warnings.map((w, i) => /* @__PURE__ */ h("div", { key: i }, w))) : null, /* @__PURE__ */ h("div", { className: "fm-plan-list" }, Object.entries(plan.plan || {}).filter(([, v]) => v.changes.length).map(([p, v]) => /* @__PURE__ */ h("div", { key: p, className: "fm-plan-item" }, /* @__PURE__ */ h("button", { className: "fm-plan-toggle", onClick: () => setOpen(open === p ? null : p) }, /* @__PURE__ */ h("b", null, p), " \xB7 ", v.changes.length, " change(s) ", open === p ? "\u25BE" : "\u25B8"), open === p ? /* @__PURE__ */ h(Fragment, null, /* @__PURE__ */ h("div", { className: "fm-changes" }, v.changes.map((c, i) => /* @__PURE__ */ h("div", { key: i }, /* @__PURE__ */ h("code", null, c)))), /* @__PURE__ */ h("pre", { className: "fm-diff" }, v.diff.split("\n").map((l, i) => /* @__PURE__ */ h("span", { key: i, className: l.startsWith("+") && !l.startsWith("+++") ? "fm-add-l" : l.startsWith("-") && !l.startsWith("---") ? "fm-del-l" : "" }, l + "\n")))) : null))), !(plan.errors && plan.errors.length) ? /* @__PURE__ */ h("footer", { className: "fm-panel-foot" }, /* @__PURE__ */ h("input", { className: "fm-summary", placeholder: "What is this change? (goes in the history)", value: summary, onChange: (e) => setSummary(e.target.value) }), needsUnlock ? /* @__PURE__ */ h("label", { className: "fm-check fm-unlock" }, /* @__PURE__ */ h("input", { type: "checkbox", checked: unlock, onChange: (e) => setUnlock(e.target.checked) }), " Unlock Smith for this apply") : null, /* @__PURE__ */ h("button", { className: "fm-btn fm-btn--primary", disabled: applying || needsUnlock && !unlock, onClick: onApply }, applying ? "Applying\u2026" : "Apply to the fleet")) : null));
    }, ModelsPage = function() {
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
        setState(s);
        setErr(null);
        setDraft((d) => {
          if (!d || !baseRef.current || same(d, baseRef.current)) {
            baseRef.current = clone(s.doc);
            return clone(s.doc);
          }
          return d;
        });
        return s;
      }).catch((e) => setErr(String(e.message || e))), []);
      const setWin = (w) => {
        saveWin(w);
        setWinState(w);
      };
      useEffect(() => {
        if (!rootEl || typeof ResizeObserver === "undefined") return void 0;
        let t = null;
        const ro = new ResizeObserver((es) => {
          const w = Math.round(es[0].contentRect.width);
          clearTimeout(t);
          t = setTimeout(() => setWidth(w), 150);
        });
        ro.observe(rootEl);
        setWidth(Math.round(rootEl.getBoundingClientRect().width));
        return () => {
          ro.disconnect();
          clearTimeout(t);
        };
      }, [rootEl]);
      const chartW = Math.max(240, width - 36);
      const bucket = pickBucket(win, chartW);
      const loadUsage = useCallback(() => {
        const q = ++useq.current;
        setULoading(true);
        return fetchJSON(`${API}/usage?window=${win}&bucket=${bucket}&tz=${encodeURIComponent(TZ)}`).then((u) => {
          if (q === useq.current) {
            setUsage(u);
            setULoading(false);
          }
        }).catch(() => {
          if (q === useq.current) setULoading(false);
        });
      }, [win, bucket]);
      useEffect(() => {
        load();
      }, []);
      useEffect(() => {
        loadUsage();
        const t = setInterval(() => {
          if (!document.hidden) loadUsage();
        }, win <= 10800 ? 3e4 : 6e4);
        return () => clearInterval(t);
      }, [loadUsage]);
      useEffect(() => {
        const t = setInterval(() => {
          if (!document.hidden) load();
        }, 15e3);
        return () => clearInterval(t);
      }, [load]);
      const base = baseRef.current;
      const dirty = !!(state && draft && base && !same(draft, base));
      const movedUnder = dirty && state.revision !== Number(base.revision || 0);
      const needsUnlock = !!(dirty && base.agents && base.agents.root && base.agents.root.locked && !same(base.agents.root, draft.agents.root));
      const say = (msg, tone) => {
        setFlash({ msg, tone });
        setTimeout(() => setFlash(null), 6e3);
      };
      const body = (isPreview) => JSON.stringify({
        doc: draft,
        base_revision: Number((baseRef.current || {}).revision || 0),
        unlock: (isPreview ? needsUnlock : unlock) ? ["root"] : [],
        summary
      });
      const preview = () => {
        setBusy(true);
        fetchJSON(`${API}/plan`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(true) }).then((r) => {
          setPlan(r);
          setBusy(false);
        }).catch((e) => {
          setBusy(false);
          say(String(e.message || e), "bad");
        });
      };
      const apply = () => {
        setBusy(true);
        fetchJSON(`${API}/apply`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(false) }).then((r) => {
          setBusy(false);
          if (r.ok) {
            setPlan(null);
            setSummary("");
            setUnlock(false);
            baseRef.current = null;
            setDraft(null);
            load().then(() => {
            });
            loadUsage();
            say(`Applied \u2014 ${r.changed.length} config(s) rewritten and verified${r.souls && r.souls.length ? ", SOULs refreshed" : ""}. Revert from History if needed.`, "ok");
          } else {
            setPlan({ ...plan || {}, errors: r.errors, warnings: r.warnings });
          }
        }).catch((e) => {
          setBusy(false);
          say(String(e.message || e), "bad");
        });
      };
      const revert = (hh) => {
        if (!window.confirm(`Put models.yaml, the configs and SOULs back as they were before \u201C${hh.summary}\u201D?`)) return;
        fetchJSON(`${API}/revert`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: hh.id }) }).then((r) => {
          if (r.ok) {
            baseRef.current = null;
            setDraft(null);
            load();
            say("Reverted \u2014 recorded as " + r.id, "ok");
          } else say((r.errors || []).join("; "), "bad");
        });
      };
      if (err && !state) return /* @__PURE__ */ h("div", { className: "fm-root", ref: setRootEl }, /* @__PURE__ */ h("div", { className: "fm-note fm-note--bad" }, "Fleet Models can't load: ", err));
      if (!state || !draft) return /* @__PURE__ */ h("div", { className: "fm-root", ref: setRootEl }, /* @__PURE__ */ h("div", { className: "fm-muted fm-loading" }, "Loading the fleet's model settings\u2026"));
      const stale = !!(usage && usage.window !== win);
      const shown = usage;
      const doc = draft;
      const TABS = [["fleet", "Fleet"], ["agent", "Agents"], ["models", "Models & hosts"], ["costs", "Costs"], ["decisions", "Rules & history"]];
      return /* @__PURE__ */ h("div", { className: "fm-root", ref: setRootEl }, /* @__PURE__ */ h("header", { className: "fm-head" }, /* @__PURE__ */ h("div", null, /* @__PURE__ */ h("h1", null, "Fleet Models"), /* @__PURE__ */ h("div", { className: "fm-sub" }, "Every agent's waterfall from one file \u2014 ", /* @__PURE__ */ h("code", null, "~/.hermes/fleet/models.yaml"), " \xB7 revision ", state.revision, " \xB7 ", state.doc.updated_by || "\u2014", " ", ago(state.doc.updated_at))), /* @__PURE__ */ h("div", { className: "fm-row" }, /* @__PURE__ */ h(Badge, { tone: "ok", title: "data_collection: deny on every OpenRouter call" }, "no-training \xB7 deny"), /* @__PURE__ */ h(Badge, { tone: state.runtime.aux_routing_seam ? "ok" : "warn", title: "helper calls follow each model's pins (fleet-models plugin)" }, state.runtime.aux_routing_seam ? "helper routing on" : "helper routing off"), state.validation.errors.length ? /* @__PURE__ */ h(Badge, { tone: "bad", title: state.validation.errors.join("\n") }, state.validation.errors.length, " rule breach") : null, /* @__PURE__ */ h("button", { className: "fm-btn", onClick: () => {
        load();
        loadUsage();
      } }, "Refresh"))), /* @__PURE__ */ h("nav", { className: "fm-tabs" }, TABS.map(([k, l]) => /* @__PURE__ */ h("button", { key: k, className: cls("fm-tab", tab === k && "is-active"), onClick: () => setTab(k) }, l))), movedUnder ? /* @__PURE__ */ h("div", { className: "fm-note fm-note--warn" }, "Someone applied a change while you were editing (now revision ", state.revision, "). Preview will refuse a stale edit \u2014 discard and redo it.") : null, tab === "fleet" || tab === "models" || tab === "costs" ? /* @__PURE__ */ h(PeriodBar, { win, setWin, usage: shown, loading: uLoading }) : null, /* @__PURE__ */ h("main", { className: cls("fm-main", stale && "is-stale") }, tab === "fleet" ? /* @__PURE__ */ h(FleetView, { state, doc, usage: shown, win: shown ? shown.window : win, onOpen: (p) => {
        setAgent(p);
        setTab("agent");
      } }) : null, tab === "agent" ? /* @__PURE__ */ h(AgentView, { state, draft, setDraft, p: agent, setP: setAgent }) : null, tab === "models" ? /* @__PURE__ */ h(ModelsView, { draft, setDraft, usage: shown, win: shown ? shown.window : win, sel: modelSel, setSel: setModelSel }) : null, tab === "costs" ? /* @__PURE__ */ h(CostsView, { doc, usage: shown, win: shown ? shown.window : win, width: chartW }) : null, tab === "decisions" ? /* @__PURE__ */ h(DecisionsView, { state, draft, setDraft, onRevert: revert }) : null), dirty ? /* @__PURE__ */ h("div", { className: "fm-dock" }, /* @__PURE__ */ h("span", null, /* @__PURE__ */ h("b", null, "Unsaved changes"), " ", /* @__PURE__ */ h("span", { className: "fm-muted" }, "\u2014 nothing reaches the fleet until you apply")), /* @__PURE__ */ h("span", { className: "fm-row" }, /* @__PURE__ */ h("button", { className: "fm-btn", onClick: () => {
        setDraft(clone(state.doc));
        baseRef.current = clone(state.doc);
      } }, "Discard"), /* @__PURE__ */ h("button", { className: "fm-btn fm-btn--primary", disabled: busy, onClick: preview }, busy ? "Checking\u2026" : "Preview & apply"))) : null, plan ? /* @__PURE__ */ h(
        PlanPanel,
        {
          plan,
          onClose: () => setPlan(null),
          onApply: apply,
          applying: busy,
          needsUnlock,
          unlock,
          setUnlock,
          summary,
          setSummary
        }
      ) : null, flash ? /* @__PURE__ */ h("div", { className: cls("fm-flash", "fm-flash--" + flash.tone) }, flash.msg) : null);
    };
    var tickLabel2 = tickLabel, isTick2 = isTick, rangeLabel2 = rangeLabel, cls2 = cls, Pill2 = Pill, Chain2 = Chain, ChainEditor2 = ChainEditor, Stat2 = Stat, Badge2 = Badge, Section2 = Section, PeriodBar2 = PeriodBar, Spark2 = Spark, TimeChart2 = TimeChart, helperGroups2 = helperGroups, AgentCard2 = AgentCard, usageByProfile2 = usageByProfile, FleetView2 = FleetView, LiveChain2 = LiveChain, SlotRow2 = SlotRow, listInput2 = listInput, parseList2 = parseList, AgentView2 = AgentView, usedBy2 = usedBy, HostTable2 = HostTable, ModelDetail2 = ModelDetail, AddModel2 = AddModel, ModelsView2 = ModelsView, CostsView2 = CostsView, BarList2 = BarList, DecisionsView2 = DecisionsView, PlanPanel2 = PlanPanel, ModelsPage2 = ModelsPage;
    const { React } = SDK;
    const h = React.createElement;
    const Fragment = React.Fragment;
    const { useState, useEffect, useCallback, useMemo, useRef } = SDK.hooks;
    const API = "/api/plugins/fleet-models";
    const fetchJSON = SDK.fetchJSON;
    const AUX_TASKS = [
      "vision",
      "compression",
      "title_generation",
      "background_review",
      "goal_judge",
      "kanban_decomposer",
      "web_extract",
      "session_search",
      "skills_hub",
      "approval",
      "flush_memories"
    ];
    const REASONING = ["", "none", "minimal", "low", "medium", "high", "xhigh", "max"];
    const SLOT_LABEL = { main: "Main loop", subagents: "Subagents", cron: "Cron jobs" };
    const TASK_LABEL = {
      vision: "Vision",
      compression: "Compression",
      title_generation: "Titles",
      background_review: "Background review",
      goal_judge: "Goal judge",
      kanban_decomposer: "Decomposer",
      web_extract: "Web extract",
      session_search: "Session search",
      skills_hub: "Skills hub",
      approval: "Approval",
      flush_memories: "Memory flush"
    };
    const clone = (o) => JSON.parse(JSON.stringify(o));
    const chainOf = (spec) => spec == null ? [] : Array.isArray(spec) ? spec : spec.chain || [];
    const reasoningOf = (spec) => spec && !Array.isArray(spec) ? spec.reasoning || "" : "";
    const withChain = (spec, chain) => spec && !Array.isArray(spec) ? { ...spec, chain } : chain;
    const money = (v, d = 4) => v == null ? "\u2014" : v === 0 ? "$0" : v < 1e-4 ? "<$0.0001" : "$" + Number(v).toFixed(v >= 10 ? 2 : d);
    const num = (v) => v == null ? "\u2014" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(v);
    const pct = (v) => v == null ? "\u2014" : v.toFixed(v >= 99.95 ? 2 : 1) + "%";
    const price = (v) => v == null ? "\u2014" : "$" + (v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));
    const ago = (ts) => {
      if (!ts) return "\u2014";
      const s = Math.max(0, Date.now() / 1e3 - (typeof ts === "string" ? Date.parse(ts) / 1e3 : ts));
      return s < 60 ? "just now" : s < 3600 ? Math.round(s / 60) + " min ago" : s < 86400 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " d ago";
    };
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const hostSlug = (t) => String(t || "").split("/")[0];
    const PERIODS = [
      ["15m", 900, "15 minutes"],
      ["30m", 1800, "30 minutes"],
      ["1h", 3600, "hour"],
      ["2h", 7200, "2 hours"],
      ["3h", 10800, "3 hours"],
      ["6h", 21600, "6 hours"],
      ["12h", 43200, "12 hours"],
      ["24h", 86400, "24 hours"],
      ["7d", 604800, "7 days"],
      ["30d", 2592e3, "30 days"]
    ];
    const NICE = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800];
    const LABEL_STEPS = NICE.concat([1209600]);
    const MIN_PITCH = 12;
    const MIN_LABEL_GAP = 36;
    const TZ = (() => {
      try {
        return Intl.DateTimeFormat().resolvedOptions().timeZone || "";
      } catch (e) {
        return "";
      }
    })();
    const naturalBucket = (w) => {
      const m = [[900, 60], [1800, 60], [3600, 120], [7200, 300], [10800, 300], [21600, 900], [43200, 1800], [86400, 3600], [604800, 21600]];
      const hit = m.find(([x]) => w <= x);
      return hit ? hit[1] : 86400;
    };
    const pickBucket = (w, width) => {
      const maxBars = Math.max(6, Math.floor(width / MIN_PITCH));
      let b = naturalBucket(w);
      while (w / b > maxBars) {
        const nx = NICE.find((x) => x > b);
        if (!nx) break;
        b = nx;
      }
      return b;
    };
    const periodName = (w) => {
      const p = PERIODS.find((x) => x[1] === w);
      return p ? p[2] : Math.round(w / 3600) + " hours";
    };
    const periodShort = (w) => {
      const p = PERIODS.find((x) => x[1] === w);
      return p ? p[0] : Math.round(w / 3600) + "h";
    };
    const lastLabel = (w) => w === 3600 ? "Last hour" : "Last " + periodName(w);
    const bucketName = (b) => b < 3600 ? b / 60 + "-minute" : b < 86400 ? b / 3600 === 1 ? "hourly" : b / 3600 + "-hour" : b === 86400 ? "daily" : b / 86400 + "-day";
    const WIN_KEY = "fleet-models.window";
    const readWin = () => {
      try {
        const v = Number(window.localStorage.getItem(WIN_KEY));
        return PERIODS.some((p) => p[1] === v) ? v : 604800;
      } catch (e) {
        return 604800;
      }
    };
    const saveWin = (v) => {
      try {
        window.localStorage.setItem(WIN_KEY, String(v));
      } catch (e) {
      }
    };
    const hhmm = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
    const dayMon = (d) => d.toLocaleDateString([], { day: "numeric", month: "short" });
    const wday = (d) => d.toLocaleDateString([], { weekday: "short" });
    if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
      window.__HERMES_PLUGINS__.register("fleet-models", ModelsPage);
    }
  }
})();
