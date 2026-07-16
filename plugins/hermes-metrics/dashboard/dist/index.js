(function () {
  "use strict";
  // hermes-metrics dashboard plugin — fleet spend + performance metrics.
  // Hand-authored IIFE against the host plugin SDK (same pattern as
  // hermes-achievements): no build step, React from the SDK.
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const h = React.createElement;

  function api(path) {
    return SDK.fetchJSON("/api/plugins/hermes-metrics" + path);
  }

  // ── formatting ────────────────────────────────────────────────────────
  function fmtUSD(v) {
    if (v == null) return "—";
    return "$" + Number(v).toFixed(2);
  }
  function fmtTok(v) {
    if (v == null) return "—";
    v = Number(v);
    if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
    return String(v);
  }
  function fmtDur(s) {
    if (s == null) return "—";
    s = Math.round(s);
    if (s < 60) return s + "s";
    if (s < 3600) return Math.round(s / 60) + "m";
    if (s < 86400) return (s / 3600).toFixed(1) + "h";
    return (s / 86400).toFixed(1) + "d";
  }
  function fmtDate(d) {
    return d ? d.slice(5) : "";
  }

  // ── tooltip (one per chart container, positioned on hover) ────────────
  function useTooltip() {
    const [tip, setTip] = React.useState(null); // {x, y, lines}
    function show(evt, lines) {
      const host = evt.currentTarget.closest(".hm-chart");
      if (!host) return;
      const rect = host.getBoundingClientRect();
      setTip({ x: evt.clientX - rect.left, y: evt.clientY - rect.top, lines: lines });
    }
    function hide() {
      setTip(null);
    }
    const node =
      tip &&
      h(
        "div",
        {
          className: "hm-tooltip",
          style: { left: Math.min(tip.x + 12, 999) + "px", top: Math.max(tip.y - 10, 0) + "px" },
        },
        tip.lines.map(function (line, i) {
          return h("div", { key: i, className: i === 0 ? "hm-tooltip-title" : "" }, line);
        })
      );
    return { show: show, hide: hide, node: node };
  }

  // ── budget meter card (status color + text label, never color alone) ──
  function MeterCard({ lane, label, usd, cap, paused }) {
    const pct = cap ? usd / cap : null;
    let status = "good", statusText = "under budget";
    if (pct != null && pct >= 1) { status = "critical"; statusText = "over budget"; }
    else if (pct != null && pct >= 0.8) { status = "warning"; statusText = "approaching cap"; }
    return h(
      "div",
      { className: "hm-card hm-meter" },
      h("div", { className: "hm-meter-head" },
        h("span", { className: "hm-label" }, label || lane),
        paused && h("span", { className: "hm-badge hm-badge-critical" }, "dispatch paused")
      ),
      h("div", { className: "hm-meter-value" },
        h("span", { className: "hm-hero" }, fmtUSD(usd)),
        cap ? h("span", { className: "hm-cap" }, " / " + fmtUSD(cap) + " today") : null
      ),
      cap
        ? h("div", { className: "hm-meter-bar" },
            h("div", {
              className: "hm-meter-fill hm-fill-" + status,
              style: { width: Math.min(100, (pct || 0) * 100) + "%" },
            })
          )
        : null,
      cap
        ? h("div", { className: "hm-meter-foot hm-status-" + status },
            Math.round((pct || 0) * 100) + "% — " + statusText
          )
        : null
    );
  }

  // ── vertical bar chart (single series or stacked lanes) ───────────────
  const LANE_ORDER = ["api_key", "personal_oauth"];

  function DailyBars({ history, labels }) {
    const tip = useTooltip();
    const days = history || [];
    const totals = days.map(function (d) {
      return LANE_ORDER.reduce(function (acc, l) { return acc + (d.lanes[l] || 0); }, 0);
    });
    const max = Math.max.apply(null, totals.concat([0.01]));
    const W = 460, H = 120, PAD = 4;
    const bw = Math.max(6, Math.floor((W - PAD * 2) / Math.max(days.length, 1)) - 2);
    return h(
      "div",
      { className: "hm-chart" },
      h("svg", { viewBox: "0 0 " + W + " " + (H + 18), className: "hm-svg" },
        days.map(function (d, i) {
          const x = PAD + i * (bw + 2);
          let y = H;
          const segs = LANE_ORDER.map(function (lane, li) {
            const v = d.lanes[lane] || 0;
            const hgt = (v / max) * (H - 8);
            y -= hgt;
            return h("rect", {
              key: lane,
              x: x, y: y, width: bw,
              height: Math.max(hgt - (li > 0 ? 2 : 0), 0),
              rx: 2,
              className: "hm-lane-" + lane,
              onMouseMove: function (e) {
                tip.show(e, [d.date || "", (labels && labels[lane] || lane) + ": " + fmtUSD(v), "total: " + fmtUSD(totals[i])]);
              },
              onMouseLeave: tip.hide,
            });
          });
          return h("g", { key: i },
            segs,
            (i === days.length - 1 || i % 3 === 0) &&
              h("text", { x: x + bw / 2, y: H + 13, className: "hm-tick", textAnchor: "middle" }, fmtDate(d.date))
          );
        })
      ),
      h("div", { className: "hm-legend" },
        LANE_ORDER.map(function (lane) {
          return h("span", { key: lane, className: "hm-legend-item" },
            h("span", { className: "hm-swatch hm-lane-" + lane }),
            (labels && labels[lane]) || lane
          );
        })
      ),
      tip.node
    );
  }

  function ThroughputBars({ throughput }) {
    const tip = useTooltip();
    const days = throughput || [];
    const max = Math.max.apply(null, days.map(function (d) { return d.done; }).concat([1]));
    const W = 460, H = 100, PAD = 4;
    const bw = Math.max(6, Math.floor((W - PAD * 2) / Math.max(days.length, 1)) - 2);
    return h(
      "div",
      { className: "hm-chart" },
      h("svg", { viewBox: "0 0 " + W + " " + (H + 18), className: "hm-svg" },
        days.map(function (d, i) {
          const hgt = (d.done / max) * (H - 8);
          const x = PAD + i * (bw + 2);
          return h("g", { key: i },
            h("rect", {
              x: x, y: H - hgt, width: bw, height: hgt, rx: 2, className: "hm-series-1",
              onMouseMove: function (e) { tip.show(e, [d.date, d.done + " task" + (d.done === 1 ? "" : "s") + " done"]); },
              onMouseLeave: tip.hide,
            }),
            d.done === max && h("text", { x: x + bw / 2, y: H - hgt - 3, className: "hm-tick", textAnchor: "middle" }, d.done),
            (i === days.length - 1 || i % 3 === 0) &&
              h("text", { x: x + bw / 2, y: H + 13, className: "hm-tick", textAnchor: "middle" }, fmtDate(d.date))
          );
        })
      ),
      tip.node
    );
  }

  // ── horizontal bar list (label + bar + value; single hue) ─────────────
  function HBarList({ rows, fmt }) {
    const max = Math.max.apply(null, rows.map(function (r) { return r.value; }).concat([0.0001]));
    return h("div", { className: "hm-hbars" },
      rows.map(function (r) {
        return h("div", { key: r.label, className: "hm-hbar-row" },
          h("span", { className: "hm-hbar-label" }, r.label),
          h("div", { className: "hm-hbar-track" },
            h("div", { className: "hm-hbar-fill hm-series-1", style: { width: (r.value / max) * 100 + "%" } })
          ),
          h("span", { className: "hm-hbar-value" }, (fmt || fmtUSD)(r.value))
        );
      })
    );
  }

  function StatTile({ label, value, hint }) {
    return h("div", { className: "hm-card hm-stat" },
      h("div", { className: "hm-label" }, label),
      h("div", { className: "hm-stat-value" }, value),
      hint && h("div", { className: "hm-hint" }, hint)
    );
  }

  // ── tables ────────────────────────────────────────────────────────────
  const OUTCOME_COLS = ["completed", "blocked", "crashed", "timed_out", "gave_up", "reclaimed"];

  function AgentsTable({ agents, refinements, usage }) {
    const profiles = Object.keys(agents || {}).sort(function (a, b) {
      return (agents[b].runs || 0) - (agents[a].runs || 0);
    });
    const usageBy = {};
    ((usage && usage.profiles) || []).forEach(function (u) { usageBy[u.profile] = u; });
    return h("div", { className: "hm-card hm-tablewrap" },
      h("table", { className: "hm-table" },
        h("thead", null, h("tr", null,
          ["profile", "runs", "ok", "blocked", "crashed", "timeout", "gave up", "avg run",
           "sent back", "reviews ✓/✗", "violations", "follow-ups/session", "tokens 7d"].map(function (c) {
            return h("th", { key: c }, c);
          })
        )),
        h("tbody", null, profiles.map(function (p) {
          const a = agents[p];
          const o = a.outcomes || {};
          const ref = (refinements && refinements.profiles && refinements.profiles[p]) || {};
          const u = usageBy[p];
          const violations = (a.events && ((a.events.protocol_violation || 0) + (a.events.completion_blocked_hallucination || 0))) || 0;
          return h("tr", { key: p },
            h("td", { className: "hm-td-name" }, p),
            h("td", null, a.runs || 0),
            h("td", { className: "hm-num-good" }, o.completed || 0),
            h("td", null, o.blocked || 0),
            h("td", { className: (o.crashed || 0) > 0 ? "hm-num-bad" : "" }, o.crashed || 0),
            h("td", { className: (o.timed_out || 0) > 0 ? "hm-num-bad" : "" }, o.timed_out || 0),
            h("td", null, o.gave_up || 0),
            h("td", null, fmtDur(a.avg_run_seconds)),
            h("td", { className: (a.sent_back || 0) > 0 ? "hm-num-warn" : "" }, a.sent_back || 0),
            h("td", null, (a.reviews_approved || 0) + " / " + (a.reviews_rejected || 0)),
            h("td", { className: violations > 0 ? "hm-num-warn" : "" }, violations),
            h("td", null, ref.followups_per_session != null ? ref.followups_per_session : "—"),
            h("td", null, u ? fmtTok(u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens) : "—")
          );
        }))
      )
    );
  }

  function RunningTable({ running }) {
    if (!running || !running.length)
      return h("div", { className: "hm-card hm-empty" }, "No tasks running right now.");
    return h("div", { className: "hm-card hm-tablewrap" },
      h("table", { className: "hm-table" },
        h("thead", null, h("tr", null, ["task", "profile", "running for", "last heartbeat"].map(function (c) {
          return h("th", { key: c }, c);
        }))),
        h("tbody", null, running.map(function (r) {
          const stale = r.heartbeat_age_seconds != null && r.heartbeat_age_seconds > 300;
          return h("tr", { key: r.id },
            h("td", { className: "hm-td-name" }, r.title || r.id),
            h("td", null, r.assignee),
            h("td", null, fmtDur(r.running_seconds)),
            h("td", { className: stale ? "hm-num-warn" : "" },
              r.heartbeat_age_seconds != null ? fmtDur(r.heartbeat_age_seconds) + " ago" : "—",
              stale ? " ⚠ stale" : "")
          );
        }))
      )
    );
  }

  // ── page ──────────────────────────────────────────────────────────────
  function MetricsPage() {
    const [data, setData] = React.useState(null);
    const [error, setError] = React.useState(null);
    const [updatedAt, setUpdatedAt] = React.useState(null);

    React.useEffect(function () {
      let alive = true;
      function load() {
        api("/overview")
          .then(function (d) {
            if (!alive) return;
            setData(d);
            setError(null);
            setUpdatedAt(new Date());
          })
          .catch(function (e) {
            if (alive) setError(String(e));
          });
      }
      load();
      const t = setInterval(load, 30000);
      return function () { alive = false; clearInterval(t); };
    }, []);

    if (error) return h("div", { className: "hm-page" }, h("div", { className: "hm-card hm-empty" }, "Metrics unavailable: " + error));
    if (!data) return h("div", { className: "hm-page" }, h("div", { className: "hm-card hm-empty" }, "Loading metrics…"));

    const spend = data.spend || {};
    const kanban = data.kanban || {};
    const agents = (data.agents && data.agents.agents) || {};
    const lanes = spend.lanes || {};
    const caps = spend.caps || {};
    const pausedLanes = (spend.throttle && spend.throttle.paused_lanes) || {};
    const laneNames = Array.from(new Set(Object.keys(caps).concat(Object.keys(lanes))));
    const topProfiles = Object.entries(spend.profiles || {})
      .map(function (e) { return { label: e[0], value: e[1].usd || 0 }; })
      .sort(function (a, b) { return b.value - a.value; })
      .slice(0, 8);
    const byStatus = kanban.by_status || {};
    const cyc = kanban.cycle_time_seconds || {};

    return h("div", { className: "hm-page" },
      h("div", { className: "hm-header" },
        h("div", null,
          h("h1", null, "Fleet Metrics"),
          h("p", { className: "hm-hint" },
            "Spend day " + (spend.date || "—") +
            (spend.throttle && spend.throttle.enabled ? " · throttle ARMED" : " · throttle in measurement mode") +
            (updatedAt ? " · updated " + updatedAt.toLocaleTimeString() : ""))
        )
      ),

      h("h2", { className: "hm-section" }, "Spend"),
      h("div", { className: "hm-row" },
        laneNames.map(function (lane) {
          return h(MeterCard, {
            key: lane, lane: lane,
            label: (spend.labels || {})[lane],
            usd: (lanes[lane] || {}).usd || 0,
            cap: caps[lane],
            paused: !!pausedLanes[lane],
          });
        }),
        h("div", { className: "hm-card hm-grow" },
          h("div", { className: "hm-label" }, "Daily spend (last " + ((spend.history || []).length) + " days)"),
          h(DailyBars, { history: spend.history, labels: spend.labels })
        )
      ),
      h("div", { className: "hm-row" },
        h("div", { className: "hm-card hm-grow" },
          h("div", { className: "hm-label" }, "Top profiles today (estimated USD)"),
          topProfiles.length ? h(HBarList, { rows: topProfiles }) : h("div", { className: "hm-empty" }, "No spend recorded today yet.")
        ),
        (spend.pricing_gaps || []).length
          ? h("div", { className: "hm-card" },
              h("div", { className: "hm-label" }, "Unpriced models"),
              h("div", { className: "hm-num-warn" }, spend.pricing_gaps.join(", ")))
          : null
      ),

      h("h2", { className: "hm-section" }, "Fleet"),
      h("div", { className: "hm-row" },
        h(StatTile, { label: "Running", value: byStatus.running || 0 }),
        h(StatTile, { label: "Queued", value: (byStatus.ready || 0) + (byStatus.todo || 0) + (byStatus.triage || 0), hint: kanban.oldest_ready_age_seconds != null ? "oldest ready " + fmtDur(kanban.oldest_ready_age_seconds) : null }),
        h(StatTile, { label: "Blocked", value: byStatus.blocked || 0 }),
        h(StatTile, { label: "In review", value: byStatus.review || 0 }),
        h(StatTile, { label: "Done (" + (kanban.window_days || 14) + "d)", value: (kanban.throughput || []).reduce(function (a, d) { return a + d.done; }, 0) }),
        h(StatTile, { label: "Cycle time p50", value: fmtDur(cyc.p50), hint: "p90 " + fmtDur(cyc.p90) + " · n=" + (cyc.n || 0) })
      ),
      h("div", { className: "hm-row" },
        h("div", { className: "hm-card hm-grow" },
          h("div", { className: "hm-label" }, "Tasks completed per day"),
          h(ThroughputBars, { throughput: kanban.throughput })
        )
      ),
      h(RunningTable, { running: kanban.running }),

      h("h2", { className: "hm-section" }, "Agents (last " + ((data.agents && data.agents.window_days) || 14) + " days)"),
      h(AgentsTable, { agents: agents, refinements: data.refinements, usage: data.usage })
    );
  }

  window.__HERMES_PLUGINS__.register("hermes-metrics", MetricsPage);
})();
