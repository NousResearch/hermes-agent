(function(){"use strict";var SDK=window.__HERMES_PLUGIN_SDK__;var React=SDK&&SDK.React;if(!SDK||!React||!window.__HERMES_PLUGINS__){return;}
(() => {
  // ../lib/src/charts.tsx
  function Sparkline({
    values,
    width = 120,
    height = 32,
    stroke = "currentColor",
    fill = "none",
    strokeWidth = 1.5,
    className
  }) {
    if (values.length === 0) {
      return /* @__PURE__ */ React.createElement("svg", { width, height, viewBox: `0 0 ${width} ${height}`, className });
    }
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    const stepX = values.length > 1 ? width / (values.length - 1) : 0;
    const points = values.map((v, i) => {
      const x = i * stepX;
      const y = height - (v - min) / range * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    const areaPoints = `0,${height} ${points} ${width},${height}`;
    return /* @__PURE__ */ React.createElement(
      "svg",
      {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        className,
        preserveAspectRatio: "none"
      },
      fill !== "none" && /* @__PURE__ */ React.createElement("polygon", { points: areaPoints, fill, opacity: 0.4 }),
      /* @__PURE__ */ React.createElement(
        "polyline",
        {
          points,
          fill: "none",
          stroke,
          strokeWidth,
          strokeLinecap: "round",
          strokeLinejoin: "round",
          vectorEffect: "non-scaling-stroke"
        }
      )
    );
  }
  function StatTile({ label, value, sub, trend, className }) {
    return /* @__PURE__ */ React.createElement(
      "div",
      {
        className,
        style: { display: "flex", flexDirection: "column", gap: "2px", minWidth: 0 }
      },
      /* @__PURE__ */ React.createElement(
        "div",
        {
          className: "text-[10px] uppercase tracking-[0.18em] text-muted-foreground",
          style: { fontFamily: "var(--theme-font-mono, monospace)" }
        },
        label
      ),
      /* @__PURE__ */ React.createElement(
        "div",
        {
          className: "text-2xl font-semibold tabular-nums",
          style: { lineHeight: 1.1 }
        },
        value
      ),
      sub && /* @__PURE__ */ React.createElement("div", { className: "text-[11px] text-muted-foreground" }, trend === "up" && "\u25B2 ", trend === "down" && "\u25BC ", sub)
    );
  }

  // src/index.tsx
  var FAILURE_REASONS = /* @__PURE__ */ new Set([
    "error",
    "failed",
    "failure",
    "timeout",
    "cancelled",
    "canceled",
    "interrupted",
    "killed"
  ]);
  var LONG_RUNNING_THRESHOLD_MS = 2 * 60 * 60 * 1e3;
  var COMPLETED_LIMIT = 50;

  function chatHref(sessionId) {
    const embedded = window.__HERMES_DASHBOARD_EMBEDDED_CHAT__;
    if (embedded) {
      return sessionId ? `/chat?resume=${encodeURIComponent(sessionId)}` : "/chat";
    }
    return "/kensei-console";
  }
  async function fetchSessions() {
    const data = await SDK.fetchJSON("/api/sessions");
    return Array.isArray(data) ? data : data.sessions ?? [];
  }
  function formatRelative(iso) {
    if (!iso) return "";
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 6e4);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }
  function formatDuration(startIso, endIso) {
    const start = new Date(startIso).getTime();
    const end = endIso ? new Date(endIso).getTime() : Date.now();
    const mins = Math.floor((end - start) / 6e4);
    if (mins < 60) return `${mins}m`;
    const hours = Math.floor(mins / 60);
    const remMins = mins % 60;
    return `${hours}h${remMins > 0 ? ` ${remMins}m` : ""}`;
  }
  function shortId(id, n = 16) {
    return id.length > n ? id.slice(0, n) + "\u2026" : id;
  }
  function isFailure(s) {
    return FAILURE_REASONS.has((s.end_reason || "").toLowerCase());
  }
  function isStale(s) {
    if (s.ended_at) return false;
    return Date.now() - new Date(s.started_at).getTime() > LONG_RUNNING_THRESHOLD_MS;
  }
  function within(iso, ms) {
    if (!iso) return false;
    return Date.now() - new Date(iso).getTime() < ms;
  }
  function failuresPerDay(sessions) {
    const days = Array(7).fill(0);
    const now = Date.now();
    for (const s of sessions) {
      if (!isFailure(s) || !s.ended_at) continue;
      const t = new Date(s.ended_at).getTime();
      const ageDays = Math.floor((now - t) / (24 * 60 * 60 * 1e3));
      if (ageDays < 0 || ageDays > 6) continue;
      days[6 - ageDays] += 1;
    }
    return days;
  }
  function endReasonVariant(reason) {
    switch ((reason || "").toLowerCase()) {
      case "completed":
      case "done":
        return "default";
      case "error":
      case "failed":
      case "timeout":
        return "destructive";
      case "cancelled":
      case "canceled":
      case "interrupted":
        return "outline";
      default:
        return "secondary";
    }
  }
  function CompactSessionRow({ s, mode }) {
    const { Badge, Button } = SDK.components;
    const variant = mode === "failure" ? "destructive" : "outline";
    const variantLabel = mode === "failure" ? s.end_reason || "failed" : `${formatDuration(s.started_at)} running`;
    return /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2 py-1.5 border-b border-border/40 last:border-0 min-w-0" }, /* @__PURE__ */ React.createElement(Badge, { variant, className: "text-[10px] shrink-0" }, variantLabel), /* @__PURE__ */ React.createElement("span", { className: "text-[10px] font-mono text-muted-foreground shrink-0" }, shortId(s.id, 18)), /* @__PURE__ */ React.createElement("span", { className: "text-xs truncate flex-1 min-w-0" }, s.title || s.source || "untitled"), /* @__PURE__ */ React.createElement("span", { className: "text-[10px] text-muted-foreground shrink-0" }, formatRelative(mode === "stale" ? s.started_at : s.ended_at)), /* @__PURE__ */ React.createElement(
      Button,
      {
        size: "sm",
        variant: "ghost",
        className: "text-xs h-6 px-2 shrink-0",
        onClick: () => {
          window.location.href = chatHref(s.id);
        }
      },
      "open"
    ));
  }
  function TriagePanel() {
    const { useState, useEffect, useCallback, useMemo } = SDK.hooks;
    const { Card, CardContent, Button, Badge } = SDK.components;
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const refresh = useCallback(async () => {
      try {
        const list = await fetchSessions();
        setSessions(list);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }, []);
    useEffect(() => {
      refresh();
      const interval = setInterval(refresh, 3e4);
      return () => clearInterval(interval);
    }, [refresh]);
    const failures24h = useMemo(
      () => sessions.filter((s) => isFailure(s) && within(s.ended_at, 24 * 60 * 60 * 1e3)).sort((a, b) => new Date(b.ended_at).getTime() - new Date(a.ended_at).getTime()),
      [sessions]
    );
    const stale = useMemo(
      () => sessions.filter(isStale).sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime()),
      [sessions]
    );
    const completed = useMemo(
      () => sessions.filter((s) => s.ended_at && !isFailure(s)).sort((a, b) => new Date(b.ended_at).getTime() - new Date(a.ended_at).getTime()).slice(0, COMPLETED_LIMIT),
      [sessions]
    );
    const failuresSpark = useMemo(() => failuresPerDay(sessions), [sessions]);
    const failures7d = useMemo(
      () => sessions.filter((s) => isFailure(s) && within(s.ended_at, 7 * 24 * 60 * 60 * 1e3)).length,
      [sessions]
    );
    const attentionTotal = failures24h.length + stale.length;
    return /* @__PURE__ */ React.createElement("div", { className: "p-6 space-y-4 max-w-7xl mx-auto" }, /* @__PURE__ */ React.createElement("header", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement(
      "h1",
      {
        className: "text-2xl font-semibold tracking-tight",
        style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
      },
      "Triage"
    ), /* @__PURE__ */ React.createElement("p", { className: "text-xs uppercase tracking-[0.18em] text-muted-foreground" }, attentionTotal, " need attention \xB7 ", completed.length, " recently completed")), /* @__PURE__ */ React.createElement(Button, { size: "sm", variant: "ghost", onClick: refresh, className: "text-xs h-7" }, "Refresh")), error && /* @__PURE__ */ React.createElement(Card, { className: "border-destructive/50" }, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-sm text-destructive" }, error)), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-3" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-4" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Needs attention"), attentionTotal > 0 && /* @__PURE__ */ React.createElement(
      Badge,
      {
        variant: failures24h.length > 0 ? "destructive" : "outline",
        className: "text-[10px] uppercase"
      },
      failures24h.length > 0 ? "action required" : "review"
    )), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-3 gap-4" }, /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Failures \xB7 24h",
        value: failures24h.length,
        sub: failures24h.length === 0 ? "all clear" : void 0
      }
    ), /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Long runners \xB7 2h+",
        value: stale.length,
        sub: stale[0] ? `oldest ${formatDuration(stale[0].started_at)}` : void 0
      }
    ), /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Failures \xB7 7d",
        value: failures7d
      }
    )), attentionTotal === 0 ? /* @__PURE__ */ React.createElement("div", { className: "rounded-md border border-border bg-muted/20 p-3 text-xs text-muted-foreground italic" }, "No failed sessions in 24h. No long-runners over 2h. Quiet.") : /* @__PURE__ */ React.createElement("div", { className: "rounded-md border border-border" }, failures24h.map((s) => /* @__PURE__ */ React.createElement(CompactSessionRow, { key: s.id, s, mode: "failure" })), stale.map((s) => /* @__PURE__ */ React.createElement(CompactSessionRow, { key: s.id, s, mode: "stale" }))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between mb-3" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Failures \xB7 7 days"), /* @__PURE__ */ React.createElement("div", { className: "text-xl font-semibold tabular-nums" }, failures7d)), /* @__PURE__ */ React.createElement("div", { style: { color: "var(--color-destructive, #d94c56)" } }, /* @__PURE__ */ React.createElement(
      Sparkline,
      {
        values: failuresSpark,
        width: 300,
        height: 48,
        strokeWidth: 2,
        fill: "currentColor",
        className: "w-full h-12"
      }
    )), /* @__PURE__ */ React.createElement("div", { className: "mt-2 flex justify-between text-[10px] text-muted-foreground font-mono" }, Array.from({ length: 7 }).map((_, i) => {
      const d = /* @__PURE__ */ new Date();
      d.setDate(d.getDate() - (6 - i));
      return /* @__PURE__ */ React.createElement("span", { key: i }, `${d.getMonth() + 1}-${d.getDate()}`);
    }))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-3" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Recently completed (", completed.length, ")"), loading && completed.length === 0 && /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground" }, "Loading\u2026")), completed.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "No completed sessions yet.") : /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "flex gap-2 overflow-x-auto pb-2",
        style: { scrollbarWidth: "thin" }
      },
      completed.map((s) => /* @__PURE__ */ React.createElement(
        "a",
        {
          key: s.id,
          href: chatHref(s.id),
          className: "flex flex-col gap-0.5 px-3 py-2 rounded-md border border-border hover:bg-accent hover:text-accent-foreground transition-colors shrink-0 min-w-[180px] max-w-[220px]"
        },
        /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ React.createElement(Badge, { variant: endReasonVariant(s.end_reason), className: "text-[9px]" }, s.end_reason || "ended"), /* @__PURE__ */ React.createElement("span", { className: "text-[10px] font-mono text-muted-foreground truncate" }, shortId(s.id, 14))),
        /* @__PURE__ */ React.createElement("div", { className: "text-xs truncate" }, s.title || s.source || "untitled"),
        /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground" }, formatRelative(s.ended_at), " \xB7 ", s.model || "?")
      ))
    ))));
  }
  window.__HERMES_PLUGINS__.register("kensei-triage", TriagePanel);
})();
})();
