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
  function Donut({
    segments,
    size = 120,
    thickness = 14,
    trackColor = "rgba(255,255,255,0.08)",
    centerLabel,
    className
  }) {
    const total = segments.reduce((sum, s) => sum + Math.max(0, s.value), 0);
    const radius = (size - thickness) / 2;
    const circumference = 2 * Math.PI * radius;
    const cx = size / 2;
    const cy = size / 2;
    let offset = 0;
    return /* @__PURE__ */ React.createElement("div", { className, style: { position: "relative", width: size, height: size } }, /* @__PURE__ */ React.createElement("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}` }, /* @__PURE__ */ React.createElement(
      "circle",
      {
        cx,
        cy,
        r: radius,
        fill: "none",
        stroke: trackColor,
        strokeWidth: thickness
      }
    ), total > 0 && segments.map((s, i) => {
      const v = Math.max(0, s.value);
      const length = v / total * circumference;
      const dash = `${length} ${circumference - length}`;
      const dashOffset = -offset;
      offset += length;
      return /* @__PURE__ */ React.createElement(
        "circle",
        {
          key: i,
          cx,
          cy,
          r: radius,
          fill: "none",
          stroke: s.color,
          strokeWidth: thickness,
          strokeDasharray: dash,
          strokeDashoffset: dashOffset,
          strokeLinecap: "butt",
          transform: `rotate(-90 ${cx} ${cy})`
        }
      );
    })), centerLabel != null && /* @__PURE__ */ React.createElement(
      "div",
      {
        style: {
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none"
        }
      },
      centerLabel
    ));
  }
  function MiniBars({
    values,
    width,
    height = 36,
    fill = "currentColor",
    gap = 2,
    className,
    ariaLabel
  }) {
    const max = Math.max(...values, 1);
    return /* @__PURE__ */ React.createElement(
      "div",
      {
        role: "img",
        "aria-label": ariaLabel,
        className,
        style: {
          display: "flex",
          alignItems: "flex-end",
          gap: `${gap}px`,
          width: width ? `${width}px` : "100%",
          height: `${height}px`
        }
      },
      values.map((v, i) => /* @__PURE__ */ React.createElement(
        "div",
        {
          key: i,
          style: {
            flex: 1,
            height: `${Math.max(2, v / max * 100)}%`,
            background: fill,
            borderRadius: "1px 1px 0 0",
            opacity: v === 0 ? 0.25 : 1
          }
        }
      ))
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

  function chatHref(sessionId) {
    const embedded = window.__HERMES_DASHBOARD_EMBEDDED_CHAT__;
    if (embedded) {
      return sessionId ? `/chat?resume=${encodeURIComponent(sessionId)}` : "/chat";
    }
    return "/kensei-console";
  }
  async function authedJson(path) {
    return SDK.fetchJSON(path);
  }
  async function fetchSessions() {
    const data = await authedJson("/api/sessions");
    return Array.isArray(data) ? data : data.sessions ?? [];
  }
  function formatNumber(n) {
    if (n < 1e3) return String(n);
    if (n < 1e6) return `${(n / 1e3).toFixed(1)}k`;
    if (n < 1e9) return `${(n / 1e6).toFixed(1)}M`;
    return `${(n / 1e9).toFixed(1)}B`;
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
  function shortId(id, n = 16) {
    return id.length > n ? id.slice(0, n) + "\u2026" : id;
  }
  function platformStateColor(state) {
    const s = state.toLowerCase();
    if (s === "connected" || s === "running" || s === "ok") return "var(--color-success, #5cf76e)";
    if (s === "connecting" || s === "starting" || s === "reconnecting") return "var(--color-warning, #f4a740)";
    if (s === "error" || s === "failed" || s === "disconnected") return "var(--color-destructive, #ff5470)";
    return "var(--color-muted-foreground, #888)";
  }
  function OverviewPanel() {
    const { useState, useEffect, useCallback, useMemo } = SDK.hooks;
    const { Card, CardHeader, CardTitle, CardContent, Button, Badge, Separator } = SDK.components;
    const [status, setStatus] = useState(null);
    const [analytics, setAnalytics] = useState(null);
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const refresh = useCallback(async () => {
      try {
        const [s, a, ss] = await Promise.all([
          authedJson("/api/status"),
          authedJson("/api/analytics/usage?days=7"),
          fetchSessions()
        ]);
        setStatus(s);
        setAnalytics(a);
        setSessions(ss);
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
    const today = useMemo(() => {
      if (!analytics?.daily?.length) return null;
      return analytics.daily[analytics.daily.length - 1];
    }, [analytics]);
    const sessionsSparkline = useMemo(
      () => analytics?.daily?.map((d) => d.sessions ?? 0) ?? [],
      [analytics]
    );
    const tokensSparkline = useMemo(
      () => analytics?.daily?.map((d) => (d.input_tokens || 0) + (d.output_tokens || 0)) ?? [],
      [analytics]
    );
    const activeSessions = useMemo(
      () => sessions.filter((s) => !s.ended_at).slice(0, 8),
      [sessions]
    );
    const modelDonut = useMemo(() => {
      const colors = ["#5e8fff", "#7ad7ff", "#5cf76e", "#f4a740", "#d94c56", "#a855f7"];
      return (analytics?.by_model || []).slice(0, 6).map((m, i) => ({
        label: m.model,
        value: m.sessions,
        color: colors[i % colors.length]
      }));
    }, [analytics]);
    const platforms = Object.entries(status?.gateway_platforms || {});
    const gatewayState = status?.gateway_state || "unknown";
    if (loading && !status) {
      return /* @__PURE__ */ React.createElement("div", { className: "p-6" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-xs text-muted-foreground" }, "Loading overview\u2026")));
    }
    return /* @__PURE__ */ React.createElement("div", { className: "p-6 space-y-4 max-w-7xl mx-auto" }, /* @__PURE__ */ React.createElement("header", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement(
      "h1",
      {
        className: "text-2xl font-semibold tracking-tight",
        style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
      },
      "Overview"
    ), /* @__PURE__ */ React.createElement("p", { className: "text-xs uppercase tracking-[0.18em] text-muted-foreground" }, "KENSEI Console \xB7 Live")), /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[10px] font-mono" }, "v", status?.version || "?"), /* @__PURE__ */ React.createElement(Button, { size: "sm", variant: "ghost", onClick: refresh, className: "text-xs h-7" }, "Refresh"))), error && /* @__PURE__ */ React.createElement(Card, { className: "border-destructive/50" }, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-sm text-destructive" }, error)), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-3" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-4" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "System Status"), /* @__PURE__ */ React.createElement(
      Badge,
      {
        variant: gatewayState === "running" ? "default" : "destructive",
        className: "text-[10px] uppercase"
      },
      gatewayState
    )), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-3 gap-4" }, /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Active sessions",
        value: status?.active_sessions ?? 0,
        sub: `${activeSessions.length} visible`
      }
    ), /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Sessions \xB7 7d",
        value: analytics?.totals?.total_sessions ?? 0
      }
    ), /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "API calls \xB7 7d",
        value: formatNumber(analytics?.totals?.total_api_calls ?? 0)
      }
    )), /* @__PURE__ */ React.createElement(Separator, null), /* @__PURE__ */ React.createElement("div", { className: "space-y-1" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.18em] text-muted-foreground" }, "Platforms"), /* @__PURE__ */ React.createElement("div", { className: "flex flex-wrap gap-2" }, platforms.length === 0 && /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "No platforms reporting."), platforms.map(([name, info]) => /* @__PURE__ */ React.createElement(
      "div",
      {
        key: name,
        className: "flex items-center gap-2 px-2 py-1 rounded-md border border-border text-xs"
      },
      /* @__PURE__ */ React.createElement(
        "span",
        {
          style: {
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: platformStateColor(info.state),
            boxShadow: `0 0 6px ${platformStateColor(info.state)}`
          }
        }
      ),
      /* @__PURE__ */ React.createElement("span", { className: "font-mono" }, name),
      /* @__PURE__ */ React.createElement("span", { className: "text-[10px] text-muted-foreground uppercase tracking-wider" }, info.state)
    )))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-3" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Today"), /* @__PURE__ */ React.createElement("div", { className: "space-y-3" }, /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Sessions",
        value: today?.sessions ?? 0
      }
    ), /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "API calls",
        value: formatNumber(today?.api_calls ?? 0)
      }
    ), /* @__PURE__ */ React.createElement(
      StatTile,
      {
        label: "Tokens in / out",
        value: /* @__PURE__ */ React.createElement("span", { className: "text-base" }, formatNumber(today?.input_tokens ?? 0), /* @__PURE__ */ React.createElement("span", { className: "text-muted-foreground" }, " / "), formatNumber(today?.output_tokens ?? 0))
      }
    ))))), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-3" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between mb-3" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Sessions \xB7 last 7 days"), /* @__PURE__ */ React.createElement("div", { className: "text-xl font-semibold tabular-nums" }, analytics?.totals?.total_sessions ?? 0)), /* @__PURE__ */ React.createElement("div", { style: { color: "var(--color-primary, #5e8fff)" } }, /* @__PURE__ */ React.createElement(
      Sparkline,
      {
        values: sessionsSparkline,
        width: 400,
        height: 56,
        strokeWidth: 2,
        fill: "currentColor",
        className: "w-full h-14"
      }
    )), /* @__PURE__ */ React.createElement("div", { className: "mt-2 flex justify-between text-[10px] text-muted-foreground font-mono" }, analytics?.daily?.map((d) => /* @__PURE__ */ React.createElement("span", { key: d.day }, d.day.slice(5)))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between mb-3" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Tokens \xB7 last 7 days"), /* @__PURE__ */ React.createElement("div", { className: "text-xl font-semibold tabular-nums" }, formatNumber(
      (analytics?.totals?.total_input ?? 0) + (analytics?.totals?.total_output ?? 0)
    ))), /* @__PURE__ */ React.createElement("div", { style: { color: "var(--color-accent, #7ad7ff)" } }, /* @__PURE__ */ React.createElement(
      MiniBars,
      {
        values: tokensSparkline,
        height: 56,
        fill: "currentColor",
        className: "w-full",
        ariaLabel: "tokens per day, 7d"
      }
    )), /* @__PURE__ */ React.createElement("div", { className: "mt-2 flex justify-between text-[10px] text-muted-foreground font-mono" }, analytics?.daily?.map((d) => /* @__PURE__ */ React.createElement("span", { key: d.day }, d.day.slice(5))))))), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-3" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-3" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Active sessions (", activeSessions.length, ")"), /* @__PURE__ */ React.createElement("a", { href: "/kensei-triage", className: "text-xs underline text-muted-foreground" }, "triage all \u2192")), activeSessions.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic py-3" }, "No active sessions. Engage from Pulse or Console.") : /* @__PURE__ */ React.createElement("div", { className: "flex flex-wrap gap-2" }, activeSessions.map((s) => /* @__PURE__ */ React.createElement(
      "a",
      {
        key: s.id,
        href: chatHref(s.id),
        className: "flex flex-col gap-0.5 px-3 py-2 rounded-md border border-border hover:bg-accent hover:text-accent-foreground transition-colors min-w-0",
        style: { maxWidth: 240 }
      },
      /* @__PURE__ */ React.createElement("div", { className: "text-[10px] font-mono text-muted-foreground truncate" }, shortId(s.id, 22)),
      /* @__PURE__ */ React.createElement("div", { className: "text-xs truncate" }, s.title || s.source || "untitled"),
      /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground" }, formatRelative(s.started_at), " \xB7 ", s.message_count, " msg")
    ))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground mb-3" }, "Sessions by model \xB7 7d"), modelDonut.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "No data.") : /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-4" }, /* @__PURE__ */ React.createElement(
      Donut,
      {
        segments: modelDonut,
        size: 120,
        thickness: 14,
        centerLabel: /* @__PURE__ */ React.createElement("div", { className: "text-center" }, /* @__PURE__ */ React.createElement("div", { className: "text-lg font-semibold tabular-nums" }, analytics?.totals?.total_sessions ?? 0), /* @__PURE__ */ React.createElement("div", { className: "text-[9px] uppercase tracking-wider text-muted-foreground" }, "total"))
      }
    ), /* @__PURE__ */ React.createElement("div", { className: "flex-1 space-y-1.5 min-w-0" }, modelDonut.map((s) => /* @__PURE__ */ React.createElement("div", { key: s.label, className: "flex items-center gap-2 text-xs" }, /* @__PURE__ */ React.createElement(
      "span",
      {
        style: {
          width: 8,
          height: 8,
          borderRadius: "2px",
          background: s.color,
          flexShrink: 0
        }
      }
    ), /* @__PURE__ */ React.createElement("span", { className: "font-mono truncate flex-1 min-w-0" }, s.label), /* @__PURE__ */ React.createElement("span", { className: "tabular-nums text-muted-foreground" }, s.value)))))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground mb-3" }, "Quick links"), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-2" }, [
      { label: "Triage", path: "/kensei-triage", desc: "failures \xB7 long-runners" },
      { label: "Pulse", path: "/kensei-pulse", desc: "per-profile activity" },
      { label: "Postiz", path: "/kensei-postiz", desc: "social posts" },
      { label: "Console", path: "/kensei-console", desc: "embedded workspace" }
    ].map((q) => /* @__PURE__ */ React.createElement(
      "a",
      {
        key: q.path,
        href: q.path,
        className: "flex flex-col gap-0.5 px-3 py-2.5 rounded-md border border-border hover:bg-accent hover:text-accent-foreground transition-colors"
      },
      /* @__PURE__ */ React.createElement(
        "div",
        {
          className: "text-sm font-semibold",
          style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
        },
        q.label
      ),
      /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground" }, q.desc)
    ))))));
  }
  window.__HERMES_PLUGINS__.register("kensei-overview", OverviewPanel);
})();
})();
