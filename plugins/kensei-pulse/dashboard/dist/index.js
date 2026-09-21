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
  async function listProfiles() {
    const data = await authedJson("/api/profiles");
    return Array.isArray(data) ? data : data.profiles ?? [];
  }
  async function listSessions() {
    const data = await authedJson("/api/sessions");
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
  function within24h(iso) {
    if (!iso) return false;
    return Date.now() - new Date(iso).getTime() < 24 * 60 * 60 * 1e3;
  }
  function sessionsPerHour(sessions) {
    const buckets = Array(24).fill(0);
    const now = Date.now();
    for (const s of sessions) {
      const t = new Date(s.started_at).getTime();
      const hoursAgo = Math.floor((now - t) / (60 * 60 * 1e3));
      if (hoursAgo < 0 || hoursAgo > 23) continue;
      buckets[23 - hoursAgo] += 1;
    }
    return buckets;
  }
  function buildLanes(profiles, sessions) {
    return profiles.map((p) => {
      const matches = sessions.filter((s) => (s.model || "") === (p.model || ""));
      const active = matches.filter((s) => !s.ended_at);
      const current = active.sort(
        (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
      )[0] || null;
      const ended = matches.filter((s) => s.ended_at).sort((a, b) => new Date(b.ended_at).getTime() - new Date(a.ended_at).getTime());
      const lastSession = current || ended[0] || null;
      const completed24h = ended.filter((s) => within24h(s.ended_at)).length;
      return {
        profile: p,
        current,
        lastSession,
        activeCount: active.length,
        completed24h,
        totalRecent: matches.length,
        sparkline: sessionsPerHour(matches)
      };
    });
  }
  function CurrentTaskBlock({ session, label }) {
    if (!session) {
      return /* @__PURE__ */ React.createElement("div", { className: "rounded-md border border-border bg-muted/20 p-2 text-[11px] text-muted-foreground italic" }, "No recent activity.");
    }
    return /* @__PURE__ */ React.createElement("div", { className: "rounded-md border border-border p-2 space-y-1" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-center justify-between gap-2" }, /* @__PURE__ */ React.createElement("span", { className: "text-[10px] uppercase tracking-wider text-muted-foreground" }, label), /* @__PURE__ */ React.createElement("span", { className: "text-[10px] text-muted-foreground" }, formatRelative(session.ended_at || session.started_at))), /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "text-xs leading-snug",
        style: {
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden"
        }
      },
      session.title || session.source || "untitled session"
    ), /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground" }, session.message_count, " msg \xB7 ", session.tool_call_count, " tool"));
  }
  function HeroProfileCard({ lane }) {
    const { Card, CardContent, Badge, Button, Separator } = SDK.components;
    const { profile, current, lastSession, activeCount, completed24h, sparkline } = lane;
    const showSession = current || lastSession;
    const sessionLabel = current ? "Running" : "Last task";
    return /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-4" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ React.createElement(
      "h2",
      {
        className: "text-lg font-semibold capitalize tracking-tight",
        style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
      },
      profile.name
    ), profile.is_default && /* @__PURE__ */ React.createElement(Badge, { variant: "default", className: "text-[10px]" }, "default"), current && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[10px] uppercase" }, "live")), /* @__PURE__ */ React.createElement("div", { className: "text-[11px] text-muted-foreground font-mono mt-0.5" }, profile.model || "\u2014", " \xB7 ", profile.provider || "\u2014"))), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-3 gap-4" }, /* @__PURE__ */ React.createElement(StatTile, { label: "Active", value: activeCount }), /* @__PURE__ */ React.createElement(StatTile, { label: "Done \xB7 24h", value: completed24h }), /* @__PURE__ */ React.createElement(StatTile, { label: "Skills", value: profile.skill_count ?? 0 })), /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between mb-1" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.18em] text-muted-foreground" }, "Activity \xB7 24h"), /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground tabular-nums" }, sparkline.reduce((a, b) => a + b, 0), " sessions")), /* @__PURE__ */ React.createElement("div", { style: { color: "var(--color-primary, #5e8fff)" } }, /* @__PURE__ */ React.createElement(
      Sparkline,
      {
        values: sparkline,
        width: 400,
        height: 48,
        strokeWidth: 1.5,
        fill: "currentColor",
        className: "w-full h-12"
      }
    ))), /* @__PURE__ */ React.createElement(Separator, null), /* @__PURE__ */ React.createElement(CurrentTaskBlock, { session: showSession, label: sessionLabel }), /* @__PURE__ */ React.createElement("div", { className: "flex gap-2" }, current && /* @__PURE__ */ React.createElement(
      Button,
      {
        size: "sm",
        variant: "outline",
        className: "flex-1 text-xs h-7",
        onClick: () => {
          window.location.href = chatHref(current.id);
        }
      },
      "Resume"
    ), /* @__PURE__ */ React.createElement(
      Button,
      {
        size: "sm",
        variant: "default",
        className: "flex-1 text-xs h-7",
        onClick: () => {
          window.location.href = chatHref();
        }
      },
      "Engage"
    ))));
  }
  function CompactProfileCard({ lane }) {
    const { Card, CardContent, Badge, Button } = SDK.components;
    const { profile, current, lastSession, activeCount, completed24h, sparkline } = lane;
    return /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 space-y-3" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-center justify-between gap-2 flex-wrap" }, /* @__PURE__ */ React.createElement("div", { className: "min-w-0" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ React.createElement("span", { className: "text-sm font-semibold capitalize truncate" }, profile.name), current && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[9px]" }, "live")), /* @__PURE__ */ React.createElement("div", { className: "text-[10px] text-muted-foreground font-mono truncate" }, profile.model || "\u2014"))), /* @__PURE__ */ React.createElement("div", { style: { color: "var(--color-primary, #5e8fff)" } }, /* @__PURE__ */ React.createElement(
      Sparkline,
      {
        values: sparkline,
        width: 200,
        height: 28,
        strokeWidth: 1.25,
        fill: "currentColor",
        className: "w-full h-7"
      }
    )), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-2 gap-1 text-[10px] font-mono" }, /* @__PURE__ */ React.createElement("div", { className: "flex justify-between" }, /* @__PURE__ */ React.createElement("span", { className: "text-muted-foreground" }, "active"), /* @__PURE__ */ React.createElement("span", { className: "tabular-nums" }, activeCount)), /* @__PURE__ */ React.createElement("div", { className: "flex justify-between" }, /* @__PURE__ */ React.createElement("span", { className: "text-muted-foreground" }, "done\xB724h"), /* @__PURE__ */ React.createElement("span", { className: "tabular-nums" }, completed24h))), (current || lastSession) && /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "text-[10px] text-muted-foreground italic",
        style: {
          display: "-webkit-box",
          WebkitLineClamp: 1,
          WebkitBoxOrient: "vertical",
          overflow: "hidden"
        }
      },
      (current || lastSession)?.title || (current || lastSession)?.source || "\u2014"
    ), /* @__PURE__ */ React.createElement("div", { className: "flex gap-1.5" }, current && /* @__PURE__ */ React.createElement(
      Button,
      {
        size: "sm",
        variant: "outline",
        className: "flex-1 text-[10px] h-6 px-2",
        onClick: () => {
          window.location.href = chatHref(current.id);
        }
      },
      "Resume"
    ), /* @__PURE__ */ React.createElement(
      Button,
      {
        size: "sm",
        variant: "default",
        className: "flex-1 text-[10px] h-6 px-2",
        onClick: () => {
          window.location.href = chatHref();
        }
      },
      "Engage"
    ))));
  }
  function PulsePanel() {
    const { useState, useEffect, useCallback, useMemo } = SDK.hooks;
    const { Card, CardContent } = SDK.components;
    const [profiles, setProfiles] = useState([]);
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const refresh = useCallback(async () => {
      try {
        const [p, s] = await Promise.all([listProfiles(), listSessions()]);
        setProfiles(p);
        setSessions(s);
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
    const lanes = useMemo(() => buildLanes(profiles, sessions), [profiles, sessions]);
    const heroLane = useMemo(
      () => lanes.find((l) => l.profile.is_default) || lanes[0] || null,
      [lanes]
    );
    const otherLanes = useMemo(
      () => lanes.filter((l) => l !== heroLane),
      [lanes, heroLane]
    );
    return /* @__PURE__ */ React.createElement("div", { className: "p-6 space-y-4 max-w-7xl mx-auto" }, /* @__PURE__ */ React.createElement("header", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement(
      "h1",
      {
        className: "text-2xl font-semibold tracking-tight",
        style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
      },
      "Pulse"
    ), /* @__PURE__ */ React.createElement("p", { className: "text-xs uppercase tracking-[0.18em] text-muted-foreground" }, profiles.length, " profiles \xB7 ", sessions.filter((s) => !s.ended_at).length, " active"))), error && /* @__PURE__ */ React.createElement(Card, { className: "border-destructive/50" }, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-sm text-destructive" }, error)), loading && profiles.length === 0 ? /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-xs text-muted-foreground" }, "Loading profiles\u2026")) : /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-3" }, heroLane && /* @__PURE__ */ React.createElement(HeroProfileCard, { lane: heroLane }), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 gap-3" }, otherLanes.slice(0, 2).map((lane) => /* @__PURE__ */ React.createElement(CompactProfileCard, { key: lane.profile.name, lane })))), otherLanes.length > 2 && /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3" }, otherLanes.slice(2).map((lane) => /* @__PURE__ */ React.createElement(CompactProfileCard, { key: lane.profile.name, lane })))));
  }
  window.__HERMES_PLUGINS__.register("kensei-pulse", PulsePanel);
})();
})();
