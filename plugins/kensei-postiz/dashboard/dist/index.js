(function(){"use strict";var SDK=window.__HERMES_PLUGIN_SDK__;var React=SDK&&SDK.React;if(!SDK||!React||!window.__HERMES_PLUGINS__){return;}
(() => {
  // ../lib/src/charts.tsx
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
  var PLUGIN_BASE = "/api/plugins/kensei-postiz";
  var STORAGE_KEY = "kensei-mc-config";
  function loadCreds() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        return {
          postizEmail: parsed.postizEmail || "",
          postizPassword: parsed.postizPassword || ""
        };
      }
    } catch {
    }
    return { postizEmail: "", postizPassword: "" };
  }
  async function pluginFetch(path, init = {}) {
    const headers = new Headers(init.headers);
    if (init.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    return SDK.authedFetch(`${PLUGIN_BASE}${path}`, { ...init, headers });
  }
  async function loginPostiz(creds) {
    const r = await pluginFetch("/login", {
      method: "POST",
      body: JSON.stringify({
        email: creds.postizEmail,
        password: creds.postizPassword,
        provider: "LOCAL"
      })
    });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`login failed (${r.status}): ${text || "no body"}`);
    }
  }
  async function fetchJson(path, init = {}) {
    const r = await pluginFetch(path, init);
    if (!r.ok) {
      const text2 = await r.text().catch(() => "");
      throw new Error(`${path} \u2192 ${r.status}${text2 ? `: ${text2.slice(0, 200)}` : ""}`);
    }
    const text = await r.text();
    if (!text) return [];
    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`${path}: non-JSON response`);
    }
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
  function isUpcoming(p) {
    if (!p.publishDate) return false;
    return new Date(p.publishDate).getTime() > Date.now();
  }
  function platformColor(provider, idx) {
    const map = {
      twitter: "#1da1f2",
      x: "#1da1f2",
      instagram: "#e1306c",
      facebook: "#1877f2",
      linkedin: "#0a66c2",
      tiktok: "#69c9d0",
      youtube: "#ff0000",
      threads: "#cccccc",
      bluesky: "#1185fe",
      mastodon: "#6364ff",
      reddit: "#ff4500",
      pinterest: "#e60023",
      discord: "#5865f2"
    };
    if (provider && map[provider.toLowerCase()]) return map[provider.toLowerCase()];
    const fallback = ["#5e8fff", "#7ad7ff", "#5cf76e", "#f4a740", "#d94c56", "#a855f7"];
    return fallback[idx % fallback.length];
  }
  function PostizPanel() {
    const { useState, useEffect, useCallback, useMemo } = SDK.hooks;
    const { Card, CardHeader, CardTitle, CardContent, Button, Badge, Separator } = SDK.components;
    const [creds] = useState(loadCreds);
    const [authState, setAuthState] = useState(
      creds.postizEmail && creds.postizPassword ? "unknown" : "no-creds"
    );
    const [authError, setAuthError] = useState(null);
    const [posts, setPosts] = useState([]);
    const [integrations, setIntegrations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const ensureAuth = useCallback(async () => {
      if (!creds.postizEmail || !creds.postizPassword) {
        setAuthState("no-creds");
        return false;
      }
      setAuthState("authenticating");
      try {
        await loginPostiz(creds);
        setAuthState("ok");
        setAuthError(null);
        return true;
      } catch (e) {
        setAuthError(e instanceof Error ? e.message : String(e));
        setAuthState("error");
        return false;
      }
    }, [creds]);
    const refresh = useCallback(async () => {
      setLoading(true);
      setError(null);
      try {
        const ok = authState === "ok" || await ensureAuth();
        if (!ok) return;
        const [postsResp, integsResp] = await Promise.all([
          fetchJson("/posts"),
          fetchJson("/integrations")
        ]);
        const postArr = Array.isArray(postsResp) ? postsResp : postsResp.posts ?? [];
        const intArr = Array.isArray(integsResp) ? integsResp : integsResp.integrations ?? [];
        setPosts(postArr);
        setIntegrations(intArr);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }, [authState, ensureAuth]);
    useEffect(() => {
      refresh();
    }, []);
    const counts = useMemo(() => {
      let scheduled = 0;
      let published = 0;
      let drafts = 0;
      let other = 0;
      for (const p of posts) {
        const state = (p.state || "").toUpperCase();
        if (state === "QUEUE" || state === "SCHEDULED") scheduled++;
        else if (state === "PUBLISHED" || state === "SENT") published++;
        else if (state === "DRAFT") drafts++;
        else other++;
      }
      return { scheduled, published, drafts, other, total: posts.length };
    }, [posts]);
    const platformDonut = useMemo(() => {
      const tally = {};
      for (const p of posts) {
        const k = p.integration?.providerIdentifier || "unknown";
        tally[k] = (tally[k] || 0) + 1;
      }
      const entries = Object.entries(tally).sort((a, b) => b[1] - a[1]);
      return entries.map(([label, value], i) => ({
        label,
        value,
        color: platformColor(label, i)
      }));
    }, [posts]);
    const upcoming = useMemo(
      () => posts.filter(isUpcoming).sort((a, b) => new Date(a.publishDate).getTime() - new Date(b.publishDate).getTime()).slice(0, 10),
      [posts]
    );
    const recent = useMemo(
      () => posts.filter((p) => !isUpcoming(p)).sort(
        (a, b) => new Date(b.publishDate || b.createdAt || 0).getTime() - new Date(a.publishDate || a.createdAt || 0).getTime()
      ).slice(0, 12),
      [posts]
    );
    const onDelete = async (id) => {
      try {
        const r = await pluginFetch(`/posts/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`delete failed: ${r.status}`);
        setPosts((prev) => prev.filter((p) => p.id !== id));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    const noCreds = authState === "no-creds";
    return /* @__PURE__ */ React.createElement("div", { className: "p-6 space-y-4 max-w-7xl mx-auto" }, /* @__PURE__ */ React.createElement("header", { className: "flex items-baseline justify-between flex-wrap gap-2" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement(
      "h1",
      {
        className: "text-2xl font-semibold tracking-tight",
        style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
      },
      "Postiz"
    ), /* @__PURE__ */ React.createElement("p", { className: "text-xs uppercase tracking-[0.18em] text-muted-foreground" }, integrations.length, " platforms \xB7 ", posts.length, " posts in window")), /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2" }, authState === "ok" && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[10px]" }, "Connected"), authState === "authenticating" && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[10px]" }, "Authenticating\u2026"), authState === "error" && /* @__PURE__ */ React.createElement(Badge, { variant: "destructive", className: "text-[10px]" }, "Auth error"), authState === "no-creds" && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[10px]" }, "No creds"), /* @__PURE__ */ React.createElement(Button, { size: "sm", variant: "ghost", onClick: refresh, disabled: loading, className: "text-xs h-7" }, "Refresh"))), noCreds && /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-sm" }, "Postiz credentials not set. Open the gear icon in the dashboard header to set them.")), authState === "error" && /* @__PURE__ */ React.createElement(Card, { className: "border-destructive/50" }, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-sm space-y-1" }, /* @__PURE__ */ React.createElement("div", { className: "font-medium text-destructive" }, "Authentication failed"), /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground font-mono break-all" }, authError))), error && authState === "ok" && /* @__PURE__ */ React.createElement(Card, { className: "border-destructive/50" }, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4 text-sm text-destructive" }, error)), authState === "ok" && /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-3" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground mb-3" }, "Pipeline"), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-2 sm:grid-cols-4 gap-4" }, /* @__PURE__ */ React.createElement(StatTile, { label: "Scheduled", value: counts.scheduled }), /* @__PURE__ */ React.createElement(StatTile, { label: "Published", value: counts.published }), /* @__PURE__ */ React.createElement(StatTile, { label: "Drafts", value: counts.drafts }), /* @__PURE__ */ React.createElement(StatTile, { label: "Other", value: counts.other, sub: `${counts.total} total` })))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground mb-3" }, "By platform"), platformDonut.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "No posts to chart.") : /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-4" }, /* @__PURE__ */ React.createElement(
      Donut,
      {
        segments: platformDonut,
        size: 104,
        thickness: 12,
        centerLabel: /* @__PURE__ */ React.createElement("div", { className: "text-center" }, /* @__PURE__ */ React.createElement("div", { className: "text-base font-semibold tabular-nums" }, counts.total), /* @__PURE__ */ React.createElement("div", { className: "text-[9px] uppercase tracking-wider text-muted-foreground" }, "posts"))
      }
    ), /* @__PURE__ */ React.createElement("div", { className: "flex-1 space-y-1 min-w-0" }, platformDonut.slice(0, 5).map((s) => /* @__PURE__ */ React.createElement("div", { key: s.label, className: "flex items-center gap-2 text-xs" }, /* @__PURE__ */ React.createElement(
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
    ), /* @__PURE__ */ React.createElement("span", { className: "truncate flex-1 min-w-0" }, s.label), /* @__PURE__ */ React.createElement("span", { className: "tabular-nums text-muted-foreground" }, s.value)))))))), /* @__PURE__ */ React.createElement("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-3" }, /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-3" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Upcoming (", upcoming.length, ")")), upcoming.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "Nothing scheduled.") : /* @__PURE__ */ React.createElement("div", { className: "space-y-2" }, upcoming.map((p) => /* @__PURE__ */ React.createElement("div", { key: p.id, className: "rounded-md border border-border p-2 space-y-1" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2 flex-wrap" }, p.state && /* @__PURE__ */ React.createElement(Badge, { variant: "secondary", className: "text-[9px]" }, p.state), p.integration?.providerIdentifier && /* @__PURE__ */ React.createElement(
      Badge,
      {
        variant: "outline",
        className: "text-[9px]",
        style: {
          color: platformColor(p.integration.providerIdentifier, 0),
          borderColor: platformColor(p.integration.providerIdentifier, 0)
        }
      },
      p.integration.providerIdentifier
    ), /* @__PURE__ */ React.createElement("span", { className: "text-[10px] text-muted-foreground ml-auto" }, p.publishDate ? new Date(p.publishDate).toLocaleString() : "")), /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "text-xs whitespace-pre-wrap",
        style: {
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden"
        }
      },
      p.content
    )))))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-5 space-y-3" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-baseline justify-between" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground" }, "Recent (", recent.length, ")")), recent.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "No posts yet.") : /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "space-y-2",
        style: { maxHeight: 480, overflowY: "auto" }
      },
      recent.map((p) => /* @__PURE__ */ React.createElement("div", { key: p.id, className: "rounded-md border border-border p-2 space-y-1" }, /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2 flex-wrap" }, p.state && /* @__PURE__ */ React.createElement(Badge, { variant: "secondary", className: "text-[9px]" }, p.state), p.integration?.providerIdentifier && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[9px]" }, p.integration.providerIdentifier), /* @__PURE__ */ React.createElement("span", { className: "text-[10px] text-muted-foreground ml-auto" }, formatRelative(p.publishDate || p.createdAt)), /* @__PURE__ */ React.createElement(
        Button,
        {
          size: "sm",
          variant: "ghost",
          className: "text-[10px] h-5 px-1.5",
          onClick: () => onDelete(p.id)
        },
        "del"
      )), /* @__PURE__ */ React.createElement(
        "div",
        {
          className: "text-xs whitespace-pre-wrap",
          style: {
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden"
          }
        },
        p.content
      )))
    )))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardContent, { className: "p-4" }, /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground mb-3" }, "Integrations (", integrations.length, ")"), integrations.length === 0 ? /* @__PURE__ */ React.createElement("div", { className: "text-xs text-muted-foreground italic" }, "None connected.") : /* @__PURE__ */ React.createElement("div", { className: "flex flex-wrap gap-2" }, integrations.map((i, idx) => /* @__PURE__ */ React.createElement(
      "div",
      {
        key: i.id,
        className: "flex items-center gap-2 px-2.5 py-1 rounded-md border border-border text-xs"
      },
      /* @__PURE__ */ React.createElement(
        "span",
        {
          style: {
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: platformColor(i.providerIdentifier, idx),
            opacity: i.disabled ? 0.4 : 1
          }
        }
      ),
      /* @__PURE__ */ React.createElement("span", { className: "truncate", style: { maxWidth: 180 } }, i.name),
      /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "text-[9px]" }, i.providerIdentifier || "\u2014")
    )))))));
  }
  window.__HERMES_PLUGINS__.register("kensei-postiz", PostizPanel);
})();
})();
