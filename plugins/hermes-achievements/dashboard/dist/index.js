(function () {
  "use strict";
  // hermes-achievements dashboard plugin
  // Originally authored by @PCinkusz — https://github.com/PCinkusz/hermes-achievements (MIT).
  // Bundled into hermes-agent. Upstream repo remains the staging ground for new
  // badges and UI iteration; the in-progress scan banner below is a small addition
  // layered on top of the original dist bundle.
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const hooks = SDK.hooks;
  const C = SDK.components;
  const cn = SDK.utils.cn;
  // useI18n is a hook so each component that needs translations calls it
  // locally (see AchievementsPage, AchievementCard, ShareDialog, LoadingPage).
  // Older host dashboards may not expose useI18n yet; fall back to a no-op
  // shim that returns en values so the bundle still renders against an older
  // host SDK.  English fallback strings live alongside each call site.
  const useI18n = SDK.useI18n || function () { return { t: { achievements: null }, locale: "en" }; };

  // Resolve a translation by dotted path (e.g. "card.share_text"); fall back to
  // the English string passed in.  Used inside components after they call
  // useI18n() so they can still render against an older host SDK that doesn't
  // expose the achievements namespace yet.
  function tx(t, path, fallback, vars) {
    let node = t && t.achievements;
    if (node) {
      const parts = path.split(".");
      for (let i = 0; i < parts.length; i++) {
        if (node && typeof node === "object" && parts[i] in node) {
          node = node[parts[i]];
        } else { node = null; break; }
      }
    }
    let str = (typeof node === "string") ? node : fallback;
    if (vars) {
      for (const k in vars) {
        str = str.replace(new RegExp("\\{" + k + "\\}", "g"), vars[k]);
      }
    }
    return str;
  }

  const LUCIDE = {"flame":"<path d=\"M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z\" />","avalanche":"<path d=\"m8 3 4 8 5-5 5 15H2L8 3z\" />\n  <path d=\"M4.14 15.08c2.62-1.57 5.24-1.43 7.86.42 2.74 1.94 5.49 2 8.23.19\" />","nodes":"<rect x=\"16\" y=\"16\" width=\"6\" height=\"6\" rx=\"1\" />\n  <rect x=\"2\" y=\"16\" width=\"6\" height=\"6\" rx=\"1\" />\n  <rect x=\"9\" y=\"2\" width=\"6\" height=\"6\" rx=\"1\" />\n  <path d=\"M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3\" />\n  <path d=\"M12 12V8\" />","rocket":"<path d=\"M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z\" />\n  <path d=\"m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z\" />\n  <path d=\"M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0\" />\n  <path d=\"M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5\" />","branch":"<line x1=\"6\" x2=\"6\" y1=\"3\" y2=\"15\" />\n  <circle cx=\"18\" cy=\"6\" r=\"3\" />\n  <circle cx=\"6\" cy=\"18\" r=\"3\" />\n  <path d=\"M18 9a9 9 0 0 1-9 9\" />","daemon":"<path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\" />\n  <path d=\"M21 3v5h-5\" />","clock":"<circle cx=\"12\" cy=\"12\" r=\"10\" />\n  <polyline points=\"12 6 12 12 16 14\" />","warning":"<path d=\"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3\" />\n  <path d=\"M12 9v4\" />\n  <path d=\"M12 17h.01\" />","wine":"<path d=\"M8 22h8\" />\n  <path d=\"M7 10h10\" />\n  <path d=\"M12 15v7\" />\n  <path d=\"M12 15a5 5 0 0 0 5-5c0-2-.5-4-2-8H9c-1.5 4-2 6-2 8a5 5 0 0 0 5 5Z\" />","scroll":"<path d=\"M15 12h-5\" />\n  <path d=\"M15 8h-5\" />\n  <path d=\"M19 17V5a2 2 0 0 0-2-2H4\" />\n  <path d=\"M8 21h12a2 2 0 0 0 2-2v-1a1 1 0 0 0-1-1H11a1 1 0 0 0-1 1v1a2 2 0 1 1-4 0V5a2 2 0 1 0-4 0v2a1 1 0 0 0 1 1h3\" />","plug":"<path d=\"m19 5 3-3\" />\n  <path d=\"m2 22 3-3\" />\n  <path d=\"M6.3 20.3a2.4 2.4 0 0 0 3.4 0L12 18l-6-6-2.3 2.3a2.4 2.4 0 0 0 0 3.4Z\" />\n  <path d=\"M7.5 13.5 10 11\" />\n  <path d=\"M10.5 16.5 13 14\" />\n  <path d=\"m12 6 6 6 2.3-2.3a2.4 2.4 0 0 0 0-3.4l-2.6-2.6a2.4 2.4 0 0 0-3.4 0Z\" />","lock":"<circle cx=\"12\" cy=\"16\" r=\"1\" />\n  <rect x=\"3\" y=\"10\" width=\"18\" height=\"12\" rx=\"2\" />\n  <path d=\"M7 10V7a5 5 0 0 1 10 0v3\" />","package_skull":"<path d=\"M21 10V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l2-1.14\" />\n  <path d=\"m7.5 4.27 9 5.15\" />\n  <polyline points=\"3.29 7 12 12 20.71 7\" />\n  <line x1=\"12\" x2=\"12\" y1=\"22\" y2=\"12\" />\n  <path d=\"m17 13 5 5m-5 0 5-5\" />","restart":"<path d=\"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8\" />\n  <path d=\"M21 3v5h-5\" />\n  <path d=\"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16\" />\n  <path d=\"M8 16H3v5\" />","key":"<path d=\"M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z\" />\n  <circle cx=\"16.5\" cy=\"7.5\" r=\".5\" fill=\"currentColor\" />","colon":"<path d=\"M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1\" />\n  <path d=\"M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1\" />","container":"<path d=\"M22 7.7c0-.6-.4-1.2-.8-1.5l-6.3-3.9a1.72 1.72 0 0 0-1.7 0l-10.3 6c-.5.2-.9.8-.9 1.4v6.6c0 .5.4 1.2.8 1.5l6.3 3.9a1.72 1.72 0 0 0 1.7 0l10.3-6c.5-.3.9-1 .9-1.5Z\" />\n  <path d=\"M10 21.9V14L2.1 9.1\" />\n  <path d=\"m10 14 11.9-6.9\" />\n  <path d=\"M14 19.8v-8.1\" />\n  <path d=\"M18 17.5V9.4\" />","melting_clock":"<line x1=\"10\" x2=\"14\" y1=\"2\" y2=\"2\" />\n  <line x1=\"12\" x2=\"15\" y1=\"14\" y2=\"11\" />\n  <circle cx=\"12\" cy=\"14\" r=\"8\" />","pencil":"<path d=\"M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z\" />\n  <path d=\"m15 5 4 4\" />","blueprint":"<path d=\"m12.99 6.74 1.93 3.44\" />\n  <path d=\"M19.136 12a10 10 0 0 1-14.271 0\" />\n  <path d=\"m21 21-2.16-3.84\" />\n  <path d=\"m3 21 8.02-14.26\" />\n  <circle cx=\"12\" cy=\"5\" r=\"2\" />","pixel":"<path d=\"M3 7V5a2 2 0 0 1 2-2h2\" />\n  <path d=\"M17 3h2a2 2 0 0 1 2 2v2\" />\n  <path d=\"M21 17v2a2 2 0 0 1-2 2h-2\" />\n  <path d=\"M7 21H5a2 2 0 0 1-2-2v-2\" />\n  <path d=\"M7 12h10\" />","ship":"<path d=\"M12 10.189V14\" />\n  <path d=\"M12 2v3\" />\n  <path d=\"M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6\" />\n  <path d=\"M19.38 20A11.6 11.6 0 0 0 21 14l-8.188-3.639a2 2 0 0 0-1.624 0L3 14a11.6 11.6 0 0 0 2.81 7.76\" />\n  <path d=\"M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1s1.2 1 2.5 1c2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1\" />","spark_cursor":"<path d=\"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z\" />\n  <path d=\"M20 3v4\" />\n  <path d=\"M22 5h-4\" />\n  <path d=\"M4 17v2\" />\n  <path d=\"M5 18H3\" />","needle":"<path d=\"M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z\" />","hammer_scroll":"<path d=\"m15 12-8.373 8.373a1 1 0 1 1-3-3L12 9\" />\n  <path d=\"m18 15 4-4\" />\n  <path d=\"m21.5 11.5-1.914-1.914A2 2 0 0 1 19 8.172V7l-2.26-2.26a6 6 0 0 0-4.202-1.756L9 2.96l.92.82A6.18 6.18 0 0 1 12 8.4V10l2 2h1.172a2 2 0 0 1 1.414.586L18.5 14.5\" />","anvil":"<path d=\"M7 10H6a4 4 0 0 1-4-4 1 1 0 0 1 1-1h4\" />\n  <path d=\"M7 5a1 1 0 0 1 1-1h13a1 1 0 0 1 1 1 7 7 0 0 1-7 7H8a1 1 0 0 1-1-1z\" />\n  <path d=\"M9 12v5\" />\n  <path d=\"M15 12v5\" />\n  <path d=\"M5 20a3 3 0 0 1 3-3h8a3 3 0 0 1 3 3 1 1 0 0 1-1 1H6a1 1 0 0 1-1-1\" />","crystal":"<path d=\"M6 3h12l4 6-10 13L2 9Z\" />\n  <path d=\"M11 3 8 9l4 13 4-13-3-6\" />\n  <path d=\"M2 9h20\" />","palace":"<line x1=\"3\" x2=\"21\" y1=\"22\" y2=\"22\" />\n  <line x1=\"6\" x2=\"6\" y1=\"18\" y2=\"11\" />\n  <line x1=\"10\" x2=\"10\" y1=\"18\" y2=\"11\" />\n  <line x1=\"14\" x2=\"14\" y1=\"18\" y2=\"11\" />\n  <line x1=\"18\" x2=\"18\" y1=\"18\" y2=\"11\" />\n  <polygon points=\"12 2 20 7 4 7\" />","dragon":"<path d=\"M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z\" />","antenna":"<path d=\"M4.9 16.1C1 12.2 1 5.8 4.9 1.9\" />\n  <path d=\"M7.8 4.7a6.14 6.14 0 0 0-.8 7.5\" />\n  <circle cx=\"12\" cy=\"9\" r=\"2\" />\n  <path d=\"M16.2 4.8c2 2 2.26 5.11.8 7.47\" />\n  <path d=\"M19.1 1.9a9.96 9.96 0 0 1 0 14.1\" />\n  <path d=\"M9.5 18h5\" />\n  <path d=\"m8 22 4-11 4 11\" />","puzzle":"<path d=\"M15.39 4.39a1 1 0 0 0 1.68-.474 2.5 2.5 0 1 1 3.014 3.015 1 1 0 0 0-.474 1.68l1.683 1.682a2.414 2.414 0 0 1 0 3.414L19.61 15.39a1 1 0 0 1-1.68-.474 2.5 2.5 0 1 0-3.014 3.015 1 1 0 0 1 .474 1.68l-1.683 1.682a2.414 2.414 0 0 1-3.414 0L8.61 19.61a1 1 0 0 0-1.68.474 2.5 2.5 0 1 1-3.014-3.015 1 1 0 0 0 .474-1.68l-1.683-1.682a2.414 2.414 0 0 1 0-3.414L4.39 8.61a1 1 0 0 1 1.68.474 2.5 2.5 0 1 0 3.014-3.015 1 1 0 0 1-.474-1.68l1.683-1.682a2.414 2.414 0 0 1 3.414 0z\" />","rewind":"<path d=\"M9 14 4 9l5-5\" />\n  <path d=\"M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11\" />","spiral":"<path d=\"M13 16a3 3 0 0 1 2.24 5\" />\n  <path d=\"M18 12h.01\" />\n  <path d=\"M18 21h-8a4 4 0 0 1-4-4 7 7 0 0 1 7-7h.2L9.6 6.4a1 1 0 1 1 2.8-2.8L15.8 7h.2c3.3 0 6 2.7 6 6v1a2 2 0 0 1-2 2h-1a3 3 0 0 0-3 3\" />\n  <path d=\"M20 8.54V4a2 2 0 1 0-4 0v3\" />\n  <path d=\"M7.612 12.524a3 3 0 1 0-1.6 4.3\" />","quote":"<path d=\"M16 3a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2 1 1 0 0 1 1 1v1a2 2 0 0 1-2 2 1 1 0 0 0-1 1v2a1 1 0 0 0 1 1 6 6 0 0 0 6-6V5a2 2 0 0 0-2-2z\" />\n  <path d=\"M5 3a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2 1 1 0 0 1 1 1v1a2 2 0 0 1-2 2 1 1 0 0 0-1 1v2a1 1 0 0 0 1 1 6 6 0 0 0 6-6V5a2 2 0 0 0-2-2z\" />","compass":"<path d=\"m16.24 7.76-1.804 5.411a2 2 0 0 1-1.265 1.265L7.76 16.24l1.804-5.411a2 2 0 0 1 1.265-1.265z\" />\n  <circle cx=\"12\" cy=\"12\" r=\"10\" />","browser":"<circle cx=\"12\" cy=\"12\" r=\"10\" />\n  <path d=\"M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20\" />\n  <path d=\"M2 12h20\" />","terminal":"<polyline points=\"4 17 10 11 4 5\" />\n  <line x1=\"12\" x2=\"20\" y1=\"19\" y2=\"19\" />","wand":"<path d=\"m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72\" />\n  <path d=\"m14 7 3 3\" />\n  <path d=\"M5 6v4\" />\n  <path d=\"M19 14v4\" />\n  <path d=\"M10 2v2\" />\n  <path d=\"M7 8H3\" />\n  <path d=\"M21 16h-4\" />\n  <path d=\"M11 3H9\" />","folder":"<path d=\"M10.7 20H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H20a2 2 0 0 1 2 2v4.1\" />\n  <path d=\"m21 21-1.9-1.9\" />\n  <circle cx=\"17\" cy=\"17\" r=\"3\" />","eye":"<path d=\"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0\" />\n  <circle cx=\"12\" cy=\"12\" r=\"3\" />","wave":"<path d=\"M2 13a2 2 0 0 0 2-2V7a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0V4a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0v-4a2 2 0 0 1 2-2\" />","swap":"<path d=\"m17 2 4 4-4 4\" />\n  <path d=\"M3 11v-1a4 4 0 0 1 4-4h14\" />\n  <path d=\"m7 22-4-4 4-4\" />\n  <path d=\"M21 13v1a4 4 0 0 1-4 4H3\" />","router":"<rect width=\"20\" height=\"8\" x=\"2\" y=\"14\" rx=\"2\" />\n  <path d=\"M6.01 18H6\" />\n  <path d=\"M10.01 18H10\" />\n  <path d=\"M15 10v4\" />\n  <path d=\"M17.84 7.17a4 4 0 0 0-5.66 0\" />\n  <path d=\"M20.66 4.34a8 8 0 0 0-11.31 0\" />","codex":"<path d=\"M10 9.5 8 12l2 2.5\" />\n  <path d=\"m14 9.5 2 2.5-2 2.5\" />\n  <rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" />","prism":"<path d=\"M6 3h12l4 6-10 13L2 9Z\" />\n  <path d=\"M11 3 8 9l4 13 4-13-3-6\" />\n  <path d=\"M2 9h20\" />","marathon":"<line x1=\"10\" x2=\"14\" y1=\"2\" y2=\"2\" />\n  <line x1=\"12\" x2=\"15\" y1=\"14\" y2=\"11\" />\n  <circle cx=\"12\" cy=\"14\" r=\"8\" />","calendar":"<path d=\"M8 2v4\" />\n  <path d=\"M16 2v4\" />\n  <rect width=\"18\" height=\"18\" x=\"3\" y=\"4\" rx=\"2\" />\n  <path d=\"M3 10h18\" />\n  <path d=\"M8 14h.01\" />\n  <path d=\"M12 14h.01\" />\n  <path d=\"M16 14h.01\" />\n  <path d=\"M8 18h.01\" />\n  <path d=\"M12 18h.01\" />\n  <path d=\"M16 18h.01\" />","moon":"<path d=\"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z\" />","cache":"<ellipse cx=\"12\" cy=\"5\" rx=\"9\" ry=\"3\" />\n  <path d=\"M3 5V19A9 3 0 0 0 21 19V5\" />\n  <path d=\"M3 12A9 3 0 0 0 21 12\" />","secret":"<path d=\"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z\" />\n  <path d=\"M9.1 9a3 3 0 0 1 5.82 1c0 2-3 3-3 3\" />\n  <path d=\"M12 17h.01\" />"};

  const tierClass = function (tier) {
    return tier ? "ha-tier-" + tier.toLowerCase() : "ha-tier-pending";
  };

  async function api(path, options) {
    const url = "/api/plugins/hermes-achievements" + path;
    const token = window.__HERMES_SESSION_TOKEN__ || "";
    const headers = { ...((options && options.headers) || {}) };
    if (token) headers["X-Hermes-Session-Token"] = token;
    const res = await fetch(url, { ...(options || {}), headers });
    if (!res.ok) {
      const text = await res.text().catch(function () { return res.statusText; });
      throw new Error(res.status + ": " + text);
    }
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch (_) {
      return null;
    }
  }

  function AchievementIcon({ icon }) {
    const svg = LUCIDE[icon] || LUCIDE.secret;
    const ref = React.useRef(null);
    React.useEffect(function () {
      if (!ref.current) return;
      const el = ref.current;
      while (el.firstChild) el.removeChild(el.firstChild);
      try {
        const doc = new DOMParser().parseFromString(
          "<svg xmlns=\"http://www.w3.org/2000/svg\">" + svg + "</svg>",
          "image/svg+xml"
        );
        if (!doc.querySelector("parsererror")) {
          Array.from(doc.documentElement.childNodes).forEach(function (n) {
            el.appendChild(document.importNode(n, true));
          });
        }
      } catch (_) {}
    }, [svg]);
    return React.createElement("svg", {
      ref: ref,
      className: "ha-lucide",
      viewBox: "0 0 24 24",
      fill: "none",
      stroke: "currentColor",
      strokeWidth: 2,
      strokeLinecap: "round",
      strokeLinejoin: "round",
      "aria-hidden": "true",
    });
  }

  const TIER_HEX = {
    "Copper": "#b87333",
    "Silver": "#c0c7d2",
    "Gold": "#f2c94c",
    "Diamond": "#67e8f9",
    "Olympian": "#c084fc",
  };

  function tierHex(tier) {
    return TIER_HEX[tier] || "#67e8f9";
  }

  // Render a LUCIDE icon path fragment into a standalone SVG string with an
  // explicit stroke color so it can be rasterized onto a <canvas> via Image.
  // The normal render path uses stroke="currentColor" which browsers honor in
  // DOM but NOT when the SVG is drawn to a canvas from a data URL.
  function iconSvgForCanvas(iconKey, strokeColor) {
    const paths = LUCIDE[iconKey] || LUCIDE.secret;
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" " +
      "stroke=\"" + strokeColor + "\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">" +
      paths + "</svg>";
  }

  function loadSvgImage(svgString) {
    return new Promise(function (resolve, reject) {
      const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = function (e) { URL.revokeObjectURL(url); reject(e); };
      img.src = url;
    });
  }

  function wrapText(ctx, text, maxWidth) {
    const words = String(text || "").split(/\s+/).filter(Boolean);
    const lines = [];
    let current = "";
    for (let i = 0; i < words.length; i++) {
      const candidate = current ? current + " " + words[i] : words[i];
      if (ctx.measureText(candidate).width <= maxWidth) {
        current = candidate;
      } else {
        if (current) lines.push(current);
        current = words[i];
      }
    }
    if (current) lines.push(current);
    return lines;
  }

  // Build a 1200x630 share card PNG for a single achievement. Returns a Blob.
  // Pure client-side render via Canvas2D — no external deps, no network.
  async function buildShareImage(achievement) {
    const W = 1200;
    const H = 630;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");

    const tier = achievement.tier || achievement.next_tier || "Copper";
    const color = tierHex(tier);

    // Background: dark charcoal with a tier-tinted radial highlight on the
    // top-left, echoing the card visual language.
    ctx.fillStyle = "#0b0d11";
    ctx.fillRect(0, 0, W, H);
    const bgGrad = ctx.createRadialGradient(260, 220, 60, 260, 220, 820);
    bgGrad.addColorStop(0, color + "33");
    bgGrad.addColorStop(0.55, color + "0a");
    bgGrad.addColorStop(1, "#0b0d1100");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    // Outer border
    ctx.strokeStyle = color + "66";
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, W - 2, H - 2);

    // Icon block — 380x380 on the left
    try {
      const svg = iconSvgForCanvas(achievement.icon || "secret", color);
      const iconImg = await loadSvgImage(svg);
      const ix = 90;
      const iy = 125;
      const isize = 380;
      // Icon glow
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = 40;
      ctx.drawImage(iconImg, ix, iy, isize, isize);
      ctx.restore();
    } catch (_) {
      // Icon render failure is non-fatal; card still useful without it.
    }

    // Right column text layout
    const rx = 520;
    const rMaxWidth = W - rx - 70;

    // Category label (kicker)
    ctx.fillStyle = "#8b95a8";
    ctx.font = "600 22px ui-monospace, 'SF Mono', Menlo, monospace";
    ctx.textBaseline = "top";
    ctx.fillText((achievement.category || "").toUpperCase(), rx, 112);

    // Achievement name — wrap to 2 lines if needed
    ctx.fillStyle = "#ffffff";
    ctx.font = "780 68px system-ui, -apple-system, 'Segoe UI', sans-serif";
    const nameLines = wrapText(ctx, achievement.name || "Achievement", rMaxWidth).slice(0, 2);
    let cursorY = 150;
    for (let i = 0; i < nameLines.length; i++) {
      ctx.fillText(nameLines[i], rx, cursorY);
      cursorY += 76;
    }

    // Tier badge pill
    const badgeLabel = tier.toUpperCase() + " TIER";
    ctx.font = "700 22px ui-monospace, 'SF Mono', Menlo, monospace";
    const badgeWidth = ctx.measureText(badgeLabel).width + 32;
    const badgeX = rx;
    const badgeY = cursorY + 14;
    const badgeH = 40;
    ctx.fillStyle = color + "1f";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(badgeX, badgeY, badgeWidth, badgeH);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(badgeLabel, badgeX + 16, badgeY + badgeH / 2 + 1);
    ctx.textBaseline = "top";

    // Description — wrap up to 3 lines
    ctx.fillStyle = "#c3cad6";
    ctx.font = "400 26px system-ui, -apple-system, 'Segoe UI', sans-serif";
    const descLines = wrapText(ctx, achievement.description || "", rMaxWidth).slice(0, 3);
    let descY = badgeY + badgeH + 28;
    for (let i = 0; i < descLines.length; i++) {
      ctx.fillText(descLines[i], rx, descY);
      descY += 34;
    }

    // Progress / stat line (if meaningful)
    const progressValue = achievement.progress;
    const threshold = achievement.next_threshold;
    let statLine = null;
    if (progressValue && threshold) {
      statLine = progressValue.toLocaleString() + " / " + threshold.toLocaleString();
    } else if (progressValue) {
      statLine = progressValue.toLocaleString();
    }
    if (statLine) {
      ctx.fillStyle = color;
      ctx.font = "700 28px ui-monospace, 'SF Mono', Menlo, monospace";
      ctx.fillText(statLine, rx, descY + 14);
    }

    // Footer watermark
    ctx.fillStyle = "#8b95a8";
    ctx.font = "600 20px ui-monospace, 'SF Mono', Menlo, monospace";
    ctx.textBaseline = "bottom";
    ctx.fillText("HERMES AGENT  ·  hermes-agent.nousresearch.com", 70, H - 40);

    // "UNLOCKED" stamp upper-right
    ctx.textBaseline = "top";
    ctx.fillStyle = color;
    ctx.font = "800 24px ui-monospace, 'SF Mono', Menlo, monospace";
    const stamp = "◆ UNLOCKED";
    const stampW = ctx.measureText(stamp).width;
    ctx.fillText(stamp, W - 70 - stampW, 70);

    return await new Promise(function (resolve, reject) {
      canvas.toBlob(function (blob) {
        if (blob) resolve(blob); else reject(new Error("canvas.toBlob returned null"));
      }, "image/png");
    });
  }

  function ShareDialog({ achievement, onClose }) {
    const { t } = useI18n();
    const [status, setStatus] = hooks.useState("rendering"); // rendering | ready | copied | error
    const [errorMsg, setErrorMsg] = hooks.useState(null);
    const [previewUrl, setPreviewUrl] = hooks.useState(null);
    const blobRef = React.useRef(null);

    hooks.useEffect(function () {
      let cancelled = false;
      let createdUrl = null;
      buildShareImage(achievement).then(function (blob) {
        if (cancelled) return;
        blobRef.current = blob;
        createdUrl = URL.createObjectURL(blob);
        setPreviewUrl(createdUrl);
        setStatus("ready");
      }).catch(function (err) {
        if (cancelled) return;
        setErrorMsg(String(err && err.message || err));
        setStatus("error");
      });
      return function () {
        cancelled = true;
        if (createdUrl) URL.revokeObjectURL(createdUrl);
      };
    }, [achievement.id]);

    function download() {
      if (!blobRef.current) return;
      const url = URL.createObjectURL(blobRef.current);
      const a = document.createElement("a");
      a.href = url;
      a.download = "hermes-achievement-" + (achievement.id || "badge") + ".png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    }

    async function copyToClipboard() {
      if (!blobRef.current) return;
      try {
        if (!navigator.clipboard || !window.ClipboardItem) {
          throw new Error(tx(t, "share.clipboard_unsupported", "Clipboard image copy not supported in this browser — use Download instead."));
        }
        await navigator.clipboard.write([
          new window.ClipboardItem({ "image/png": blobRef.current }),
        ]);
        setStatus("copied");
        setTimeout(function () { setStatus("ready"); }, 1800);
      } catch (err) {
        setErrorMsg(String(err && err.message || err));
        setStatus("error");
      }
    }

    // Build the pre-filled tweet text. Keep it short so X doesn't truncate
    // when the user hasn't attached the PNG yet — they'll copy-image and
    // paste in the same flow.
    function tweetText() {
      const tierPart = achievement.tier ? (achievement.tier + " tier ") : "";
      const tmpl = tx(t, "share.tweet_text", "Just unlocked {tier_part}\"{name}\" in Hermes Agent ☤", {
        tier_part: tierPart,
        name: achievement.name,
      });
      return tmpl + "\n\n@NousResearch · https://hermes-agent.nousresearch.com";
    }

    function shareOnX() {
      const url = "https://x.com/intent/post?text=" + encodeURIComponent(tweetText());
      window.open(url, "_blank", "noopener,noreferrer");
    }

    return React.createElement("div", {
      className: "ha-share-backdrop",
      onClick: function (e) { if (e.target === e.currentTarget) onClose(); },
    },
      React.createElement("div", { className: "ha-share-dialog", role: "dialog", "aria-label": tx(t, "share.dialog_label", "Share achievement") },
        React.createElement("div", { className: "ha-share-head" },
          React.createElement("strong", null, tx(t, "share.header", "Share: {name}", { name: achievement.name })),
          React.createElement("button", { className: "ha-share-close", onClick: onClose, "aria-label": tx(t, "share.close", "Close") }, "×")
        ),
        React.createElement("div", { className: "ha-share-preview" },
          status === "rendering" && React.createElement("div", { className: "ha-share-placeholder" }, tx(t, "share.rendering", "Rendering…")),
          previewUrl && React.createElement("img", { src: previewUrl, alt: tx(t, "share.card_alt", "{name} share card", { name: achievement.name }) })
        ),
        status === "error" && React.createElement("div", { className: "ha-share-error" }, errorMsg || tx(t, "share.error_generic", "Something went wrong.")),
        React.createElement("div", { className: "ha-share-actions" },
          React.createElement("button", {
            className: "ha-share-btn ha-share-btn-primary",
            onClick: shareOnX,
            title: tx(t, "share.x_title", "Opens X with a pre-filled post"),
          }, tx(t, "share.x_button", "Share on X")),
          React.createElement("button", {
            className: "ha-share-btn",
            onClick: copyToClipboard,
            disabled: status !== "ready" && status !== "copied",
            title: tx(t, "share.copy_title", "Copy the image to paste into your post"),
          }, status === "copied" ? tx(t, "share.copied", "Copied ✓") : tx(t, "share.copy_button", "Copy image")),
          React.createElement("button", {
            className: "ha-share-btn",
            onClick: download,
            disabled: status !== "ready" && status !== "copied",
          }, tx(t, "share.download_button", "Download PNG"))
        ),
        React.createElement("p", { className: "ha-share-hint" },
          tx(t, "share.hint", "Share on X opens a pre-filled post in a new tab. Click Copy image first if you want the 1200×630 badge attached — X lets you paste it right into the tweet composer. Download PNG saves the file for use anywhere.")
        )
      )
    );
  }

  function StatCard(props) {
    return React.createElement(C.Card, { className: "ha-stat" },
      React.createElement(C.CardContent, { className: "ha-stat-content" },
        React.createElement("div", { className: "ha-stat-label" }, props.label),
        React.createElement("div", { className: "ha-stat-value" }, props.value),
        props.hint && React.createElement("div", { className: "ha-stat-hint" }, props.hint)
      )
    );
  }

  function TierLegend() {
    return React.createElement("div", { className: "ha-tier-legend" },
      ["Copper", "Silver", "Gold", "Diamond", "Olympian"].map(function (tier, index, arr) {
        return React.createElement(React.Fragment, { key: tier },
          React.createElement("span", { className: "ha-tier-step ha-tier-" + tier.toLowerCase() },
            React.createElement("i", null),
            tier
          ),
          index < arr.length - 1 && React.createElement("span", { className: "ha-tier-arrow" }, "→")
        );
      })
    );
  }


  function LoadingSkeletonCard(props) {
    return React.createElement(C.Card, { className: "ha-card ha-skeleton-card ha-tier-pending" },
      React.createElement(C.CardContent, { className: "ha-card-content" },
        React.createElement("div", { className: "ha-card-head" },
          React.createElement("div", { className: "ha-skeleton ha-skeleton-icon" }),
          React.createElement("div", { className: "ha-skeleton-stack" },
            React.createElement("div", { className: "ha-skeleton ha-skeleton-title" }),
            React.createElement("div", { className: "ha-skeleton ha-skeleton-meta" })
          ),
          React.createElement("div", { className: "ha-badges" },
            React.createElement("div", { className: "ha-skeleton ha-skeleton-badge" }),
            React.createElement("div", { className: "ha-skeleton ha-skeleton-badge ha-skeleton-badge-short" })
          )
        ),
        React.createElement("div", { className: "ha-skeleton ha-skeleton-line" }),
        React.createElement("div", { className: "ha-skeleton ha-skeleton-line ha-skeleton-line-short" }),
        React.createElement("div", { className: "ha-skeleton ha-skeleton-criteria" }),
        React.createElement("div", { className: "ha-evidence-slot" }, React.createElement("div", { className: "ha-skeleton ha-skeleton-evidence" })),
        React.createElement("div", { className: "ha-progress-row" },
          React.createElement("div", { className: "ha-skeleton ha-skeleton-progress" }),
          React.createElement("div", { className: "ha-skeleton ha-skeleton-progress-text" })
        )
      )
    );
  }

  function LoadingPage() {
    const { t } = useI18n();
    return React.createElement("div", { className: "ha-page ha-page-loading" },
      React.createElement("section", { className: "ha-hero ha-loading-hero" },
        React.createElement("div", null,
          React.createElement("div", { className: "ha-kicker" }, tx(t, "hero.kicker", "Agentic Gamerscore")),
          React.createElement("h1", null, tx(t, "hero.title", "Hermes Achievements")),
          React.createElement("p", null, tx(t, "hero.scan_subtitle", "Scanning Hermes session history. First scan can take 5–10 seconds on large histories."))
        ),
        React.createElement("div", { className: "ha-scan-status", role: "status", "aria-live": "polite" },
          React.createElement("span", { className: "ha-scan-pulse", "aria-hidden": "true" }),
          React.createElement("div", null,
            React.createElement("strong", null, tx(t, "scan.building_headline", "Building achievement profile…")),
            React.createElement("p", null, tx(t, "scan.building_detail", "Reading sessions, tool calls, model metadata, and unlock state."))
          )
        )
      ),
      React.createElement("div", { className: "ha-stats" },
        [
          { key: "stats.unlocked", fallback: "Unlocked" },
          { key: "stats.discovered", fallback: "Discovered" },
          { key: "stats.secrets", fallback: "Secrets" },
          { key: "stats.highest_tier", fallback: "Highest tier" },
          { key: "stats.latest", fallback: "Latest" },
        ].map(function (entry) {
          const label = tx(t, entry.key, entry.fallback);
          return React.createElement(C.Card, { key: entry.key, className: "ha-stat ha-skeleton-stat" },
            React.createElement(C.CardContent, { className: "ha-stat-content" },
              React.createElement("div", { className: "ha-stat-label" }, label),
              React.createElement("div", { className: "ha-skeleton ha-skeleton-stat-value" }),
              React.createElement("div", { className: "ha-skeleton ha-skeleton-stat-hint" })
            )
          );
        })
      ),
      React.createElement("section", { className: "ha-guide ha-loading-guide" },
        React.createElement("div", null,
          React.createElement("strong", null, tx(t, "guide.scan_status_header", "Scan status")),
          React.createElement("p", null, tx(t, "guide.scan_status_body", "Hermes is scanning local history once, then cards will appear automatically. Nothing is stuck if this takes a few seconds."))
        ),
        React.createElement("div", null,
          React.createElement("strong", null, tx(t, "guide.what_scanned_header", "What is scanned")),
          React.createElement("p", null, tx(t, "guide.what_scanned_body", "Sessions, tool calls, model metadata, errors, achievements, and local unlock state."))
        )
      ),
      React.createElement("section", { className: "ha-grid" }, [0, 1, 2, 3, 4, 5].map(function (i) {
        return React.createElement(LoadingSkeletonCard, { key: i });
      }))
    );
  }


  function AchievementCard({ achievement }) {
    const { t } = useI18n();
    const unlocked = achievement.unlocked;
    const progress = achievement.progress || 0;
    const pct = achievement.progress_pct || (unlocked ? 100 : 0);
    const state = achievement.state || (unlocked ? "unlocked" : "discovered");
    const stateLabel = state === "unlocked"
      ? tx(t, "state.unlocked", "Unlocked")
      : (state === "secret" ? tx(t, "state.secret", "Secret") : tx(t, "state.discovered", "Discovered"));
    const targetTier = achievement.next_tier || achievement.tier;
    let tierLabel;
    if (achievement.tier) {
      tierLabel = achievement.tier;
    } else if (targetTier) {
      tierLabel = tx(t, "tier.target", "Target {tier}", { tier: targetTier });
    } else if (state === "secret") {
      tierLabel = tx(t, "tier.hidden", "Hidden");
    } else if (unlocked) {
      tierLabel = tx(t, "tier.complete", "Complete");
    } else {
      tierLabel = tx(t, "tier.objective", "Objective");
    }
    const progressText = state === "secret"
      ? tx(t, "progress.hidden", "hidden")
      : (progress + (achievement.next_threshold ? " / " + achievement.next_threshold : ""));
    const [shareOpen, setShareOpen] = hooks.useState(false);
    return React.createElement(C.Card, { className: cn("ha-card", "ha-state-" + state, tierClass(achievement.tier || achievement.next_tier)) },
      React.createElement(C.CardContent, { className: "ha-card-content" },
        React.createElement("div", { className: "ha-card-head" },
          React.createElement("div", { className: "ha-icon" }, React.createElement(AchievementIcon, { icon: achievement.icon || "secret" })),
          React.createElement("div", { className: "ha-card-title-wrap" },
            React.createElement("div", { className: "ha-card-title" }, achievement.name),
            React.createElement("div", { className: "ha-card-category" }, achievement.category)
          ),
          React.createElement("div", { className: "ha-badges" },
            React.createElement("span", { className: "ha-state-badge" }, stateLabel),
            React.createElement("span", { className: "ha-tier-badge" }, tierLabel),
            state === "unlocked" && React.createElement("button", {
              className: "ha-share-trigger",
              onClick: function () { setShareOpen(true); },
              title: tx(t, "card.share_title", "Share this achievement"),
              "aria-label": tx(t, "card.share_label", "Share {name}", { name: achievement.name }),
            }, tx(t, "card.share_text", "Share"))
          )
        ),
        React.createElement("p", { className: "ha-description" }, achievement.description),
        achievement.criteria && React.createElement("details", { className: "ha-criteria" },
          React.createElement("summary", null, state === "secret"
            ? tx(t, "card.how_to_reveal", "How to reveal")
            : tx(t, "card.what_counts", "What counts")),
          React.createElement("p", null, achievement.criteria)
        ),
        React.createElement("div", { className: "ha-evidence-slot" },
          achievement.evidence ? React.createElement("div", { className: "ha-evidence" },
            React.createElement("span", { className: "ha-evidence-label" }, tx(t, "card.evidence_label", "Evidence")),
            React.createElement("span", { className: "ha-evidence-title" }, achievement.evidence.title || achievement.evidence.session_id || tx(t, "card.evidence_session_fallback", "session"))
          ) : React.createElement("div", { className: "ha-evidence ha-evidence-empty", "aria-hidden": "true" }, tx(t, "card.no_evidence", "No evidence yet"))
        ),
        React.createElement("div", { className: "ha-progress-row" },
          React.createElement("div", { className: "ha-progress-track" },
            React.createElement("div", { className: "ha-progress-fill", style: { width: Math.max(state === "secret" ? 0 : 3, Math.min(100, pct)) + "%" } })
          ),
          React.createElement("span", { className: "ha-progress-text" }, progressText)
        )
      ),
      shareOpen && React.createElement(ShareDialog, {
        achievement: achievement,
        onClose: function () { setShareOpen(false); },
      })
    );
  }

  function AchievementsPage() {
    const { t } = useI18n();
    const [data, setData] = hooks.useState(null);
    const [loading, setLoading] = hooks.useState(true);
    const [error, setError] = hooks.useState(null);
    const [category, setCategory] = hooks.useState("All");
    const [visibility, setVisibility] = hooks.useState("all");

    function load() {
      setLoading(true);
      api("/achievements")
        .then(function (payload) { setData(payload); setError((payload && payload.error) || null); })
        .catch(function (err) { setError(String(err)); })
        .finally(function () { setLoading(false); });
    }
    // refresh() re-fetches without flipping the loading state — used by the
    // auto-poller during an in-progress background scan so the page updates
    // with growing unlock counts instead of flashing the loading skeleton.
    function refresh() {
      api("/achievements")
        .then(function (payload) { setData(payload); setError((payload && payload.error) || null); })
        .catch(function (err) { setError(String(err)); });
    }
    hooks.useEffect(load, []);

    // Auto-poll while the backend is still scanning. scan_meta.mode is
    // "pending" on the very first request (no cache yet) and "in_progress"
    // while the background thread is publishing partial snapshots. Once it
    // flips to "full" or "incremental" the scan is done and we stop polling.
    const scanMode = (data && data.scan_meta && data.scan_meta.mode) || null;
    const scanInFlight = scanMode === "pending" || scanMode === "in_progress";
    hooks.useEffect(function () {
      if (!scanInFlight) return undefined;
      const id = setInterval(refresh, 4000);
      return function () { clearInterval(id); };
    }, [scanInFlight]);

    const achievements = (data && data.achievements) || [];
    const categories = ["All"].concat(Array.from(new Set(achievements.map(function (a) { return a.category; }))));
    const visible = achievements.filter(function (a) {
      if (category !== "All" && a.category !== category) return false;
      if (visibility === "unlocked" && a.state !== "unlocked") return false;
      if (visibility === "discovered" && a.state !== "discovered") return false;
      if (visibility === "secret" && a.state !== "secret") return false;
      return true;
    });
    const unlocked = achievements.filter(function (a) { return a.state === "unlocked"; });
    const discovered = achievements.filter(function (a) { return a.state === "discovered"; });
    const secret = achievements.filter(function (a) { return a.state === "secret"; });
    const latest = unlocked.slice().sort(function (a, b) { return (b.unlocked_at || 0) - (a.unlocked_at || 0); }).slice(0, 5);
    const highest = ["Olympian", "Diamond", "Gold", "Silver", "Copper"].find(function (tier) { return unlocked.some(function (a) { return a.tier === tier; }); }) || tx(t, "stats.none_yet", "None yet");

    // Build the in-progress scan banner once so the JSX below stays readable.
    // Shows nothing when the scan is idle. When a scan is running it renders
    // a pulsing status row with "X / Y sessions · Z%" and a filling bar, so
    // the user gets continuous visual feedback during long cold scans on
    // large session databases (can take several minutes on 8000+ sessions).
    let scanBanner = null;
    if (scanInFlight) {
      const meta = (data && data.scan_meta) || {};
      const scanned = Number(meta.sessions_scanned_so_far || meta.sessions_total || 0);
      const total = Number(meta.sessions_expected_total || 0);
      const pct = total > 0 ? Math.max(0, Math.min(100, Math.floor((scanned / total) * 100))) : 0;
      const headline = scanMode === "pending"
        ? tx(t, "scan.starting_headline", "Starting achievement scan…")
        : tx(t, "scan.building_headline", "Building achievement profile…");
      const detail = total > 0
        ? tx(t, "scan.progress_detail", "Scanned {scanned} of {total} sessions · {pct}%. Badges unlock as more history streams in.", {
            scanned: scanned.toLocaleString(),
            total: total.toLocaleString(),
            pct: String(pct),
          })
        : tx(t, "scan.idle_detail", "Reading sessions, tool calls, model metadata, and unlock state. Badges appear here as they unlock.");
      scanBanner = React.createElement("section", { className: "ha-scan-banner", role: "status", "aria-live": "polite" },
        React.createElement("div", { className: "ha-scan-banner-head" },
          React.createElement("span", { className: "ha-scan-pulse", "aria-hidden": "true" }),
          React.createElement("div", { className: "ha-scan-banner-text" },
            React.createElement("strong", null, headline),
            React.createElement("p", null, detail)
          )
        ),
        total > 0 && React.createElement("div", { className: "ha-scan-progress-track", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": pct },
          React.createElement("div", { className: "ha-scan-progress-fill", style: { width: pct + "%" } })
        )
      );
    }

    if (loading) {
      return React.createElement(LoadingPage, null);
    }

    // Translate the "All" category pill but keep the underlying state ("All")
    // as the canonical key the API matches against.
    const allCategoryLabel = tx(t, "filters.all_categories", "All");
    const visibilityLabels = {
      all: tx(t, "filters.visibility_all", "all"),
      unlocked: tx(t, "filters.visibility_unlocked", "unlocked"),
      discovered: tx(t, "filters.visibility_discovered", "discovered"),
      secret: tx(t, "filters.visibility_secret", "secret"),
    };

    return React.createElement("div", { className: "ha-page" },
      React.createElement("section", { className: "ha-hero" },
        React.createElement("div", null,
          React.createElement("div", { className: "ha-kicker" }, tx(t, "hero.kicker", "Agentic Gamerscore")),
          React.createElement("h1", null, tx(t, "hero.title", "Hermes Achievements")),
          React.createElement("p", null, tx(t, "hero.subtitle", "Collectible Hermes badges earned from real session history. Known unfinished achievements are shown as Discovered; Secret achievements stay hidden until the first matching behavior appears."))
        ),
        React.createElement(C.Button, { onClick: load, className: "ha-refresh" }, tx(t, "actions.rescan", "Rescan"))
      ),
      scanBanner,
      error && React.createElement(C.Card, { className: "ha-error" }, React.createElement(C.CardContent, null, String(error))),
      React.createElement("div", { className: "ha-stats" },
        React.createElement(StatCard, { label: tx(t, "stats.unlocked", "Unlocked"), value: (data ? data.unlocked_count : 0) + " / " + (data ? data.total_count : 0), hint: tx(t, "stats.unlocked_hint", "earned badges") }),
        React.createElement(StatCard, { label: tx(t, "stats.discovered", "Discovered"), value: discovered.length, hint: tx(t, "stats.discovered_hint", "known, not earned yet") }),
        React.createElement(StatCard, { label: tx(t, "stats.secrets", "Secrets"), value: secret.length, hint: tx(t, "stats.secrets_hint", "hidden until first signal") }),
        React.createElement(StatCard, { label: tx(t, "stats.highest_tier", "Highest tier"), value: highest, hint: tx(t, "stats.highest_tier_hint", "Copper → Silver → Gold → Diamond → Olympian") }),
        React.createElement(StatCard, { label: tx(t, "stats.latest", "Latest"), value: latest[0] ? latest[0].name : tx(t, "stats.none_yet", "None yet"), hint: latest[0] ? latest[0].category : tx(t, "stats.latest_hint_empty", "run Hermes more") })
      ),
      React.createElement("section", { className: "ha-guide" },
        React.createElement("div", null,
          React.createElement("strong", null, tx(t, "guide.tiers_header", "Tiers")),
          React.createElement(TierLegend, null)
        ),
        React.createElement("div", null,
          React.createElement("strong", null, tx(t, "guide.secret_header", "Secret achievements")),
          React.createElement("p", null, tx(t, "guide.secret_body", "Secrets hide their exact trigger. Once Hermes sees a related signal, the card becomes Discovered and shows its requirement."))
        )
      ),
      React.createElement("div", { className: "ha-toolbar" },
        React.createElement("div", { className: "ha-pills" }, categories.map(function (cat) {
          // Render the localized "All" pill but keep the underlying value
          // unchanged so the filter logic still compares against "All".
          const pillLabel = cat === "All" ? allCategoryLabel : cat;
          return React.createElement("button", { key: cat, onClick: function () { setCategory(cat); }, className: cat === category ? "active" : "" }, pillLabel);
        })),
        React.createElement("div", { className: "ha-pills" }, ["all", "unlocked", "discovered", "secret"].map(function (v) {
          return React.createElement("button", { key: v, onClick: function () { setVisibility(v); }, className: v === visibility ? "active" : "" }, visibilityLabels[v] || v);
        }))
      ),
      latest.length > 0 && React.createElement("section", { className: "ha-latest" },
        React.createElement("h2", null, tx(t, "latest.header", "Recent unlocks")),
        React.createElement("div", { className: "ha-latest-row" }, latest.map(function (a) {
          return React.createElement("div", { key: a.id, className: cn("ha-chip", tierClass(a.tier)) },
            React.createElement("span", { className: "ha-chip-icon" }, React.createElement(AchievementIcon, { icon: a.icon || "secret" })),
            a.name
          );
        }))
      ),
      visibility === "secret" && visible.length === 0 && React.createElement(C.Card, { className: "ha-secret-empty" },
        React.createElement(C.CardContent, { className: "ha-secret-empty-content" },
          React.createElement("strong", null, tx(t, "empty.no_secrets_header", "No hidden secrets left in this scan.")),
          React.createElement("p", null, tx(t, "empty.no_secrets_body", "Clue: secrets usually start from unusual failure or power-user patterns — port conflicts, permission walls, missing env vars, YAML mistakes, Docker collisions, rollback/checkpoint use, cache hits, or tiny fixes after lots of red text."))
        )
      ),
      React.createElement("section", { className: "ha-grid" }, visible.map(function (a) {
        return React.createElement(AchievementCard, { key: a.id, achievement: a });
      }))
    );
  }

  window.__HERMES_PLUGINS__.register("hermes-achievements", AchievementsPage);
})();
