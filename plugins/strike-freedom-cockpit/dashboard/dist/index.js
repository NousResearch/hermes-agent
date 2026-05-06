/**
 * Strike Freedom Cockpit — dashboard plugin demo.
 *
 * A slot-only plugin (manifest sets tab.hidden: true) that populates
 * three shell slots when the user has the ``strike-freedom`` theme
 * selected (or any theme that picks layoutVariant: cockpit):
 *
 *   - sidebar       → MS-STATUS panel: ENERGY / SHIELD / POWER bars,
 *                     ZGMF-X20A identity line, pilot block, hero
 *                     render (from --theme-asset-hero when the theme
 *                     provides one).
 *   - header-left   → COMPASS faction crest (uses --theme-asset-crest
 *                     if provided, falls back to a geometric SVG).
 *   - footer-right  → COSMIC ERA tagline that replaces the default
 *                     footer org line.
 *
 * The plugin demonstrates every extension point added alongside the
 * slot system: registerSlot, tab.hidden, reading theme asset CSS vars
 * from plugin code, and rendering above the built-in route content.
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const PLUGINS = window.__HERMES_PLUGINS__;
  if (!SDK || !PLUGINS || !PLUGINS.registerSlot) {
    // Old dashboard bundle without slot support — bail silently rather
    // than breaking the page.
    return;
  }

  const { React } = SDK;
  const { useState, useEffect } = SDK.hooks;
  const { api } = SDK;

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------

  /** Read a CSS custom property from :root. Empty string when unset. */
  function cssVar(name) {
    if (typeof document === "undefined") return "";
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /** Segmented chip progress bar — 10 cells filled proportionally to value. */
  function TelemetryBar(props) {
    const { label, value, color } = props;
    const cells = [];
    for (let i = 0; i < 10; i++) {
      const filled = Math.round(value / 10) > i;
      cells.push(
        React.createElement("span", {
          key: i,
          style: {
            flex: 1,
            height: 8,
            background: filled ? color : "rgba(255,255,255,0.06)",
            transition: "background 200ms",
            clipPath: "polygon(2px 0, 100% 0, calc(100% - 2px) 100%, 0 100%)",
          },
        }),
      );
    }
    return React.createElement(
      "div",
      { style: { display: "flex", flexDirection: "column", gap: 4 } },
      React.createElement(
        "div",
        {
          style: {
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.65rem",
            letterSpacing: "0.12em",
            opacity: 0.75,
          },
        },
        React.createElement("span", null, label),
        React.createElement("span", { style: { color, fontWeight: 700 } }, value + "%"),
      ),
      React.createElement(
        "div",
        { style: { display: "flex", gap: 2 } },
        cells,
      ),
    );
  }

  // ---------------------------------------------------------------------
  // Sidebar: MS-STATUS panel
  // ---------------------------------------------------------------------

  function SidebarSlot() {
    // Pull live-ish numbers from the status API so the plugin isn't just
    // a static decoration. Fall back to full bars if the API is slow /
    // unavailable.
    const [status, setStatus] = useState(null);
    useEffect(function () {
      let cancel = false;
      api.getStatus()
        .then(function (s) { if (!cancel) setStatus(s); })
        .catch(function () {});
      return function () { cancel = true; };
    }, []);

    // Map real status signals to HUD telemetry. Energy/shield/power
    // aren't literal concepts on a software agent, so we read them from
    // adjacent signals: active sessions, gateway connected-platforms,
    // and agent-online health.
    const energy = status && status.gateway_online ? 92 : 18;
    const shield = status && status.connected_platforms
      ? Math.min(100, 40 + (status.connected_platforms.length * 15))
      : 70;
    const power = status && status.active_sessions
      ? Math.min(100, 55 + (status.active_sessions.length * 10))
      : 87;

    const hero = cssVar("--theme-asset-hero");

    return React.createElement(
      "div",
      {
        style: {
          padding: "1rem 0.75rem",
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          fontFamily: "var(--theme-font-display, sans-serif)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          fontSize: "0.65rem",
        },
      },
      // Header line
      React.createElement(
        "div",
        {
          style: {
            borderBottom: "1px solid rgba(64,200,255,0.3)",
            paddingBottom: 8,
            display: "flex",
            flexDirection: "column",
            gap: 2,
          },
        },
        React.createElement("span", { style: { opacity: 0.6 } }, "ms status"),
        React.createElement("span", { style: { fontWeight: 700, fontSize: "0.85rem" } }, "zgmf-x20a"),
        React.createElement("span", { style: { opacity: 0.6, fontSize: "0.6rem" } }, "strike freedom"),
      ),
      // Hero slot — only renders when the theme provides one.
      hero
        ? React.createElement("div", {
            style: {
              width: "100%",
              aspectRatio: "3 / 4",
              backgroundImage: hero,
              backgroundSize: "contain",
              backgroundPosition: "center",
              backgroundRepeat: "no-repeat",
              opacity: 0.85,
            },
            "aria-hidden": true,
          })
        : React.createElement("div", {
            style: {
              width: "100%",
              aspectRatio: "3 / 4",
              border: "1px dashed rgba(64,200,255,0.25)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "0.55rem",
              opacity: 0.4,
            },
          }, "hero slot — set assets.hero in theme"),
      // Pilot block
      React.createElement(
        "div",
        {
          style: {
            borderTop: "1px solid rgba(64,200,255,0.18)",
            borderBottom: "1px solid rgba(64,200,255,0.18)",
            padding: "8px 0",
            display: "flex",
            flexDirection: "column",
            gap: 2,
          },
        },
        React.createElement("span", { style: { opacity: 0.5, fontSize: "0.55rem" } }, "pilot"),
        React.createElement("span", { style: { fontWeight: 700 } }, "hermes agent"),
        React.createElement("span", { style: { opacity: 0.5, fontSize: "0.55rem" } }, "compass"),
      ),
      // Telemetry bars
      React.createElement(TelemetryBar, { label: "energy",  value: energy, color: "#ffce3a" }),
      React.createElement(TelemetryBar, { label: "shield",  value: shield, color: "#3fd3ff" }),
      React.createElement(TelemetryBar, { label: "power",   value: power,  color: "#ff3a5e" }),
      // System online
      React.createElement(
        "div",
        {
          style: {
            marginTop: 4,
            padding: "6px 8px",
            border: "1px solid rgba(74,222,128,0.4)",
            color: "#4ade80",
            textAlign: "center",
            fontWeight: 700,
            fontSize: "0.6rem",
          },
        },
        status && status.gateway_online ? "system online" : "system offline",
      ),
    );
  }

  // ---------------------------------------------------------------------
  // Header-left: COMPASS crest
  // ---------------------------------------------------------------------

  function HeaderCrestSlot() {
    const crest = cssVar("--theme-asset-crest");
    const inner = crest
      ? React.createElement("div", {
          style: {
            width: 28,
            height: 28,
            backgroundImage: crest,
            backgroundSize: "contain",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
          },
          "aria-hidden": true,
        })
      : React.createElement(
          "svg",
          {
            width: 28,
            height: 28,
            viewBox: "0 0 28 28",
            fill: "none",
            stroke: "currentColor",
            strokeWidth: 1.5,
            "aria-hidden": true,
          },
          React.createElement("path", { d: "M14 2 L26 14 L14 26 L2 14 Z" }),
          React.createElement("path", { d: "M14 8 L20 14 L14 20 L8 14 Z" }),
          React.createElement("circle", { cx: 14, cy: 14, r: 2, fill: "currentColor" }),
        );
    return React.createElement(
      "div",
      {
        style: {
          display: "flex",
          alignItems: "center",
          paddingLeft: 12,
          paddingRight: 8,
          color: "var(--color-accent, #3fd3ff)",
        },
      },
      inner,
    );
  }

  // ---------------------------------------------------------------------
  // Footer-right: COSMIC ERA tagline
  // ---------------------------------------------------------------------

  function FooterTaglineSlot() {
    return React.createElement(
      "span",
      {
        style: {
          fontFamily: "var(--theme-font-display, sans-serif)",
          fontSize: "0.6rem",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          opacity: 0.75,
          mixBlendMode: "plus-lighter",
        },
      },
      "compass hermes systems / cosmic era 71",
    );
  }

  // ---------------------------------------------------------------------
  // Hidden tab placeholder — tab.hidden=true means this never renders in
  // the nav, but we still register something sensible in case someone
  // manually navigates to /strike-freedom-cockpit (e.g. via a bookmark).
  // ---------------------------------------------------------------------

  function HiddenPage() {
    return React.createElement(
      "div",
      { style: { padding: "2rem", opacity: 0.6, fontSize: "0.8rem" } },
      "Strike Freedom cockpit is a slot-only plugin — it populates the sidebar, header, and footer instead of showing a tab page.",
    );
  }

  // ---------------------------------------------------------------------
  // Registration
  // ---------------------------------------------------------------------

  const NAME = "strike-freedom-cockpit";
  PLUGINS.register(NAME, HiddenPage);
  PLUGINS.registerSlot(NAME, "sidebar", SidebarSlot);
  PLUGINS.registerSlot(NAME, "header-left", HeaderCrestSlot);
  PLUGINS.registerSlot(NAME, "footer-right", FooterTaglineSlot);
})();
