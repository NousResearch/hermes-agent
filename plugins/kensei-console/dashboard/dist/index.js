(function () {
  "use strict";
  var SDK = window.__HERMES_PLUGIN_SDK__;
  var PLUGINS = window.__HERMES_PLUGINS__;
  if (!SDK || !PLUGINS) return;

  var React = SDK.React;
  var hooks = SDK.hooks || {};
  var useEffect = hooks.useEffect;
  var useState = hooks.useState;

  var NAME = "kensei-console";
  var STORAGE_KEY = "kensei-mc-config";
  var DEFAULT_TAILSCALE = "https://kensei-prod.taild3d5e0.ts.net/?chrome=0";

  // Resolve the Workspace URL with this priority:
  //   1. ?ws=<url> in current location (per-tab override)
  //   2. window.__KENSEI_WORKSPACE_URL__ (runtime override)
  //   3. localStorage kensei-mc-config.workspaceUrl (set in KENSEI Settings)
  //   4. Tailscale Serve URL (default)
  // Loopback fallback was dropped: when the browser hits the dashboard via SSH
  // tunnel or Tailscale, hostname is "localhost"/"127.0.0.1" but :3002 lives on
  // the VPS not the laptop. The Tailscale URL works from VPS, laptop, and phone.
  function resolveWorkspaceUrl() {
    try {
      var params = new URLSearchParams(window.location.search);
      var override = params.get("ws");
      if (override) return override;
    } catch (e) {}
    if (typeof window.__KENSEI_WORKSPACE_URL__ === "string" && window.__KENSEI_WORKSPACE_URL__) {
      return window.__KENSEI_WORKSPACE_URL__;
    }
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        var cfg = JSON.parse(raw);
        if (cfg && typeof cfg.workspaceUrl === "string" && cfg.workspaceUrl.trim()) {
          return cfg.workspaceUrl.trim();
        }
      }
    } catch (e) {}
    return DEFAULT_TAILSCALE;
  }

  function WorkspacePage() {
    var s = useState({ src: resolveWorkspaceUrl(), loaded: false, errored: false });
    var state = s[0];
    var setState = s[1];

    useEffect(function () {
      // Most cross-origin iframe load failures don't fire onError reliably,
      // so we fall back to a "show diagnostic UI after 6s if not loaded".
      var t = setTimeout(function () {
        setState(function (prev) {
          if (prev.loaded) return prev;
          return Object.assign({}, prev, { errored: true });
        });
      }, 6000);
      return function () { clearTimeout(t); };
    }, []);

    var iframe = React.createElement("iframe", {
      src: state.src,
      title: "Hermes Workspace",
      // Dashboard wraps non-chat plugin routes without flex-fill, so height:100%
      // collapses to 0. Anchor to viewport height minus the dashboard header
      // (~5rem) and bottom padding (~2rem) to fill the visible area.
      style: { width: "100%", height: "calc(100vh - 7rem)", minHeight: "60vh", border: 0, display: "block" },
      onLoad: function () {
        setState(function (prev) { return Object.assign({}, prev, { loaded: true, errored: false }); });
      },
      onError: function () {
        setState(function (prev) { return Object.assign({}, prev, { errored: true }); });
      },
    });

    var diagnostic = state.errored && !state.loaded
      ? React.createElement(
          "div",
          { className: "p-6 max-w-2xl mx-auto space-y-4 text-sm" },
          React.createElement("h2", { className: "text-base font-semibold" }, "Workspace iframe didn't load"),
          React.createElement(
            "p",
            { className: "text-muted-foreground" },
            "Tried: ",
            React.createElement(
              "code",
              { className: "font-mono break-all" },
              state.src,
            ),
          ),
          React.createElement(
            "p",
            { className: "text-muted-foreground" },
            "Likely causes: hermes-workspace isn't running on :3002, the URL isn't reachable from this browser, or its CSP/auth blocks the embed.",
          ),
          React.createElement(
            "p",
            { className: "text-muted-foreground" },
            "Set a different URL in ",
            React.createElement(
              "a",
              { href: "/kensei-settings", className: "underline" },
              "KENSEI Settings",
            ),
            " (Workspace iframe URL) — most likely your Tailscale Serve URL.",
          ),
          React.createElement(
            "div",
            { className: "flex gap-2" },
            React.createElement(
              "a",
              {
                href: state.src,
                target: "_blank",
                rel: "noreferrer",
                className: "inline-flex items-center px-3 py-1.5 text-xs rounded-md border border-border hover:bg-accent",
              },
              "Open in new tab",
            ),
          ),
        )
      : null;

    return React.createElement(
      "div",
      { className: "w-full", style: { display: "flex", flexDirection: "column" } },
      iframe,
      diagnostic,
    );
  }

  PLUGINS.register(NAME, WorkspacePage);
})();
