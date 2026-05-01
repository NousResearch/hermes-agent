/**
 * Example Dashboard Plugin
 *
 * Demonstrates how to build a dashboard plugin using the Hermes Plugin SDK.
 * No build step needed — this is a plain IIFE that uses globals from the SDK.
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button } = SDK.components;
  const { useState, useEffect } = SDK.hooks;
  const { cn } = SDK.utils;

  function ExamplePage() {
    const [greeting, setGreeting] = useState(null);
    const [loading, setLoading] = useState(false);

    function fetchGreeting() {
      setLoading(true);
      SDK.fetchJSON("/api/plugins/example/hello")
        .then(function (data) { setGreeting(data.message); })
        .catch(function () { setGreeting("(backend not available)"); })
        .finally(function () { setLoading(false); });
    }

    return React.createElement("div", { className: "flex flex-col gap-6" },
      // Header card
      React.createElement(Card, null,
        React.createElement(CardHeader, null,
          React.createElement("div", { className: "flex items-center gap-3" },
            React.createElement(CardTitle, { className: "text-lg" }, "Example Plugin"),
            React.createElement(Badge, { variant: "outline" }, "v1.0.0"),
          ),
        ),
        React.createElement(CardContent, { className: "flex flex-col gap-4" },
          React.createElement("p", { className: "text-sm text-muted-foreground" },
            "This is an example dashboard plugin. It demonstrates using the Plugin SDK to build ",
            "custom tabs with React components, connect to backend API routes, and integrate with ",
            "the existing Hermes UI system.",
          ),
          React.createElement("div", { className: "flex items-center gap-3" },
            React.createElement(Button, {
              onClick: fetchGreeting,
              disabled: loading,
              className: cn(
                "inline-flex items-center gap-2 border border-border bg-background/40 px-4 py-2",
                "text-sm font-courier transition-colors hover:bg-foreground/10 cursor-pointer",
              ),
            }, loading ? "Loading..." : "Call Backend API"),
            greeting && React.createElement("span", {
              className: "text-sm font-courier text-muted-foreground",
            }, greeting),
          ),
        ),
      ),

      // Info card about the SDK
      React.createElement(Card, null,
        React.createElement(CardHeader, null,
          React.createElement(CardTitle, { className: "text-base" }, "Plugin SDK Reference"),
        ),
        React.createElement(CardContent, null,
          React.createElement("div", { className: "grid gap-3 text-sm" },
            React.createElement("div", { className: "flex flex-col gap-1 border border-border p-3" },
              React.createElement("span", { className: "font-medium" }, "window.__HERMES_PLUGIN_SDK__.React"),
              React.createElement("span", { className: "text-muted-foreground text-xs" }, "React instance — use instead of importing react"),
            ),
            React.createElement("div", { className: "flex flex-col gap-1 border border-border p-3" },
              React.createElement("span", { className: "font-medium" }, "window.__HERMES_PLUGIN_SDK__.hooks"),
              React.createElement("span", { className: "text-muted-foreground text-xs" }, "useState, useEffect, useCallback, useMemo, useRef, useContext, createContext"),
            ),
            React.createElement("div", { className: "flex flex-col gap-1 border border-border p-3" },
              React.createElement("span", { className: "font-medium" }, "window.__HERMES_PLUGIN_SDK__.components"),
              React.createElement("span", { className: "text-muted-foreground text-xs" }, "Card, Badge, Button, Input, Label, Select, Separator, Tabs, etc."),
            ),
            React.createElement("div", { className: "flex flex-col gap-1 border border-border p-3" },
              React.createElement("span", { className: "font-medium" }, "window.__HERMES_PLUGIN_SDK__.api"),
              React.createElement("span", { className: "text-muted-foreground text-xs" }, "Hermes API client — getStatus(), getSessions(), etc."),
            ),
            React.createElement("div", { className: "flex flex-col gap-1 border border-border p-3" },
              React.createElement("span", { className: "font-medium" }, "window.__HERMES_PLUGIN_SDK__.utils"),
              React.createElement("span", { className: "text-muted-foreground text-xs" }, "cn(), timeAgo(), isoTimeAgo()"),
            ),
          ),
        ),
      ),
    );
  }

  // Register this plugin — the dashboard picks it up automatically.
  window.__HERMES_PLUGINS__.register("example", ExamplePage);

  // ─────────────────────────────────────────────────────────────────────
  // Page-scoped slot demo: inject a small banner at the top of /sessions.
  //
  // Built-in pages expose named slots (<page>:top, <page>:bottom) that
  // plugins can populate without overriding the whole route. The
  // manifest lists the slots we use in its `slots` array so the shell
  // knows to render <PluginSlot name="sessions:top" /> there.
  // ─────────────────────────────────────────────────────────────────────
  function SessionsTopBanner() {
    return React.createElement(Card, {
      className: "border-dashed",
    },
      React.createElement(CardContent, { className: "flex items-center gap-3 py-2" },
        React.createElement(Badge, { variant: "outline" }, "Example"),
        React.createElement("span", {
          className: "text-xs text-muted-foreground",
        }, "This banner was injected into the Sessions page by the example plugin via the ",
          React.createElement("code", { className: "font-courier" }, "sessions:top"),
          " slot."),
      ),
    );
  }

  window.__HERMES_PLUGINS__.registerSlot("example", "sessions:top", SessionsTopBanner);
})();
