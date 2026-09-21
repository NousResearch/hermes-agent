(function(){"use strict";var SDK=window.__HERMES_PLUGIN_SDK__;var React=SDK&&SDK.React;if(!SDK||!React||!window.__HERMES_PLUGINS__){return;}
(() => {
  // src/index.tsx
  var STORAGE_KEY = "kensei-mc-config";
  var DEFAULT_CONFIG = {
    gatewayApiKey: "",
    postizEmail: "",
    postizPassword: "",
    workspaceUrl: ""
  };
  function loadConfig() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
    } catch {
    }
    return DEFAULT_CONFIG;
  }
  function saveConfig(cfg) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
  }
  function SettingsPanel() {
    const { useState } = SDK.hooks;
    const { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Label, Separator } = SDK.components;
    const [config, setConfig] = useState(loadConfig);
    const [saved, setSaved] = useState(false);
    const onSave = () => {
      saveConfig(config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2e3);
    };
    const set = (key, val) => setConfig((prev) => ({ ...prev, [key]: val }));
    return /* @__PURE__ */ React.createElement("div", { className: "p-6 space-y-6 max-w-2xl mx-auto" }, /* @__PURE__ */ React.createElement("header", { className: "space-y-1" }, /* @__PURE__ */ React.createElement("h1", { className: "text-xl font-semibold tracking-tight" }, "KENSEI Settings"), /* @__PURE__ */ React.createElement("p", { className: "text-sm text-muted-foreground" }, "Credentials for KENSEI plugins. Stored in localStorage on this device.")), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardHeader, null, /* @__PURE__ */ React.createElement(CardTitle, null, "Workspace iframe URL")), /* @__PURE__ */ React.createElement(CardContent, { className: "space-y-3" }, /* @__PURE__ */ React.createElement("p", { className: "text-sm text-muted-foreground" }, "URL the Workspace tab embeds. Leave blank to auto-detect (uses", /* @__PURE__ */ React.createElement("code", { className: "mx-1" }, `<protocol>://<hostname>:3002/?chrome=0`), "). Set this if you access the dashboard through a tunnel and the auto-detect points at the wrong host."), /* @__PURE__ */ React.createElement("div", { className: "space-y-2" }, /* @__PURE__ */ React.createElement(Label, { htmlFor: "kensei-workspace-url" }, "URL"), /* @__PURE__ */ React.createElement(
      Input,
      {
        id: "kensei-workspace-url",
        type: "url",
        value: config.workspaceUrl,
        onChange: (e) => set("workspaceUrl", e.target.value),
        placeholder: "https://kensei-prod.taild3d5e0.ts.net/?chrome=0"
      }
    )))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardHeader, null, /* @__PURE__ */ React.createElement(CardTitle, null, "Postiz")), /* @__PURE__ */ React.createElement(CardContent, { className: "space-y-3" }, /* @__PURE__ */ React.createElement("p", { className: "text-sm text-muted-foreground" }, "For the Postiz plugin to authenticate against your local Postiz instance on :4007."), /* @__PURE__ */ React.createElement("div", { className: "space-y-2" }, /* @__PURE__ */ React.createElement(Label, { htmlFor: "kensei-postiz-email" }, "Email"), /* @__PURE__ */ React.createElement(
      Input,
      {
        id: "kensei-postiz-email",
        type: "email",
        autoComplete: "email",
        value: config.postizEmail,
        onChange: (e) => set("postizEmail", e.target.value),
        placeholder: "you@example.com"
      }
    )), /* @__PURE__ */ React.createElement("div", { className: "space-y-2" }, /* @__PURE__ */ React.createElement(Label, { htmlFor: "kensei-postiz-password" }, "Password"), /* @__PURE__ */ React.createElement(
      Input,
      {
        id: "kensei-postiz-password",
        type: "password",
        autoComplete: "current-password",
        value: config.postizPassword,
        onChange: (e) => set("postizPassword", e.target.value),
        placeholder: "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022"
      }
    )))), /* @__PURE__ */ React.createElement(Card, null, /* @__PURE__ */ React.createElement(CardHeader, null, /* @__PURE__ */ React.createElement(CardTitle, null, "Gateway")), /* @__PURE__ */ React.createElement(CardContent, { className: "space-y-3" }, /* @__PURE__ */ React.createElement("p", { className: "text-sm text-muted-foreground" }, "For plugins that hit the Hermes Gateway directly (not via dashboard token)."), /* @__PURE__ */ React.createElement("div", { className: "space-y-2" }, /* @__PURE__ */ React.createElement(Label, { htmlFor: "kensei-gateway-key" }, "API key"), /* @__PURE__ */ React.createElement(
      Input,
      {
        id: "kensei-gateway-key",
        type: "password",
        autoComplete: "off",
        value: config.gatewayApiKey,
        onChange: (e) => set("gatewayApiKey", e.target.value),
        placeholder: "hermes API server key"
      }
    )))), /* @__PURE__ */ React.createElement(Separator, null), /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-3" }, /* @__PURE__ */ React.createElement(Button, { onClick: onSave }, "Save"), saved && /* @__PURE__ */ React.createElement(Badge, { variant: "outline", className: "border-emerald-500/40 text-emerald-300" }, "Saved")));
  }
  function GearIcon() {
    return React.createElement(
      "svg",
      {
        width: 16,
        height: 16,
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: 2,
        strokeLinecap: "round",
        strokeLinejoin: "round",
        "aria-hidden": "true"
      },
      React.createElement("path", {
        d: "M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
      }),
      React.createElement("circle", { cx: 12, cy: 12, r: 3 })
    );
  }
  function SettingsHeaderWidget() {
    const { Button } = SDK.components;
    return React.createElement(
      Button,
      {
        size: "icon",
        variant: "ghost",
        title: "KENSEI Settings",
        "aria-label": "KENSEI Settings",
        onClick: () => {
          window.location.href = "/kensei-settings";
        }
      },
      React.createElement(GearIcon)
    );
  }
  window.__HERMES_PLUGINS__.register("kensei-settings", SettingsPanel);
  window.__HERMES_PLUGINS__.registerSlot("kensei-settings", "header-right", SettingsHeaderWidget);
})();
})();
