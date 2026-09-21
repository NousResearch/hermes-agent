(function(){"use strict";var SDK=window.__HERMES_PLUGIN_SDK__;var React=SDK&&SDK.React;if(!SDK||!React||!window.__HERMES_PLUGINS__){return;}
(() => {
  // src/index.tsx
  function readLogoUrl() {
    try {
      const raw = getComputedStyle(document.documentElement).getPropertyValue("--theme-asset-logo-raw").trim();
      return raw.replace(/^["']|["']$/g, "");
    } catch {
      return "";
    }
  }
  function BrandWidget() {
    const { useState, useEffect } = SDK.hooks;
    const [logoUrl, setLogoUrl] = useState(readLogoUrl);
    useEffect(() => {
      const target = document.documentElement;
      const observer = new MutationObserver(() => {
        const next = readLogoUrl();
        setLogoUrl((prev) => prev === next ? prev : next);
      });
      observer.observe(target, { attributes: true, attributeFilter: ["style", "class", "data-theme"] });
      return () => observer.disconnect();
    }, []);
    return /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2.5 min-w-0" }, logoUrl && /* @__PURE__ */ React.createElement(
      "img",
      {
        src: logoUrl,
        alt: "",
        className: "h-7 w-7 shrink-0",
        style: { objectFit: "contain" }
      }
    ), /* @__PURE__ */ React.createElement("div", { className: "leading-tight min-w-0" }, /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "font-semibold text-base tracking-[0.04em] truncate",
        style: { fontFamily: "var(--theme-font-display, var(--theme-font-sans))" }
      },
      "KENSEI"
    ), /* @__PURE__ */ React.createElement("div", { className: "text-[10px] uppercase tracking-[0.20em] text-muted-foreground truncate" }, "Console")));
  }
  window.__HERMES_PLUGINS__.registerSlot("kensei-brand", "header-left", BrandWidget);
})();
})();
