// Boardstate dashboard tab for Hermes.
//
// A thin React component (host React from the Plugin SDK — never bundles its own)
// that mounts the real `<boardstate-view>` custom element and drives it over the
// networked WebSocket transport to the plugin's sidecar. The element definitions
// live in the vendored `@boardstate/lit/browser` bundle (loaded once as a static
// asset the host serves); `createWsTransport` (from `@boardstate/core`) is bundled
// into this file. React/react-dom are NOT bundled — everything comes from
// `window.__HERMES_PLUGIN_SDK__`.

import { createWsTransport, type WsTransport } from "@boardstate/core";
import { BS_TO_HERMES, aliasChain, themeBase } from "./theme";
import { TEMPLATES } from "./templates";
import { withOperatorGate } from "./operator-transport";
import skinCss from "./skin-web.css"; // esbuild text loader → string

const SDK = window.__HERMES_PLUGIN_SDK__!;
const React = SDK.React;

// Authenticated WS endpoint on the Hermes dashboard origin; the backend
// (`plugin_api.py`) bridges it to the loopback sidecar. `buildWsUrl` attaches the
// correct auth query param (loopback token / gated single-use ticket).
const WS_PATH = "/api/plugins/boardstate/ws";
// The privileged operator endpoint (approve/confirm/deny). Same-origin authed fetch — the
// browser WS + MCP can NOT reach the operator verbs, this route is the only path.
const OPERATOR_PATH = "/api/plugins/boardstate/operator";

// POST an operator decision through the authed plugin_api route; resolve the sidecar's raw
// RPC result. `fetchJSON` attaches the session credential + throws on a 401/403/refusal.
async function sendOperator(method: string, params: unknown): Promise<unknown> {
  const res = (await SDK.fetchJSON(OPERATOR_PATH, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ method, params }),
  })) as { result?: unknown };
  return res?.result;
}
// Static asset served by the host from `dashboard/vendor/` — registers
// `<boardstate-view>` and every built-in widget renderer.
const BUNDLE_URL = "/dashboard-plugins/boardstate/vendor/boardstate-browser.js";

// Load the element bundle exactly once, even across tab remounts.
let bundlePromise: Promise<unknown> | null = null;
function ensureBundle(): Promise<unknown> {
  if (!bundlePromise) {
    // Runtime dynamic import of a static URL — kept out of the esbuild graph so the
    // browser (not the bundler) fetches the vendored asset.
    const url = BUNDLE_URL;
    bundlePromise = import(/* @vite-ignore */ url).catch((err) => {
      bundlePromise = null;
      throw err;
    });
  }
  return bundlePromise;
}

type ViewElement = HTMLElement & {
  transport?: unknown;
  connected?: boolean;
  basePath?: string;
  operator?: boolean;
};

// Inject the Hermes web skin once (class-level rules the tokens can't express).
// Idempotent, like the desktop plugin's ensureCss.
let skinInjected = false;
function ensureSkin(): void {
  if (skinInjected || document.querySelector("style[data-boardstate-skin]")) {
    skinInjected = true;
    return;
  }
  const style = document.createElement("style");
  style.setAttribute("data-boardstate-skin", "");
  style.textContent = skinCss as unknown as string;
  document.head.appendChild(style);
  skinInjected = true;
}

// ── Hermes theme adapter (DOM glue; pure mapping lives in ./theme) ───────────
// A var() alias re-resolves whenever Hermes rewrites its own tokens, so live
// palette swaps repaint the board with zero JS; only the light/dark base is
// computed here, from the host's painted background.
function applyHermesTheme(view: HTMLElement): void {
  const bodyBg = getComputedStyle(document.body).backgroundColor || "rgb(0,0,0)";
  view.setAttribute("data-theme", themeBase(bodyBg));
  for (const [bsVar, hermesVars] of Object.entries(BS_TO_HERMES)) {
    view.style.setProperty(bsVar, aliasChain(hermesVars));
  }
  // Non-color skin tokens: inherit the host font, flatten tiles (no shadow), adopt
  // Hermes radii, and make the tile translucent so the board reads as native chrome.
  view.style.setProperty("--bs-font-sans", getComputedStyle(document.body).fontFamily);
  view.style.setProperty("--bs-shadow-md", "none");
  view.style.setProperty("--bs-radius-lg", "0.5rem");
  view.style.setProperty("--bs-radius-md", "0.375rem");
  view.style.setProperty("--bs-radius-sm", "0.25rem");
  // Translucent tile (the Kanban look) — ONLY when the host actually defines the
  // card token. A fixed fallback color here would force a dark card onto light
  // non-Hermes hosts; skipping the override keeps the bundle's own theme-correct card.
  const hostCard = getComputedStyle(document.documentElement)
    .getPropertyValue("--color-card")
    .trim();
  if (hostCard) {
    view.style.setProperty("--bs-card", "color-mix(in srgb, var(--color-card) 85%, transparent)");
  } else {
    view.style.removeProperty("--bs-card");
  }
}

function observeHermesTheme(view: HTMLElement): () => void {
  // The var() aliases auto-follow token value changes; the observer only needs to
  // re-evaluate the light/dark base when a palette swap flips the background.
  const obs = new MutationObserver(() => applyHermesTheme(view));
  obs.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class", "style", "data-theme"],
  });
  obs.observe(document.body, { attributes: true, attributeFilter: ["class", "style"] });
  return () => obs.disconnect();
}

function BoardPage() {
  const hostRef = React.useRef<HTMLDivElement | null>(null);
  const transportRef = React.useRef<WsTransport | undefined>(undefined);
  const [status, setStatus] = React.useState<"connecting" | "live" | "error">("connecting");
  const [detail, setDetail] = React.useState<string>("");
  const [applying, setApplying] = React.useState<string>("");

  // Apply a template by replacing the workspace doc through the authed bridge.
  // `dashboard.workspace.replace` is a non-operator mutation (same RPC the Import
  // button uses); the template's data-source builtins self-bind to live Hermes data.
  const applyTemplate = React.useCallback(async (id: string, name: string, docValue: unknown) => {
    const transport = transportRef.current;
    if (!transport) return;
    if (!window.confirm(`Replace the current board with the "${name}" template?`)) return;
    setApplying(id);
    try {
      await transport.request("dashboard.workspace.replace", { doc: docValue, actor: "user" });
    } catch (err) {
      window.alert(`Could not apply template: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setApplying("");
    }
  }, []);

  React.useEffect(() => {
    let disposed = false;
    let transport: WsTransport | undefined;
    let view: ViewElement | undefined;
    let disposeTheme: (() => void) | undefined;

    ensureSkin();
    (async () => {
      try {
        await ensureBundle();
        const wsUrl = await SDK.buildWsUrl(WS_PATH);
        if (disposed) return;
        // Wrap the live transport so the four operator verbs route to the plugin_api operator
        // endpoint (the only privileged path), while everything else rides the WS. `operator:
        // true` then renders the approve/confirm affordances enabled.
        transport = withOperatorGate(createWsTransport(wsUrl), sendOperator);
        transportRef.current = transport;
        view = document.createElement("boardstate-view") as ViewElement;
        view.transport = transport;
        view.connected = true;
        view.operator = true;
        // Built-in widgets resolve from the bundle; approved custom widgets would
        // resolve under the sidecar's own /widgets route (out of scope for v1).
        view.basePath = "";
        // Follow the active Hermes palette (light/dark base + `--bs-*` aliases)
        // and keep following it across live palette swaps.
        applyHermesTheme(view);
        disposeTheme = observeHermesTheme(view);
        view.style.display = "block";
        view.style.minHeight = "70vh";
        hostRef.current?.appendChild(view);
        transport.ready
          .then(() => {
            if (!disposed) setStatus("live");
          })
          .catch((err: unknown) => {
            if (!disposed) {
              setStatus("error");
              setDetail(err instanceof Error ? err.message : String(err));
            }
          });
      } catch (err) {
        if (!disposed) {
          setStatus("error");
          setDetail(err instanceof Error ? err.message : String(err));
        }
      }
    })();

    return () => {
      disposed = true;
      disposeTheme?.();
      transportRef.current = undefined;
      try {
        transport?.close();
      } catch {
        /* already closed */
      }
      if (view && view.parentNode) view.parentNode.removeChild(view);
    };
  }, []);

  const dot =
    status === "live" ? "#6aa84f" : status === "error" ? "#e06c75" : "#d0a94f";

  return React.createElement(
    "div",
    { className: "bs-plugin-root", style: { display: "flex", flexDirection: "column", gap: "8px" } },
    React.createElement(
      "div",
      {
        className: "bs-plugin-status",
        style: { display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", opacity: 0.8 },
      },
      React.createElement("span", {
        style: {
          width: "8px",
          height: "8px",
          borderRadius: "50%",
          background: dot,
          display: "inline-block",
        },
      }),
      React.createElement(
        "span",
        null,
        status === "live"
          ? "Board connected"
          : status === "error"
            ? `Board unavailable${detail ? `: ${detail}` : ""}`
            : "Connecting to board…",
      ),
    ),
    // Template quick-start: one click swaps in a live-bound board. Only offered once
    // the board is connected (the picker needs the transport).
    status === "live"
      ? React.createElement(
          "div",
          {
            className: "bs-template-picker",
            style: { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px", fontSize: "12px" },
          },
          React.createElement("span", { style: { opacity: 0.7, marginRight: "2px" } }, "Templates:"),
          ...TEMPLATES.map((tpl) =>
            React.createElement(
              "button",
              {
                key: tpl.id,
                type: "button",
                title: tpl.summary,
                disabled: applying !== "",
                onClick: () => applyTemplate(tpl.id, tpl.name, tpl.doc),
                style: {
                  cursor: applying ? "default" : "pointer",
                  padding: "3px 10px",
                  borderRadius: "var(--bs-radius-md, 6px)",
                  border: "1px solid var(--color-border, #2a2a33)",
                  background: applying === tpl.id ? "var(--color-muted, #23232b)" : "transparent",
                  color: "inherit",
                  opacity: applying && applying !== tpl.id ? 0.5 : 1,
                },
              },
              applying === tpl.id ? "Applying…" : tpl.name,
            ),
          ),
        )
      : null,
    React.createElement("div", { ref: hostRef, className: "bs-view-host", style: { flex: 1 } }),
  );
}

// Register the tab component under the manifest name.
window.__HERMES_PLUGINS__!.register("boardstate", BoardPage as never);
