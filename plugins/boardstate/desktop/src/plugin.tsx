// Hermes DESKTOP app plugin — the Board as a first-class desktop page.
//
// The desktop plugin loader executes this file as ESM in the renderer realm and only
// resolves `@hermes/plugin-sdk` and `react*`; every other import is REJECTED. So unlike
// the web tab (which loads the vendored `<boardstate-view>` bundle as a separate static
// asset), everything here — createWsTransport, the Lit element definitions, the CSS,
// the theme map, the templates — is INLINED by esbuild into a single `plugin.js`.
//
// Backend: identical to the web plugin. The desktop app spawns the same Python backend,
// so `/api/plugins/boardstate/*` (WS bridge + MCP proxy + sidecar) already exists. We
// build the same WS URL the web tab uses, sourcing base + token from the desktop bridge
// (`window.hermesDesktop.getConnection()`) instead of the web SDK's `buildWsUrl`.

import { host, ROUTES_AREA, SIDEBAR_NAV_AREA } from "@hermes/plugin-sdk";
import { useCallback, useEffect, useRef, useState } from "react";
import { createWsTransport, type WsTransport } from "@boardstate/core";
import "@boardstate/lit/browser"; // side effect: registers <boardstate-view> + builtins
import boardstateCss from "../vendor/boardstate.css"; // esbuild text loader → string
import skinDesktopCss from "./skin-desktop.css"; // Hermes DESKTOP skin (macOS language)
import { BS_TO_DESKTOP, aliasChain, themeBase } from "../src/theme";
import { TEMPLATES } from "../src/templates";
import { withOperatorGate } from "../src/operator-transport";

type Connection = { baseUrl: string; token: string; authMode?: string };
declare global {
  interface Window {
    hermesDesktop?: { getConnection?: () => Promise<Connection | null> };
  }
}

// Inject the Boardstate stylesheet once (the desktop app has no manifest `css` hook).
let cssInjected = false;
function ensureCss(): void {
  if (cssInjected || document.querySelector("style[data-boardstate]")) {
    cssInjected = true;
    return;
  }
  const style = document.createElement("style");
  style.setAttribute("data-boardstate", "");
  // Boardstate base sheet first, then the DESKTOP skin (macOS design language) so
  // the skin's scoped class-level rules win over the bundle's own defaults.
  style.textContent = `${boardstateCss as unknown as string}\n${skinDesktopCss as unknown as string}`;
  document.head.appendChild(style);
  cssInjected = true;
}

// Same var()-alias theme adapter as the web tab, against the desktop `--ui-*` tokens.
function applyDesktopTheme(view: HTMLElement): void {
  const bg = getComputedStyle(document.body).backgroundColor || "rgb(0,0,0)";
  view.setAttribute("data-theme", themeBase(bg));
  for (const [bsVar, uiVars] of Object.entries(BS_TO_DESKTOP)) {
    view.style.setProperty(bsVar, aliasChain(uiVars));
  }
  // macOS design language: adopt the app's own card rounding (--radius-xl ≈ 9.6px),
  // tighter control radii, a single subtle elevation shadow (not the bundle's heavy
  // two-layer default), and the host system font stack (SF on macOS).
  view.style.setProperty("--bs-radius-lg", "var(--radius-xl, 10px)");
  view.style.setProperty("--bs-radius-md", "0.375rem");
  view.style.setProperty("--bs-radius-sm", "0.25rem");
  view.style.setProperty("--bs-shadow-md", "0 1px 3px rgba(0,0,0,0.10)");
  view.style.setProperty("--bs-font-sans", getComputedStyle(document.body).fontFamily);
}

type ViewElement = HTMLElement & { transport?: unknown; connected?: boolean; basePath?: string; operator?: boolean };

/** POST an operator decision through the plugin's own namespaced REST door (`ctx.rest` →
 *  `/api/plugins/boardstate/operator`); resolve the sidecar's raw RPC result. */
type OperatorRest = <T>(path: string, opts?: { method?: string; body?: unknown }) => Promise<T>;

function BoardPage({ operatorRest }: { operatorRest?: OperatorRest }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const transportRef = useRef<WsTransport | undefined>(undefined);
  const [status, setStatus] = useState<"connecting" | "live" | "error">("connecting");
  const [detail, setDetail] = useState("");
  const [applying, setApplying] = useState("");

  const applyTemplate = useCallback(async (name: string, doc: unknown) => {
    const transport = transportRef.current;
    if (!transport) return;
    if (!window.confirm(`Replace the current board with the "${name}" template?`)) return;
    setApplying(name);
    try {
      await transport.request("dashboard.workspace.replace", { doc, actor: "user" });
    } catch (err) {
      host.notify?.({ kind: "error", message: `Template failed: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setApplying("");
    }
  }, []);

  useEffect(() => {
    ensureCss();
    let disposed = false;
    let transport: WsTransport | undefined;
    let view: ViewElement | undefined;
    let obs: MutationObserver | undefined;

    (async () => {
      const conn = await window.hermesDesktop?.getConnection?.().catch(() => null);
      if (disposed) return;
      if (!conn) {
        setStatus("error");
        setDetail("No desktop gateway connection.");
        return;
      }
      if (conn.authMode === "oauth") {
        // WS tickets are single-use / core-managed on OAuth remotes; the token-URL WS
        // won't authenticate. Surface it rather than half-connect. (Poll fallback: TODO.)
        setStatus("error");
        setDetail("The live board needs a local gateway (OAuth remote not yet supported).");
        return;
      }
      const wsBase = conn.baseUrl.replace(/^http/, "ws");
      const wsUrl = `${wsBase}/api/plugins/boardstate/ws?token=${encodeURIComponent(conn.token)}`;
      // Route the four operator verbs through the plugin_api operator endpoint via the
      // desktop REST door (the WS + MCP stay blocked); everything else rides the WS.
      const sendOperator = async (method: string, params: unknown): Promise<unknown> => {
        if (!operatorRest) throw new Error("operator endpoint unavailable");
        const res = await operatorRest<{ result?: unknown }>("/operator", { method: "POST", body: { method, params } });
        return res?.result;
      };
      transport = withOperatorGate(createWsTransport(wsUrl), sendOperator);
      transportRef.current = transport;
      view = document.createElement("boardstate-view") as ViewElement;
      view.transport = transport;
      view.connected = true;
      view.operator = true;
      // The desktop page runs on a file:// origin, so the custom-widget asset base must
      // be ABSOLUTE. The backend hands out a tokenized root-level base (iframes can't
      // carry auth headers); fetched here through the plugin's authed REST door. No
      // base ⇒ builtins-only, no errors.
      try {
        const ab = operatorRest ? await operatorRest<{ base?: string }>("/assets-base", { method: "GET" }) : undefined;
        view.basePath = ab?.base ? `${conn.baseUrl.replace(/\/+$/, "")}${ab.base}` : "";
      } catch {
        view.basePath = "";
      }
      applyDesktopTheme(view);
      obs = new MutationObserver(() => view && applyDesktopTheme(view));
      obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "style", "data-theme"] });
      obs.observe(document.body, { attributes: true, attributeFilter: ["class", "style"] });
      view.style.display = "block";
      view.style.height = "100%";
      hostRef.current?.appendChild(view);
      transport.ready
        .then(() => !disposed && setStatus("live"))
        .catch((err: unknown) => {
          if (!disposed) {
            setStatus("error");
            setDetail(err instanceof Error ? err.message : String(err));
          }
        });
    })();

    return () => {
      disposed = true;
      obs?.disconnect();
      transportRef.current = undefined;
      try {
        transport?.close();
      } catch {
        /* already closed */
      }
      if (view && view.parentNode) view.parentNode.removeChild(view);
    };
  }, []);

  const dotColor = status === "live" ? "var(--ui-green, #6aa84f)" : status === "error" ? "var(--ui-red, #e06c75)" : "var(--ui-yellow, #d0a94f)";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 8, padding: 12 }}>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8, fontSize: 12 }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: dotColor, display: "inline-block" }} />
        <span style={{ opacity: 0.8 }}>
          {status === "live" ? "Board connected" : status === "error" ? `Board unavailable${detail ? `: ${detail}` : ""}` : "Connecting to board…"}
        </span>
        {status === "live" ? (
          <span style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 6, marginLeft: 8 }}>
            <span style={{ opacity: 0.7 }}>Templates:</span>
            {TEMPLATES.map((tpl) => (
              <button
                key={tpl.id}
                type="button"
                title={tpl.summary}
                disabled={applying !== ""}
                onClick={() => applyTemplate(tpl.name, tpl.doc)}
                style={{
                  cursor: applying ? "default" : "pointer",
                  padding: "3px 10px",
                  borderRadius: 6,
                  border: "1px solid var(--ui-stroke-secondary, #2a2a33)",
                  background: applying === tpl.id ? "var(--ui-row-active-background, #23232b)" : "transparent",
                  color: "inherit",
                  opacity: applying && applying !== tpl.name ? 0.5 : 1,
                }}
              >
                {applying === tpl.name ? "Applying…" : tpl.name}
              </button>
            ))}
          </span>
        ) : null}
      </div>
      <div ref={hostRef} style={{ flex: 1, minHeight: 0 }} />
    </div>
  );
}

export default {
  id: "boardstate",
  name: "Board",
  register(ctx: {
    register: (c: { id: string; area: unknown; data?: unknown; render?: () => unknown }) => void;
    rest?: OperatorRest;
  }) {
    // A full page in the workspace pane… The plugin's namespaced REST door (`ctx.rest`) is
    // threaded in as the operator transport's send path (→ /api/plugins/boardstate/operator).
    ctx.register({ id: "board-route", area: ROUTES_AREA, data: { path: "/board" }, render: () => <BoardPage operatorRest={ctx.rest} /> });
    // …reachable from a sidebar nav row.
    ctx.register({ id: "board-nav", area: SIDEBAR_NAV_AREA, data: { path: "/board", label: "Board", codicon: "dashboard" } });
  },
};
