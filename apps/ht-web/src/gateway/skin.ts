// Maps the gateway `skin` payload onto the frontend's brand state + CSS vars,
// so branding is server-driven (gateway.ready / skin.changed) rather than
// hardcoded. Pure: returns a resolved object; the caller writes CSS vars.

import type { GatewaySkin } from "./types";

export interface ResolvedSkin {
  agentName: string;
  promptSymbol: string;
  helpHeader: string;
  /** CSS custom properties to set on :root, e.g. { "--ht-accent": "#8E7CFF" }. */
  cssVars: Record<string, string>;
}

export const DEFAULT_SKIN: ResolvedSkin = {
  agentName: "HT AI Agent",
  promptSymbol: "❯",
  helpHeader: "Commands",
  cssVars: {
    "--ht-accent": "#8E7CFF",
    "--ht-accent-strong": "#C9BFFF",
    "--ht-border": "#5B4BEA",
  },
};

// Gateway color keys → our CSS var names. Only the handful a chat UI needs.
const COLOR_MAP: Record<string, string> = {
  ui_accent: "--ht-accent",
  banner_accent: "--ht-accent",
  banner_title: "--ht-accent-strong",
  banner_border: "--ht-border",
  ui_error: "--ht-error",
  ui_ok: "--ht-ok",
};

export function resolveSkin(skin: GatewaySkin | undefined | null): ResolvedSkin {
  if (!skin) return DEFAULT_SKIN;
  const branding = skin.branding ?? {};
  const colors = skin.colors ?? {};
  const cssVars: Record<string, string> = { ...DEFAULT_SKIN.cssVars };

  for (const [gatewayKey, cssVar] of Object.entries(COLOR_MAP)) {
    const value = colors[gatewayKey];
    if (typeof value === "string" && /^#[0-9a-fA-F]{3,8}$/.test(value)) {
      cssVars[cssVar] = value;
    }
  }

  const promptRaw = (branding.prompt_symbol ?? "").trim();

  return {
    agentName: branding.agent_name || DEFAULT_SKIN.agentName,
    promptSymbol: promptRaw || DEFAULT_SKIN.promptSymbol,
    helpHeader: skin.help_header || branding.help_header || DEFAULT_SKIN.helpHeader,
    cssVars,
  };
}

/** Write resolved CSS vars onto a target element (defaults to document root). */
export function applySkinVars(skin: ResolvedSkin, target?: HTMLElement): void {
  const el = target ?? (typeof document !== "undefined" ? document.documentElement : null);
  if (!el) return;
  for (const [key, value] of Object.entries(skin.cssVars)) {
    el.style.setProperty(key, value);
  }
}
