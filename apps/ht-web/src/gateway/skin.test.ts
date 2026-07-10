import { describe, it, expect } from "vitest";
import { resolveSkin, DEFAULT_SKIN } from "./skin";

describe("resolveSkin", () => {
  it("returns defaults for a missing skin", () => {
    expect(resolveSkin(undefined)).toEqual(DEFAULT_SKIN);
    expect(resolveSkin(null)).toEqual(DEFAULT_SKIN);
  });

  it("maps branding and valid hex colors from the gateway payload", () => {
    const r = resolveSkin({
      branding: { agent_name: "HT AI Agent", prompt_symbol: " ❯ " },
      colors: { ui_accent: "#8E7CFF", banner_title: "#C9BFFF" },
      help_header: "Commands",
    });
    expect(r.agentName).toBe("HT AI Agent");
    expect(r.promptSymbol).toBe("❯");
    expect(r.cssVars["--ht-accent"]).toBe("#8E7CFF");
    expect(r.cssVars["--ht-accent-strong"]).toBe("#C9BFFF");
  });

  it("ignores non-hex color values (defends against garbage payloads)", () => {
    const r = resolveSkin({ colors: { ui_accent: "javascript:alert(1)" } });
    expect(r.cssVars["--ht-accent"]).toBe(DEFAULT_SKIN.cssVars["--ht-accent"]);
  });

  it("falls back to the default name when branding omits agent_name", () => {
    expect(resolveSkin({ colors: {} }).agentName).toBe("HT AI Agent");
  });
});
