import { describe, it, expect } from "vitest";
import { NAV_ITEMS } from "./nav";

describe("NAV_ITEMS", () => {
  it("has a chat route at / and unique paths", () => {
    const chat = NAV_ITEMS.filter((n) => n.group === "chat");
    expect(chat).toHaveLength(1);
    expect(chat[0]!.to).toBe("/");
    const paths = NAV_ITEMS.map((n) => n.to);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it("management routes are all under a distinct path and non-empty", () => {
    const manage = NAV_ITEMS.filter((n) => n.group === "manage");
    expect(manage.length).toBeGreaterThan(0);
    for (const item of manage) {
      expect(item.to.startsWith("/")).toBe(true);
      expect(item.to).not.toBe("/");
      expect(item.label.length).toBeGreaterThan(0);
      expect(item.glyph.length).toBeGreaterThan(0);
    }
  });
});
