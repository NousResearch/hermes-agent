import { describe, it, expect } from "vitest";
import { shouldRefreshSessions } from "./session-refresh";

describe("shouldRefreshSessions", () => {
  it("returns false on the first poll (no baseline yet)", () => {
    expect(shouldRefreshSessions(null, "s2")).toBe(false);
  });

  it("returns false when the current response has no sessions", () => {
    expect(shouldRefreshSessions("s1", null)).toBe(false);
    expect(shouldRefreshSessions(null, null)).toBe(false);
  });

  it("returns false when the newest session id is unchanged", () => {
    expect(shouldRefreshSessions("s1", "s1")).toBe(false);
  });

  it("returns true when a new session appears at the head of the list", () => {
    expect(shouldRefreshSessions("s1", "s2")).toBe(true);
  });
});
