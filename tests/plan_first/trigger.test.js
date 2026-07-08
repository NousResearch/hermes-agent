/**
 * trigger.test.js — Invariant-based tests for plan-trigger.js.
 *
 * Tests are organized by behavior category, not as a snapshot of current
 * inputs/outputs. Each group asserts a behavioral invariant:
 *   • Override: explicit override words always force "skip"
 *   • Skip triggers: atomic tasks produce "skip"
 *   • Plan triggers: multi-step tasks produce "plan"
 *   • Word boundaries: compound words don't falsely trigger
 *   • Edge cases: empty input, whitespace, punctuation variations
 */

import { describe, it, expect } from "vitest";
import { classifyPrompt } from "../../scripts/plan-trigger.js";

// ── Invariant: Overrides always produce "skip" ──────────────────────────

describe("override — explicit override words force skip", () => {
  it('"go" alone returns skip', () => {
    expect(classifyPrompt("go")).toBe("skip");
  });

  it('"just do it" returns skip', () => {
    expect(classifyPrompt("just do it")).toBe("skip");
  });

  it('"skip plan" prefix returns skip', () => {
    expect(classifyPrompt("skip plan")).toBe("skip");
  });

  it('"skip plan and build X" returns skip despite "build" trigger', () => {
    expect(classifyPrompt("skip plan and build the server")).toBe("skip");
  });

  it("override takes priority over all other triggers", () => {
    // "skip plan" should win over "set up", "build", "migrate"
    expect(classifyPrompt("skip plan set up the database")).toBe("skip");
    expect(classifyPrompt("skip plan migrate the schema")).toBe("skip");
    expect(classifyPrompt("skip plan why is this broken")).toBe("skip");
  });
});

// ── Invariant: Atomic/well-understood tasks produce "skip" ──────────────

describe("skip triggers — atomic tasks execute directly", () => {
  it('"rename" prefix returns skip', () => {
    expect(classifyPrompt("rename foo to bar")).toBe("skip");
    expect(classifyPrompt("rename the function to calculateTotal")).toBe("skip");
    expect(classifyPrompt("rename the variable")).toBe("skip");
  });

  it('"what" prefix returns skip (informational)', () => {
    expect(classifyPrompt("what does line 42 do")).toBe("skip");
    expect(classifyPrompt("what is this variable")).toBe("skip");
    expect(classifyPrompt("what does this function return")).toBe("skip");
  });

  it('"read" prefix returns skip', () => {
    expect(classifyPrompt("read file F")).toBe("skip");
    expect(classifyPrompt("read the config")).toBe("skip");
    expect(classifyPrompt("read src/index.js")).toBe("skip");
  });

  it('"commit" prefix returns skip', () => {
    expect(classifyPrompt("commit this")).toBe("skip");
    expect(classifyPrompt("commit my changes")).toBe("skip");
    expect(classifyPrompt("commit the work")).toBe("skip");
  });
});

// ── Invariant: Multi-step / build / investigate tasks produce "plan" ────

describe("plan triggers — complex tasks enter plan-first mode", () => {
  it('"set up" prefix returns plan', () => {
    expect(classifyPrompt("set up X")).toBe("plan");
    expect(classifyPrompt("set up the database")).toBe("plan");
    expect(classifyPrompt("set up CI/CD pipeline")).toBe("plan");
  });

  it('word-boundary "build" returns plan', () => {
    expect(classifyPrompt("build Y")).toBe("plan");
    expect(classifyPrompt("build the API server")).toBe("plan");
    expect(classifyPrompt("build the React frontend")).toBe("plan");
  });

  it('"migrate" prefix returns plan', () => {
    expect(classifyPrompt("migrate Z")).toBe("plan");
    expect(classifyPrompt("migrate the database")).toBe("plan");
    expect(classifyPrompt("migrate from v1 to v2")).toBe("plan");
  });

  it('"make a plan" phrase returns plan', () => {
    expect(classifyPrompt("make a plan")).toBe("plan");
    expect(classifyPrompt("please make a plan for this feature")).toBe("plan");
    expect(classifyPrompt("can you make a plan")).toBe("plan");
  });

  it('"why" prefix returns plan (debug/investigation)', () => {
    expect(classifyPrompt("why is this broken")).toBe("plan");
    expect(classifyPrompt("why is the build failing")).toBe("plan");
    expect(classifyPrompt("why does this crash")).toBe("plan");
  });
});

// ── Invariant: Word-boundary awareness prevents false matches ──────────

describe("word boundaries — compound words do not falsely trigger", () => {
  it('"rebuild" does not match the "build" trigger', () => {
    expect(classifyPrompt("rebuild the cache")).toBe("skip");
  });

  it('"building" does not match the "build" trigger', () => {
    expect(classifyPrompt("building the image failed")).toBe("skip");
  });

  it('"setup" (no space) does not match "set up" trigger', () => {
    expect(classifyPrompt("setup the environment")).toBe("skip");
  });

  it('"migration" does not match "migrate" prefix', () => {
    expect(classifyPrompt("migration plan")).toBe("skip");
  });

  it('"commitment" does not match "commit"', () => {
    expect(classifyPrompt("commitment to quality")).toBe("skip");
  });

  it('"readiness" does not match "read"', () => {
    expect(classifyPrompt("readiness check")).toBe("skip");
  });

  it('"builder" does not match "build"', () => {
    expect(classifyPrompt("builder pattern")).toBe("skip");
  });
});

// ── Invariant: Edge cases handle gracefully ─────────────────────────────

describe("edge cases — empty, whitespace, variations", () => {
  it("empty string returns skip", () => {
    expect(classifyPrompt("")).toBe("skip");
  });

  it("whitespace-only returns skip", () => {
    expect(classifyPrompt("   ")).toBe("skip");
  });

  it("null/undefined returns skip", () => {
    expect(classifyPrompt(null)).toBe("skip");
    expect(classifyPrompt(undefined)).toBe("skip");
  });

  it("arbitrary text with no keyword match returns skip", () => {
    expect(classifyPrompt("hello world")).toBe("skip");
    expect(classifyPrompt("check the docs")).toBe("skip");
    expect(classifyPrompt("list files")).toBe("skip");
  });

  it("case insensitivity: uppercase works same as lowercase", () => {
    expect(classifyPrompt("BUILD my project")).toBe("plan");
    expect(classifyPrompt("RENAME foo")).toBe("skip");
    expect(classifyPrompt("WHAT is this")).toBe("skip");
    expect(classifyPrompt("GO")).toBe("skip");
  });
});
