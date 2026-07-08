#!/usr/bin/env node
/**
 * plan-trigger.js — Trigger classifier for the plan-first workflow.
 *
 * Determines whether a user prompt should enter plan-first mode ("plan")
 * or execute directly ("skip").
 *
 * Usage (ESM):
 *   import { classifyPrompt } from "./plan-trigger.js";
 *   classifyPrompt("build my project"); // => "plan"
 *
 * Usage (CLI):
 *   node plan-trigger.js "build my project"
 */

/**
 * Classify a user message as "plan" (enter plan-first mode) or "skip"
 * (execute directly).
 *
 * Priority order:
 *   1. Override keywords — checked FIRST, force "skip" regardless
 *   2. Skip triggers — atomic/well-understood tasks (rename, read, commit, ask what)
 *   3. Plan triggers — multi-step / investigative / build tasks
 *   4. Default — "skip"
 *
 * All keyword detection is word-boundary aware (e.g. "rebuild" does NOT
 * match the "build" trigger, "building" does not either).
 *
 * @param {string} message - The user's raw prompt.
 * @returns {"skip"|"plan"} Classification decision.
 */
export function classifyPrompt(message) {
  const msg = (message ?? "").trim().toLowerCase();

  // ── Step 1: Override check (highest priority) ────────────────────────
  // Explicit user override: skip plan-first regardless of other triggers.
  if (msg === "go" || msg === "just do it" || /^(skip plan\b)/.test(msg)) {
    return "skip";
  }

  // ── Step 2: Skip triggers ────────────────────────────────────────────
  // These are atomic / well-understood tasks the user wants done now.

  // "rename foo to bar" — straightforward edit
  if (/^rename\b/.test(msg)) return "skip";
  // "what does line 42 do" — asking for explanation
  if (/^what\b/.test(msg)) return "skip";
  // "read file F" — just reading
  if (/^read\b/.test(msg)) return "skip";
  // "commit this" — just commit
  if (/^commit\b/.test(msg)) return "skip";

  // ── Step 3: Plan triggers ────────────────────────────────────────────
  // These are multi-step, investigative, or build tasks.

  // "set up X" — requires setup
  if (/^set\s+up\b/.test(msg)) return "plan";
  // "build Y" — building (word-boundary safe: "rebuild" and "building" don't match)
  if (/\bbuild\b/.test(msg)) return "plan";
  // "migrate Z" — migration
  if (/^migrate\b/.test(msg)) return "plan";
  // "make a plan" — explicit request
  if (/\bmake\s+a\s+plan\b/.test(msg)) return "plan";
  // "why is this broken" — debugging / investigation
  if (/^why\b/.test(msg)) return "plan";

  // ── Default: skip ────────────────────────────────────────────────────
  return "skip";
}

// ── CLI entry point ────────────────────────────────────────────────────
const isCLI = process.argv[1]?.endsWith("plan-trigger.js");
if (isCLI) {
  const input = process.argv.slice(2).join(" ");
  if (!input) {
    console.error("Usage: plan-trigger.js <message>");
    process.exit(1);
  }
  process.stdout.write(classifyPrompt(input) + "\n");
}
