/**
 * Session filtering for memory-tdai.
 *
 * Decides whether a session should be ignored by the memory plugin
 * (capture, recall, pipeline scheduling). All skip rules are compiled
 * into a flat list of matchers at construction time — zero per-call overhead.
 */

// ============================
// Types
// ============================

export interface AgentHookContext {
  sessionKey?: string;
  sessionId?: string;
  trigger?: string;
}

type SessionKeyMatcher = (sessionKey: string) => boolean;

// ============================
// Non-interactive trigger detection
// ============================

const SKIP_TRIGGERS = new Set(["cron", "heartbeat", "automation", "schedule"]);

/**
 * Returns true when the hook was fired by a non-interactive trigger
 * (heartbeat, cron job, automation, etc.) — these produce no meaningful
 * user conversation and should not be captured or counted.
 */
export function isNonInteractiveTrigger(trigger?: string, sessionKey?: string): boolean {
  if (trigger && SKIP_TRIGGERS.has(trigger.toLowerCase())) return true;
  if (sessionKey) {
    if (/:cron:/i.test(sessionKey) || /:heartbeat:/i.test(sessionKey)) return true;
  }
  return false;
}

// ============================
// Built-in skip rules (always active)
// ============================

/**
 * Hard-coded matchers that identify internal / non-user sessions.
 * These are always applied regardless of user configuration.
 */
const BUILTIN_MATCHERS: SessionKeyMatcher[] = [
  // Scene extraction runner sessions
  (key) => key.includes(":memory-scene-extract-"),
  // OpenClaw subagent sessions
  (key) => key.includes(":subagent:"),
  // Temporary / internal utility sessions (e.g. temp:slug-generator)
  (key) => key.startsWith("temp:"),
];

// ============================
// Glob → matcher compiler
// ============================

/**
 * Turn a simple glob pattern (only `*` supported) into a matcher
 * that tests the full sessionKey.
 *
 * Since sessionKeys look like `agent:<agentId>:...`, we match the
 * glob against the whole key so users can write patterns like
 * `bench-judge-*` (matched anywhere) or more specific ones.
 */
function globToMatcher(pattern: string): SessionKeyMatcher {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  const re = new RegExp(escaped);
  return (key) => re.test(key);
}

// ============================
// SessionFilter
// ============================

/**
 * Unified filter: construct once at plugin startup, then call
 * `shouldSkip(sessionKey)` or `shouldSkipCtx(ctx)` at each gate.
 */
export class SessionFilter {
  private readonly matchers: SessionKeyMatcher[];

  constructor(excludeAgents: string[] = []) {
    // Merge built-in rules + user-configured exclude patterns into one flat list
    const userMatchers = excludeAgents
      .map((p) => p.trim())
      .filter((p) => p.length > 0)
      .map(globToMatcher);

    this.matchers = [...BUILTIN_MATCHERS, ...userMatchers];
  }

  /** Should this sessionKey be skipped? */
  shouldSkip(sessionKey: string): boolean {
    return this.matchers.some((m) => m(sessionKey));
  }

  /** Should this hook context be skipped? */
  shouldSkipCtx(ctx: AgentHookContext): boolean {
    if (!ctx.sessionKey) return true;
    if (ctx.sessionId?.startsWith("memory-")) return true;
    if (isNonInteractiveTrigger(ctx.trigger, ctx.sessionKey)) return true;
    return this.shouldSkip(ctx.sessionKey);
  }
}
