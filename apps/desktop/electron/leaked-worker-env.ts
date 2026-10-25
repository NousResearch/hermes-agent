// A Desktop launched from inside a kanban worker / delegate child (e.g. `hermes desktop` or an
// update reopening the app with `open` from a worker's terminal) inherits that worker's identity env. Every backend, terminal
// and CLI the Desktop spawns would then believe it is that worker's delegate child: kanban tools
// refuse ("delegate_task child contexts cannot mutate Kanban") and the board is pinned to the
// worker's DB. The Desktop is never a worker, so drop the
// whole agent-scope family at startup. Mirrors agent/delegation_context.py
// (strip_agent_scope_env); tests/hermes_cli/test_agent_scope_env.py fails on drift.

// Presence of any of these means the env was inherited from a worker or delegate child.
export const WORKER_MARKER_KEYS = [
  'HERMES_DELEGATED_CHILD_CONTEXT',
  'HERMES_KANBAN_TASK',
  'HERMES_KANBAN_WORKSPACE',
  'HERMES_KANBAN_RUN_ID',
  'HERMES_KANBAN_CLAIM_LOCK'
] as const

export const AGENT_SCOPE_EXACT_KEYS = [
  'HERMES_DELEGATED_CHILD_CONTEXT',
  'HERMES_SINGLE_QUERY_SESSION',
  'HERMES_SPAWN',
  'HERMES_SUPERVISED_CHILD'
] as const

export const AGENT_SCOPE_PREFIXES = [
  'HERMES_KANBAN_',
  'HERMES_SESSION_',
  'HERMES_CRON_',
  'HERMES_PARENT_',
  'HERMES_RPC_',
  'HERMES_KERNEL_'
] as const

// Operator tunables sharing those prefixes are config, not identity: keep them.
export const AGENT_SCOPE_TUNABLE_RE = /(_TIMEOUT|_TIMEOUT_S|_POLL_S|_SECONDS|_MAX_PARALLEL)$/

export function isAgentScopeEnvKey(key: string): boolean {
  if ((AGENT_SCOPE_EXACT_KEYS as readonly string[]).includes(key)) {
    return true
  }

  return AGENT_SCOPE_PREFIXES.some(prefix => key.startsWith(prefix)) && !AGENT_SCOPE_TUNABLE_RE.test(key)
}

/** Remove leaked agent-scope keys from `env` in place; returns the removed key names. Without a
 * worker marker nothing is removed: a board pin or session var set alone is an operator choice. */
export function scrubLeakedWorkerEnv(env: NodeJS.ProcessEnv = process.env): string[] {
  if (!WORKER_MARKER_KEYS.some(key => Boolean(env[key]))) {
    return []
  }

  const removed = Object.keys(env).filter(isAgentScopeEnvKey)

  for (const key of removed) {
    delete env[key]
  }

  return removed
}
