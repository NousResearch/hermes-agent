/**
 * Isolation Context — three-dimensional tenancy for L0/L1/Profile data.
 *
 * Every write into L0_conversations / l1_records / profiles MUST carry a
 * full IsolationContext (user_id, agent_id, session_id). Every query MUST
 * accept an IsolationFilter so the caller controls which dimensions are
 * narrowed.
 *
 * See `docs/l0l3-tenant-isolation-design.md` for the full design rationale.
 */

/** Default bucket used when caller does not provide explicit isolation fields. */
export const DEFAULT_ISOLATION_ID = "default";
/** Placeholder used when migrating legacy data that has no user/agent assignment. */
export const LEGACY_ISOLATION_PLACEHOLDER = DEFAULT_ISOLATION_ID;

/**
 * Full isolation context required for any write into L0/L1/Profile.
 *
 * - `userId` / `agentId` / `sessionId` are MANDATORY. Empty strings are
 *   rejected unless legacy_compat_mode is enabled (see `assertIsolation`).
 * - `taskId` is an optional business dimension for L0/L1 filtering. It must
 *   never replace sessionId/sessionKey because L1 extraction is session-based.
 */
export interface IsolationContext {
  teamId?: string;
  userId: string;
  agentId: string;
  sessionId: string;
  taskId?: string;
  /** Optional secondary aggregation key kept for legacy callers. */
  sessionKey?: string;
}

/**
 * Isolation filter for queries. Any unset / undefined field means "do not
 * narrow on this dimension". This matches the SQL convention of `WHERE x = ?`
 * being skipped when the parameter is absent.
 */
export interface IsolationFilter {
  teamId?: string;
  userId?: string;
  agentId?: string;
  sessionId?: string;
  taskId?: string;
  sessionKey?: string;
}

/** Runtime config governing isolation enforcement. */
export interface IsolationConfig {
  /** When true (default) writes without full IsolationContext throw. */
  enforce: boolean;
  /** When true the writer fills missing fields with `__legacy__` instead of throwing. */
  legacyCompatMode: boolean;
  /** Placeholder used when legacyCompatMode fills missing fields. */
  legacyPlaceholder: string;
}

export const DEFAULT_ISOLATION_CONFIG: IsolationConfig = {
  enforce: true,
  legacyCompatMode: false,
  legacyPlaceholder: LEGACY_ISOLATION_PLACEHOLDER,
};

/**
 * Validate (and optionally repair) an IsolationContext.
 *
 * @returns A normalised context with `userId/agentId/sessionId` guaranteed
 *          non-empty.
 * @throws  When `enforce=true` and `legacyCompatMode=false` and any of the
 *          three mandatory fields is empty/missing.
 */
export function assertIsolation(
  ctx: Partial<IsolationContext> | undefined,
  config: IsolationConfig = DEFAULT_ISOLATION_CONFIG,
): IsolationContext {
  const teamId = (ctx?.teamId ?? "").trim() || undefined;
  const userId = (ctx?.userId ?? "").trim();
  const agentId = (ctx?.agentId ?? "").trim();
  const sessionId = (ctx?.sessionId ?? "").trim();
  const taskId = ctx?.taskId ?? undefined;
  const sessionKey = ctx?.sessionKey ?? undefined;

  const missing: string[] = [];
  if (!userId) missing.push("userId");
  if (!agentId) missing.push("agentId");
  if (!sessionId) missing.push("sessionId");

  if (missing.length === 0) {
    return { teamId, userId, agentId, sessionId, taskId, sessionKey };
  }

  const placeholder = config.legacyCompatMode ? config.legacyPlaceholder : DEFAULT_ISOLATION_ID;
  return {
    teamId,
    userId: userId || placeholder,
    agentId: agentId || placeholder,
    sessionId: sessionId || placeholder,
    taskId,
    sessionKey,
  };
}

export class IsolationError extends Error {
  constructor(message: string, public readonly missingFields: string[]) {
    super(message);
    this.name = "IsolationError";
  }
}

/**
 * Build a SQL WHERE clause fragment + params for IsolationFilter.
 *
 * Returns:
 *   - `clause`: e.g. `user_id = ? AND agent_id = ?` (no leading WHERE / AND)
 *   - `params`: positional bindings in the same order as the clause
 *
 * Pass `tablePrefix` (e.g. "l1.") if joining multiple tables.
 */
export function buildIsolationWhere(
  filter: IsolationFilter | undefined,
  tablePrefix = "",
): { clause: string; params: string[] } {
  if (!filter) return { clause: "", params: [] };
  const parts: string[] = [];
  const params: string[] = [];
  if (filter.teamId !== undefined) {
    parts.push(`${tablePrefix}team_id = ?`);
    params.push(filter.teamId);
  }
  if (filter.userId !== undefined) {
    parts.push(`${tablePrefix}user_id = ?`);
    params.push(filter.userId);
  }
  if (filter.agentId !== undefined) {
    parts.push(`${tablePrefix}agent_id = ?`);
    params.push(filter.agentId);
  }
  if (filter.sessionId !== undefined) {
    parts.push(`${tablePrefix}session_id = ?`);
    params.push(filter.sessionId);
  }
  if (filter.taskId !== undefined) {
    parts.push(`${tablePrefix}task_id = ?`);
    params.push(filter.taskId);
  }
  if (filter.sessionKey !== undefined) {
    parts.push(`${tablePrefix}session_key = ?`);
    params.push(filter.sessionKey);
  }
  return { clause: parts.join(" AND "), params };
}

/**
 * Check whether a row honours an IsolationFilter (post-retrieve re-check).
 * Used as a safety net after vector / FTS recall, in case the underlying
 * store cannot push the filter down (e.g. older TCVDB collection).
 */
export function rowMatchesIsolation(
  row: { team_id?: string; user_id?: string; agent_id?: string; session_id?: string; task_id?: string; session_key?: string },
  filter: IsolationFilter | undefined,
): boolean {
  if (!filter) return true;
  if (filter.teamId !== undefined && row.team_id !== filter.teamId) return false;
  if (filter.userId !== undefined && row.user_id !== filter.userId) return false;
  if (filter.agentId !== undefined && row.agent_id !== filter.agentId) return false;
  if (filter.sessionId !== undefined && row.session_id !== filter.sessionId) return false;
  if (filter.taskId !== undefined && row.task_id !== filter.taskId) return false;
  if (filter.sessionKey !== undefined && row.session_key !== filter.sessionKey) return false;
  return true;
}
