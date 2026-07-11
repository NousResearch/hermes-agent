import { type ApprovalRuntimeMode, coerceApprovalMode, setApprovalMode, setYoloActive } from '@/store/session'

export type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/**
 * Toggle per-session YOLO (approval bypass) via gateway `config.set` — the same
 * session-scoped flag as the TUI's Shift+Tab. It does NOT touch the global
 * `approvals.mode` config, so CLI / TUI / cron behavior is unaffected.
 */
export async function setSessionYolo(
  requestGateway: GatewayRequester,
  sessionId: string,
  enabled: boolean
): Promise<boolean> {
  const result = await requestGateway<{ value?: string }>('config.set', {
    key: 'yolo',
    session_id: sessionId,
    value: enabled ? '1' : '0'
  })

  const active = result?.value === '1'

  setYoloActive(active)

  return active
}

/**
 * Toggle GLOBAL YOLO (approval bypass) via gateway `config.set` with
 * `scope: 'global'`. This flips the persistent `approvals.mode` in config.yaml
 * between `off` (bypass on) and `manual` (bypass off), affecting every session,
 * the CLI, the TUI, and cron — and it survives restarts. Triggered by
 * Shift+clicking the status-bar zap.
 */
export async function setGlobalYolo(
  requestGateway: GatewayRequester,
  enabled: boolean
): Promise<boolean> {
  const result = await requestGateway<{ value?: string }>('config.set', {
    key: 'yolo',
    scope: 'global',
    value: enabled ? '1' : '0'
  })

  const active = result?.value === '1'

  setYoloActive(active)

  return active
}

/**
 * Persist a GLOBAL gating approvals.mode via gateway `config.set` — the two
 * restrictive tiers of the composer's approval pill (hc-514):
 *   manual → gate only detected-dangerous commands
 *   smart  → LLM risk judge decides when to ask
 * Persistent and global (approvals.mode has no per-session form), so it also
 * changes the CLI / TUI / cron default. `off` is deliberately NOT accepted
 * here: the desktop must never persist an unrestricted global default — the
 * pill's "full access" tier arms the session-scoped `setSessionYolo` override
 * instead (temporary, dies with the session). The narrowed parameter type is
 * the static guarantee.
 */
export async function applyApprovalMode(
  requestGateway: GatewayRequester,
  mode: Exclude<ApprovalRuntimeMode, 'off'>
): Promise<ApprovalRuntimeMode> {
  const result = await requestGateway<{ value?: string }>('config.set', {
    key: 'approvals.mode',
    value: mode
  })

  const next = coerceApprovalMode(result?.value)

  setApprovalMode(next)

  return next
}
