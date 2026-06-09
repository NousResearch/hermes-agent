import { setYoloActive } from '@/store/session'
import type { ApprovalMode } from '@/types/hermes'

export type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

const APPROVAL_MODES: ReadonlySet<string> = new Set(['manual', 'smart', 'off'])

export function normalizeApprovalMode(value: unknown): ApprovalMode | null {
  const mode = String(value ?? '')
    .trim()
    .toLowerCase()

  return APPROVAL_MODES.has(mode) ? (mode as ApprovalMode) : null
}

export async function getGlobalApprovalMode(requestGateway: GatewayRequester): Promise<ApprovalMode> {
  const result = await requestGateway<{ value?: string }>('config.get', {
    key: 'approvals.mode'
  })
  const mode = normalizeApprovalMode(result?.value) ?? 'manual'

  return mode
}

export async function setGlobalApprovalMode(
  requestGateway: GatewayRequester,
  mode: ApprovalMode
): Promise<ApprovalMode> {
  const result = await requestGateway<{ value?: string }>('config.set', {
    key: 'approvals.mode',
    value: mode
  })
  const activeMode = normalizeApprovalMode(result?.value) ?? mode

  return activeMode
}

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
