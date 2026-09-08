import { $gateway } from '@/store/gateway'
import { $activeSessionId, setYoloActive } from '@/store/session'

export type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/** Writer into a session's runtime slice — `updateSessionState` from the
 *  session-state cache owner, registered once by the wiring. */
export type SessionYoloSliceWriter = (sessionId: string, yolo: boolean) => void

let sliceWriter: SessionYoloSliceWriter | null = null

/**
 * The cache that owns per-session runtime state registers its writer here so
 * every YOLO toggle (slash, ⌘K, status bar, session-create) lands in the
 * session's own slice. That slice is the authority every later
 * `updateSessionState` re-syncs the composer atoms from
 * (`syncRuntimeMetadataToView`) — an atom-only write let the very next
 * transcript append (`/yolo`'s own "YOLO on" system line) flip the indicator
 * straight back off.
 */
export function registerSessionYoloSliceWriter(writer: SessionYoloSliceWriter | null): () => void {
  sliceWriter = writer

  return () => {
    if (sliceWriter === writer) {
      sliceWriter = null
    }
  }
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

  sliceWriter?.(sessionId, active)

  // A background tile's toggle must never repaint the primary's indicator.
  if (!sliceWriter || sessionId === $activeSessionId.get()) {
    setYoloActive(active)
  }

  return active
}

/**
 * Toggle GLOBAL YOLO (approval bypass) via gateway `config.set` with
 * `scope: 'global'`. This flips the persistent `approvals.mode` in config.yaml
 * between `off` (bypass on) and `manual` (bypass off), affecting every session,
 * the CLI, the TUI, and cron — and it survives restarts. Triggered by
 * Shift+clicking the status-bar zap.
 */
export async function setGlobalYolo(requestGateway: GatewayRequester, enabled: boolean): Promise<boolean> {
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
 * Set YOLO to an explicit state from a surface that has no React context — the
 * ⌘K rows. `useSlashCommand` keeps its own `requestGateway` (it already holds
 * one, with the reconnect handling), so this reaches the active gateway
 * directly rather than growing a second requester abstraction.
 *
 * With no session yet the flag is armed locally; the session-create path
 * (use-session-actions) applies it on the first message, exactly as a bare
 * `/yolo` in a fresh draft does.
 */
export async function setYoloEnabled(enabled: boolean): Promise<boolean> {
  const sessionId = $activeSessionId.get()

  if (!sessionId) {
    setYoloActive(enabled)

    return enabled
  }

  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Hermes gateway unavailable')
  }

  return setSessionYolo((method, params) => gateway.request(method, params), sessionId, enabled)
}
