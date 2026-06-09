import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import {
  $desktopYoloDefault,
  $yoloActive,
  DEFAULT_DESKTOP_YOLO_ACTIVE,
  setDesktopYoloDefaultActive,
  setYoloActive
} from '@/store/session'

export type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function desktopYoloDefaultFromConfig(config: { desktop?: unknown }): boolean {
  const desktop = isRecord(config.desktop) ? config.desktop : {}

  return typeof desktop.yolo_default === 'boolean' ? desktop.yolo_default : DEFAULT_DESKTOP_YOLO_ACTIVE
}

export async function setDesktopYoloDefault(enabled: boolean): Promise<boolean> {
  const config = await getHermesConfigRecord()
  const desktop = isRecord(config.desktop) ? { ...config.desktop } : {}

  await saveHermesConfig({
    ...config,
    desktop: {
      ...desktop,
      yolo_default: enabled
    }
  })

  setDesktopYoloDefaultActive(enabled)

  return enabled
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

export async function setDesktopYoloMode(
  requestGateway: GatewayRequester,
  sessionId: string | null,
  enabled: boolean
): Promise<boolean> {
  const previousDefault = $desktopYoloDefault.get()
  const previousActive = $yoloActive.get()

  setDesktopYoloDefaultActive(enabled)
  setYoloActive(enabled)

  try {
    await setDesktopYoloDefault(enabled)

    if (sessionId) {
      await setSessionYolo(requestGateway, sessionId, enabled)
    }

    setYoloActive(enabled)

    return enabled
  } catch (err) {
    setDesktopYoloDefaultActive(previousDefault)
    setYoloActive(previousActive)

    try {
      await setDesktopYoloDefault(previousDefault)
    } catch {
      // Keep the original failure as the actionable error.
    }

    if (sessionId) {
      try {
        await setSessionYolo(requestGateway, sessionId, previousActive)
      } catch {
        // Best-effort rollback only.
      }
    }

    throw err
  }
}
