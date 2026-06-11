import { translateNow } from '@/i18n'
import { requestGatewayForEndpoint, requestGatewayForProfile } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import { remoteSessionEndpoint } from '@/store/remote-sessions'
import { $archivedSessions, $cronSessions, $localDeviceName, $messagingSessions, $sessions } from '@/store/session'
import type { SessionInfo, SessionResumeResponse } from '@/types/hermes'

export interface BulkRuntimeResult {
  ok: string[]
  failed: Array<{ sessionId: string; reason: unknown }>
}

type RuntimeAction = 'halt' | 'prompt' | 'steer'

interface RuntimeTarget {
  endpoint: null | string
  profile: string
  sessionId: string
}

function uniqueSessionIds(sessionIds: readonly string[]): string[] {
  return [...new Set(sessionIds.map(id => id.trim()).filter(Boolean))]
}

function findStoredSession(sessionId: string): SessionInfo | null {
  for (const rows of [$sessions.get(), $messagingSessions.get(), $cronSessions.get(), $archivedSessions.get()]) {
    const session = rows.find(row => row.id === sessionId || row._lineage_root_id === sessionId)

    if (session) {
      return session
    }
  }

  return null
}

function runtimeTarget(sessionId: string): RuntimeTarget {
  const endpoint = remoteSessionEndpoint(sessionId)
  const stored = findStoredSession(sessionId)

  return {
    endpoint,
    profile: normalizeProfileKey(stored?.profile),
    sessionId
  }
}

function requestTarget<T>(target: RuntimeTarget, method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (target.endpoint) {
    return requestGatewayForEndpoint<T>(target.endpoint, method, params)
  }

  return requestGatewayForProfile<T>(target.profile, method, params)
}

function viewerDeviceParams(): { viewer_device?: string } {
  const name = $localDeviceName.get()

  return name ? { viewer_device: name } : {}
}

function remoteSenderParams(target: RuntimeTarget): { sender_device?: string } {
  const name = $localDeviceName.get()

  return target.endpoint && name ? { sender_device: name } : {}
}

async function resumeRuntimeSession(target: RuntimeTarget): Promise<string> {
  const resumed = await requestTarget<SessionResumeResponse>(target, 'session.resume', {
    session_id: target.sessionId,
    cols: 96,
    ...viewerDeviceParams(),
    ...(target.endpoint ? {} : { profile: target.profile })
  })

  return resumed.session_id || target.sessionId
}

function toastSuccess(action: RuntimeAction, count: number) {
  const message =
    action === 'prompt'
      ? translateNow('sidebar.bulk.promptedToast', count)
      : action === 'steer'
        ? translateNow('sidebar.bulk.steeredToast', count)
        : translateNow('sidebar.bulk.haltedToast', count)

  notify({ durationMs: 2_500, kind: 'success', message })
}

function toastFailure(action: RuntimeAction, count: number, reason: unknown) {
  const message =
    action === 'prompt'
      ? translateNow('sidebar.bulk.promptFailed', count)
      : action === 'steer'
        ? translateNow('sidebar.bulk.steerFailed', count)
        : translateNow('sidebar.bulk.haltFailed', count)

  notifyError(reason, message)
}

async function runBulkRuntime(
  action: RuntimeAction,
  sessionIds: readonly string[],
  run: (target: RuntimeTarget) => Promise<void>
): Promise<BulkRuntimeResult> {
  const ok: string[] = []
  const failed: BulkRuntimeResult['failed'] = []

  for (const sessionId of uniqueSessionIds(sessionIds)) {
    const target = runtimeTarget(sessionId)

    try {
      await run(target)
      ok.push(sessionId)
    } catch (reason) {
      failed.push({ reason, sessionId })
    }
  }

  if (ok.length) {
    toastSuccess(action, ok.length)
  }

  if (failed.length) {
    toastFailure(action, failed.length, failed[0].reason)
  }

  return { failed, ok }
}

export function promptStoredSessions(sessionIds: readonly string[], rawText: string): Promise<BulkRuntimeResult> {
  const text = rawText.trim()

  if (!text) {
    return Promise.resolve({ failed: [], ok: [] })
  }

  return runBulkRuntime('prompt', sessionIds, async target => {
    const runtimeId = await resumeRuntimeSession(target)

    await requestTarget(target, 'prompt.submit', {
      session_id: runtimeId,
      text,
      ...remoteSenderParams(target)
    })
  })
}

export function steerStoredSessions(sessionIds: readonly string[], rawText: string): Promise<BulkRuntimeResult> {
  const text = rawText.trim()

  if (!text) {
    return Promise.resolve({ failed: [], ok: [] })
  }

  return runBulkRuntime('steer', sessionIds, async target => {
    const runtimeId = await resumeRuntimeSession(target)
    const result = await requestTarget<{ status?: string }>(target, 'session.steer', { session_id: runtimeId, text })

    if (result?.status === 'rejected') {
      throw new Error('Session has no live tool window to steer')
    }
  })
}

export function haltStoredSessions(sessionIds: readonly string[]): Promise<BulkRuntimeResult> {
  return runBulkRuntime('halt', sessionIds, async target => {
    const runtimeId = await resumeRuntimeSession(target)

    await requestTarget(target, 'session.interrupt', { session_id: runtimeId })
  })
}
