import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect } from 'react'

import { gatewayEventCompletedFileDiff } from '@/lib/gateway-events'
import {
  type BrowserDomActionPayload,
  type BrowserDriveAction,
  type BrowserDrivePayload,
  type BrowserElementFingerprint,
  driveBrowser
} from '@/store/browser'
import {
  $previewTarget,
  $sessionPreviewRegistry,
  beginPreviewServerRestart,
  completePreviewServerRestart,
  progressPreviewServerRestart,
  requestPreviewReload,
  restoreRightRailForSession,
  setPreviewTarget
} from '@/store/preview'
import { $currentCwd } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

type EventHandler = (event: RpcEvent) => void

interface PreviewRoutingOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  baseHandleGatewayEvent: EventHandler
  currentCwd: string
  currentView: string
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  routedSessionId: string | null
  selectedStoredSessionId: string | null
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

function activePreviewSessionId(
  activeSessionIdRef: MutableRefObject<string | null>,
  routedSessionId: string | null,
  selectedStoredSessionId: string | null
): string {
  return selectedStoredSessionId || routedSessionId || activeSessionIdRef.current || ''
}

const BROWSER_DRIVE_ACTIONS = new Set<BrowserDriveAction>([
  'act',
  'goBack',
  'goForward',
  'navigate',
  'open',
  'reload',
  'snapshot'
])

function browserDomAction(value: unknown): BrowserDomActionPayload | undefined {
  const record = asRecord(value)
  const kind = typeof record.kind === 'string' ? record.kind : ''

  if (!['click', 'press', 'scroll', 'select', 'setValue', 'type'].includes(kind)) {
    return undefined
  }

  return {
    kind: kind as BrowserDomActionPayload['kind'],
    ...(typeof record.amount === 'number' ? { amount: record.amount } : {}),
    ...(record.direction === 'down' || record.direction === 'left' || record.direction === 'right' || record.direction === 'up'
      ? { direction: record.direction }
      : {}),
    ...(typeof record.expectedUrl === 'string' ? { expectedUrl: record.expectedUrl } : {}),
    ...(typeof record.index === 'number' ? { index: record.index } : {}),
    ...(typeof record.key === 'string' ? { key: record.key } : {}),
    ...(typeof record.selector === 'string' ? { selector: record.selector } : {}),
    ...(browserElementFingerprint(record.target) ? { target: browserElementFingerprint(record.target) } : {}),
    ...(typeof record.text === 'string' ? { text: record.text } : {}),
    ...(typeof record.value === 'string' ? { value: record.value } : {})
  }
}

function browserElementFingerprint(value: unknown): BrowserElementFingerprint | undefined {
  const record = asRecord(value)
  const fingerprint: BrowserElementFingerprint = {}

  for (const key of ['ariaLabel', 'href', 'id', 'name', 'placeholder', 'role', 'tag', 'text', 'type'] as const) {
    if (typeof record[key] === 'string' && record[key]) {
      fingerprint[key] = record[key]
    }
  }

  return Object.keys(fingerprint).length ? fingerprint : undefined
}

function browserDrivePayload(payload: unknown): BrowserDrivePayload | null {
  const record = asRecord(payload)
  const action = typeof record.action === 'string' ? record.action : ''

  if (!BROWSER_DRIVE_ACTIONS.has(action as BrowserDriveAction)) {
    return null
  }

  return {
    action: action as BrowserDriveAction,
    ...(record.domAction ? { domAction: browserDomAction(record.domAction) } : {}),
    ...(typeof record.requestId === 'string' ? { requestId: record.requestId } : {}),
    ...(typeof record.title === 'string' ? { title: record.title } : {}),
    ...(typeof record.url === 'string' ? { url: record.url } : {})
  }
}

export function usePreviewRouting({
  activeSessionIdRef,
  baseHandleGatewayEvent,
  currentCwd,
  currentView,
  requestGateway,
  routedSessionId,
  selectedStoredSessionId
}: PreviewRoutingOptions) {
  const previewRegistry = useStore($sessionPreviewRegistry)
  const previewSessionId = activePreviewSessionId(activeSessionIdRef, routedSessionId, selectedStoredSessionId)

  // Restore a *user-opened* preview when its session becomes active. Tool
  // results no longer auto-register/open a preview — the inline preview card in
  // the tool row is the only entry point, so HTML artifacts never pop the rail
  // open on their own.
  useEffect(() => {
    if (currentView !== 'chat') {
      setPreviewTarget(null)

      return
    }

    restoreRightRailForSession(previewSessionId)
  }, [currentView, previewRegistry, previewSessionId])

  const restartPreviewServer = useCallback(
    async (url: string, context?: string) => {
      const sessionId = activeSessionIdRef.current

      if (!sessionId) {
        throw new Error('No active session for background restart')
      }

      const cwd = $currentCwd.get() || currentCwd || ''

      const result = await requestGateway<{ task_id?: string }>('preview.restart', {
        context: context || undefined,
        cwd: cwd || undefined,
        session_id: sessionId,
        url
      })

      const taskId = result.task_id || ''

      if (!taskId) {
        throw new Error('Background restart did not return a task id')
      }

      beginPreviewServerRestart(taskId, url)

      return taskId
    },
    [activeSessionIdRef, currentCwd, requestGateway]
  )

  const answerBrowserRequest = useCallback(
    async (event: RpcEvent) => {
      const request = asRecord(event.payload)
      const requestId = typeof request.request_id === 'string' ? request.request_id : ''

      if (!requestId) {
        return
      }

      const respond = async (result: Record<string, unknown>) => {
        await requestGateway('browser.respond', {
          request_id: requestId,
          text: JSON.stringify(result)
        })
      }

      const sessionId = event.session_id || ''

      if (!sessionId || sessionId !== activeSessionIdRef.current) {
        await respond({
          error: 'Browser requests can only control the active Desktop session',
          ok: false
        })

        return
      }

      const browser = window.hermesDesktop?.browser
      const operation = typeof request.operation === 'string' ? request.operation : ''
      const payload = asRecord(request.payload)

      try {
        if (!browser) {
          throw new Error('Desktop browser bridge is unavailable')
        }

        const settledSnapshot = async ({
          previousUrl = '',
          requestedUrl = '',
          requireUrlChange = false
        }: {
          previousUrl?: string
          requestedUrl?: string
          requireUrlChange?: boolean
        } = {}) => {
          let lastError = 'Desktop browser snapshot failed'

          for (let attempt = 0; attempt < 40; attempt += 1) {
            await new Promise(resolve => window.setTimeout(resolve, attempt === 0 ? 250 : 200))
            const state = await browser.getState?.(sessionId)

            if (state?.loading) {
              continue
            }

            try {
              const snapshot = await browser.snapshot?.(sessionId)

              if (snapshot?.ok) {
                const stillPreviousUrl = Boolean(previousUrl) && snapshot.url === previousUrl

                if (
                  stillPreviousUrl &&
                  (requireUrlChange || (Boolean(requestedUrl) && requestedUrl !== previousUrl))
                ) {
                  continue
                }

                return snapshot
              }

              lastError = snapshot?.error || lastError
            } catch (error) {
              lastError = error instanceof Error ? error.message : String(error)
            }
          }

          throw new Error(lastError)
        }

        if (operation === 'snapshot') {
          if (!browser.snapshot) {
            throw new Error('Desktop browser snapshot API is unavailable')
          }

          await respond({ ok: true, snapshot: await browser.snapshot(sessionId) })

          return
        }

        if (operation === 'navigate') {
          const url = typeof payload.url === 'string' ? payload.url : ''

          if (!url || !browser.navigate) {
            throw new Error('Desktop browser navigate request is invalid')
          }

          const previousState = await browser.getState?.(sessionId)
          await browser.navigate(url, sessionId)
          await respond({
            ok: true,
            snapshot: await settledSnapshot({ previousUrl: previousState?.url, requestedUrl: url })
          })

          return
        }

        if (operation === 'back') {
          if (!browser.goBack) {
            throw new Error('Desktop browser back API is unavailable')
          }

          const previousState = await browser.getState?.(sessionId)
          await browser.goBack(sessionId)
          await respond({
            ok: true,
            snapshot: await settledSnapshot({ previousUrl: previousState?.url, requireUrlChange: true })
          })

          return
        }

        if (operation === 'action') {
          const action = browserDomAction(payload.action)

          if (!action || !browser.act) {
            throw new Error('Desktop browser action request is invalid')
          }

          const actionResult = await browser.act(action, sessionId)

          if (!actionResult?.ok) {
            await respond({
              action: actionResult,
              error: actionResult?.error || 'Desktop browser action failed',
              ok: false
            })

            return
          }

          await respond({ action: actionResult, ok: true, snapshot: await settledSnapshot() })

          return
        }

        throw new Error(`Unsupported Desktop browser operation: ${operation || '(missing)'}`)
      } catch (error) {
        await respond({
          error: error instanceof Error ? error.message : String(error),
          ok: false
        })
      }
    },
    [activeSessionIdRef, requestGateway]
  )

  const handleDesktopGatewayEvent = useCallback<EventHandler>(
    event => {
      baseHandleGatewayEvent(event)

      if (event.type === 'preview.restart.complete') {
        const { task_id, text } = asRecord(event.payload)

        if (typeof task_id === 'string' && task_id) {
          completePreviewServerRestart(task_id, typeof text === 'string' ? text : '')
        }
      } else if (event.type === 'preview.restart.progress') {
        const { task_id, text } = asRecord(event.payload)

        if (typeof task_id === 'string' && task_id) {
          progressPreviewServerRestart(task_id, typeof text === 'string' ? text : '')
        }
      } else if (event.type === 'browser.request') {
        void answerBrowserRequest(event)
      } else if (event.type === 'browser.drive') {
        if (event.session_id && event.session_id !== activeSessionIdRef.current) {
          return
        }

        const payload = browserDrivePayload(event.payload)

        if (payload) {
          driveBrowser({ ...payload, sessionId: event.session_id || activeSessionIdRef.current || undefined })
        }
      }

      if (event.session_id && event.session_id !== activeSessionIdRef.current) {
        return
      }

      // Only refresh an already-open live preview when a file changes; never
      // open one unprompted. (Preview links are surfaced from the tool row into
      // the status stack — see tool-fallback.tsx.)
      if ($previewTarget.get()?.kind === 'url' && gatewayEventCompletedFileDiff(event)) {
        requestPreviewReload()
      }
    },
    [activeSessionIdRef, answerBrowserRequest, baseHandleGatewayEvent]
  )

  return { handleDesktopGatewayEvent, restartPreviewServer }
}
