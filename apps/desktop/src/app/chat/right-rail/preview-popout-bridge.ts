/**
 * Cross-window bridge for the popped-out Browser.
 *
 * Each Electron window is its own renderer, so the webview registries that
 * drive_preview / read_preview use live only in the window that mounts the
 * pane. After pop-out that is `?win=browser`, while the chat window still owns
 * the active-session gate for agent tools. This channel lets the gated chat
 * window ask the pop-out to run the live act/read and return the result.
 */

import type { PreviewActAction, PreviewActResult } from '@/lib/preview-act/act-in-page'
import { isBrowserWindow } from '@/store/windows'

import { actOnActivePreview } from './preview-act'
import { activePreviewNav } from './preview-nav'
import { readActivePreview, type PreviewReadOptions, type PreviewReadResult } from './preview-reader'
import { activePreviewScriptRunner } from './preview-script-runner'

const CHANNEL = 'hermes:preview-popout'

const ACT_TIMEOUT_MS = 20_000
const READ_TIMEOUT_MS = 8_000

type ActPayload = Omit<PreviewActAction, 'kind'> & { kind: string }

type BridgeRequest =
  | { id: string; kind: 'act'; payload: ActPayload }
  | { id: string; kind: 'read'; payload: PreviewReadOptions }

type BridgeResponse =
  | { id: string; kind: 'act'; result: PreviewActResult }
  | { id: string; kind: 'read'; result: PreviewReadResult | null }
  | { id: string; kind: 'error'; error: string }

let channel: BroadcastChannel | null = null
let responderInstalled = false
let seq = 0

function getChannel(): BroadcastChannel | null {
  if (typeof BroadcastChannel === 'undefined') {
    return null
  }

  if (!channel) {
    channel = new BroadcastChannel(CHANNEL)
  }

  return channel
}

/** True when this renderer has a live webview (or nav handle) for the active tab. */
export function hasLivePreviewSurface(): boolean {
  return Boolean(activePreviewScriptRunner() || activePreviewNav())
}

function nextId(prefix: string): string {
  seq += 1

  return `${prefix}-${Date.now()}-${seq}`
}

function askPopout<T>(
  request: BridgeRequest,
  timeoutMs: number,
  pick: (response: BridgeResponse) => T | undefined
): Promise<T | null> {
  const bus = getChannel()

  if (!bus) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    const timer = window.setTimeout(() => {
      bus.removeEventListener('message', onMessage)
      resolve(null)
    }, timeoutMs)

    const onMessage = (event: MessageEvent<BridgeResponse | BridgeRequest>) => {
      const data = event.data

      // Ignore our own request echo (same-window tests / unusual hosts) and
      // unrelated traffic. Only a response carries `result` or `error`.
      if (!data || data.id !== request.id) {
        return
      }

      if (!('result' in data) && data.kind !== 'error') {
        return
      }

      window.clearTimeout(timer)
      bus.removeEventListener('message', onMessage)

      if (data.kind === 'error') {
        resolve(null)

        return
      }

      resolve(pick(data) ?? null)
    }

    bus.addEventListener('message', onMessage)
    bus.postMessage(request)
  })
}

/** Ask the browser pop-out to run drive_preview. Null when no pop-out answers. */
export function requestPopoutPreviewAct(payload: ActPayload): Promise<PreviewActResult | null> {
  return askPopout(
    { id: nextId('act'), kind: 'act', payload },
    ACT_TIMEOUT_MS,
    response => (response.kind === 'act' ? response.result : undefined)
  )
}

/** Ask the browser pop-out to run read_preview. Null when no pop-out answers. */
export function requestPopoutPreviewRead(payload: PreviewReadOptions = {}): Promise<PreviewReadResult | null> {
  return askPopout(
    { id: nextId('read'), kind: 'read', payload },
    READ_TIMEOUT_MS,
    response => (response.kind === 'read' ? response.result : undefined)
  )
}

/**
 * Browser pop-out only: answer act/read requests from the chat window.
 * Safe to call more than once; installs a single listener.
 */
export function installPopoutPreviewResponder(): () => void {
  if (!isBrowserWindow() || responderInstalled) {
    return () => {}
  }

  const bus = getChannel()

  if (!bus) {
    return () => {}
  }

  responderInstalled = true

  const onMessage = (event: MessageEvent<BridgeRequest | BridgeResponse>) => {
    const data = event.data

    if (!data?.id || !('payload' in data) || (data.kind !== 'act' && data.kind !== 'read')) {
      return
    }

    void (async () => {
      try {
        if (data.kind === 'act') {
          const result = await actOnActivePreview(data.payload)
          bus.postMessage({ id: data.id, kind: 'act', result } satisfies BridgeResponse)

          return
        }

        const result = await readActivePreview(data.payload)
        bus.postMessage({ id: data.id, kind: 'read', result } satisfies BridgeResponse)
      } catch (error) {
        bus.postMessage({
          id: data.id,
          kind: 'error',
          error: error instanceof Error ? error.message : String(error)
        } satisfies BridgeResponse)
      }
    })()
  }

  bus.addEventListener('message', onMessage)

  return () => {
    bus.removeEventListener('message', onMessage)
    responderInstalled = false
  }
}
