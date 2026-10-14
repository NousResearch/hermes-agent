/**
 * Cross-window bridge for the popped-out Browser.
 *
 * Each Electron window is its own renderer, so the webview registries that
 * drive_preview / read_preview use live only in the window that mounts the
 * pane. After pop-out that is `?win=browser`, while the chat window still owns
 * the active-session gate for agent tools (`server-requests.ts`
 * WINDOW_OWNED_REQUESTS). This channel lets the gated chat window ask the
 * pop-out to run the live act/read and return the result.
 *
 * Transport: the same main-process IPC relay the Comment Mode handoff uses
 * (`window.hermesDesktop.windowRelay`). Packaged builds load renderers over
 * `file://`, where BroadcastChannel origin semantics must not be trusted;
 * BroadcastChannel stays only as a dev/browser fallback (dev serves the
 * renderer over http, so same-origin channels work there).
 */

import type { PreviewActAction, PreviewActResult } from '@/lib/preview-act/act-in-page'
import { previewTabIdsVisibleTo } from '@/store/preview'
import type { PreviewOwner } from '@/store/preview-ownership'
import { isBrowserWindow, windowBrowserTabId } from '@/store/windows'

import { actOnActivePreview } from './preview-act'
import { activePreviewNav } from './preview-nav'
import { type PreviewReadOptions, type PreviewReadResult, readActivePreview } from './preview-reader'
import { activePreviewScriptRunner } from './preview-script-runner'

const CHANNEL = 'hermes:preview-popout'

const ACT_TIMEOUT_MS = 20_000
const READ_TIMEOUT_MS = 8_000

type ActPayload = Omit<PreviewActAction, 'kind'> & { kind: string }

/** `tabIds`: the tabs the requesting session may see, resolved in the chat
 *  window (it alone knows compression rotations and session tiles). A pop-out
 *  answers only when the tab it shows is among them; absent = unscoped. */
type BridgeRequest = { id: string; tabIds?: string[] } & (
  { kind: 'act'; payload: ActPayload } | { kind: 'read'; payload: PreviewReadOptions }
)

type BridgeResponse =
  | { id: string; kind: 'act'; result: PreviewActResult }
  | { id: string; kind: 'read'; result: PreviewReadResult | null }
  | { id: string; kind: 'error'; error: string }

let seq = 0

type RelayBus = {
  post: (message: unknown) => void
  subscribe: (listener: (data: unknown) => void) => () => void
}

let cachedBus: RelayBus | null = null

function getBus(): RelayBus | null {
  if (cachedBus) {
    return cachedBus
  }

  const desktopRelay = typeof window !== 'undefined' ? window.hermesDesktop?.windowRelay : undefined

  if (desktopRelay?.onMessage && desktopRelay.send) {
    cachedBus = {
      post: message => desktopRelay.send(message),
      subscribe: listener => desktopRelay.onMessage(payload => listener(payload))
    }

    return cachedBus
  }

  if (typeof BroadcastChannel === 'undefined') {
    return null
  }

  const channel = new BroadcastChannel(CHANNEL)

  cachedBus = {
    post: message => channel.postMessage(message),
    subscribe: listener => {
      const onMessage = (event: MessageEvent<unknown>) => listener(event.data)
      channel.addEventListener('message', onMessage)

      return () => channel.removeEventListener('message', onMessage)
    }
  }

  return cachedBus
}

/** True when this renderer has a live webview (or nav handle) for the active
 *  tab among those `owner` (omitted = the focused session) may see. */
export function hasLivePreviewSurface(owner?: PreviewOwner): boolean {
  return Boolean(activePreviewScriptRunner(owner) || activePreviewNav(owner))
}

/** Scope a request to `owner`'s tabs (undefined = an unscoped request). */
const scopeFor = (owner: PreviewOwner | undefined): { tabIds?: string[] } =>
  owner === undefined ? {} : { tabIds: previewTabIdsVisibleTo(owner) }

function nextId(prefix: string): string {
  seq += 1

  return `${prefix}-${Date.now()}-${seq}`
}

function askPopout<T>(
  request: BridgeRequest,
  timeoutMs: number,
  pick: (response: BridgeResponse) => T | undefined
): Promise<T | null> {
  const bus = getBus()

  if (!bus) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    let stop: (() => void) | undefined

    const timer = window.setTimeout(() => {
      stop?.()
      resolve(null)
    }, timeoutMs)

    stop = bus.subscribe(data => {
      if (!data || typeof data !== 'object') {
        return
      }

      const response = data as Partial<BridgeResponse> & { id?: unknown }

      // Ignore our own request echo (same-window test buses / unusual hosts)
      // and unrelated traffic. Only a response carries `result` or `error`.
      if (response.id !== request.id) {
        return
      }

      if (!('result' in response) && response.kind !== 'error') {
        return
      }

      window.clearTimeout(timer)
      stop?.()

      if (response.kind === 'error') {
        resolve(null)

        return
      }

      resolve(pick(response as BridgeResponse) ?? null)
    })

    bus.post(request)
  })
}

/** Ask the browser pop-out to run drive_preview for `owner` (the requesting
 *  session's stored id). Null when no pop-out showing one of its tabs answers. */
export function requestPopoutPreviewAct(payload: ActPayload, owner?: PreviewOwner): Promise<PreviewActResult | null> {
  return askPopout({ id: nextId('act'), kind: 'act', payload, ...scopeFor(owner) }, ACT_TIMEOUT_MS, response =>
    response.kind === 'act' ? response.result : undefined
  )
}

/** Ask the browser pop-out to run read_preview for `owner`. Null when no
 *  pop-out showing one of its tabs answers. */
export function requestPopoutPreviewRead(
  payload: PreviewReadOptions = {},
  owner?: PreviewOwner
): Promise<PreviewReadResult | null> {
  return askPopout({ id: nextId('read'), kind: 'read', payload, ...scopeFor(owner) }, READ_TIMEOUT_MS, response =>
    response.kind === 'read' ? response.result : undefined
  )
}

/**
 * Browser pop-out only: answer act/read requests from the chat window.
 * Safe to call more than once; installs a single listener.
 */
export function installPopoutPreviewResponder(): () => void {
  if (!isBrowserWindow()) {
    return () => {}
  }

  const bus = getBus()

  if (!bus) {
    return () => {}
  }

  const stop = bus.subscribe(data => {
    if (!data || typeof data !== 'object') {
      return
    }

    const request = data as Partial<BridgeRequest> & { id?: unknown; kind?: unknown }

    if (
      typeof request.id !== 'string' ||
      (request.kind !== 'act' && request.kind !== 'read') ||
      !('payload' in request)
    ) {
      return
    }

    // Another session's request: stay silent (an answer — even an error —
    // would win the race against a pop-out that does show that session's tab).
    if (Array.isArray(request.tabIds) && !request.tabIds.includes(windowBrowserTabId() ?? '')) {
      return
    }

    const id = request.id

    void (async () => {
      try {
        if (request.kind === 'act') {
          const result = await actOnActivePreview(request.payload as ActPayload)
          bus.post({ id, kind: 'act', result } satisfies BridgeResponse)

          return
        }

        const result = await readActivePreview(request.payload as PreviewReadOptions)
        bus.post({ id, kind: 'read', result } satisfies BridgeResponse)
      } catch (error) {
        bus.post({
          id,
          kind: 'error',
          error: error instanceof Error ? error.message : String(error)
        } satisfies BridgeResponse)
      }
    })()
  })

  return stop
}
