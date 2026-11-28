import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ATTACHMENTS_SLOT, RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import type { ClientSessionState } from '@/app/types'
import { type ComposerAttachment, stashSessionDraft } from '@/store/composer'
import { setPreviewDirty } from '@/store/preview-edit'
import { $gatewayState } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { retainVoiceStreamPaint } from '@/store/voice-stream-paint'

import { reloadOntoServedBundle, setBundleReloadForTests, startServedBundleWatch } from './webapp-bundle'

type BundleWindow = { __HERMES_WEBAPP_BUNDLE__?: string; hermesDesktop?: unknown }
const win = window as unknown as BundleWindow

const reload = vi.fn()
let served: { current: boolean; id: null | string } | Error = { current: true, id: 'new' }

const api = vi.fn(async () => {
  if (served instanceof Error) {
    throw served
  }

  return served
})

function bootOn(id: string | undefined): void {
  if (id === undefined) {
    delete win.__HERMES_WEBAPP_BUNDLE__
  } else {
    win.__HERMES_WEBAPP_BUNDLE__ = id
  }
}

// A mounted composer editor the way the app renders one (main, a session tile or an inline edit):
// its text is in the DOM only until the draft debounce saves it.
function mountEditor(text: string, { hidden = false } = {}): HTMLElement {
  const pane = document.createElement('div')

  if (hidden) {
    pane.dataset.paneHidden = ''
  }

  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.contentEditable = 'true'
  editor.textContent = text
  pane.append(editor)
  document.body.append(pane)

  return editor
}

let releaseVoice: () => void = () => undefined

function mountDialog(): void {
  const dialog = document.createElement('div')
  dialog.setAttribute('role', 'dialog')
  document.body.append(dialog)
}

function mountAttachmentCard(): void {
  const list = document.createElement('div')
  list.dataset.slot = ATTACHMENTS_SLOT
  document.body.append(list)
}

const busy = (patch: Partial<ClientSessionState>) => ({
  s1: { busy: false, needsInput: false, ...patch } as ClientSessionState
})

beforeEach(() => {
  win.hermesDesktop = { api }
  setBundleReloadForTests(reload)
  served = { current: true, id: 'new' }
  bootOn('old')
})

afterEach(() => {
  reload.mockReset()
  api.mockClear()
  setBundleReloadForTests(null)
  sessionStorage.clear()
  $sessionStates.set({})
  document.body.replaceChildren()
  stashSessionDraft('s-other', '', [])
  $gatewayState.set('idle')
  window.location.hash = ''
  delete win.__HERMES_WEBAPP_BUNDLE__
  delete win.hermesDesktop
})

describe('reloadOntoServedBundle', () => {
  it('reloads an idle tab once its gateway serves a new, current bundle', async () => {
    await reloadOntoServedBundle()

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/webapp/bundle' }))
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('keeps a working tab when the served bundle is a stale one a rebuild could not replace', async () => {
    served = { current: false, id: 'new' }
    await reloadOntoServedBundle()

    expect(reload).not.toHaveBeenCalled()
  })

  it('does nothing while the tab already runs the served bundle, or no bundle is served', async () => {
    served = { current: true, id: 'old' }
    await reloadOntoServedBundle()
    served = { current: false, id: null }
    await reloadOntoServedBundle()
    served = new Error('gateway offline')
    await reloadOntoServedBundle()

    expect(reload).not.toHaveBeenCalled()
  })

  it('never reloads a page no Webapp gateway named a bundle for (native Desktop, a dev server)', async () => {
    bootOn(undefined)
    await reloadOntoServedBundle()

    expect(api).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
  })

  it('reloads at most once per served bundle, so a page that keeps booting the old one cannot loop', async () => {
    await reloadOntoServedBundle()
    // The reload landed on a page that still names the old bundle (a cached page in a proxy).
    await reloadOntoServedBundle()

    expect(reload).toHaveBeenCalledTimes(1)

    served = { current: true, id: 'newer' }
    await reloadOntoServedBundle()

    expect(reload).toHaveBeenCalledTimes(2)
  })

  it('never reloads without storage for the loop guard', async () => {
    vi.stubGlobal('sessionStorage', {
      getItem: () => null,
      setItem: () => {
        throw new Error('denied')
      }
    })

    try {
      await reloadOntoServedBundle()
    } finally {
      vi.unstubAllGlobals()
    }

    expect(reload).not.toHaveBeenCalled()
  })

  it.each([
    ['a running turn', () => $sessionStates.set(busy({ busy: true }))],
    ['a turn the backend reports live', () => $sessionStates.set(busy({ turnLive: true }))],
    ['a sent prompt awaiting its reply', () => $sessionStates.set(busy({ awaitingResponse: true }))],
    ['a prompt waiting on the user', () => $sessionStates.set(busy({ needsInput: true }))],
    ['text typed in the main composer', () => mountEditor('half a thought')],
    ['text in an inline edit or a session tile', () => (mountEditor(''), mountEditor('an edit'))],
    ['text in a composer on a hidden tab', () => mountEditor('kept tab', { hidden: true })],
    ['an attachment card in any composer', () => mountAttachmentCard()],
    [
      "an attachment stashed with another session's draft",
      () => stashSessionDraft('s-other', '', [{ id: 'a' } as ComposerAttachment])
    ],
    ['a page other than the chat open', () => (window.location.hash = '#/settings')],
    ['a dialog open over the chat', () => mountDialog()],
    ['a file edit not saved yet', () => setPreviewDirty('file:///notes.md', true)],
    ['a voice conversation', () => (releaseVoice = retainVoiceStreamPaint())]
  ])('leaves a tab with %s alone, and reloads it on the next check once it is idle', async (_name, makeBusy) => {
    makeBusy()
    await reloadOntoServedBundle()

    expect(reload).not.toHaveBeenCalled()

    $sessionStates.set({})
    document.body.replaceChildren()
    mountEditor('  ')
    stashSessionDraft('s-other', '', [])
    window.location.hash = '#/s1'
    setPreviewDirty('file:///notes.md', false)
    releaseVoice()
    releaseVoice = () => undefined
    await reloadOntoServedBundle()

    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('reloads for each id at most once, so served ids flipping back and forth cannot loop', async () => {
    const order = ['new', 'old', 'new', 'old']

    for (const id of order) {
      served = { current: true, id }
      bootOn(id === 'new' ? 'old' : 'new')
      await reloadOntoServedBundle()
    }

    expect(reload).toHaveBeenCalledTimes(2)
  })

  it('reloads once when a reconnect check and the minute check overlap', async () => {
    await Promise.all([reloadOntoServedBundle(), reloadOntoServedBundle()])

    expect(reload).toHaveBeenCalledTimes(1)
  })
})

describe('startServedBundleWatch', () => {
  it('asks on every gateway (re)connect and once a minute', async () => {
    vi.useFakeTimers()

    try {
      served = { current: true, id: 'old' }
      const stop = startServedBundleWatch()
      $gatewayState.set('open')
      expect(api).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(60_000)
      expect(api).toHaveBeenCalledTimes(2)

      stop()
      await vi.advanceTimersByTimeAsync(60_000)
      $gatewayState.set('closed')
      $gatewayState.set('open')
      expect(api).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('watches nothing on a page no Webapp gateway served', () => {
    bootOn(undefined)
    const stop = startServedBundleWatch()
    $gatewayState.set('open')
    stop()

    expect(api).not.toHaveBeenCalled()
  })
})
