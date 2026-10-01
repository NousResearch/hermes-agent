import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

import { BrowserWindow, screen } from 'electron'

import {
  ASK_CHOICE_FADE_MS,
  ASK_CHOICE_REQUEST_MAX_AGE_MS,
  askChoiceBoundsForDisplay,
  type AskChoiceRequest,
  parseAskChoiceRequest
} from './ask-choice'
import { attachRendererConsoleCapture } from './renderer-log'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'

interface AskChoiceWindowOptions {
  devServer?: string
  loadWindowUrl: (window: BrowserWindow, url: string, label: string) => void
  log: (message: string) => void
  preloadPath: string
  rendererIndex: () => string
  wireWindow: (window: BrowserWindow) => void
  requestPath: string
  answerPath: string
  pollMs?: number
}

// A small, always-on-top, focusable dialog that pops up on the PRIMARY (right)
// display to ask the user a short multiple-choice question, instead of the
// question living in the chat window. Driven by the `ask_choice` tool via a
// request/response file pair (same bridge as the scanline scope file):
//
//   1. The tool writes a request file (question + options + request_id).
//   2. This controller polls it; on a fresh request it shows the card and hands
//      the request to the renderer over IPC.
//   3. The user clicks an option (or presses Esc) → the renderer sends the
//      response back over IPC.
//   4. This controller writes the matching answer file and removes the request
//      file, then hides the card. The tool polls the answer file and returns.
//
// The window is a REAL interactive surface (focusable, clickable) — unlike the
// click-through scanline overlay, the user must click a button. It is
// always-on-top so it sits above whatever they were looking at, but it is a
// transient prompt, not a persistent window (skipTaskbar, no frame, no shadow).
const CARD_WIDTH = 380
const CARD_HEIGHT = 440

export function createAskChoiceWindowController({
  devServer,
  loadWindowUrl,
  log,
  preloadPath,
  rendererIndex,
  wireWindow,
  requestPath,
  answerPath,
  pollMs = 300
}: AskChoiceWindowOptions) {
  let window: BrowserWindow | null = null
  let hideTimer: NodeJS.Timeout | null = null
  // The request_id we last showed, so the poll doesn't re-spawn for the same ask.
  let activeRequestId: string | null = null

  const url = () => {
    if (devServer) {
      return `${devServer.endsWith('/') ? devServer.slice(0, -1) : devServer}/?win=askchoice#/`
    }

    return `${pathToFileURL(rendererIndex()).toString()}?win=askchoice#/`
  }

  const cardBounds = () => askChoiceBoundsForDisplay(screen.getPrimaryDisplay(), CARD_WIDTH, CARD_HEIGHT)

  const sendRequest = (request: AskChoiceRequest) => {
    if (!window || window.isDestroyed()) {
      return
    }

    window.webContents.send('hermes:ask-choice:request', request)
  }

  const spawn = (request: AskChoiceRequest) => {
    const next = new BrowserWindow({
      ...cardBounds(),
      alwaysOnTop: true,
      backgroundColor: '#00000000',
      // Interactive: the user must click an option. focusable + type:'panel'
      // (not a click-through overlay) so mouse clicks land reliably on Windows,
      // matching the listen overlay and every other clickable utility window.
      focusable: true,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      hiddenInMissionControl: true,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: true,
      transparent: true,
      type: 'panel',
      webPreferences: {
        backgroundThrottling: false,
        contextIsolation: true,
        devTools: true,
        nodeIntegration: false,
        preload: preloadPath,
        sandbox: true
      }
    })

    next.setAlwaysOnTop(true, 'screen-saver')
    wireWindow(next)
    installWindowRendererLifecycle(next, { kind: 'ask-choice', callbacks: { log } })
    attachRendererConsoleCapture(next, 'ask-choice', log)

    next.on('closed', () => {
      if (window === next) {
        window = null
        activeRequestId = null
      }
    })

    next.once('ready-to-show', () => {
      if (!next.isDestroyed()) {
        // Re-assert bounds at show time (Windows clamps a fresh frameless window
        // to the work area) so the card is truly centered on the full display.
        next.setBounds(cardBounds())
        next.show()
        next.focus()
        sendRequest(request)
      }
    })

    loadWindowUrl(next, url(), 'Hermes question')

    return next
  }

  // Show (or re-show) the card for a request, spawning if needed.
  const showFor = (request: AskChoiceRequest) => {
    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (!window || window.isDestroyed()) {
      window = spawn(request)
    } else {
      window.setBounds(cardBounds())
      window.show()
      window.focus()
      sendRequest(request)
    }

    activeRequestId = request.request_id
  }

  const hide = () => {
    activeRequestId = null

    if (!window || window.isDestroyed()) {
      return
    }

    if (hideTimer) {
      clearTimeout(hideTimer)
    }

    // Let the fade-out land before the window hides — but only if no new request
    // has arrived in the meantime (a new ask re-sets activeRequestId and shows
    // the card again).
    hideTimer = setTimeout(() => {
      hideTimer = null

      if (window && !window.isDestroyed() && activeRequestId === null) {
        window.hide()
      }
    }, ASK_CHOICE_FADE_MS)
  }

  // Read the current request, if any + fresh. Returns the request to act on,
  // or null for "no dialog should be showing right now".
  const pollRequest = () => {
    if (!existsSync(requestPath)) {
      // Request gone (the tool finished, timed out, or was cancelled) → fade out.
      if (activeRequestId !== null) {
        hide()
      }

      return
    }

    let raw: string

    try {
      raw = readFileSync(requestPath, 'utf8')
    } catch {
      return
    }

    const request = parseAskChoiceRequest(raw)

    if (!request) {
      return
    }

    if (request.ts > 0 && Date.now() - request.ts > ASK_CHOICE_REQUEST_MAX_AGE_MS) {
      // Stale leftover (the tool crashed). Drop the file so the poll is clean
      // and fade the card if it's showing this ghost request.
      try {
        rmSync(requestPath, { force: true })
      } catch {
        // best effort
      }

      if (activeRequestId === request.request_id) {
        hide()
      }

      return
    }

    if (request.request_id !== activeRequestId) {
      showFor(request)
    }
  }

  // Shared: persist the outcome to the answer file (for the tool), retire the
  // request file so the poll + tool both see the dialog as closed, then fade.
  const persist = (request_id: string, choice?: string, cancelled?: boolean) => {
    writeFileSync(
      answerPath,
      JSON.stringify({
        request_id,
        ...(choice ? { choice } : {}),
        ...(cancelled ? { cancelled: true } : {}),
        ts: Date.now()
      }),
      'utf8'
    )

    try {
      rmSync(requestPath, { force: true })
    } catch {
      // best effort
    }

    hide()
  }

  // A button click or number key: the user picked an option.
  const respond = (payload: { request_id?: unknown; choice?: unknown }) => {
    const request_id = typeof payload.request_id === 'string' ? payload.request_id : null
    const choice = typeof payload.choice === 'string' ? payload.choice : undefined

    // Safety: only accept a response for the request we're currently showing.
    if (request_id && activeRequestId && request_id !== activeRequestId) {
      return
    }

    if (!choice) {
      return
    }

    persist(request_id ?? activeRequestId ?? '', choice)
  }

  // Esc / the user dismissed without picking — record a cancel so the tool can
  // report it accurately instead of waiting out the timeout.
  const cancel = (payload: { request_id?: unknown }) => {
    const request_id = typeof payload.request_id === 'string' ? payload.request_id : null

    if (request_id && activeRequestId && request_id !== activeRequestId) {
      return
    }

    if (activeRequestId === null) {
      return
    }

    persist(activeRequestId, undefined, true)
  }

  let poll: NodeJS.Timeout | null = setInterval(pollRequest, pollMs)
  poll.unref?.()

  const close = () => {
    if (poll) {
      clearInterval(poll)
      poll = null
    }

    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (window && !window.isDestroyed()) {
      window.close()
    }

    window = null
    activeRequestId = null
  }

  return {
    close,
    cancel,
    respond,
    // Exposed for tests / diagnostics.
    _pollNow: pollRequest
  }
}
