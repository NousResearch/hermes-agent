/**
 * Side chat — the floating window `/btw` opens.
 *
 * `/btw <question>` answers from a SNAPSHOT of the conversation without
 * touching its history, alternation or prompt cache (agent/side_question.py).
 * That is exactly the shape of an aside, so the desktop gives it an aside's
 * surface: a small always-on-top window beside the app instead of a system
 * line spliced into the transcript the user is still reading.
 *
 * The window carries NO gateway connection, for the same reason Quick Entry
 * doesn't: the primary renderer already owns the session, its profile and the
 * socket that session is bound to. The side window hands its question to main,
 * main forwards it to the primary renderer, the primary renderer calls the
 * SAME `prompt.btw` RPC the inline command always called, and the `btw.complete`
 * event it already subscribes to travels back the same road. One RPC, one
 * event, one owner — the window is a view, not a second client.
 *
 * Everything Electron-free lives here so the parts that actually break a user —
 * placement on an odd display, and the payload validation standing between two
 * windows — are unit-testable without booting Electron. main.ts owns the
 * BrowserWindow and the IPC wiring.
 */

import { pathToFileURL } from 'node:url'

// A column, not a second app: wide enough for prose and a code snippet, tall
// enough to hold a few exchanges. The user resizes from here; nothing in the
// renderer grows the OS window.
const SIDE_CHAT_WINDOW_WIDTH = 420
const SIDE_CHAT_WINDOW_HEIGHT = 620
const SIDE_CHAT_MIN_WIDTH = 320
const SIDE_CHAT_MIN_HEIGHT = 320

// Breathing room from the work-area edge so the window reads as floating over
// the desktop rather than as a docked panel welded to the screen border.
const SIDE_CHAT_EDGE_MARGIN = 24

/**
 * Where the side chat opens on a given work area: pinned to the RIGHT edge and
 * vertically centered, so it lands beside the app window rather than over the
 * transcript the question is about.
 *
 * Both axes are clamped into the work area, and the size is capped by it: a
 * 620px-tall window on a 1280x540 projector would otherwise open with its
 * composer below the bottom of the screen — un-typeable, which for a window
 * whose entire purpose is typing a follow-up means the feature is simply gone.
 */
export function sideChatWindowBounds(workArea?: {
  height: number
  width: number
  x: number
  y: number
}): { height: number; width: number; x: number; y: number } {
  const width = Math.min(SIDE_CHAT_WINDOW_WIDTH, workArea?.width ?? SIDE_CHAT_WINDOW_WIDTH)
  const height = Math.min(SIDE_CHAT_WINDOW_HEIGHT, workArea?.height ?? SIDE_CHAT_WINDOW_HEIGHT)

  if (!workArea) {
    return { height, width, x: 0, y: 0 }
  }

  // Margin only while it actually fits — on a work area exactly as wide as the
  // window, insisting on the gap would push the window off-screen.
  const margin = Math.max(0, Math.min(SIDE_CHAT_EDGE_MARGIN, workArea.width - width))
  const x = Math.round(workArea.x + workArea.width - width - margin)
  const y = Math.round(workArea.y + Math.max(0, (workArea.height - height) / 2))

  return { height, width, x, y }
}

/**
 * Build the side chat's renderer URL.
 *
 * Same query-before-hash contract as `buildHudWindowUrl` / `buildSessionWindowUrl`:
 * `?win=side` MUST sit in the search string before the '#', or HashRouter
 * swallows it as part of the route.
 *
 * Deliberately carries NO session id and NO profile. Unlike the HUD, this
 * window never talks to a gateway, so it has nothing to resolve an id against —
 * the parent session travels as IPC context (`hermes:side-chat:context`)
 * instead, and stays a string the window only ever echoes back to main.
 */
export function buildSideChatWindowUrl({
  devServer,
  rendererIndexPath
}: { devServer?: null | string; rendererIndexPath?: string } = {}): string {
  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/?win=side#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}?win=side#/`
}

/** The parent conversation a side chat is asking about. */
export interface SideChatContext {
  /** Optional first question, from `/btw <question>`. Blank for a bare `/btw`. */
  question: string
  /** Runtime session id of the parent chat. Every ask is scoped to it. */
  sessionId: string
  /** Parent chat's title, shown in the window header for orientation. */
  title: string
}

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/**
 * Validate an open request from the primary renderer.
 *
 * `sessionId` is required and is the whole point of the check: `prompt.btw`
 * snapshots a specific conversation, so a request without one could only ever
 * produce an aside about nothing. Rejecting it here keeps a window that can
 * never answer from opening at all.
 */
export function normalizeSideChatContext(raw: unknown): null | SideChatContext {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const sessionId = asText(record.sessionId).trim()

  if (!sessionId) {
    return null
  }

  return { question: asText(record.question), sessionId, title: asText(record.title).trim() }
}

/** A question typed in the side chat, on its way to the primary renderer. */
export interface SideChatAsk {
  /** Renderer-minted id correlating this ask with its reply. */
  askId: string
  sessionId: string
  text: string
}

/**
 * Validate an ask from the side window.
 *
 * `askId` correlates the reply, and main cannot mint one: a reply that arrived
 * without a matching ask would have nowhere to render, so an ask without an id
 * is dropped rather than sent into a conversation that can never show it.
 */
export function normalizeSideChatAsk(raw: unknown): null | SideChatAsk {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const askId = asText(record.askId).trim()
  const sessionId = asText(record.sessionId).trim()
  const text = asText(record.text)

  if (!askId || !sessionId || !text.trim()) {
    return null
  }

  return { askId, sessionId, text }
}

/** An answer (or failure) travelling back to the side window. */
export interface SideChatReply {
  askId: string
  /** Set when the ask could not be answered; `text` is then empty. */
  error: string
  text: string
}

/**
 * Validate a reply from the primary renderer.
 *
 * An empty `text` is kept when `error` is set — a failed aside must still land
 * in the window as a failure, because the alternative is a question that sits
 * "thinking" forever with no way for the user to learn it never will.
 */
export function normalizeSideChatReply(raw: unknown): null | SideChatReply {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const askId = asText(record.askId).trim()
  const error = asText(record.error).trim()
  const text = asText(record.text)

  if (!askId || (!text.trim() && !error)) {
    return null
  }

  return { askId, error, text }
}

export {
  SIDE_CHAT_EDGE_MARGIN,
  SIDE_CHAT_MIN_HEIGHT,
  SIDE_CHAT_MIN_WIDTH,
  SIDE_CHAT_WINDOW_HEIGHT,
  SIDE_CHAT_WINDOW_WIDTH
}
