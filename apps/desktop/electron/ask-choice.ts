// On-screen choice dialog: request/response file model + geometry. Mirrors
// `scanline.ts` (the screen-analysis sweep) — the same file-poll bridge between
// the backend and the desktop main process, but for a two-way question the user
// answers by clicking.
//
// The `ask_choice` tool (tools/ask_choice_tool.py) writes a request file to the
// HERMES_HOME root (%LOCALAPPDATA%\hermes on Windows):
//   { "request_id": hex, "question": string, "options": string[], "ts": epochMs }
// The desktop main process polls it and, while one is present + fresh, shows a
// small always-on-top dialog card with one button per option. When the user
// clicks (or presses Esc) the dialog writes a matching answer file:
//   { "request_id": hex, "choice"?: string, "cancelled"?: true, "ts": epochMs }
// and removes the request. The tool polls the answer file and returns.

// Request/response filenames, written directly in the HERMES_HOME root (the
// backend's `get_hermes_home()` and the desktop's `resolveHermesHome()` both
// resolve the same directory on Windows).
export const ASK_CHOICE_REQUEST_FILENAME = 'ask-choice-request.json'
export const ASK_CHOICE_ANSWER_FILENAME = 'ask-choice-answer.json'

// A request file older than this is a stale leftover (the tool crashed before
// it read the answer) — the dialog drops it so a ghost prompt can't stick.
// Generous: the tool's own default wait is 90s, so a request that's ~45s old is
// almost certainly still being answered; we only reap the truly dead.
export const ASK_CHOICE_REQUEST_MAX_AGE_MS = 45_000

export const ASK_CHOICE_FADE_MS = 200

export interface AskChoiceRequest {
  request_id: string
  question: string
  options: string[]
  ts: number
}

export interface AskChoiceAnswer {
  request_id: string
  choice?: string
  cancelled?: boolean
  ts: number
}

export interface DisplayLike {
  bounds: { height: number; width: number; x: number; y: number }
  internal?: boolean
  id?: number
}

export function resolveAskChoiceRequestPath(hermesHome: string): string {
  return hermesHome.replace(/[\\/]+$/, '') + `/${ASK_CHOICE_REQUEST_FILENAME}`
}

export function resolveAskChoiceAnswerPath(hermesHome: string): string {
  return hermesHome.replace(/[\\/]+$/, '') + `/${ASK_CHOICE_ANSWER_FILENAME}`
}

export function parseAskChoiceRequest(raw: string): AskChoiceRequest | null {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>

    if (typeof parsed.request_id !== 'string' || parsed.request_id.length === 0) {
      return null
    }

    const question = typeof parsed.question === 'string' ? parsed.question : ''
    const rawOptions = Array.isArray(parsed.options) ? parsed.options : []

    const options = rawOptions
      .filter((o): o is string => typeof o === 'string' && o.trim().length > 0)
      .map(o => o.slice(0, 64))

    if (options.length < 2) {
      return null
    }

    const ts = typeof parsed.ts === 'number' ? parsed.ts : 0

    return { request_id: parsed.request_id, question: question.slice(0, 240), options, ts }
  } catch {
    return null
  }
}

// Center a fixed-size card on the PRIMARY display (the user's right screen).
// Electron reports DIP bounds; the window is created in the same DIP space.
export function askChoiceBoundsForDisplay(primary: DisplayLike, cardWidth: number, cardHeight: number) {
  const b = primary.bounds

  return {
    height: Math.round(cardHeight),
    width: Math.round(cardWidth),
    x: Math.round(b.x + b.width / 2 - cardWidth / 2),
    y: Math.round(b.y + b.height / 2 - cardHeight / 2)
  }
}
