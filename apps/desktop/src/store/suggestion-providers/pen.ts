import { translateNow } from '@/i18n'
import { getAllSessionMessages } from '@/hermes'
import { type ComposerSuggestion, offerSuggestions, registerDraftProvider } from '@/store/composer-suggestions'
import { openPenCanvas, refreshPenStatus } from '@/store/pen'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

/**
 * Pen canvas draft provider: the draft talks about designing on a canvas, so
 * offer to slide the pen.dev drawer out — a fresh canvas, or one of their
 * .pen files via the native picker. Both are one-click-does-the-whole-thing
 * (side-effecting) pills per the suggestion contract.
 *
 * Self-limiting: only offers while pen.dev is installed AND no canvas is
 * already open (the drawer on screen means the suggestion is done), and only
 * on completed whole-word triggers. Ambiguous words ("design", "draw") get no
 * bare keyword — coding chats are full of design docs and design systems.
 */

const STATUS_TTL_MS = 30_000

// Whole words that mean "I want a canvas". Deliberately narrow; "pen" alone
// is a writing implement, "design" alone is a document.
const KEYWORDS = ['canvas', 'pencil', 'pen.dev', 'mockup', 'mock-up', 'wireframe']

let statusAt = 0
let statusUsable = false

async function penUsable(): Promise<boolean> {
  if (Date.now() - statusAt < STATUS_TTL_MS) {
    return statusUsable
  }

  const status = await refreshPenStatus()

  statusAt = Date.now()
  statusUsable = Boolean(status?.available && status.openDocuments.length === 0)

  return statusUsable
}

/** Whole-word, completed-word trigger test, exported for tests. A hit only
 *  counts when at least one character follows the match — the debounce firing
 *  mid-word under the caret is not intent. */
export function penTrigger(text: string): null | string {
  const haystack = text.toLowerCase()

  for (const keyword of KEYWORDS) {
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const match = new RegExp(`(?<![\\p{L}\\p{N}])${escaped}(?![\\p{L}\\p{N}-])`, 'u').exec(haystack)

    if (match && match.index + keyword.length < haystack.length) {
      return keyword
    }
  }

  return null
}

function copy(key: string): string {
  return translateNow(`composer.penSuggestions.${key}`)
}

function newCanvasSuggestion(): ComposerSuggestion {
  return {
    doneLabel: copy('done'),
    doneTip: copy('doneTip'),
    icon: 'edit',
    id: 'new-canvas',
    invoke: async ({ cancelled }) => {
      const doc = await openPenCanvas()

      if (!doc && !cancelled()) {
        throw new Error(copy('openFailed'))
      }

      statusAt = 0
    },
    label: copy('newCanvas'),
    provider: 'pen',
    tip: copy('newCanvasTip'),
    workingLabel: copy('working'),
    workingTip: copy('workingTip')
  }
}

function openFileSuggestion(): ComposerSuggestion {
  return {
    doneLabel: copy('done'),
    doneTip: copy('doneTip'),
    icon: 'folder-opened',
    id: 'open-file',
    invoke: async ({ cancelled }) => {
      const paths = await window.hermesDesktop?.selectPaths({
        filters: [{ extensions: ['pen'], name: 'Pen Design Files' }]
      })
      const file = paths?.[0]

      if (!file || cancelled()) {
        return
      }

      const doc = await openPenCanvas({ path: file })

      if (!doc && !cancelled()) {
        throw new Error(copy('openFailed'))
      }

      statusAt = 0
    },
    label: copy('openFile'),
    provider: 'pen',
    tip: copy('openFileTip'),
    workingLabel: copy('working'),
    workingTip: copy('workingTip')
  }
}

registerDraftProvider('pen', async ({ text }) => {
  if (!penTrigger(text)) {
    return []
  }

  if (!(await penUsable())) {
    return []
  }

  // Only surface the file variant when the user plausibly has .pen files —
  // cheap heuristic: they mentioned opening/existing work.
  const wantsExisting = /\b(open|existing|my|load)\b/iu.test(text)

  return wantsExisting ? [openFileSuggestion(), newCanvasSuggestion()] : [newCanvasSuggestion(), openFileSuggestion()]
})

/**
 * Reopen pill — the session HAS a canvas and it isn't on screen.
 *
 * This one is an EVENT offering, not a draft one: it's a fact about session
 * state, not something to infer from what's being typed. A canvas you already
 * made for this chat should be one click away without having to describe it
 * again — that's the whole point of tying it to the session.
 *
 * Self-limiting by construction: withdrawn the moment the canvas is open
 * (offer []), so it can't sit there duplicating what's already visible.
 */
function reopenSuggestion(name: string, minedPath: null | string = null): ComposerSuggestion {
  return {
    doneLabel: copy('done'),
    doneTip: copy('doneTip'),
    icon: 'edit',
    id: 'reopen-canvas',
    invoke: async ({ cancelled, sessionId }) => {
      if (!sessionId) {
        return
      }

      // A mined (transcript-recovered) canvas has no tie entry to restore —
      // open it by path instead, which records a FRESH tie so the store is
      // healed for every future launch.
      const restored = minedPath
        ? await openPenCanvas({ path: minedPath }, sessionId)
        : await (await import('@/store/pen')).restorePenCanvas(sessionId)

      if (!restored && !cancelled()) {
        throw new Error(copy('openFailed'))
      }

      statusAt = 0
    },
    label: copy('reopen').replace('{name}', name),
    provider: 'pen',
    tip: copy('reopenTip').replace('{name}', name),
    workingLabel: copy('working'),
    workingTip: copy('workingTip')
  }
}

/** Transcript fallback: the SESSION ITSELF is the durable record of its
 *  canvases — pen tool results and chat text carry the .pen paths (that is
 *  exactly how the Artifacts page ties pens to chats). The side tie-store is
 *  a fast cache that has already missed once (draft-chat hole); when it has
 *  no answer, mine the transcript.
 *
 *  Direction matters: we search the transcript FOR each library path, never
 *  extract path-shaped strings FROM the transcript. Real canvas names carry
 *  spaces ("Untitled 8.pen") and appear percent-encoded in file:// URIs —
 *  both defeat forward extraction (found empirically: a session whose
 *  transcript mentioned its canvas twice matched neither form). Matching
 *  known library paths in both raw and URI-encoded shapes is immune to
 *  either, and only ever offers files that still exist. */
async function canvasPathFromTranscript(sessionId: string): Promise<null | string> {
  try {
    const [{ messages }, library] = await Promise.all([
      getAllSessionMessages(sessionId),
      window.hermesDesktop?.pen?.library().catch(() => null) ?? Promise.resolve(null)
    ])

    const items = library?.items ?? []

    if (!messages?.length || items.length === 0) {
      return null
    }

    // Each library canvas, with every spelling a transcript might contain.
    const needles = items.map(item => ({
      path: item.path,
      forms: [item.path, encodeURI(item.path)]
    }))

    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i]
      const blobs: string[] = []

      if (typeof message.content === 'string') {
        blobs.push(message.content)
      }

      const calls = Array.isArray((message as { tool_calls?: unknown }).tool_calls)
        ? ((message as { tool_calls?: Array<{ function?: { arguments?: unknown } }> }).tool_calls ?? [])
        : []

      for (const call of calls) {
        const args = call?.function?.arguments

        if (typeof args === 'string') {
          blobs.push(args)
        }
      }

      for (const blob of blobs) {
        for (const needle of needles) {
          if (needle.forms.some(form => blob.includes(form))) {
            return needle.path
          }
        }
      }
    }
  } catch {
    // Mining is best-effort; the pill just doesn't appear.
  }

  return null
}

/** The bus keys offerings by the COMPOSER'S session id, and the composer
 *  wears a different identity than the route when a session has been
 *  compacted/rotated: the route (and this refresh) speaks the durable stored
 *  id, the live tile's strip reads under the tip id. Proven live: a pill
 *  offered under the stored id sat invisible while the identical pill under
 *  the tile id rendered instantly. Publish to every identity the
 *  conversation currently wears — the strip dedupes by provider:id, so the
 *  double write can never double-render. */
function pillTargets(sessionId: string): string[] {
  const targets = new Set([sessionId])

  if ($selectedStoredSessionId.get() === sessionId) {
    const active = $activeSessionId.get()

    if (active) {
      targets.add(active)
    }
  }

  return [...targets]
}

/** Re-evaluate the reopen pill for the active session. Called on session
 *  switch and whenever a canvas opens or closes. */
export async function refreshPenSessionSuggestion(sessionId: null | string): Promise<void> {
  const pen = window.hermesDesktop?.pen

  if (!pen?.session || !sessionId) {
    return
  }

  const [entry, status] = await Promise.all([
    pen.session(sessionId).catch(() => null),
    pen.status().catch(() => null)
  ])

  const offerAll = (suggestions: ComposerSuggestion[]) => {
    for (const target of pillTargets(sessionId)) {
      offerSuggestions(target, 'pen', suggestions)
    }
  }

  // Its canvas is already on screen — nothing to offer.
  if ((status?.openDocuments.length ?? 0) > 0) {
    offerAll([])

    return
  }

  // Tie store first (fast, covers the normal case), transcript second (the
  // durable record — catches canvases the store missed).
  let path = entry?.path ?? null

  if (!entry) {
    path = await canvasPathFromTranscript(sessionId)

    if (!path) {
      offerAll([])

      return
    }
  }

  const name = path ? (path.split('/').pop() || '').replace(/\.pen$/, '') : ''

  offerAll([reopenSuggestion(name || copy('untitledCanvas'), entry ? null : path)])
}
