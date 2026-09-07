import { translateNow } from '@/i18n'
import { getAllSessionMessages } from '@/hermes'
import { type ComposerSuggestion, offerSuggestions, registerDraftProvider } from '@/store/composer-suggestions'
import { openPenCanvas, refreshPenStatus } from '@/store/pen'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

const STATUS_TTL_MS = 30_000

const KEYWORDS = ['canvas', 'pencil', 'pen.dev', 'mockup', 'mock-up', 'wireframe']

let statusAt = 0
let statusUsable = false

async function penUsable(): Promise<boolean> {
  if (Date.now() - statusAt < STATUS_TTL_MS) {
    return statusUsable
  }

  const status = await refreshPenStatus()

  statusAt = Date.now()
  statusUsable = Boolean(status && status.openDocuments.length === 0)

  return statusUsable
}

/** Whole-word hit only when something follows the match (debounce mid-word is not intent). */
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

/** Search the transcript for known library paths (raw + URI-encoded). Don't extract paths from text. */
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

/** Offer under stored id and live tile id — compacted sessions wear both. */
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

  if ((status?.openDocuments.length ?? 0) > 0) {
    offerAll([])

    return
  }

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
