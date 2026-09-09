import { atom } from 'nanostores'

/**
 * The composer suggestion bus — a generic, session-scoped feed for the pill
 * strip above the composer (rendered by `composer/suggestion-pills.tsx`).
 *
 * Providers, not sources baked into the store: anything can offer a pill —
 * draft keywords (the MCP directory was the first), session state, tool
 * results, connection events. Two provider shapes:
 *
 * - **Draft providers** react to what the user is typing. Registered with
 *   `registerDraftProvider`, they run inside the existing debounced draft
 *   sampler (600ms, change-gated) and return suggestions for that session's
 *   draft. Pure-ish: given a draft, they decide; the bus handles debounce,
 *   session scoping, reference identity, and the cap.
 * - **Event providers** push and withdraw suggestions directly via
 *   `offerSuggestion` / `withdrawSuggestion` from wherever their signal
 *   lives (a store listener, a gateway event handler). The bus applies the
 *   same session scoping and cap.
 *
 * The UX contract every suggestion signs (see PR #85036's pills):
 * session-scoped, capped at MAX_SUGGESTIONS with draft suggestions ranked
 * after event ones, self-limiting (a suggestion exists only while its
 * trigger condition holds — providers withdraw it, there is NO dismiss
 * affordance), and one-click: `invoke` runs the whole action with the pill
 * narrating idle → working → done. No suggestion may block or shift the
 * composer.
 */
export interface ComposerSuggestion {
  /** Stable per-suggestion identity, unique within its provider
   *  (e.g. the server name, the skill name). */
  id: string
  /** Provider that offered it; `${provider}:${id}` is the bus-wide key. */
  provider: string
  /** Pill label, already localized ("Add Atlassian"). */
  label: string
  /** Tooltip explaining WHY this is being suggested. */
  tip: string
  /** Brand identity for the glyph slot; falls back to `icon`. */
  brand?: string
  /** Codicon name when there is no brand glyph (default: lightbulb). */
  icon?: string
  /** Runs the whole action; the pill shows `workingLabel` while it's
   *  in flight and `doneLabel` on success. Reject to return to idle
   *  (provider surfaces its own error toast). */
  invoke: (context: { cancelled: () => boolean; sessionId: string | null }) => Promise<void>
  /** Label while `invoke` runs ("Connecting Atlassian…"). */
  workingLabel: string
  /** Tooltip while working; clicking a working pill requests cancel. */
  workingTip: string
  /** Label after `invoke` resolves ("Added Atlassian"). */
  doneLabel: string
  /** Tooltip once done. */
  doneTip: string
}

export const MAX_SUGGESTIONS = 2

/** Bus-wide key: provider-namespaced so two providers can't collide. */
export const suggestionKey = (suggestion: Pick<ComposerSuggestion, 'id' | 'provider'>): string =>
  `${suggestion.provider}:${suggestion.id}`

/** Suggestions keyed by RUNTIME session id, exactly like
 *  `$composerActionsBySession`: drafts are per-session state, so suggestions
 *  derived from them are too. */
export const $composerSuggestionsBySession = atom<Record<string, ComposerSuggestion[]>>({})

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

// Everything the pill actually paints. Compared field-by-field rather than by
// key alone: a provider rebuilds its suggestion objects on every sample, so
// key equality is true constantly, and treating that as "no change" pins the
// FIRST object forever — the strip then paints a stale tip and, worse, calls a
// stale `invoke` closure. Comparing the rendered copy keeps the cheap bail-out
// for the common case (same draft, same match) while letting a genuinely
// changed offer through.
const RENDERED: readonly (keyof ComposerSuggestion)[] = [
  'brand',
  'doneLabel',
  'doneTip',
  'icon',
  'label',
  'tip',
  'workingLabel',
  'workingTip'
]

const sameSuggestions = (a: readonly ComposerSuggestion[], b: readonly ComposerSuggestion[]) =>
  a.length === b.length &&
  a.every((x, i) => {
    const y = b[i]!

    return suggestionKey(x) === suggestionKey(y) && RENDERED.every(field => x[field] === y[field])
  })

function write(sessionId: string | null | undefined, suggestions: ComposerSuggestion[]): void {
  const key = keyFor(sessionId)
  const current = $composerSuggestionsBySession.get()
  const existing = current[key] ?? []

  // Unchanged sets keep their reference so the strip doesn't re-render.
  if (sameSuggestions(existing, suggestions)) {
    return
  }

  const next = { ...current }

  if (suggestions.length > 0) {
    next[key] = suggestions
  } else {
    delete next[key]
  }

  $composerSuggestionsBySession.set(next)
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

export interface DraftProviderContext {
  sessionId: string | null
  /** The draft text at sample time. */
  text: string
}

export type DraftProvider = (context: DraftProviderContext) => Promise<ComposerSuggestion[]>

const draftProviders = new Map<string, DraftProvider>()

/** Register a provider that derives suggestions from the draft. Runs inside
 *  the composer's debounced sampler; results replace that provider's previous
 *  offerings for the session. Returns an unregister fn (HMR hygiene). */
export function registerDraftProvider(name: string, provider: DraftProvider): () => void {
  draftProviders.set(name, provider)

  return () => {
    draftProviders.delete(name)
  }
}

// Event-provider offerings, merged with draft results on every write.
// Keyed session → provider → suggestions.
const eventOfferings = new Map<string, Map<string, ComposerSuggestion[]>>()

/** Offer suggestions from an event provider (session state, tool results,
 *  connection events…). Replaces that provider's previous offerings for the
 *  session; providers withdraw by offering []. */
export function offerSuggestions(
  sessionId: string | null | undefined,
  provider: string,
  suggestions: ComposerSuggestion[]
): void {
  const key = keyFor(sessionId)
  let providers = eventOfferings.get(key)

  if (!providers) {
    providers = new Map()
    eventOfferings.set(key, providers)
  }

  if (suggestions.length > 0) {
    providers.set(provider, suggestions)
  } else {
    providers.delete(provider)
  }

  publish(sessionId ?? null)
}

// Last draft-provider results per session, merged with event offerings.
const draftOfferings = new Map<string, ComposerSuggestion[]>()

// ---------------------------------------------------------------------------
// Declined ledger (VS Code's hardest-won recommendation lesson: never keep
// re-firing at someone who has seen the offer and not taken it)
// ---------------------------------------------------------------------------

// Times a suggestion key was WITHDRAWN without ever being invoked, per
// session. A pill the user watched appear and let die N times is a declined
// offer — stop making it for the rest of the session. Session-scoped and
// in-memory on purpose: a fresh session is a fresh chance, and acting on a
// pill clears its count.
const IGNORED_LIMIT = 3
const ignoredCounts = new Map<string, Map<string, number>>()
const shown = new Map<string, Set<string>>()

/** The pill strip marks a suggestion invoked so its withdrawal isn't
 *  miscounted as the user ignoring it. */
export function markSuggestionInvoked(sessionId: string | null | undefined, key: string): void {
  ignoredCounts.get(keyFor(sessionId))?.delete(key)
  shown.get(keyFor(sessionId))?.delete(key)
}

const quieted = (sessionKey: string, key: string): boolean =>
  (ignoredCounts.get(sessionKey)?.get(key) ?? 0) >= IGNORED_LIMIT

// Suggestions visible in the last publish that vanish uninvoked get a strike.
function recordWithdrawals(sessionKey: string, next: readonly ComposerSuggestion[]): void {
  const previous = shown.get(sessionKey)

  if (previous) {
    const surviving = new Set(next.map(suggestionKey))

    for (const key of previous) {
      if (!surviving.has(key)) {
        const counts = ignoredCounts.get(sessionKey) ?? new Map<string, number>()

        counts.set(key, (counts.get(key) ?? 0) + 1)
        ignoredCounts.set(sessionKey, counts)
      }
    }
  }

  shown.set(sessionKey, new Set(next.map(suggestionKey)))
}

/** Event offerings first (they carry session/tool state, stronger signal
 *  than draft keywords), then draft matches, capped. */
function publish(sessionId: string | null): void {
  const key = keyFor(sessionId)
  const event = [...(eventOfferings.get(key)?.values() ?? [])].flat()
  const draft = draftOfferings.get(key) ?? []
  const seen = new Set<string>()
  const merged: ComposerSuggestion[] = []

  for (const suggestion of [...event, ...draft]) {
    const k = suggestionKey(suggestion)

    if (!seen.has(k) && !quieted(key, k)) {
      seen.add(k)
      merged.push(suggestion)
    }

    if (merged.length >= MAX_SUGGESTIONS) {
      break
    }
  }

  recordWithdrawals(key, merged)
  write(sessionId, merged)
}

// ---------------------------------------------------------------------------
// Draft sampling (the composer's runtime subscription feeds this)
// ---------------------------------------------------------------------------

const SAMPLE_DEBOUNCE_MS = 600

// Per-session debounce/generation so a tile composer's sampling never stomps
// the primary's (each session settles independently).
const sampleTimers = new Map<string, number>()
const sampleGenerations = new Map<string, number>()

/**
 * Feed a session's draft snapshot to the draft providers. Called from that
 * composer's runtime subscription on every change, but internally debounced
 * and change-gated: the store only writes when the session's merged set
 * actually differs, so typing within a line costs nothing downstream.
 */
export function sampleComposerDraft(sessionId: string | null | undefined, text: string): void {
  const key = keyFor(sessionId)

  window.clearTimeout(sampleTimers.get(key))

  const generation = (sampleGenerations.get(key) ?? 0) + 1
  sampleGenerations.set(key, generation)

  // Too short to mean anything — clear draft offerings without running providers.
  if (text.trim().length < 3) {
    draftOfferings.delete(key)
    publish(sessionId ?? null)

    return
  }

  sampleTimers.set(
    key,
    window.setTimeout(() => {
      void Promise.all(
        [...draftProviders.values()].map(provider =>
          provider({ sessionId: sessionId ?? null, text }).catch((): ComposerSuggestion[] => [])
        )
      ).then(results => {
        // A newer sample for THIS session superseded this one mid-flight.
        if (generation !== sampleGenerations.get(key)) {
          return
        }

        draftOfferings.set(key, results.flat())
        publish(sessionId ?? null)
      })
    }, SAMPLE_DEBOUNCE_MS)
  )
}

/** Drop a session's suggestions outright (composer unmount / session leave).
 *  Draft offerings die with the draft; event offerings persist — their
 *  providers own that lifecycle and withdraw on their own signal. */
export function clearDraftSuggestions(sessionId: string | null | undefined): void {
  const key = keyFor(sessionId)

  window.clearTimeout(sampleTimers.get(key))
  draftOfferings.delete(key)
  publish(sessionId ?? null)
}
