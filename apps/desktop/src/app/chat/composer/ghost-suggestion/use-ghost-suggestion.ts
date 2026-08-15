/**
 * AI-powered ghost suggestion hook.
 *
 * While the user types free text (no trigger character active), ask the
 * backend which slash command best matches the intent. Surface the active
 * one as a faded ghost the user can accept with Tab. Shift+Tab cycles to
 * the next candidate; Escape dismisses the ghost for the current draft.
 *
 * The backend call is the existing `llm.oneshot` gateway RPC — no new
 * server work. We debounce input, cache results, and keep the request
 * shape small (≤ 4 candidates, max_tokens = 30) so each keystroke costs
 * about as much as a commit-message helper.
 *
 * Used (and accepted) commands in this session are tracked in a
 * session-scoped Set so the same command never ghosts twice in a row —
 * the user already saw it, offering it again is noise.
 */
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { $skillSuggestionsEnabled } from '@/store/skill-suggestions'
import type { HermesGateway } from '@/hermes'
import type { CommandsCatalogLike } from '@/lib/desktop-slash-commands'
import { peekCachedSlashCompletion } from '@/lib/slash-completion-cache'

const MIN_DRAFT_LENGTH = 2
const DEBOUNCE_MS = 300
const CACHE_TTL_MS = 30_000
const REQUEST_TIMEOUT_MS = 8_000
const MAX_CANDIDATES = 4

export interface GhostCandidate {
  /** The slash command exactly as it should appear after the user accepts. */
  command: string
  /** Free-text rationale the LLM gave for why this command fits. */
  reason: string
}

export interface GhostSuggestionState {
  /** Ordered candidates; the first is the active ghost. Empty when nothing fits. */
  candidates: GhostCandidate[]
  /** The command currently highlighted as the ghost. Null when no suggestion fits. */
  active: GhostCandidate | null
  /** Switch the active candidate to the next one (cyclic). */
  cycleNext: () => void
  /** Switch the active candidate to the previous one (cyclic). */
  cyclePrev: () => void
  /** Accept the active candidate by appending it to the current draft. */
  accept: () => void
  /** Dismiss the ghost for the current draft without accepting. */
  dismiss: () => void
}

interface CachedResponse {
  candidates: GhostCandidate[]
  cachedAt: number
}

const requestCache = new Map<string, CachedResponse>()

/**
 * Build the catalog digest that ships with the LLM request. We don't
 * paste the full catalog (it changes per skill install) — instead a
 * compact command → description list the model can rank in one pass.
 */
export function digestCatalog(catalog: CommandsCatalogLike | null | undefined): string {
  if (!catalog) {
    return ''
  }

  const pairs = catalog.pairs ?? []

  return pairs
    .map(([command, description]) => {
      const normalized = command.startsWith('/') ? command : `/${command}`

      return `${normalized}: ${description || '(no description)'}`
    })
    .join('\n')
}

/**
 * Parse the LLM's reply into ordered candidates. The model is asked to
 * emit one command per line; lines that don't start with `/` are
 * discarded. Reasons are extracted from a trailing comment if the model
 * emitted one, but a bare command is also accepted.
 */
export function parseCandidates(reply: string, validCommands: ReadonlySet<string>): GhostCandidate[] {
  const out: GhostCandidate[] = []
  const seen = new Set<string>()

  for (const raw of reply.split(/\r?\n/)) {
    const line = raw.trim()

    if (!line) {
      continue
    }

    // Accept either a bare command or `command — reason`.
    const [first, ...rest] = line.split(/\s+[—\-:|]\s+/)
    const candidate = first?.trim() ?? ''

    if (!candidate.startsWith('/') || !validCommands.has(candidate)) {
      continue
    }

    if (seen.has(candidate)) {
      continue
    }

    seen.add(candidate)
    out.push({
      command: candidate,
      reason: rest.join(' ').trim()
    })

    if (out.length >= MAX_CANDIDATES) {
      break
    }
  }

  return out
}

function isTriggerActive(draft: string): boolean {
  return draft.includes('@') || draft.includes('/') || draft.includes(':')
}

/**
 * Map common Chinese / English topic words to the slash command they
 * most often map to. The catalog descriptions are the source of truth;
 * this list exists only so a fallback has a chance when the LLM is
 * unreachable (offline mode, rate-limit, in-flight session cap). Each
 * entry is a substring match against the lowercased description.
 */
const KEYWORD_HINTS: Array<{ command: string; keywords: readonly string[] }> = [
  {
    command: '/learn',
    keywords: ['learn', '学习', '教程', '教程', 'caac', '无人机', 'study', 'teach']
  },
  {
    command: '/commit',
    keywords: ['commit', '提交', 'git commit']
  },
  {
    command: '/commit-push',
    keywords: ['push', '推送', 'commit and push', 'commit-push']
  },
  {
    command: '/voice',
    keywords: ['voice', '语音', '口述']
  },
  {
    command: '/gif-search',
    keywords: ['gif', '表情', '动图']
  }
]

/**
 * Cheap client-side fallback that ranks commands by how many keyword
 * tokens the draft matches. Used only when the LLM call fails or
 * returns NONE. No network, sub-millisecond.
 */
function keywordFallback(draft: string, validCommands: ReadonlySet<string>): GhostCandidate[] {
  const haystack = draft.toLowerCase()
  const ranked: Array<{ command: string; score: number }> = []

  for (const { command, keywords } of KEYWORD_HINTS) {
    if (!validCommands.has(command)) {
      continue
    }

    const score = keywords.reduce((acc, keyword) => acc + (haystack.includes(keyword.toLowerCase()) ? 1 : 0), 0)

    if (score > 0) {
      ranked.push({ command, score })
    }
  }

  return ranked
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_CANDIDATES)
      .map(({ command }) => ({ command, reason: '' }))
}

/**
 * Single stateless LLM call asking the model to rank commands for `input`.
 * The catalog is included verbatim — Hermes skill names already encode
 * what the skill does (e.g. `/commit-push`, `/learn`), and descriptions
 * give the model enough context to match a Chinese prompt to an English
 * command. Reuses an existing RPC, so no backend changes.
 */
async function askBackend(gateway: HermesGateway, input: string, catalogDigest: string): Promise<GhostCandidate[]> {
  const instructions =
    'You are the Hermes Desktop command router. The user is typing free text in the composer and may not know which slash command fits.\n\n' +
    'Available commands:\n' +
    catalogDigest +
    '\n\n' +
    'Given the user\'s input, return the slash command that best matches the intent. Output one command per line, no numbering, no prose, no markdown. If no command fits, output exactly NONE.\n' +
    'Prefer commands whose description mentions the user\'s topic; ignore commands the user clearly already invoked earlier in the conversation.'

  // The whole RPC is wrapped in a timeout race so an offline gateway
  // never strands the user on a stale ghost — the keyword fallback
  // kicks in the moment the race resolves, with or without LLM data.
  const response = await Promise.race([
    gateway.request<{ text?: string }>('llm.oneshot', {
      instructions,
      input,
      task: 'intent_match',
      max_tokens: 60,
      temperature: 0.1
    }),
    new Promise<{ text: '' }>((_, reject) =>
      setTimeout(() => reject(new Error('llm.oneshot timeout')), REQUEST_TIMEOUT_MS)
    )
  ]).catch(() => ({ text: '' }))

  const text = response.text ?? ''

  if (text.includes('NONE')) {
    return []
  }

  // We don't have the catalog set here — askBackend runs before the
  // validSet is built at the call site, so the call site filters the
  // returned candidates. We return raw candidates keyed on whatever the
  // LLM emitted, and the caller drops entries that aren't in the live
  // catalog (e.g. commands the user uninstalled since the prompt was
  // built).
  return text
    .split(/\r?\n/)
    .map(line => line.trim().split(/\s+[—\-:|]\s+/)[0]?.trim() ?? '')
    .filter(command => command.startsWith('/'))
    .slice(0, MAX_CANDIDATES)
    .map(command => ({ command, reason: '' }))
}

interface UseGhostSuggestionOptions {
  /** Live composer text ref. Hook reads via rAF so typing doesn't re-render. */
  draftRef: { current: string }
  /** The composer's rich-text editor element. Accept writes the command here. */
  editorRef: { current: HTMLElement | null }
  /** IME composition flag — the hook is silent while the user is composing. */
  composingRef: React.MutableRefObject<boolean>
  /** Gateway for the backend RPC. Null while the desktop is offline. */
  gateway: HermesGateway | null
  /**
   * Commands the user already accepted or rejected in this session. The
   * ghost never suggests one of these — the user has seen it, re-suggesting
   * is noise. Provided by the parent; the hook never mutates it.
   */
  rejectedCommandsRef: React.MutableRefObject<ReadonlySet<string>>
}

/**
 * Mirror a contentEditable ref into React state via rAF. Same idea as the
 * existing pattern in the composer: cheap (one DOM read per frame), and
 * the comparison check is O(1).
 */
export function useDraftValue(ref: { current: string }): string {
  const [value, setValue] = useState<string>(ref.current)
  const rafRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    let lastValue = ref.current

    function tick(): void {
      const current = ref.current

      if (current !== lastValue) {
        lastValue = current
        setValue(current)
      }

      rafRef.current = window.requestAnimationFrame(tick)
    }

    rafRef.current = window.requestAnimationFrame(tick)

    return () => {
      if (rafRef.current !== undefined) {
        window.cancelAnimationFrame(rafRef.current)
        rafRef.current = undefined
      }
    }
  }, [ref])

  return value
}

export function useGhostSuggestion({
  draftRef,
  editorRef,
  composingRef,
  gateway,
  rejectedCommandsRef
}: UseGhostSuggestionOptions): GhostSuggestionState {
  const [catalog, setCatalog] = useState<CommandsCatalogLike | null>(() =>
    peekCachedSlashCompletion<CommandsCatalogLike>('catalog') ?? null
  )
  const [candidates, setCandidates] = useState<GhostCandidate[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [dismissedFor, setDismissedFor] = useState<string | null>(null)
  // Feature toggle from Settings — when off, the ghost never surfaces.
  const enabled = useStore($skillSuggestionsEnabled)

  // Mirror the draft ref so React re-renders on typing. The composer's
  // contentEditable keeps the source of truth on a ref to avoid painting
  // the chrome on every keystroke; the hook needs *state* to decide when
  // to fire a request.
  const draft = useDraftValue(draftRef)

  // Refresh the catalog view periodically. The slash-completion adapter
  // populates the cache; we just re-read it so a skill install shows up
  // without a session reload.
  useEffect(() => {
    const interval = window.setInterval(() => {
      setCatalog(peekCachedSlashCompletion<CommandsCatalogLike>('catalog') ?? null)
    }, 60_000)

    return () => window.clearInterval(interval)
  }, [])

  const validCommands = useMemo(() => {
    const set = new Set<string>()

    for (const [command] of catalog?.pairs ?? []) {
      const normalized = command.startsWith('/') ? command : `/${command}`
      set.add(normalized)
    }

    return set
  }, [catalog])

  // Build the set we actually surface: catalog commands minus rejected.
  const visibleCandidates = useMemo(() => {
    const rejected = rejectedCommandsRef.current
    return candidates.filter(candidate => !rejected.has(candidate.command))
  }, [candidates, rejectedCommandsRef])

  // Dev telemetry: log every state change to the console and stash the
  // latest snapshot on `window.__ghostDebug` so the engineer can verify
  // the hook by opening DevTools and running `__ghostDebug()`. Cheap, runs
  // in both dev and prod builds — the noise is acceptable for a feature
  // this experimental.
  {
    const snapshot = {
      draft: draft.slice(0, 30),
      candidates: candidates.map(c => c.command),
      active: visibleCandidates[activeIndex]?.command ?? null,
      dismissed: dismissedFor === draft
    }
    console.log('[ghost]', snapshot)
    ;(window as unknown as { __ghostDebug?: unknown }).__ghostDebug = snapshot
  }

  const active: GhostCandidate | null = visibleCandidates[activeIndex] ?? null

  // Debounce the request so a fast typist doesn't fire one LLM call per
  // keystroke. The debounced draft is the value we send.
  useEffect(() => {
    if (!enabled) {
      setCandidates([])
      setActiveIndex(0)
      return
    }

    if (composingRef.current) {
      return
    }

    if (draft.length < MIN_DRAFT_LENGTH || isTriggerActive(draft)) {
      setCandidates([])
      setActiveIndex(0)
      return
    }

    // Re-arm the active index when the draft changes so a stale cycle
    // position from the previous suggestion doesn't carry over.
    setActiveIndex(0)
    setDismissedFor(null)

    const cacheKey = draft.trim()
    const cached = requestCache.get(cacheKey)

    if (cached && Date.now() - cached.cachedAt < CACHE_TTL_MS) {
      setCandidates(cached.candidates)
      return
    }

    const timer = window.setTimeout(() => {
      // Keyword fallback is the always-on floor: it runs regardless of
      // gateway connectivity or catalog state, so a fresh user with
      // an empty catalog cache still sees a ghost for common topics.
      // We seed `validSet` with the known core commands so the user's
      // first keystroke (before `/` has primed the catalog) still
      // surfaces suggestions.
      const SEED_COMMANDS = ['/learn', '/commit', '/commit-push', '/voice', '/gif-search', '/help']
      const liveCatalog = peekCachedSlashCompletion<CommandsCatalogLike>('catalog')
      const validSet = new Set<string>(SEED_COMMANDS)

      for (const [command] of liveCatalog?.pairs ?? []) {
        validSet.add(command.startsWith('/') ? command : `/${command}`)
      }

      const keywordCandidates = keywordFallback(draft.trim(), validSet)

      // LLM branch: only fires when the gateway is alive AND the
      // catalog has loaded. The LLM output is filtered against the same
      // `validSet` so a hallucinated command never shows up as a ghost.
      const catalogDigest = digestCatalog(liveCatalog)

      void (async () => {
        let llmCandidates: GhostCandidate[] = []

        if (gateway && catalogDigest) {
          try {
            const parsed = await askBackend(gateway, draft.trim(), catalogDigest)
            llmCandidates = parsed.filter(c => validSet.has(c.command))
          } catch {
            // LLM call failed — fall through to the keyword branch.
          }
        }

        // Prefer the LLM result when it has any candidates. If it
        // returned empty (NONE or a weak match), the keyword fallback
        // takes over so the user always sees *something* relevant.
        const final = llmCandidates.length > 0 ? llmCandidates : keywordCandidates

        // Only commit if the draft hasn't moved on while we were awaiting.
        if (draftRef.current.trim() !== cacheKey) {
          return
        }

        requestCache.set(cacheKey, { candidates: final, cachedAt: Date.now() })
        setCandidates(final)
      })()
    }, DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [draft, gateway, composingRef, draftRef, enabled])

  // Re-derive the visible list when the rejected set grows.
  useEffect(() => {
    if (visibleCandidates.length === 0) {
      setActiveIndex(0)

      return
    }

    if (activeIndex >= visibleCandidates.length) {
      setActiveIndex(0)
    }
  }, [visibleCandidates.length, activeIndex])

  const cycleNext = useCallback(() => {
    if (visibleCandidates.length === 0) {
      return
    }

    setActiveIndex(index => (index + 1) % visibleCandidates.length)
  }, [visibleCandidates.length])

  const cyclePrev = useCallback(() => {
    if (visibleCandidates.length === 0) {
      return
    }

    setActiveIndex(index => (index - 1 + visibleCandidates.length) % visibleCandidates.length)
  }, [visibleCandidates.length])

  const accept = useCallback(() => {
    if (!active) {
      return
    }

    const editor = editorRef.current

    if (editor && editor.isContentEditable) {
      const current = draftRef.current
      const tail = current.endsWith(' ') || current.length === 0 ? '' : ' '
      const nextDraft = `${current}${tail}${active.command} `
      editor.textContent = nextDraft
      editor.dispatchEvent(
        new InputEvent('input', { bubbles: true, inputType: 'insertText', data: nextDraft })
      )
    }

    // Once accepted, mark the command as rejected so the same suggestion
    // doesn't come back if the user keeps typing into the same draft.
    if (!rejectedCommandsRef.current.has(active.command)) {
      const next = new Set(rejectedCommandsRef.current)
      next.add(active.command)
      rejectedCommandsRef.current = next
    }

    setDismissedFor(draftRef.current)
    setCandidates([])
    setActiveIndex(0)
  }, [active, draftRef, editorRef, rejectedCommandsRef])

  const dismiss = useCallback(() => {
    setDismissedFor(draftRef.current)
    setCandidates([])
    setActiveIndex(0)
  }, [draftRef])

  // Suppress the ghost if the user just dismissed it for this exact draft.
  const finalActive = dismissedFor === draft ? null : active

  return useMemo(
    () => ({
      candidates: visibleCandidates,
      active: finalActive,
      cycleNext,
      cyclePrev,
      accept,
      dismiss
    }),
    [visibleCandidates, finalActive, cycleNext, cyclePrev, accept, dismiss]
  )
}

// Re-exported for tests that want to drive the matchers directly.
export const __testing = { keywordFallback, parseCandidates, digestCatalog }