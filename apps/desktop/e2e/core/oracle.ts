/**
 * The transcript oracle.
 *
 * Invariant, checked after every transition: each persisted user/assistant
 * message is rendered EXACTLY ONCE, in persisted order; nothing scenario-made
 * is rendered that is not persisted; the persisted assistant rows are exactly
 * what the provider streamed; and on the wire, each turn's concatenated
 * message.delta text equals what the provider streamed for that turn (and its
 * message.complete text equals the final completion).
 *
 * Every scenario text carries a unique marker (`U<n>-<nonce>` for the user,
 * `A<n>-<nonce>` / `A<n>i-<nonce>` for replies, `R<n>-<nonce>` for reasoning),
 * so "exactly once" is a count over rendered text, independent of how the
 * renderer groups bubbles (live vs hydrated grouping legitimately differs).
 *
 * Final-state checks converge with a deadline (hydration is asynchronous);
 * transient duplicates are caught separately by an in-page MutationObserver
 * sampler that is never allowed to see a marker twice.
 */

import { expect, type Page } from '@playwright/test'

import type { GatewayEventFrame, PersistedMessage, WsRecorder } from './harness'
import { persistedTranscript } from './harness'
import type { RecordedCompletion, ScriptedProvider } from './provider'

export const ANY_MARKER_RE = /\b[UAR]\d+i?-[a-z0-9]{3,}\b/g
const FIRST_MARKER_RE = /\b[UAR]\d+i?-[a-z0-9]{3,}\b/

const norm = (text: string) => text.replace(/\s+/g, ' ').trim()

function countOccurrences(haystack: string, needle: string): number {
  if (!needle) {
    return 0
  }

  let count = 0
  let at = haystack.indexOf(needle)

  while (at !== -1) {
    count++
    at = haystack.indexOf(needle, at + needle.length)
  }

  return count
}

function countMarker(haystack: string, marker: string): number {
  return (haystack.match(new RegExp(`\\b${marker}\\b`, 'g')) ?? []).length
}

/** The user marker a reply/reasoning marker answers: A3-x / A3i-x / R3-x → U3-x. */
export function userMarkerFor(marker: string): string {
  return marker.replace(/^[AR](\d+)i?-/, 'U$1-')
}

// ─── Transient-duplicate sampler ────────────────────────────────────────

/** Idempotent; re-run after every reload (the observer dies with the document). */
export async function installDuplicateSampler(page: Page): Promise<void> {
  await page.evaluate(source => {
    const w = window as any

    if (w.__coreSampler) {
      return
    }

    const re = new RegExp(source, 'g')
    w.__coreSampler = { samples: 0, violations: [] as { marker: string; count: number; text: string }[] }
    let scheduled = false

    const sample = () => {
      scheduled = false
      const viewports = [...document.querySelectorAll('[data-slot="aui_thread-viewport"]')] as HTMLElement[]

      for (const viewport of viewports) {
        if (viewport.getClientRects().length === 0) {
          continue
        }

        const text = viewport.innerText
        w.__coreSampler.samples++
        const counts = new Map<string, number>()

        for (const match of text.match(re) ?? []) {
          counts.set(match, (counts.get(match) ?? 0) + 1)
        }

        for (const [marker, count] of counts) {
          if (count > 1 && w.__coreSampler.violations.length < 20) {
            // Where the copies live: one bubble with doubled text vs two bubbles.
            const bubbles = (
              [
                ...viewport.querySelectorAll(
                  '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]'
                )
              ] as HTMLElement[]
            )
              .filter(el => el.innerText.includes(marker))
              .map(el => `${el.getAttribute('data-slot')}#${el.getAttribute('data-message-id') ?? el.id ?? ''}`)

            w.__coreSampler.violations.push({
              marker,
              count,
              at: Math.round(performance.now()),
              route: location.hash,
              bubbles,
              text: text.replace(/\s+/g, ' ').slice(0, 600)
            })
          }
        }
      }
    }

    new MutationObserver(() => {
      if (!scheduled) {
        scheduled = true
        requestAnimationFrame(sample)
      }
    }).observe(document.body, { childList: true, subtree: true, characterData: true })
  }, ANY_MARKER_RE.source)
}

async function samplerViolations(page: Page): Promise<{ samples: number; violations: any[] }> {
  return page.evaluate(() => (window as any).__coreSampler ?? { samples: 0, violations: [] })
}

// ─── Rendered view ──────────────────────────────────────────────────────

interface RenderedView {
  text: string
  userBubbles: string[]
}

async function renderedView(page: Page): Promise<RenderedView> {
  return page.evaluate(() => {
    const viewport = ([...document.querySelectorAll('[data-slot="aui_thread-viewport"]')] as HTMLElement[]).find(
      el => el.getClientRects().length > 0
    )

    if (!viewport) {
      return { text: '', userBubbles: [] }
    }

    return {
      text: viewport.innerText.replace(/\s+/g, ' '),
      userBubbles: ([...viewport.querySelectorAll('[data-slot="aui_user-message-root"]')] as HTMLElement[]).map(el =>
        el.innerText.replace(/\s+/g, ' ').trim()
      )
    }
  })
}

// ─── Wire check ─────────────────────────────────────────────────────────

interface WireTurn {
  sessionId: string
  deltas: string
  reasoning: string
  complete: null | string
}

/**
 * Backend-truth turns: events deduplicated by (runtime session, seq) across
 * every socket (the backend stamps seq once before fan-out), split at
 * message.start, in seq order.
 */
export function wireTurns(events: GatewayEventFrame[]): WireTurn[] {
  const unique = new Map<string, GatewayEventFrame>()

  for (const event of events) {
    if (event.seq === null || !event.sessionId) {
      continue
    }

    const key = `${event.sessionId}#${event.seq}`

    if (!unique.has(key)) {
      unique.set(key, event)
    }
  }

  const bySession = new Map<string, GatewayEventFrame[]>()

  for (const event of unique.values()) {
    const list = bySession.get(event.sessionId) ?? []
    list.push(event)
    bySession.set(event.sessionId, list)
  }

  const turns: WireTurn[] = []

  for (const [sessionId, list] of bySession) {
    list.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0))
    let turn: null | WireTurn = null

    for (const event of list) {
      if (event.type === 'message.start') {
        turn = { sessionId, deltas: '', reasoning: '', complete: null }
        turns.push(turn)
      } else if (!turn) {
        continue
      } else if (event.type === 'message.delta') {
        turn.deltas += String(event.payload?.text ?? '')
      } else if (event.type === 'reasoning.delta') {
        turn.reasoning += String(event.payload?.text ?? '')
      } else if (event.type === 'message.complete') {
        turn.complete = String(event.payload?.text ?? '')
      }
    }
  }

  return turns
}

/**
 * Split a turn's streamed text at scenario markers: every scripted completion
 * starts with its own marker, so each segment is one completion's output.
 */
function segments(text: string, re: RegExp): { marker: string; text: string }[] {
  const out: { marker: string; text: string }[] = []
  const flat = norm(text)
  const hits = [...flat.matchAll(new RegExp(re.source, 'g'))]

  hits.forEach((hit, i) => {
    const end = i + 1 < hits.length ? hits[i + 1]!.index : flat.length
    out.push({ marker: hit[0], text: flat.slice(hit.index, end).trim() })
  })

  return out
}

/**
 * Backend stream integrity per turn: the concatenated message.delta (and
 * reasoning.delta) text is exactly the provider's output, segment by segment —
 * each completion once, whole (or a prefix when the backend itself hung up on
 * it, e.g. a steer), in order — and message.complete equals the final
 * completion. Only whitespace between tool-iteration segments may differ.
 */
function isSubsequence(words: string[], of: string[]): boolean {
  let i = 0

  for (const word of of) {
    if (i < words.length && words[i] === word) {
      i++
    }
  }

  return i === words.length
}

function wireViolations(
  ws: WsRecorder,
  provider: ScriptedProvider,
  userMarkers: Set<string>,
  lossy: Set<string>
): string[] {
  const problems: string[] = []
  const byOpening = new Map<string, RecordedCompletion>()

  for (const completion of provider.completions) {
    const opening = FIRST_MARKER_RE.exec(completion.sentText)?.[0]
    const reasoningOpening = FIRST_MARKER_RE.exec(completion.sentReasoning)?.[0]

    if (opening) {
      byOpening.set(opening, completion)
    }

    if (reasoningOpening) {
      byOpening.set(reasoningOpening, completion)
    }
  }

  for (const turn of wireTurns(ws.events)) {
    const textSegments = segments(turn.deltas, /\bA\d+i?-[a-z0-9]{3,}\b/)

    if (textSegments.length === 0 || !userMarkers.has(userMarkerFor(textSegments.at(-1)!.marker))) {
      continue
    }

    // A turn that straddled an injected socket drop may have frames that went
    // to the dead socket (the renderer recovers them from the persisted
    // transcript, which the render oracle checks). Its frames must still be an
    // in-order subsequence of what was streamed — never doubled or foreign.
    const gapped = lossy.has(userMarkerFor(textSegments.at(-1)!.marker))

    if (turn.complete === null && !gapped) {
      problems.push(`wire ${textSegments.at(-1)!.marker}: turn never completed`)

      continue
    }

    const seen = new Set<string>()

    const check = (kind: string, segs: { marker: string; text: string }[], pick: (c: RecordedCompletion) => string) => {
      segs.forEach((seg, i) => {
        const completion = byOpening.get(seg.marker)

        if (!completion) {
          problems.push(`wire: ${kind} segment ${seg.marker} was never streamed by the provider`)

          return
        }

        if (seen.has(`${kind}:${seg.marker}`)) {
          problems.push(`wire: ${kind} segment ${seg.marker} delivered twice in one turn`)
        }

        seen.add(`${kind}:${seg.marker}`)
        const sent = norm(pick(completion))
        const partialAllowed = completion.aborted && i < segs.length - 1

        if (gapped) {
          if (!isSubsequence(seg.text.split(' '), sent.split(' '))) {
            problems.push(
              `wire: ${kind} ${JSON.stringify(seg.text)} is not an in-order subsequence of ${JSON.stringify(sent)}`
            )
          }
        } else if (partialAllowed ? !sent.startsWith(seg.text) : seg.text !== sent) {
          problems.push(
            `wire: ${kind} ${JSON.stringify(seg.text)} != provider ${JSON.stringify(sent)}${completion.aborted ? ' (aborted)' : ''}`
          )
        }
      })
    }

    check('message.delta', textSegments, c => c.sentText)
    check('reasoning.delta', segments(turn.reasoning, /\bR\d+-[a-z0-9]{3,}\b/), c => c.sentReasoning)

    const final = byOpening.get(textSegments.at(-1)!.marker)

    if (final && turn.complete !== null && norm(turn.complete) !== norm(final.sentText)) {
      problems.push(
        `wire: message.complete ${JSON.stringify(turn.complete)} != final completion ${JSON.stringify(final.sentText)}`
      )
    }
  }

  return problems
}

// ─── The oracle ─────────────────────────────────────────────────────────

export interface OracleTarget {
  /** Stored session id (the route id). */
  sessionId: string
  profile?: string
  /** User markers this session must contain (guards against an empty/wrong session passing vacuously). */
  expectUserMarkers: string[]
  /** User markers whose turn straddled an injected socket drop (wire frames may be gapped, never doubled). */
  lossyWire?: string[]
}

function transcriptViolations(persisted: PersistedMessage[], view: RenderedView, target: OracleTarget): string[] {
  const problems: string[] = []
  const rows = persisted.filter(m => (m.role === 'user' || m.role === 'assistant') && norm(m.content))
  const persistedMarkers = new Set<string>()

  for (const marker of target.expectUserMarkers) {
    if (!rows.some(row => row.role === 'user' && countMarker(row.content, marker) === 1)) {
      problems.push(`persisted: user message ${marker} missing from the session`)
    }
  }

  for (const marker of target.expectUserMarkers) {
    if (rows.filter(row => row.role === 'user' && row.content.includes(marker)).length > 1) {
      problems.push(`persisted: user message ${marker} stored more than once`)
    }
  }

  let cursor = -1

  for (const row of rows) {
    const content = norm(row.content)

    for (const marker of content.match(ANY_MARKER_RE) ?? []) {
      persistedMarkers.add(marker)
    }

    const occurrences = countOccurrences(view.text, content)

    if (occurrences !== 1) {
      problems.push(`rendered ${occurrences}x (want 1): ${row.role} ${JSON.stringify(content.slice(0, 80))}`)

      continue
    }

    const at = view.text.indexOf(content)

    if (at < cursor) {
      problems.push(`order: ${row.role} ${JSON.stringify(content.slice(0, 40))} rendered before an earlier message`)
    }

    cursor = at
  }

  const renderedMarkers = view.text.match(ANY_MARKER_RE) ?? []
  const counts = new Map<string, number>()

  for (const marker of renderedMarkers) {
    counts.set(marker, (counts.get(marker) ?? 0) + 1)
  }

  for (const [marker, count] of counts) {
    if (count > 1) {
      problems.push(`marker ${marker} rendered ${count}x`)
    }

    // Reasoning is not part of the display content; every other marker must be persisted.
    if (!marker.startsWith('R') && !persistedMarkers.has(marker)) {
      problems.push(`rendered but not persisted: ${marker}`)
    }
  }

  const userRows = rows.filter(row => row.role === 'user').map(row => norm(row.content))

  if (view.userBubbles.length !== userRows.length) {
    problems.push(`user bubbles ${view.userBubbles.length} != persisted user rows ${userRows.length}`)
  }

  return problems
}

/**
 * Assert the invariant for the session currently on screen. Converges on the
 * final state (deadline), then requires zero transient duplicates since the
 * sampler was installed and a clean wire.
 */
export async function assertTranscriptOracle(
  page: Page,
  ws: WsRecorder,
  provider: ScriptedProvider,
  target: OracleTarget,
  label: string
): Promise<void> {
  let last: { problems: string[]; persisted: PersistedMessage[]; view: RenderedView } = {
    problems: ['not evaluated'],
    persisted: [],
    view: { text: '', userBubbles: [] }
  }

  // Wire first: wait until every expected turn's message.complete is on the
  // wire (it can trail the persisted row under load), so the DOM check below
  // also covers whatever the renderer does on completion. Duplicate/garbled
  // frames never heal, so polling cannot mask them.
  await expect
    .poll(() => wireViolations(ws, provider, new Set(target.expectUserMarkers), new Set(target.lossyWire ?? [])), {
      timeout: 60_000,
      intervals: [250, 500, 1000],
      message: `wire integrity [${label}]`
    })
    .toEqual([])

  await expect
    .poll(
      async () => {
        const persisted = await persistedTranscript(page, target.sessionId, target.profile)
        const view = await renderedView(page)
        last = { problems: transcriptViolations(persisted, view, target), persisted, view }

        return last.problems
      },
      { timeout: 60_000, intervals: [250, 500, 1000], message: `transcript oracle [${label}]` }
    )
    .toEqual([])
    .catch(error => {
      throw new Error(
        `transcript oracle [${label}] failed:\n  ${last.problems.join('\n  ')}\n` +
          `persisted: ${JSON.stringify(last.persisted.map(m => [m.role, m.content.slice(0, 60)]))}\n` +
          `rendered: ${JSON.stringify(last.view.text.slice(0, 1500))}\n(${(error as Error).message.split('\n')[0]})`
      )
    })

  const transient = await samplerViolations(page)
  expect(transient.violations, `transient duplicate render during [${label}] (${transient.samples} samples)`).toEqual(
    []
  )
}
