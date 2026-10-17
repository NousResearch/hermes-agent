import type { MutableRefObject } from 'react'

import type { ClientSessionState, SessionRedirectResponse } from '@/app/types'
import { runtimeTranslations } from '@/i18n/runtime'
import type { ChatMessage } from '@/lib/chat-messages'
import { setComposerDraft } from '@/store/composer'
import { enqueueQueuedPrompt, getQueuedPrompts, serverOwnsComposerQueue } from '@/store/composer-queue'
import { $sessions, resolveComposerSessionKey } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import type { SubmissionDestination } from './submission-destination'
import { type GatewayRequest, isTargetSessionBusy, type SubmitTextOptions } from './utils'

// Typed /undo, /retry, /steer and /queue: one behaviour per command on every surface, mirroring the
// Ink TUI on the same shared gateway (ui-tui/src/app/slash/commands/core.ts +
// canonicalSessionCommands.ts). The owner refuses them as `slash.exec`, so a canonical session runs
// /undo and /retry as the revision-fenced `session.mutate rewind` (Desktop's `session.rewind`),
// /steer as `session.steer`, and /queue on the composer queue the busy composer already uses.

const COMPACTION_MARKERS = ['[CONTEXT COMPACTION', '[CONTEXT SUMMARY]:', '[END OF PRIOR CONTEXT']
const TEXT_PARTS = new Set(['input_text', 'output_text', 'text'])
const PREVIEW_CHARS = 50

interface SnapshotRow {
  archived?: unknown
  content?: unknown
  display_kind?: unknown
  role?: unknown
  row_id?: unknown
}

interface RewindReceipt {
  rewound_count?: number
  status?: string
  target_message?: { content?: unknown } | null
}

interface SlashOutput {
  render: (text: string) => void
  sessionId: string
  storedSessionId: null | string
}

export interface SessionCommandEnv<C extends { arg: string }> {
  activeSessionIdRef: MutableRefObject<null | string>
  busyRef: MutableRefObject<boolean>
  destination: SubmissionDestination
  requestGateway: GatewayRequest
  /** Legacy (non-owner) gateways still serve these through the slash worker. */
  runExec: (ctx: C) => Promise<void>
  submitPromptText: (rawText: string, options?: SubmitTextOptions) => Promise<boolean>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => ClientSessionState
  withSlashOutput: (ctx: C) => Promise<null | SlashOutput>
}

const copy = () => runtimeTranslations().promptNotices.sessionCommands

/** Clip a queued/steered prompt for the confirmation line (Ink's previewOf). */
const previewOf = (text: string) => `${text.slice(0, PREVIEW_CHARS)}${text.length > PREVIEW_CHARS ? '…' : ''}`

const errorText = (err: unknown) => `error: ${err instanceof Error ? err.message : String(err)}`

/** A human-authored user turn (`split_user_originated_turn`): no display-only kind other than a
 *  typed /steer, a durable row id, and not a compaction handoff carrier. */
function isUserTurn(row: SnapshotRow): boolean {
  return (
    row.role === 'user' &&
    (!row.display_kind || row.display_kind === 'steer') &&
    Number.isSafeInteger(row.row_id) &&
    // A live row only: a row a compression archived is history the snapshot shows, never a turn to rewind to.
    !row.archived &&
    !(typeof row.content === 'string' && COMPACTION_MARKERS.some(marker => String(row.content).includes(marker)))
  )
}

/** Lossless /retry text (`retryable_user_text`): '' when the turn has no text, null when it carries
 *  media, unknown or annotated parts (no replay protocol: fail closed). */
export function retryText(content: unknown): null | string {
  const parts = typeof content === 'string' ? [content] : Array.isArray(content) ? content : [null]
  let text = ''

  for (const part of parts) {
    if (typeof part === 'string') {
      text += part
    } else if (
      part &&
      typeof part === 'object' &&
      TEXT_PARTS.has(part.type) &&
      typeof part.text === 'string' &&
      Object.keys(part).length === 2
    ) {
      text += part.text
    } else {
      return null
    }
  }

  return text.trim() ? text : ''
}

/** The Nth newest user turn of the owner's snapshot, or null when there are fewer. */
export function userTurnFromEnd(rows: readonly SnapshotRow[], turnsBack: number): null | SnapshotRow {
  const turns = rows.filter(isUserTurn)

  return turns.length >= turnsBack ? turns[turns.length - turnsBack] : null
}

/** Drop the rewound rows from the rendered transcript: everything from the first message bound to a
 *  durable row at or after the target. Unbound bubbles fall back to the Nth newest user bubble. */
export function truncateRewoundMessages(messages: ChatMessage[], rowId: number, turnsBack: number): ChatMessage[] {
  let cut = messages.findIndex(
    message => (message.rowId ?? -1) >= rowId || message.parts.some(part => (part.sourceRowId ?? -1) >= rowId)
  )

  if (cut < 0) {
    const users = messages.flatMap((message, index) => (message.role === 'user' && !message.hidden ? [index] : []))
    cut = users[users.length - turnsBack] ?? -1
  }

  return cut < 0 ? messages : messages.slice(0, cut)
}

export function sessionCommandHandlers<C extends { arg: string }>(env: SessionCommandEnv<C>) {
  const { activeSessionIdRef, busyRef, destination, requestGateway, submitPromptText, updateSessionState } = env

  // Rewind the Nth newest user turn on the owner. Resolves to the receipt's rewound text, or
  // undefined after it already rendered why nothing was rewound.
  async function rewind(
    out: SlashOutput,
    turnsBack: number,
    intent: 'retry' | 'undo'
  ): Promise<undefined | { count: number; content: unknown }> {
    const { render, sessionId, storedSessionId } = out
    const nothing = intent === 'retry' ? copy().retryNothing : copy().undoNothing
    // The resume snapshot carries every row's durable id and primes the CAS revision/generation the
    // rewind mutation presents, as Ink's mutateCanonical does.
    const snapshot = await requestGateway<{ messages?: SnapshotRow[] }>('session.resume', { session_id: sessionId })
    const target = userTurnFromEnd(Array.isArray(snapshot?.messages) ? snapshot.messages : [], turnsBack)

    if (!target) {
      return void render(nothing)
    }

    if (intent === 'retry') {
      const text = retryText(target.content)

      if (text === null) {
        return void render(copy().retryUnsupported)
      }

      if (!text) {
        return void render(nothing)
      }
    }

    const rowId = target.row_id as number

    const receipt = await requestGateway<RewindReceipt>('session.rewind', {
      session_id: sessionId,
      target_message_id: rowId
    })

    const count = Number(receipt?.rewound_count ?? 0)

    if (receipt?.status === 'nothing' || count <= 0) {
      return void render(nothing)
    }

    updateSessionState(
      sessionId,
      state => ({ ...state, messages: truncateRewoundMessages(state.messages, rowId, turnsBack) }),
      storedSessionId
    )

    return { count, content: receipt?.target_message?.content }
  }

  // The next-turn queue the composer uses while busy: the owner's durable FIFO for a canonical
  // session (a `queued` admission), the local composer queue otherwise.
  async function enqueue(out: SlashOutput, text: string): Promise<boolean> {
    const { sessionId, storedSessionId } = out

    if (serverOwnsComposerQueue(storedSessionId ?? sessionId)) {
      return submitPromptText(text, { destination, fromQueue: true, sessionId, storedSessionId })
    }

    return Boolean(enqueueQueuedPrompt(queueKeyFor(out), { attachments: [], text }))
  }

  const canonical = (out: SlashOutput) => serverOwnsComposerQueue(out.storedSessionId ?? out.sessionId)

  return {
    undo: async (ctx: C) => {
      const out = await env.withSlashOutput(ctx)

      if (!out) {
        return
      }

      if (!canonical(out)) {
        return env.runExec(ctx)
      }

      const arg = ctx.arg.trim()

      if (arg && !/^[1-9]\d*$/.test(arg)) {
        return out.render(copy().undoUsage)
      }

      try {
        const rewound = await rewind(out, Number(arg || 1), 'undo')

        if (!rewound) {
          return
        }

        // Main's /undo: the removed message goes back into the composer it was typed in.
        const text = typeof rewound.content === 'string' ? rewound.content : (retryText(rewound.content) ?? '')

        if (text.trim() && activeSessionIdRef.current === out.sessionId) {
          setComposerDraft(text)
        }

        out.render(rewound.count === 1 ? copy().undidOne('1') : copy().undidOther(String(rewound.count)))
      } catch (err) {
        out.render(errorText(err))
      }
    },
    retry: async (ctx: C) => {
      const out = await env.withSlashOutput(ctx)

      if (!out) {
        return
      }

      if (!canonical(out)) {
        return env.runExec(ctx)
      }

      try {
        const rewound = await rewind(out, 1, 'retry')

        if (!rewound) {
          return
        }

        // Main's /retry: the rewound turn's own text, not whatever this view typed last.
        const text = retryText(rewound.content)

        if (!text) {
          return out.render(copy().retryNothing)
        }

        await submitPromptText(text, { destination, sessionId: out.sessionId, storedSessionId: out.storedSessionId })
      } catch (err) {
        out.render(errorText(err))
      }
    },
    steer: async (ctx: C) => {
      const out = await env.withSlashOutput(ctx)

      if (!out) {
        return
      }

      const text = ctx.arg.trim()

      if (!text) {
        return out.render(copy().steerUsage)
      }

      // No running turn: queue the words as the next turn so they are not lost (Ink, gateway).
      if (!isTargetSessionBusy($sessionStates.get(), out.sessionId, busyRef.current)) {
        return out.render((await enqueue(out, text)) ? copy().steerNoActiveTurn(previewOf(text)) : copy().queueFailed)
      }

      let accepted = false

      try {
        const result = await requestGateway<SessionRedirectResponse>('session.steer', {
          session_id: out.sessionId,
          text
        })

        accepted = result?.status === 'queued'
      } catch {
        accepted = false
      }

      if (accepted) {
        return out.render(copy().steerQueued(previewOf(text)))
      }

      // The turn ended before the steer landed (#64578): keep the words as the next turn.
      out.render((await enqueue(out, text)) ? copy().steerRejected : copy().queueFailed)
    },
    queue: async (ctx: C) => {
      const out = await env.withSlashOutput(ctx)

      if (!out) {
        return
      }

      const text = ctx.arg.trim()

      if (!text) {
        const count = getQueuedPrompts(queueKeyFor(out)).length

        return out.render(count === 1 ? copy().countOne('1') : copy().countOther(String(count)))
      }

      out.render((await enqueue(out, text)) ? copy().queued(previewOf(text)) : copy().queueFailed)
    }
  }
}

/** The composer queue key for the target session (queue-if-busy.ts's resolution). */
function queueKeyFor({ sessionId, storedSessionId }: SlashOutput): string {
  const stored = storedSessionId ?? $sessionStates.get()[sessionId]?.storedSessionId ?? null

  return resolveComposerSessionKey(stored, $sessions.get()) || stored || sessionId
}
