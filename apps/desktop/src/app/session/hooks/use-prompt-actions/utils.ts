import type { AppendMessage } from '@assistant-ui/react'

import { translateNow, type Translations } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { type CommandsCatalogLike, filterDesktopCommandsCatalog } from '@/lib/desktop-slash-commands'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import type { ComposerAttachment } from '@/store/composer'

export type GatewayRequest = <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export function isSessionIdCandidate(value: string): boolean {
  const trimmed = value.trim()

  return /^\d{8}_\d{6}_[A-Fa-f0-9]{6}$/.test(trimmed) || /^[A-Fa-f0-9]{32}$/.test(trimmed)
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error(translateNow('desktop.audioReadFailed')))
      }
    })
    reader.addEventListener('error', () => reject(reader.error || new Error(translateNow('desktop.audioReadFailed'))))
    reader.readAsDataURL(blob)
  })
}

export function isProviderSetupError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)

  return isProviderSetupErrorMessage(message)
}

export function inlineErrorMessage(error: unknown, fallback: string): string {
  const raw = error instanceof Error ? error.message : typeof error === 'string' ? error : fallback

  return (raw.match(/Error invoking remote method '[^']+': Error: (.+)$/)?.[1] ?? raw).replace(/^Error:\s*/, '').trim()
}

export function isSessionNotFoundError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /session not found/i.test(message)
}

/**
 * Thrown when a stale-session recovery resumed successfully but the caller's
 * drift check says the user has since moved on (profile swap, route rebind,
 * a different chat in the foreground). The retry is deliberately NOT attempted:
 * landing it would run the prompt against a session the user is no longer
 * looking at. Callers unwind through their own abort path (#66889).
 */
export class SessionRecoveryAborted extends Error {
  constructor(
    readonly reason: string,
    readonly recoveredSessionId: string
  ) {
    super(`session recovery aborted: ${reason}`)
    this.name = 'SessionRecoveryAborted'
  }
}

export interface SessionRecoveryDeps {
  requestGateway: GatewayRequest
  /**
   * Owning profile for a stored session. A resume without it lands on
   * whichever gateway is active and forks the conversation into the wrong
   * profile's DB (#67603).
   *
   * Injected rather than imported so this module stays free of the session
   * store and the REST layer: the default implementation reaches through
   * `resolveStoredSession` → `getSession()`, a real fetch that makes any unit
   * test of this helper depend on leftover `$sessions` / `$profiles` state.
   */
  resolveProfile?: (storedSessionId: string) => Promise<string | undefined>
  /**
   * Publish the fresh live id. Implementations must update BOTH the hot ref
   * and the `$activeSessionId` atom — a ref-only write leaves the atom
   * pointing at the dead runtime and every atom-reading surface desyncs
   * (#62471).
   */
  onRecovered?: (liveSessionId: string) => void
  /**
   * Non-null reason ⇒ abort instead of retrying. Evaluated AFTER the resume
   * and BEFORE the retry, because the resume is the slow await during which a
   * profile switch or route rebind can land.
   */
  driftReason?: () => null | string
}

async function defaultResolveProfile(storedSessionId: string): Promise<string | undefined> {
  // Lazy so utils.ts has no init-time cycle with use-session-actions.
  const { resolveSessionProfile } = await import('../use-session-actions/utils')

  return resolveSessionProfile(storedSessionId)
}

/**
 * Re-register a durable stored session after the gateway dropped its
 * in-memory runtime id (sleep/wake, remote backend restart, long idle).
 * Returns the fresh live id, or null when the resume yields none.
 */
export async function resumeStoredRuntimeSession(
  storedSessionId: string,
  deps: SessionRecoveryDeps
): Promise<null | string> {
  const resolveProfile = deps.resolveProfile ?? defaultResolveProfile
  const profile = await resolveProfile(storedSessionId)

  const resumed = await deps.requestGateway<{ session_id: string }>('session.resume', {
    session_id: storedSessionId,
    source: 'desktop',
    omit_messages: true,
    ...(profile ? { profile } : {})
  })

  return resumed?.session_id ?? null
}

/**
 * Single resolver for "the runtime session id I hold is dead."
 *
 * Every session-scoped RPC needs this, not just `prompt.submit`. Attach,
 * `/compress`, checkpoint restore, and interrupt all run against the same
 * runtime id and all used to surface a raw "session not found" after sleep —
 * while plain text silently recovered, which is why the bug reads as "text
 * works, images don't."
 *
 * Runs `call(sessionId)`. On a stale-session error it resumes the stored
 * session ONCE, republishes the fresh id, and retries. Bounded to a single
 * retry: a second failure is a real error, not a stale binding.
 *
 * A resume that itself 404s (a never-persisted first-submit draft has no DB
 * row until its first successful submit) rethrows the ORIGINAL error rather
 * than the confusing secondary one (#67539).
 */
export async function withSessionNotFoundResume<T>(
  sessionId: string,
  storedSessionId: null | string | undefined,
  call: (liveSessionId: string) => Promise<T>,
  deps: SessionRecoveryDeps,
  options?: { alsoTimeout?: boolean }
): Promise<{ recovered: boolean; result: T; sessionId: string }> {
  try {
    return { recovered: false, result: await call(sessionId), sessionId }
  } catch (err) {
    // A starved backend loop rejects with a timeout that is indistinguishable
    // from a dead runtime on the client side (#55578). Opt-in per caller:
    // submit recovers from it, a compress/attach retry should not mask a
    // genuinely slow LLM-bound call.
    const recoverable = isSessionNotFoundError(err) || (Boolean(options?.alsoTimeout) && isGatewayTimeoutError(err))

    if (!recoverable || !storedSessionId) {
      throw err
    }

    let recoveredId: null | string

    try {
      recoveredId = await resumeStoredRuntimeSession(storedSessionId, deps)
    } catch {
      throw err
    }

    if (!recoveredId) {
      throw err
    }

    const drift = deps.driftReason?.()

    if (drift) {
      throw new SessionRecoveryAborted(drift, recoveredId)
    }

    deps.onRecovered?.(recoveredId)

    return { recovered: true, result: await call(recoveredId), sessionId: recoveredId }
  }
}

/**
 * Is the session a prompt is about to run against currently mid-turn?
 *
 * The foreground `busyRef` is NOT the answer. It mirrors whatever chat is on
 * screen, while submit and slash both resolve their target through
 * `resolveTargetSessionId` — routinely a different session (a tile, a route
 * rebind, a session created by this very call). Reading the foreground flag
 * therefore gates one session's send on another session's turn: a stale
 * foreground `true` (a warm resume of a still-running chat leaves one behind)
 * blocks an IDLE target and reports "session busy" about a session doing
 * nothing, and the converse lets a background send fire mid-turn.
 *
 * The published per-session state is authoritative. A known target with no
 * slice yet is idle — never inherit another session's leftover foreground
 * flag (focusing B while A runs). Fall back to the foreground flag only for
 * a true draft (no session id), where that flag must be the focused view's
 * busy, not a process-global lock.
 */
export function isTargetSessionBusy(
  sessionStates: Record<string, { busy: boolean }>,
  sessionId: null | string,
  foregroundBusy: boolean
): boolean {
  if (!sessionId) {
    return foregroundBusy
  }

  return Boolean(sessionStates[sessionId]?.busy)
}

// Gateway JSON-RPC calls reject with "request timed out: <method>" when the
// backend event loop is starved (e.g. a poller spin or a heavy async-injected
// turn). For prompt.submit this is indistinguishable from a dead runtime
// session on the client side — recovery must treat it like one (#55578):
// resume the SELECTED stored session and retry, instead of surfacing an error
// that leads to a null activeSessionId and a silently minted new session.
export function isGatewayTimeoutError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /request timed out/i.test(message)
}

// The gateway refuses prompt.submit while a turn is running (4009 "session
// busy"). It's a transient concurrency guard, never a user-facing error: a
// submit racing the settle edge (or a rewind interrupting mid-turn) just waits
// a beat for the turn to wind down, then lands. Bounded so a genuinely stuck
// turn still surfaces eventually.
export const SESSION_BUSY_RETRY_TIMEOUT_MS = 6_000
export const SESSION_BUSY_RETRY_INTERVAL_MS = 150

export function isSessionBusyError(error: unknown): boolean {
  return /session busy/i.test(error instanceof Error ? error.message : String(error))
}

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

// Retry a gateway call across transient "session busy" so it never reaches the
// user — the turn settles within the deadline and the call lands.
export async function withSessionBusyRetry<T>(call: () => Promise<T>): Promise<T> {
  const deadline = Date.now() + SESSION_BUSY_RETRY_TIMEOUT_MS

  for (;;) {
    try {
      return await call()
    } catch (err) {
      if (isSessionBusyError(err) && Date.now() < deadline) {
        await sleep(SESSION_BUSY_RETRY_INTERVAL_MS)

        continue
      }

      throw err
    }
  }
}

// After Stop, the renderer clears busy immediately while the gateway may still
// be winding down. Edit/restore that only checks busy then submits without
// interrupt-first and hits 4009 session busy. A short per-session cooldown
// keeps interrupt-first on for that window (#83855).
export const RECENT_INTERRUPT_COOLDOWN_MS = 3_000

const _recentlyInterruptedUntil = new Map<string, number>()

export function markSessionRecentlyInterrupted(sessionId: string, now = Date.now()): void {
  if (!sessionId) {
    return
  }

  _recentlyInterruptedUntil.set(sessionId, now + RECENT_INTERRUPT_COOLDOWN_MS)
}

export function isSessionRecentlyInterrupted(sessionId: string, now = Date.now()): boolean {
  const until = _recentlyInterruptedUntil.get(sessionId)

  if (until === undefined) {
    return false
  }

  if (now >= until) {
    _recentlyInterruptedUntil.delete(sessionId)

    return false
  }

  return true
}

export function clearSessionRecentlyInterrupted(sessionId?: string): void {
  if (sessionId) {
    _recentlyInterruptedUntil.delete(sessionId)

    return
  }

  _recentlyInterruptedUntil.clear()
}

/** Whether a rewind/edit should interrupt before submit — busy OR recent Stop. */
export function shouldInterruptBeforeRewind(opts: { busy: boolean; sessionId: string; now?: number }): boolean {
  return opts.busy || isSessionRecentlyInterrupted(opts.sessionId, opts.now)
}

// Hard guard: at most one prompt.submit in flight per session. Every submit
// path — user Enter, queue drain, busy-retry, slash fallthrough — funnels
// through submitPromptText. Without this, a stalled turn (e.g. a context-bloated
// session whose first call hangs) let the SAME prompt launch several real turns
// at once (the "message stacked 5×" bug). Keyed by stored/active session id.
// Entries expire so a hung submit cannot permanently block the session (#83855).
export const SUBMIT_IN_FLIGHT_TTL_MS = 30_000

const _submitInFlightAt = new Map<string, number>()

export function isSubmitInFlight(key: string, now = Date.now()): boolean {
  const acquiredAt = _submitInFlightAt.get(key)

  if (acquiredAt === undefined) {
    return false
  }

  if (now - acquiredAt >= SUBMIT_IN_FLIGHT_TTL_MS) {
    _submitInFlightAt.delete(key)

    return false
  }

  return true
}

/** Returns true when the lock was acquired; false when another fresh hold blocks. */
export function acquireSubmitInFlight(key: string, now = Date.now()): boolean {
  if (isSubmitInFlight(key, now)) {
    return false
  }

  _submitInFlightAt.set(key, now)

  return true
}

export function releaseSubmitInFlight(key: string): void {
  _submitInFlightAt.delete(key)
}

export function clearSubmitInFlight(): void {
  _submitInFlightAt.clear()
}

export function base64FromDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(',')

  return comma >= 0 ? dataUrl.slice(comma + 1) : ''
}

export function imageFilenameFromPath(filePath: string): string {
  return filePath.split(/[\\/]/).filter(Boolean).pop() || 'image.png'
}

// Remote gateway: the local composer-image file lives on THIS machine's disk,
// not the gateway's, so read the bytes here and upload them via
// image.attach_bytes. Returns null when the file can't be read.
//
// `cachedDataUrl` is the attachment's `previewUrl` when the composer already
// read the file for the chip thumbnail — that preview is the FULL file as a
// base64 data URL (attachmentPreviewDataUrl → readFileDataUrl), not a
// downscaled copy, so reusing it skips a second disk read + IPC round-trip of
// the same bytes at submit. Only a `;base64,` data URL qualifies; anything
// else falls through to the disk read.
export async function readImageForRemoteAttach(
  filePath: string,
  cachedDataUrl?: string
): Promise<{ contentBase64: string; filename: string } | null> {
  if (cachedDataUrl?.includes(';base64,')) {
    const cached = base64FromDataUrl(cachedDataUrl)

    if (cached) {
      return { contentBase64: cached, filename: imageFilenameFromPath(filePath) }
    }
  }

  const dataUrl = await window.hermesDesktop?.readFileDataUrl(filePath)
  const contentBase64 = dataUrl ? base64FromDataUrl(dataUrl) : ''

  return contentBase64 ? { contentBase64, filename: imageFilenameFromPath(filePath) } : null
}

// Read a non-image file as a data URL for upload via file.attach. Returns null
// when the desktop bridge can't read the file (e.g. it was moved/deleted).
// Prefer the attach-specific IPC (256 MiB) so remote uploads are not stuck on
// the preview/Settings default; fall back for older Electron shells.
export async function readFileDataUrlForAttach(filePath: string): Promise<string | null> {
  const reader = window.hermesDesktop?.readFileDataUrlForAttach ?? window.hermesDesktop?.readFileDataUrl

  if (!reader) {
    return null
  }

  const dataUrl = await reader(filePath)

  return dataUrl || null
}

// The attach/preview IPC base64-loads the whole file into memory and rejects
// with a raw "file is too large (N bytes; limit M bytes)" string when over
// cap. In remote mode every attachment's bytes go through that read, so a big
// file surfaces that internal message verbatim in the failure toast. Translate
// it into a friendly "too large to upload to the remote gateway" line, parsing
// the limit out of the message so it tracks the real cap. Non-cap errors pass
// through unchanged.
export function friendlyRemoteAttachError(err: unknown, label: string): Error {
  const message = err instanceof Error ? err.message : String(err)

  if (!/too large/i.test(message)) {
    return err instanceof Error ? err : new Error(message)
  }

  const limitBytes = Number(message.match(/limit (\d+) bytes/)?.[1])
  const cap = Number.isFinite(limitBytes) && limitBytes > 0 ? ` (max ${Math.floor(limitBytes / (1024 * 1024))} MB)` : ''

  return new Error(`${label} is too large to upload to the remote gateway${cap}.`)
}

export function renderCommandsCatalog(catalog: CommandsCatalogLike, copy: Translations['desktop']): string {
  const desktopCatalog = filterDesktopCommandsCatalog(catalog)

  const sections = desktopCatalog.categories?.length
    ? desktopCatalog.categories
    : [{ name: copy.desktopCommands, pairs: desktopCatalog.pairs ?? [] }]

  const body = sections
    .filter(section => section.pairs.length > 0)
    .map(section => {
      const rows = section.pairs.map(([cmd, desc]) => `${cmd.padEnd(18)} ${desc}`)

      return [`${section.name}:`, ...rows].join('\n')
    })
    .join('\n\n')

  const tail = [
    desktopCatalog.skill_count ? copy.skillCommandsAvailable(desktopCatalog.skill_count) : '',
    desktopCatalog.warning ? copy.warningLine(desktopCatalog.warning) : ''
  ]
    .filter(Boolean)
    .join('\n')

  return [body || 'No desktop commands available.', tail].filter(Boolean).join('\n\n')
}

export function slashStatusText(command: string, output: string): string {
  return [`slash:${command}`, output.trim()].filter(Boolean).join('\n')
}

/**
 * Format the JSON reply from a gateway RPC surfaced as `kind: 'rpc'` in
 * desktop-slash-commands.ts. Kept here (instead of slash.ts) so it has no
 * React / store dependencies and can be unit-tested in isolation.
 *
 * The renderer follows the field conventions shared by the gateway handlers
 * we route to today:
 * - `session.compress`: { summary: { headline, token_line, note, noop } }
 *   (only used if /compress ever moves to `rpc`; it's an `action` today
 *   because it needs transcript replacement)
 * - `session.status`:   { output: "<multi-line plain text>" }
 * - `session.save`:     { file: "<absolute path>" }
 * - `session.usage`:    { calls, input, output, total, credits_lines? }
 * - `session.steer`:    { status: 'queued' | 'rejected', text }
 * - `process.stop`:     { killed: boolean }
 * - `agents.list`:      { processes: [{ session_id, command, status, uptime }] }
 *
 * Any RPC whose response doesn't match these shapes falls through to a
 * JSON.stringify dump so we never silently swallow data.
 */
export function renderRpcResult(response: unknown, name: string): string {
  if (!response || typeof response !== 'object') {
    return ''
  }

  const r = response as Record<string, unknown>

  const summary = r.summary as { headline?: string; token_line?: string; note?: string; noop?: boolean } | undefined

  if (summary && typeof summary === 'object' && typeof summary.headline === 'string' && summary.headline) {
    const lines: string[] = [`${summary.noop ? '' : '✓ '}${summary.headline}`]

    if (summary.token_line) {
      lines.push(`  ${summary.token_line}`)
    }

    if (summary.note) {
      lines.push(`  ${summary.note}`)
    }

    return lines.join('\n')
  }

  // session.steer — { status: 'queued' | 'rejected', text }
  if (r.status === 'queued' || r.status === 'rejected') {
    const text = typeof r.text === 'string' ? r.text : ''

    if (r.status === 'queued') {
      return text ? `Steered · "${text}" queued for next tool call` : 'Steered next tool call'
    }

    return 'Steer rejected — agent declined input'
  }

  // process.stop — { killed: number }
  if ('killed' in r && typeof r.killed === 'number') {
    return r.killed > 0
      ? `Stopped ${r.killed} background process${r.killed === 1 ? '' : 'es'}.`
      : 'No background processes to stop.'
  }

  // session.save — { file }
  if (typeof r.file === 'string' && r.file) {
    return `Saved transcript to ${r.file}`
  }

  // session.status — { output }
  if (typeof r.output === 'string' && r.output) {
    return r.output
  }

  // session.usage — { calls, input, output, total, credits_lines? }
  if ('total' in r || 'input' in r || 'output' in r || 'calls' in r) {
    const calls = Number(r.calls ?? 0)
    const input = Number(r.input ?? 0)
    const output = Number(r.output ?? 0)
    const total = Number(r.total ?? 0)

    const lines: string[] = [
      `Usage: ${calls.toLocaleString()} calls · ${input.toLocaleString()} in / ${output.toLocaleString()} out · ${total.toLocaleString()} total`
    ]

    if (Array.isArray(r.credits_lines)) {
      for (const credit of r.credits_lines) {
        if (typeof credit === 'string' && credit.trim()) {
          lines.push(credit.trim())
        }
      }
    }

    return lines.join('\n')
  }

  // agents.list — { processes: [{ session_id, command, status, uptime }] }
  if (Array.isArray(r.processes)) {
    if (r.processes.length === 0) {
      return 'No background tasks running.'
    }

    return r.processes
      .map(p => {
        if (!p || typeof p !== 'object') {
          return ''
        }

        const proc = p as Record<string, unknown>
        const status = typeof proc.status === 'string' ? proc.status : 'unknown'
        const command = typeof proc.command === 'string' ? proc.command : ''
        const sessionId = typeof proc.session_id === 'string' ? proc.session_id : ''
        const uptime = proc.uptime

        const meta: string[] = []

        if (typeof uptime === 'number' && uptime >= 0) {
          meta.push(`${uptime}s`)
        }

        if (sessionId) {
          meta.push(sessionId)
        }

        const tail = meta.length ? ` (${meta.join(' · ')})` : ''

        return `• [${status}] ${command}${tail}`
      })
      .filter(Boolean)
      .join('\n')
  }

  // Generic fallback — keeps us honest if the gateway adds new fields we
  // haven't shaped yet.
  return `/${name}: ${JSON.stringify(r)}`
}

export function appendText(message: AppendMessage): string {
  return message.content
    .map(part => ('text' in part ? part.text : ''))
    .join('')
    .trim()
}

/** The one visible-user filter every user-ordinal computation must share —
 *  truncate ordinals, ordinal→index resolution, and survivor-rowId rebinding
 *  all rely on counting exactly the same turns. */
export function isVisibleUserMessage(message: ChatMessage): boolean {
  return message.role === 'user' && !message.hidden
}

/**
 * A user turn whose submit failed: the optimistic bubble stayed in the
 * transcript (followed by an assistant error), but the turn never reached the
 * gateway, so it does not exist in backend history. Every backend-facing
 * user-turn count must skip these or every later ordinal overshoots the
 * gateway's index and the rewind mis-aims / gets refused (#41275, #86573).
 */
export function isFailedUserTurn(messages: readonly ChatMessage[], index: number): boolean {
  const next = messages[index + 1]

  return next?.role === 'assistant' && Boolean(next.error)
}

/**
 * Indices of the user turns the backend also knows about — visible AND not
 * failed. This is the ONE ordinal space shared with the gateway: truncate
 * ordinals, ordinal→index resolution, survivor-rowId rebinding, and durable
 * row-id resolution all iterate exactly this list.
 */
export function visibleUserMessageIndices(messages: readonly ChatMessage[]): number[] {
  const indices: number[] = []

  for (let index = 0; index < messages.length; index += 1) {
    if (isVisibleUserMessage(messages[index]) && !isFailedUserTurn(messages, index)) {
      indices.push(index)
    }
  }

  return indices
}

export function visibleUserOrdinal(messages: readonly ChatMessage[], end: number): number {
  return visibleUserMessageIndices(messages).filter(index => index < end).length
}

export function visibleUserIndexAtOrdinal(messages: readonly ChatMessage[], targetOrdinal: number): number {
  const indices = visibleUserMessageIndices(messages)

  return targetOrdinal >= 0 && targetOrdinal < indices.length ? indices[targetOrdinal] : -1
}

export interface SubmitTextOptions {
  attachments?: ComposerAttachment[]
  /** The composer scope key that was actually loaded when this text was
   *  submitted (see use-composer-draft's activeQueueSessionKeyRef). Compared
   *  against the resolved submit target in sessionContextDrift — a mismatch
   *  means the composer and the session-side refs disagreed about which
   *  session this send belongs to (#59305). Omit for non-composer submits
   *  (queue drain, steer, external submit requests): the check is a no-op
   *  without it. */
  composerScope?: string | null
  /** What the transcript shows for this send, when it differs from the text
   *  the agent receives. A `/skill` invocation expands into the whole skill
   *  body — model-facing scaffolding the UI must never render — so the slash
   *  dispatcher passes the invocation (`/work fix the leak`) here. */
  displayText?: string
  /** `hidden` types the persisted user row (display_kind) so no bubble
   *  renders anywhere — the off-screen path for widget intents. The agent
   *  still receives the text as a normal user turn. */
  displayKind?: 'hidden'
  fromQueue?: boolean
  /** Runtime session id to submit into. Queue drains pass this so a
   *  backgrounded/source session cannot be replaced by the current foreground
   *  session between enqueue and drain. */
  sessionId?: string | null
  /** Stable stored session id for optimistic/cache updates and stale-runtime
   *  recovery. Distinct from the runtime session id minted by the gateway. */
  storedSessionId?: string | null
}
