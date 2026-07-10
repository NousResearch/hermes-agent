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

// Hard guard: at most one prompt.submit in flight per session. Every submit
// path — user Enter, queue drain, busy-retry, slash fallthrough — funnels
// through submitPromptText. Without this, a stalled turn (e.g. a context-bloated
// session whose first call hangs) let the SAME prompt launch several real turns
// at once (the "message stacked 5×" bug). Keyed by stored/active session id.
export const _submitInFlight = new Set<string>()

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
export async function readImageForRemoteAttach(
  filePath: string
): Promise<{ contentBase64: string; filename: string } | null> {
  const dataUrl = await window.hermesDesktop?.readFileDataUrl(filePath)
  const contentBase64 = dataUrl ? base64FromDataUrl(dataUrl) : ''

  return contentBase64 ? { contentBase64, filename: imageFilenameFromPath(filePath) } : null
}

// Read a non-image file as a data URL for upload via file.attach. Returns null
// when the desktop bridge can't read the file (e.g. it was moved/deleted).
export async function readFileDataUrlForAttach(filePath: string): Promise<string | null> {
  const reader = window.hermesDesktop?.readFileDataUrl

  if (!reader) {
    return null
  }

  const dataUrl = await reader(filePath)

  return dataUrl || null
}

// The readFileDataUrl IPC base64-loads the whole file into memory and is
// hard-capped (DATA_URL_READ_MAX_BYTES, 16 MB) in electron/hardening.ts, which
// rejects with a raw "file is too large (N bytes; limit M bytes)" string. In
// remote mode every attachment's bytes go through that read, so a big file
// surfaces that internal message verbatim in the failure toast. Translate it
// into a friendly "too large to upload to the remote gateway" line, parsing the
// limit out of the message so it tracks the real cap. Non-cap errors pass
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

  // process.stop — { killed: boolean }
  if ('killed' in r && typeof r.killed === 'boolean') {
    return r.killed ? 'Stopped all background processes.' : 'No background processes to stop.'
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
        if (!p || typeof p !== 'object') return ''
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
  // haven't shaped yet. Empty `name` keeps the line short in the transcript.
  return `/${name}: ${JSON.stringify(r)}`
}

export function appendText(message: AppendMessage): string {
  return message.content
    .map(part => ('text' in part ? part.text : ''))
    .join('')
    .trim()
}

export function visibleUserOrdinal(messages: readonly ChatMessage[], end: number): number {
  return messages.slice(0, end).filter(m => m.role === 'user' && !m.hidden).length
}

export function visibleUserIndexAtOrdinal(messages: readonly ChatMessage[], targetOrdinal: number): number {
  let ordinal = 0

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]

    if (message.role !== 'user' || message.hidden) {
      continue
    }

    if (ordinal === targetOrdinal) {
      return index
    }

    ordinal += 1
  }

  return -1
}

export interface SubmitTextOptions {
  attachments?: ComposerAttachment[]
  fromQueue?: boolean
}
