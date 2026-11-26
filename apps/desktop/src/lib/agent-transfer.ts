/**
 * Transfer this agent into a Hermes Cloud agent: `hermes backup` on the source, stream the archive
 * chunk by chunk to the cloud agent, which restores it with `hermes import --keep-host` (it keeps
 * its own Nous Portal sign-in, model and provider, Hermes Cloud tools and scheduler).
 * Pure async over an injected `request`, so it is tested without Electron. Never opens a dialog.
 */

import type { HermesApiRequest } from '@/global'

/** Which gateway and profile a call acts on. Captured once per flow, so a profile switch
 *  mid-transfer cannot split it across two profiles. */
export interface TransferEndpoint {
  /** A registry connection id; null is the window's own backend. */
  connectionId: null | string
  profile: string
}

export type TransferRequestFn = <T>(owner: TransferEndpoint, request: HermesApiRequest) => Promise<T>

export interface ActionStatus {
  exit_code: null | number
  lines: string[]
  running: boolean
  /** A keep-host import's undo point once it has written it; still present if the import fails. */
  undo_archive?: null | string
  /** The messaging platforms a transfer's restore switched off (`hermes import --messaging-off`). */
  messaging_off?: string[]
  /** Set when the restore finished but the agent's gateway did not start again. */
  gateway_down?: string
}

export type TransferPhase = 'backup' | 'restore' | 'upload'

export interface TransferProgress {
  phase: TransferPhase
  /** 0..1 while copying; undefined while a step has no measure. */
  fraction?: number
  line?: string
}

export interface TransferResult {
  /** The cloud agent's own backup, taken before the restore: what Undo puts back. */
  undoArchive: string
  /** The messaging bots that came over switched off on the cloud agent (their names). */
  messagingOff: string[]
  /** The files arrived but the cloud agent's gateway did not start again: it needs a restart. */
  gatewayDown: boolean
}

/** The restore failed after it was requested. With `undoArchive` set the cloud agent had already
 *  taken its undo point, so it may be half-restored and Undo must be offered. */
export class TransferRestoreError extends Error {
  readonly undoArchive: null | string

  constructor(message: string, undoArchive: null | string) {
    super(message)
    this.name = 'TransferRestoreError'
    this.undoArchive = undoArchive
  }
}

/** A Hermes Cloud agent hosts one profile. */
export const cloudEndpoint = (connectionId: string): TransferEndpoint => ({ connectionId, profile: 'default' })

export const CHUNK_BYTES = 8 * 1024 * 1024
const RETRIES = 5
/** Times one copy may re-sync after the target lost part of the upload. */
const RESYNCS = 5
const POLL_MS = 1_000
/** A restoring cloud agent pauses its gateway briefly; status reads may fail meanwhile. */
const POLL_FAILURES = 10

const BACKUP_CHANGED = 'The backup changed while it was being read. Try again.'

export interface TransferTiming {
  sleep: (ms: number) => Promise<void>
}

const realTiming: TransferTiming = { sleep: ms => new Promise(resolve => setTimeout(resolve, ms)) }

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error))

const httpStatus = (error: unknown) => Number(/^(\d{3}):/.exec(errorText(error))?.[1] ?? 0)

/** A 4xx other than timeout/rate limit answers the same on every attempt. */
function isFinal(error: unknown): boolean {
  const status = httpStatus(error)

  return status >= 400 && status < 500 && status !== 408 && status !== 429
}

async function retry<T>(work: () => Promise<T>, timing: TransferTiming, signal?: AbortSignal): Promise<T> {
  let lastError: unknown

  for (let attempt = 0; attempt < RETRIES; attempt += 1) {
    signal?.throwIfAborted()

    try {
      return await work()
    } catch (error) {
      if (isFinal(error)) {
        throw error
      }

      lastError = error
      await timing.sleep(Math.min(8000, 500 * 2 ** attempt))
    }
  }

  throw lastError
}

export interface WaitOptions {
  /** Called once, by the first status poll that carries an undo point. */
  onUndoArchive?: (archive: string) => void
  signal?: AbortSignal
  timing?: TransferTiming
  tolerateFailures?: number
}

/** Poll an action until it stops and return its last status; a non-zero exit is an error carrying
 *  its last lines. */
export async function waitForAction(
  request: TransferRequestFn,
  owner: TransferEndpoint,
  name: string,
  onLine: (line: string) => void,
  { onUndoArchive, signal, timing = realTiming, tolerateFailures = 0 }: WaitOptions = {}
): Promise<ActionStatus> {
  let failures = 0
  let undoReported = false

  for (;;) {
    await timing.sleep(POLL_MS)
    signal?.throwIfAborted()

    let status: ActionStatus

    try {
      status = await request<ActionStatus>(owner, {
        path: `/api/actions/${encodeURIComponent(name)}/status?lines=40`
      })
      failures = 0
    } catch (error) {
      failures += 1

      if (failures > tolerateFailures) {
        throw error
      }

      continue
    }

    if (status.undo_archive && !undoReported) {
      undoReported = true
      onUndoArchive?.(status.undo_archive)
    }

    const last = status.lines.at(-1)

    if (last) {
      onLine(last)
    }

    if (!status.running) {
      if (status.exit_code !== 0) {
        throw new Error(failureText(status.lines) || `${name} failed.`)
      }

      return status
    }
  }
}

/** What went wrong, from an action's log: the import's own warning list when it printed one (the
 *  files it could not restore), else the last lines. */
export function failureText(lines: string[]): string {
  const start = lines.findIndex(line => /^\s*Warnings \(\d+ files? skipped\):/.test(line))

  if (start >= 0) {
    const listed = lines.slice(start + 1).filter(line => /^\s+\S/.test(line) && !/^\s+Target:/.test(line))

    return [lines[start].trim(), ...listed.slice(0, 5).map(line => line.trim())].join('\n')
  }

  return lines.slice(-3).join('\n')
}

/** `hermes backup` on `owner`; resolves to the archive path on that backend. */
export async function runBackup(
  request: TransferRequestFn,
  owner: TransferEndpoint,
  onLine: (line: string) => void = () => undefined,
  { signal, timing = realTiming }: { signal?: AbortSignal; timing?: TransferTiming } = {}
): Promise<string> {
  const started = await request<{ archive?: string; name: string }>(owner, {
    body: {},
    method: 'POST',
    path: '/api/ops/backup'
  })

  if (!started.archive) {
    throw new Error('The backup did not say where it was written.')
  }

  await waitForAction(request, owner, started.name, onLine, { signal, timing })

  return started.archive
}

interface SourceChunk {
  data: string
  done: boolean
  offset: number
  size: number
}

interface TargetChunk {
  received: number
  upload_id: string
}

/** Decoded length of a base64 string, without decoding it. */
export function base64Bytes(data: string): number {
  if (!data) {
    return 0
  }

  const padding = data.endsWith('==') ? 2 : data.endsWith('=') ? 1 : 0

  return (data.length / 4) * 3 - padding
}

/** For a 409 offset_gap: the target's `received`, or -1 when the answer names none. Null for any
 *  other error. */
function offsetGap(error: unknown): null | number {
  const text = errorText(error)

  if (httpStatus(error) !== 409 || !text.includes('offset_gap')) {
    return null
  }

  const received = /"received"\s*:\s*(\d+)/.exec(text)

  return received ? Number(received[1]) : -1
}

/**
 * Stream `archive` from `source` into a transfer upload on `target`, holding one chunk at a time:
 * each base64 chunk the source answers is posted to the target as is.
 */
export async function copyArchive(
  request: TransferRequestFn,
  source: TransferEndpoint,
  target: TransferEndpoint,
  archive: string,
  onFraction: (fraction: number) => void,
  { signal, timing = realTiming }: { signal?: AbortSignal; timing?: TransferTiming } = {}
): Promise<{ size: number; uploadId: string }> {
  let uploadId: null | string = null
  let offset = 0
  let size: null | number = null
  let resyncs = 0

  for (;;) {
    const at = offset

    const part = await retry(
      () =>
        request<SourceChunk>(source, {
          path: `/api/ops/backup/chunk?archive=${encodeURIComponent(archive)}&offset=${at}&length=${CHUNK_BYTES}`,
          timeoutMs: 300_000
        }),
      timing,
      signal
    )

    if (size !== null && part.size !== size) {
      throw new Error(BACKUP_CHANGED)
    }

    size = part.size
    const length = base64Bytes(part.data)

    if (length > 0) {
      const id: null | string = uploadId

      try {
        const sent: TargetChunk = await retry(
          (): Promise<TargetChunk> =>
            request<TargetChunk>(target, {
              body: { data: part.data, offset: at, upload_id: id },
              method: 'POST',
              path: '/api/ops/transfer/chunk',
              timeoutMs: 300_000
            }),
          timing,
          signal
        )

        uploadId = sent.upload_id
      } catch (error) {
        const received = offsetGap(error)

        if (received === null || resyncs >= RESYNCS) {
          throw error
        }

        resyncs += 1
        // Resume from what the target holds; without that figure, start a fresh upload.
        const resume: boolean = received >= 0 && received <= at && uploadId !== null
        offset = resume ? received : 0
        uploadId = resume ? uploadId : null

        continue
      }
    }

    offset = at + length
    onFraction(size ? Math.min(1, offset / size) : 1)

    if (part.done || length === 0) {
      break
    }
  }

  if (offset !== size || !uploadId) {
    throw new Error(BACKUP_CHANGED)
  }

  return { size, uploadId }
}

/** Put the cloud agent back the way it was, from the backup it took before the restore. */
export async function undoTransfer(
  request: TransferRequestFn,
  target: TransferEndpoint,
  undoArchive: string,
  onLine: (line: string) => void = () => undefined,
  timing: TransferTiming = realTiming
): Promise<void> {
  if (!undoArchive) {
    throw new Error('The cloud agent did not keep a backup to undo with.')
  }

  const started = await request<{ name: string }>(target, {
    body: { archive: undoArchive, force: true, keep_host: true },
    method: 'POST',
    path: '/api/ops/import'
  })

  await waitForAction(request, target, started.name || 'import', onLine, { timing, tolerateFailures: POLL_FAILURES })
}

export interface TransferOptions {
  /** The cloud agent's undo point, the moment its restore reports it (also when it later fails). */
  onUndoArchive?: (archive: string) => void
  /** Honoured until the restore is requested: a restore cannot be canceled. */
  signal?: AbortSignal
  timing?: TransferTiming
}

/** The whole transfer: check the target can receive one, back up the source, stream it over,
 *  restore on the target. */
export async function transferAgent(
  request: TransferRequestFn,
  source: TransferEndpoint,
  target: TransferEndpoint,
  onProgress: (progress: TransferProgress) => void,
  { onUndoArchive, signal, timing = realTiming }: TransferOptions = {}
): Promise<TransferResult> {
  signal?.throwIfAborted()
  const status = await request<null | { import_keep_host?: number }>(target, { path: '/api/status' })

  if (!status?.import_keep_host) {
    throw new Error('This cloud agent needs a Hermes update before it can receive a transfer.')
  }

  onProgress({ phase: 'backup' })
  const archive = await runBackup(request, source, line => onProgress({ line, phase: 'backup' }), { signal, timing })

  onProgress({ fraction: 0, phase: 'upload' })

  const { size, uploadId } = await copyArchive(
    request,
    source,
    target,
    archive,
    fraction => onProgress({ fraction, phase: 'upload' }),
    { signal, timing }
  )

  signal?.throwIfAborted()
  onProgress({ phase: 'restore' })

  const started = await request<{ name?: string }>(target, {
    body: { size, upload_id: uploadId },
    method: 'POST',
    path: '/api/ops/transfer/restore',
    timeoutMs: 300_000
  })

  let undoArchive: null | string = null
  let finished: ActionStatus

  try {
    finished = await waitForAction(
      request,
      target,
      started.name || 'import',
      line => onProgress({ line, phase: 'restore' }),
      {
        onUndoArchive: archivePath => {
          undoArchive = archivePath
          onUndoArchive?.(archivePath)
        },
        timing,
        tolerateFailures: POLL_FAILURES
      }
    )
  } catch (error) {
    throw new TransferRestoreError(errorText(error), undoArchive)
  }

  if (!undoArchive) {
    throw new TransferRestoreError('The cloud agent did not report its undo point.', null)
  }

  return { gatewayDown: Boolean(finished.gateway_down), messagingOff: finished.messaging_off ?? [], undoArchive }
}

export interface UndoPoint {
  archive: string
  target: TransferEndpoint
}

/**
 * The undo point Undo restores, per cloud agent. The FIRST one recorded wins: a retry after a
 * failed restore makes the cloud agent take a new undo point of its half-restored self, which must
 * never replace the one taken before anything changed.
 */
export function createUndoLedger() {
  const points = new Map<string, UndoPoint>()

  return {
    clear: (agentId: string) => void points.delete(agentId),
    get: (agentId: string): undefined | UndoPoint => points.get(agentId),
    record: (agentId: string, point: UndoPoint) => {
      if (!points.has(agentId)) {
        points.set(agentId, point)
      }
    }
  }
}

export type UndoLedger = ReturnType<typeof createUndoLedger>

/** An error's user-facing text: the bridge rejects with `"<status>: <json detail>"`. */
export function transferErrorMessage(error: unknown): string {
  const text = errorText(error)
  const match = /"(?:message|detail)"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(text)

  return match ? JSON.parse(`"${match[1]}"`) : text.replace(/^\d{3}:\s*/, '')
}
