/**
 * Transfer this agent into a Hermes Cloud agent: `hermes backup` on the source, read the archive
 * back in chunks, upload it to the cloud agent, which restores it with `hermes import --keep-host`
 * (it keeps its own Nous Portal sign-in, model and provider, Hermes Cloud tools and scheduler).
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
}

export interface ImportStarted {
  archive?: string
  name: string
  ok?: boolean
  undo_archive?: string
  uploaded_bytes?: number
}

export type TransferPhase = 'backup' | 'read' | 'restore' | 'upload'

export interface TransferProgress {
  phase: TransferPhase
  /** 0..1 while reading or uploading; undefined while a step has no measure. */
  fraction?: number
  line?: string
}

export interface TransferResult {
  /** The cloud agent's own backup, taken before the restore: what Undo puts back. */
  undoArchive: string
}

/** A Hermes Cloud agent hosts one profile. */
export const cloudEndpoint = (connectionId: string): TransferEndpoint => ({ connectionId, profile: 'default' })

export const CHUNK_BYTES = 8 * 1024 * 1024
const RETRIES = 5
const POLL_MS = 1_000
/** A restoring cloud agent pauses its gateway briefly; status reads may fail meanwhile. */
const POLL_FAILURES = 10

export interface TransferTiming {
  sleep: (ms: number) => Promise<void>
}

const realTiming: TransferTiming = { sleep: ms => new Promise(resolve => setTimeout(resolve, ms)) }

async function retry<T>(work: () => Promise<T>, timing: TransferTiming, signal?: AbortSignal): Promise<T> {
  let lastError: unknown

  for (let attempt = 0; attempt < RETRIES; attempt += 1) {
    signal?.throwIfAborted()

    try {
      return await work()
    } catch (error) {
      lastError = error
      await timing.sleep(Math.min(8000, 500 * 2 ** attempt))
    }
  }

  throw lastError
}

/** Poll an action until it stops; a non-zero exit is an error carrying its last lines. */
export async function waitForAction(
  request: TransferRequestFn,
  owner: TransferEndpoint,
  name: string,
  onLine: (line: string) => void,
  { timing = realTiming, tolerateFailures = 0 }: { timing?: TransferTiming; tolerateFailures?: number } = {}
): Promise<void> {
  let failures = 0

  for (;;) {
    await timing.sleep(POLL_MS)

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

    const last = status.lines.at(-1)

    if (last) {
      onLine(last)
    }

    if (!status.running) {
      if (status.exit_code !== 0) {
        throw new Error(status.lines.slice(-3).join('\n') || `${name} failed.`)
      }

      return
    }
  }
}

/** `hermes backup` on `owner`; resolves to the archive path on that backend. */
export async function runBackup(
  request: TransferRequestFn,
  owner: TransferEndpoint,
  onLine: (line: string) => void = () => undefined,
  timing: TransferTiming = realTiming
): Promise<string> {
  const started = await request<{ archive?: string; name: string }>(owner, {
    body: {},
    method: 'POST',
    path: '/api/ops/backup'
  })

  if (!started.archive) {
    throw new Error('The backup did not say where it was written.')
  }

  await waitForAction(request, owner, started.name, onLine, { timing })

  return started.archive
}

const fromBase64 = (data: string) => Uint8Array.from(atob(data), c => c.charCodeAt(0))

/** Read an archive off `owner` in 8 MiB chunks, each retried with backoff. */
export async function readArchive(
  request: TransferRequestFn,
  owner: TransferEndpoint,
  archive: string,
  onFraction: (fraction: number) => void,
  { signal, timing = realTiming }: { signal?: AbortSignal; timing?: TransferTiming } = {}
): Promise<Uint8Array> {
  let bytes: Uint8Array | null = null
  let offset = 0

  for (;;) {
    signal?.throwIfAborted()
    const at = offset

    const part = await retry(
      () =>
        request<{ data: string; done: boolean; offset: number; size: number }>(owner, {
          path: `/api/ops/backup/chunk?archive=${encodeURIComponent(archive)}&offset=${at}&length=${CHUNK_BYTES}`,
          timeoutMs: 300_000
        }),
      timing,
      signal
    )

    bytes ??= new Uint8Array(part.size)
    const chunk = fromBase64(part.data)
    bytes.set(chunk, at)
    offset = at + chunk.length
    onFraction(part.size ? offset / part.size : 1)

    if (part.done || chunk.length === 0) {
      break
    }
  }

  if (offset !== bytes.length) {
    throw new Error('The backup changed while it was being read. Try again.')
  }

  return bytes
}

/** Upload the archive to the cloud agent and wait for its restore to finish. */
export async function uploadAndRestore(
  request: TransferRequestFn,
  target: TransferEndpoint,
  bytes: Uint8Array,
  onLine: (line: string) => void,
  timing: TransferTiming = realTiming,
  onUploaded: () => void = () => undefined
): Promise<TransferResult> {
  const started = await request<ImportStarted>(target, {
    method: 'POST',
    path: '/api/ops/import-upload?keep_host=true',
    timeoutMs: 1_800_000,
    upload: {
      bytes: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
      contentType: 'application/zip',
      filename: 'hermes-transfer.zip'
    }
  })

  onUploaded()
  await waitForAction(request, target, started.name || 'import', onLine, { timing, tolerateFailures: POLL_FAILURES })

  return { undoArchive: started.undo_archive ?? '' }
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

/** The whole transfer: back up the source, read it, upload, restore on the target. */
export async function transferAgent(
  request: TransferRequestFn,
  source: TransferEndpoint,
  target: TransferEndpoint,
  onProgress: (progress: TransferProgress) => void,
  { signal, timing = realTiming }: { signal?: AbortSignal; timing?: TransferTiming } = {}
): Promise<TransferResult> {
  onProgress({ phase: 'backup' })
  const archive = await runBackup(request, source, line => onProgress({ line, phase: 'backup' }), timing)
  signal?.throwIfAborted()

  const bytes = await readArchive(
    request,
    source,
    archive,
    fraction => onProgress({ fraction: fraction / 2, phase: 'upload' }),
    { signal, timing }
  )

  signal?.throwIfAborted()
  onProgress({ fraction: 0.5, phase: 'upload' })

  // The upload is one request; its half of the bar fills when the cloud agent answers.
  return uploadAndRestore(
    request,
    target,
    bytes,
    line => onProgress({ line, phase: 'restore' }),
    timing,
    () => onProgress({ phase: 'restore' })
  )
}

/** An error's user-facing text: the bridge rejects with `"<status>: <json detail>"`. */
export function transferErrorMessage(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error)
  const match = /"(?:message|detail)"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(text)

  return match ? JSON.parse(`"${match[1]}"`) : text.replace(/^\d{3}:\s*/, '')
}
