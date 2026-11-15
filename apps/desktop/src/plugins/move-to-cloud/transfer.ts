/**
 * The transfer engine: build a bundle on this computer, stream it to a Hermes Cloud agent in
 * checksummed chunks, finish, apply. Pure async over an injected `request` so it is tested
 * without Electron, and every network step is resumable:
 *
 * - The upload id is keyed by the bundle's sha256 on the receiver, so starting again after a
 *   crash, a closed lid or a dropped Wi-Fi picks up at the receiver's `received` offset.
 * - A chunk that fails is retried with backoff; a 409 `offset_gap` re-reads the receiver's
 *   offset instead of guessing.
 * - A 422 `digest_mismatch` (the bytes arrived damaged) restarts the upload once from zero.
 */

const CHUNK = 2 * 1024 * 1024
const READ = 4 * 1024 * 1024
const RETRIES = 5

/** Which gateway and profile a call acts on. Captured once per flow, so a profile switch
 *  mid-transfer cannot move the tail of it onto another profile's data. */
export interface MoveEndpoint {
  /** A registry connection id; null is the window's own backend. */
  connectionId: null | string
  profile: string
}

/** A Hermes Cloud agent hosts one profile. */
export const cloudEndpoint = (connectionId: string): MoveEndpoint => ({ connectionId, profile: 'default' })

export type MoveRequestFn = <T>(req: {
  connectionId: null | string
  profile: string
  path: string
  method?: 'DELETE' | 'GET' | 'POST'
  body?: unknown
  timeoutMs?: number
}) => Promise<T>

export interface BundleInventory {
  name: null | string
  bytes: number
  sha256: string
  skills: string[]
  skills_skipped: { skill: string; bytes: number; reason: string }[]
  memory_entries: Record<string, number>
  cron_jobs: { id: string; name: string; needs_this_machine: string[] }[]
  sessions: number
  sessions_skipped: number
  mcp_servers: { name: string; portable: boolean; reason: null | string }[]
  keys_to_reenter: { key: string; category: string; label: string }[]
  has_soul: boolean
  redacted: Record<string, number>
}

export interface MovePreview {
  skills: { add: string[]; keep_existing: string[] }
  cron: { add: { id: string; name: string }[]; keep_existing: { id: string }[] }
  sessions: number
  soul: { action: 'keep_existing' | 'none' | 'replace'; incoming?: boolean }
  memory: Record<string, number>
  config_keys?: string[]
  /** What the receiving agent has now: what a replace would take away. */
  target?: { skills: number; cron_jobs: number; conversations: number; soul_customized: boolean }
}

/** The parts a move carries, in the order the receiver applies them (`move_receive.STEPS`). */
export const PARTS = ['memory', 'soul', 'skills', 'cron', 'sessions', 'config'] as const
export type Part = (typeof PARTS)[number]

export interface MoveState {
  id: string
  status: 'applied' | 'applying' | 'partial' | 'ready' | 'undone' | 'uploading'
  bytes: number
  received: number
  preview?: MovePreview
  steps?: Record<string, MoveStep>
  result?: { failed_steps?: string[]; undo_problems?: string[] }
  /** Non-empty after an undo that could not put everything back (retryable). */
  undo_problems?: string[]
}

export interface MoveStep {
  status: 'done' | 'failed'
  error?: string
  /** Items in a finished step that did not make it (e.g. one unreadable conversation). */
  failed?: number
  errors?: string[]
  [k: string]: unknown
}

export interface StepProblem {
  step: string
  failed: number
  errors: string[]
}

export interface MoveOutcome {
  complete: boolean
  steps: StepProblem[]
  undoProblems: string[]
}

/** What an apply or undo answer means for the user: done, or what is left to retry. */
export function moveOutcome(state: MoveState): MoveOutcome {
  const steps = Object.entries(state.steps ?? {}).flatMap(([step, s]): StepProblem[] =>
    s.status === 'failed'
      ? [{ step, failed: s.failed ?? 0, errors: s.error ? [s.error] : [] }]
      : s.failed
        ? [{ step, failed: s.failed, errors: s.errors ?? [] }]
        : []
  )

  if (state.status === 'undone') {
    const undoProblems = state.undo_problems ?? state.result?.undo_problems ?? []

    return { complete: undoProblems.length === 0, steps: [], undoProblems }
  }

  return { complete: state.status === 'applied' && steps.length === 0, steps, undoProblems: [] }
}

export interface Progress {
  phase: 'apply' | 'build' | 'finish' | 'read' | 'upload'
  done: number
  total: number
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

/** The refusal code a route answered with (`{detail: {code}}` inside the bridge's error text). */
export function refusalCode(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error)
  const match = /"code"\s*:\s*"([a-z_]+)"/.exec(text)

  return match?.[1] ?? ''
}

export function refusalMessage(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error)
  const match = /"message"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(text)

  return match ? JSON.parse(`"${match[1]}"`) : text.replace(/^\d{3}:\s*/, '')
}

async function retry<T>(work: () => Promise<T>, isFinal: (error: unknown) => boolean): Promise<T> {
  let lastError: unknown

  for (let attempt = 0; attempt < RETRIES; attempt += 1) {
    try {
      return await work()
    } catch (error) {
      lastError = error

      if (isFinal(error)) {
        throw error
      }

      await sleep(Math.min(8000, 500 * 2 ** attempt))
    }
  }

  throw lastError
}

// Refusals that a retry cannot change: answer them now.
const FINAL = new Set(['too_large', 'bad_zip', 'bad_manifest', 'not_found', 'applied', 'not_ready'])
const final = (error: unknown) => FINAL.has(refusalCode(error))

export async function inventory(
  request: MoveRequestFn,
  source: MoveEndpoint,
  includeHistory: boolean
): Promise<BundleInventory> {
  return request<BundleInventory>({
    ...source,
    path: '/api/move/bundle',
    method: 'POST',
    body: { include_history: includeHistory, dry_run: true },
    timeoutMs: 300_000
  })
}

export async function buildAndRead(
  request: MoveRequestFn,
  source: MoveEndpoint,
  includeHistory: boolean,
  onProgress: (p: Progress) => void,
  signal?: AbortSignal
): Promise<{ inventory: BundleInventory; bytes: Uint8Array }> {
  onProgress({ phase: 'build', done: 0, total: 1 })

  const built = await request<BundleInventory>({
    ...source,
    path: '/api/move/bundle',
    method: 'POST',
    body: { include_history: includeHistory },
    timeoutMs: 600_000
  })

  if (!built.name) {
    throw new Error('The bundle was not written.')
  }

  const bytes = new Uint8Array(built.bytes)
  let offset = 0

  while (offset < built.bytes) {
    signal?.throwIfAborted()

    const part = await retry(
      () =>
        request<{ data: string; done: boolean }>({
          ...source,
          path: `/api/move/bundle/${built.name}?offset=${offset}&length=${READ}`
        }),
      final
    )

    const chunk = Uint8Array.from(atob(part.data), c => c.charCodeAt(0))
    bytes.set(chunk, offset)
    offset += chunk.length
    onProgress({ phase: 'read', done: offset, total: built.bytes })

    if (part.done || chunk.length === 0) {
      break
    }
  }

  if (offset !== built.bytes) {
    throw new Error('The bundle changed while it was being read. Try again.')
  }

  return { inventory: built, bytes }
}

function toBase64(bytes: Uint8Array): string {
  let binary = ''

  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }

  return btoa(binary)
}

export async function upload(
  request: MoveRequestFn,
  target: MoveEndpoint,
  bytes: Uint8Array,
  sha256: string,
  onProgress: (p: Progress) => void,
  signal?: AbortSignal
): Promise<MoveState> {
  for (let restart = 0; restart < 2; restart += 1) {
    let state = await retry(
      () =>
        request<MoveState>({
          ...target,
          path: '/api/move/uploads',
          method: 'POST',
          body: { size: bytes.length, sha256 }
        }),
      final
    )

    if (state.status === 'ready' || state.status === 'applied' || state.status === 'partial') {
      return state
    }

    let offset = state.received

    while (offset < bytes.length) {
      signal?.throwIfAborted()
      onProgress({ phase: 'upload', done: offset, total: bytes.length })
      const at = offset

      state = await retry(async () => {
        try {
          return await request<MoveState>({
            ...target,
            path: `/api/move/uploads/${state.id}/chunk`,
            method: 'POST',
            body: { offset: at, data: toBase64(bytes.subarray(at, at + CHUNK)) }
          })
        } catch (error) {
          if (refusalCode(error) === 'offset_gap' || refusalCode(error) === 'not_uploading') {
            return request<MoveState>({ ...target, path: `/api/move/uploads/${state.id}` })
          }

          throw error
        }
      }, final)

      if (state.status !== 'uploading') {
        break
      }

      offset = state.received
    }

    onProgress({ phase: 'finish', done: bytes.length, total: bytes.length })

    try {
      return await retry(
        () => request<MoveState>({ ...target, path: `/api/move/uploads/${state.id}/finish`, method: 'POST' }),
        error => final(error) || refusalCode(error) === 'digest_mismatch'
      )
    } catch (error) {
      if (refusalCode(error) !== 'digest_mismatch' || restart > 0) {
        throw error
      }
    }
  }

  throw new Error('The bundle kept arriving damaged. Check your connection and try again.')
}

/** Apply a received bundle. `replace` names the parts where this computer's copy wins over the
 *  cloud agent's; empty merges everything (the cloud agent's own copies win). The receiver refuses
 *  a replace without `confirm_replace`, which only the confirmation step sets. */
export async function apply(
  request: MoveRequestFn,
  target: MoveEndpoint,
  moveId: string,
  replace: readonly Part[] = []
): Promise<MoveState> {
  const body = replace.length ? { replace, confirm_replace: true } : {}

  return retry(
    () =>
      request<MoveState>({
        ...target,
        path: `/api/move/uploads/${moveId}/apply`,
        method: 'POST',
        body,
        timeoutMs: 900_000
      }),
    final
  )
}

export async function undo(request: MoveRequestFn, target: MoveEndpoint, moveId: string): Promise<MoveState> {
  return request<MoveState>({ ...target, path: `/api/move/uploads/${moveId}/undo`, method: 'POST', body: {} })
}

export async function cleanupLocal(request: MoveRequestFn, source: MoveEndpoint, name: null | string): Promise<void> {
  if (name) {
    await request({ ...source, path: `/api/move/bundle/${name}`, method: 'DELETE' }).catch(() => undefined)
  }
}
