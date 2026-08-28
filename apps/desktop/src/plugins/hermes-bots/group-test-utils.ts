/**
 * A scripted gateway for the group-chat room-engine tests.
 *
 * A member turn is a session-scoped RPC SEQUENCE (create/resume → attach →
 * prompt.submit → poll), so nearly every room contract is only observable in
 * what those RPCs saw. This models the parts of the gateway's session
 * lifecycle those contracts depend on:
 *
 *  - the STORED id is the durable identity and every `session.resume` mints a
 *    FRESH runtime id, which is why a turn that loses its socket mid-sequence
 *    has to recover through the stored id (#93602);
 *  - a missing session answers JSON-RPC 4007 — the real "genuinely not found"
 *    code that `ensureGroupChatSession`'s fail-closed branch keys on, so a
 *    plain failure here is NOT read as "no session, mint a new one";
 *  - `host.requestProfile` takes a per-request socket lease and disposes the
 *    socket at refcount 0, the reap that made routed members go silent.
 *
 * Its predecessors sliced `plugin.js` out of the bundle with string offsets
 * and ran it under `vm`; this drives the real modules through the real
 * `host` surface instead.
 */

import type { PluginContext } from '@hermes/plugin-sdk'
import { vi } from 'vitest'

/** One message in a scripted session transcript, in the gateway's own shape. */
export interface ScriptedMessage {
  content: string
  role: string
}

export interface ScriptedSession {
  messages: ScriptedMessage[]
  profile: string
  runtime: string
  stored: string
  title: string
}

/** What the scripted member "said" this turn: one reply, or the whole tail of
 *  messages a real turn can append (a substantive answer plus a continuation
 *  nudge and its synthetic pass, #94376). */
export type TurnReply = ScriptedMessage[] | string

export interface ScriptedTurn {
  /** 1-based index across every prompt.submit the gateway has seen. */
  n: number
  profile: string
  prompt: string
  session: ScriptedSession
}

export type TurnScript = (turn: ScriptedTurn) => Promise<TurnReply> | TurnReply

/** Every prompt.submit the gateway received, with the session it landed on. */
export interface PromptCall {
  profile: string
  prompt: string
  runtime: string
  stored: string
  title: string
}

/** Every attachment staging RPC, and how many submits preceded it. */
export interface AttachCall {
  data: string
  filename: string
  method: string
  order: number
  profile: null | string
  runtime: string
}

export interface RpcCall {
  method: string
  params: Record<string, unknown>
  /** Socket refcount after this request released its own lease. */
  refcountAfter: number
}

export interface GatewayOptions {
  /** Per profile: report inflight/running on its first N `session.resume`s. */
  busyResumes?: Record<string, number>
  /** Per profile: carry `pending_approval` on its first `until` resumes. */
  approvalUntil?: Record<string, { payload: Record<string, unknown>; until: number }>
  /** Per profile: carry `pending_clarify` on its first `until` resumes. */
  clarifyUntil?: Record<string, { payload: Record<string, unknown>; until: number }>
  /** Land a competing writer's `ui_meta` under `key` during the FIRST
   *  `profiles.configure`, then reject it as a CAS conflict — the race the
   *  sync worker's pull-merge-retry exists for. */
  conflictOnce?: { key: string; value: unknown }
  /** Reject every prompt.submit with this — a fatal, non-recoverable failure. */
  failEverySubmitWith?: unknown
  /** Reject only the FIRST prompt.submit — the 4001 reap the retry recovers. */
  failFirstSubmitWith?: unknown
  /** Fired on each post-submit poll, so a test can land a stop mid-turn. */
  onResumePoll?: (polls: number) => void
  /** Report the member inflight for the first N post-submit polls. */
  pollsBusy?: number
  turn?: TurnScript
}

/** A gateway-shaped rejection: `.code` is what the engine branches on. */
function gatewayError(message: string, code: number) {
  return Object.assign(new Error(message), { code })
}

export interface ScriptedGateway {
  /** Attachment staging RPCs, in order. */
  attaches: AttachCall[]
  /** prompt.submit calls, in order. */
  calls: PromptCall[]
  /** Times the socket refcount fell to zero — each one reaps a runtime session. */
  disposals: () => number
  /** The mock `host` the `@hermes/plugin-sdk` mock should expose. */
  host: Record<string, unknown>
  /** Every RPC, for methods the typed recorders above don't cover. */
  rpc: RpcCall[]
  /** Filter `rpc` by method. */
  rpcFor: (method: string) => RpcCall[]
  /** Live socket refcount — zero between turns, never zero during one. */
  refcount: () => number
  /** Sessions by stored id, so a test can pre-seed a finished transcript. */
  sessions: Map<string, ScriptedSession>
  /** Plugin storage writes, by key. */
  storage: Map<string, unknown>
  /** `retain`/`release`/method names in call order — the lease lifetime. */
  timeline: string[]
  /** The default profile's `ui_meta`, as `profiles.list` reports it. */
  uiMeta: Record<string, unknown>
  /** CAS revisions for `uiMeta`, advanced by every applied configure. */
  uiMetaRevisions: Record<string, number>
}

export function createGroupGateway(options: GatewayOptions = {}): ScriptedGateway {
  const { turn = () => '(pass)' } = options
  const sessions = new Map<string, ScriptedSession>()
  const runtimeToStored = new Map<string, string>()
  const titleToStored = new Map<string, string>()
  const resumesByProfile = new Map<string, number>()
  const calls: PromptCall[] = []
  const attaches: AttachCall[] = []
  const rpc: RpcCall[] = []
  const timeline: string[] = []
  const storage = new Map<string, unknown>()
  const uiMeta: Record<string, unknown> = {}
  const uiMetaRevisions: Record<string, number> = {}
  let sequence = 0
  let submits = 0
  let polls = 0
  let refcount = 0
  let disposals = 0
  let conflicted = false

  const resolveSession = (profile: unknown, target: unknown) => {
    const key = String(target ?? '')

    const stored =
      runtimeToStored.get(key) || (sessions.has(key) ? key : titleToStored.get(`${String(profile)}::${key}`))

    return stored ? sessions.get(stored) || null : null
  }

  const handle = async (method: string, params: Record<string, unknown>): Promise<unknown> => {
    if (method === 'profiles.list') {
      return {
        profiles: [{ name: 'default', ui_meta: { ...uiMeta }, ui_meta_revisions: { ...uiMetaRevisions } }]
      }
    }

    if (method === 'profiles.configure') {
      if (options.conflictOnce && !conflicted) {
        conflicted = true
        const { key, value } = options.conflictOnce
        uiMeta[key] = value
        uiMetaRevisions[key] = (uiMetaRevisions[key] || 0) + 1

        return {
          applied: {
            ui_meta: false,
            ui_meta_conflicts: { [key]: { actual: uiMetaRevisions[key], expected: uiMetaRevisions[key] - 1 } },
            ui_meta_revisions: { ...uiMetaRevisions }
          }
        }
      }

      const expected = params.ui_meta_expected_revisions as Record<string, number> | undefined
      const incoming = (params.ui_meta || {}) as Record<string, unknown>

      if (expected) {
        for (const key of Object.keys(incoming)) {
          if ((uiMetaRevisions[key] || 0) !== expected[key]) {
            return {
              applied: {
                ui_meta: false,
                ui_meta_conflicts: { [key]: { actual: uiMetaRevisions[key] || 0, expected: expected[key] } }
              }
            }
          }
        }
      }

      for (const [key, value] of Object.entries(incoming)) {
        if (value === null) {
          delete uiMeta[key]
        } else {
          uiMeta[key] = value
        }

        uiMetaRevisions[key] = (uiMetaRevisions[key] || 0) + 1
      }

      return { applied: { ui_meta: true, ui_meta_revisions: { ...uiMetaRevisions } } }
    }

    if (method === 'session.create') {
      sequence += 1
      const profile = String(params.profile ?? '')
      const title = String(params.title ?? '')

      const session: ScriptedSession = {
        messages: [],
        profile,
        runtime: `rt-${profile}-${sequence}`,
        stored: `sid-${profile}-${sequence}`,
        title
      }

      sessions.set(session.stored, session)
      runtimeToStored.set(session.runtime, session.stored)
      titleToStored.set(`${profile}::${title}`, session.stored)

      return { message_count: 0, messages: [], session_id: session.runtime, stored_session_id: session.stored }
    }

    if (method === 'session.resume') {
      const session = resolveSession(params.profile, params.session_id)

      if (!session) {
        throw gatewayError(`session not found: ${String(params.session_id)}`, 4007)
      }

      // Every resume mints a fresh runtime id; the stored id is the durable
      // identity. Old runtime ids stay resolvable so an in-flight turn that
      // still holds one is not spuriously reaped by the harness itself.
      sequence += 1
      session.runtime = `rt-${session.profile}-${sequence}`
      runtimeToStored.set(session.runtime, session.stored)

      const seen = (resumesByProfile.get(session.profile) || 0) + 1
      resumesByProfile.set(session.profile, seen)
      let busy = Boolean(options.busyResumes?.[session.profile] && seen <= options.busyResumes[session.profile])

      if (session.messages.length > 0) {
        polls += 1
        busy = busy || Boolean(options.pollsBusy && polls <= options.pollsBusy)
        options.onResumePoll?.(polls)
      }

      const clarify = options.clarifyUntil?.[session.profile]
      const approval = options.approvalUntil?.[session.profile]

      return {
        inflight: busy,
        message_count: busy ? 0 : session.messages.length,
        messages: busy || params.omit_messages ? [] : [...session.messages],
        running: false,
        session_id: session.runtime,
        session_key: session.stored,
        ...(clarify && seen <= clarify.until ? { pending_clarify: clarify.payload } : {}),
        ...(approval && seen <= approval.until ? { pending_approval: approval.payload } : {})
      }
    }

    if (method === 'image.attach_bytes' || method === 'pdf.attach' || method === 'file.attach') {
      const session = resolveSession(null, params.session_id)

      if (!session) {
        throw gatewayError(`session-scoped RPC rejected: ${String(params.session_id)} not in memory`, 4001)
      }

      attaches.push({
        data: String(params.content_base64 ?? params.data_url ?? ''),
        filename: String(params.filename ?? params.name ?? ''),
        method,
        order: calls.length,
        profile: session.profile,
        runtime: String(params.session_id ?? '')
      })

      return method === 'file.attach'
        ? { attached: true, ref_text: `@file:attachments/${String(params.name ?? 'attachment')}` }
        : { attached: true }
    }

    if (method === 'prompt.submit') {
      submits += 1

      if (options.failEverySubmitWith) {
        throw options.failEverySubmitWith
      }

      if (submits === 1 && options.failFirstSubmitWith) {
        throw options.failFirstSubmitWith
      }

      const session = resolveSession(null, params.session_id)

      if (!session) {
        throw gatewayError(`session-scoped RPC rejected: ${String(params.session_id)} not in memory`, 4001)
      }

      const prompt = String(params.text ?? '')
      session.messages.push({ content: prompt, role: 'user' })
      calls.push({
        profile: session.profile,
        prompt,
        runtime: session.runtime,
        stored: session.stored,
        title: session.title
      })
      const reply = await turn({ n: calls.length, profile: session.profile, prompt, session })

      for (const message of typeof reply === 'string' ? [{ content: reply, role: 'assistant' }] : reply) {
        session.messages.push(message)
      }

      return {}
    }

    return {}
  }

  const record = async (method: string, params: Record<string, unknown>) => {
    try {
      return await handle(method, params)
    } finally {
      rpc.push({ method, params, refcountAfter: refcount })
    }
  }

  const host: Record<string, unknown> = {
    activeConnectionId: () => 'local',
    notify: vi.fn(),
    notifyError: vi.fn(),
    request: async (method: string, params: Record<string, unknown> = {}) => record(method, params),
    requestProfile: async (_route: unknown, method: string, params: Record<string, unknown> = {}) => {
      refcount += 1

      try {
        return await handle(method, params)
      } finally {
        refcount -= 1

        if (refcount === 0) {
          disposals += 1
        }

        rpc.push({ method, params, refcountAfter: refcount })
        timeline.push(method)
      }
    },
    retainProfile: async () => {
      timeline.push('retain')
      refcount += 1
      let released = false

      return () => {
        if (released) {
          return
        }

        released = true
        timeline.push('release')
        refcount -= 1

        if (refcount === 0) {
          disposals += 1
        }
      }
    },
    setWorkspaceScope: vi.fn(),
    state: {
      connectionId: { get: () => 'local', listen: () => () => undefined },
      gateway: { get: () => 'open', listen: () => () => undefined },
      profile: { get: () => 'default', listen: () => () => undefined }
    }
  }

  return {
    attaches,
    calls,
    disposals: () => disposals,
    host,
    refcount: () => refcount,
    rpc,
    rpcFor: (method: string) => rpc.filter(entry => entry.method === method),
    sessions,
    storage,
    timeline,
    uiMeta,
    uiMetaRevisions
  }
}

/** The `@hermes/plugin-sdk` surface the group modules actually reach for.
 *  `atom` must be the real nanostores one — the room store IS its atoms — and
 *  it has to come from the CURRENT module generation, so this is built inside
 *  the `vi.mock` factory rather than hoisted alongside it. */
export async function pluginSdkMock(host: Record<string, unknown>) {
  const nanostores = await import('nanostores')

  return {
    atom: nanostores.atom,
    // Feature-detected SDK members: the modules read them off the namespace
    // and fall back when absent, but vitest rejects a namespace access with
    // no matching export at all — so they have to be present and undefined.
    BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: undefined,
    blobatarSvg: undefined,
    computed: nanostores.computed,
    createBudgetedLoop: undefined,
    host,
    SkillsView: undefined,
    Streamdown: undefined,
    queryClient: { invalidateQueries: () => undefined },
    useQuery: () => ({ data: [], isLoading: false }),
    useValue: <T>(store: { get: () => T }) => store.get()
  }
}

/** Plugin storage backed by a plain map, for `setPluginCtx`. */
export function scriptedStorage(storage: Map<string, unknown>): PluginContext {
  return {
    storage: {
      get: async (key: string) => storage.get(key) ?? null,
      set: async (key: string, value: unknown) => {
        storage.set(key, structuredClone(value))
      }
    }
    // The modules under test only ever reach for ctx.storage; the rest of the
    // host-supplied context has no bearing on room state.
  } as unknown as PluginContext
}

/** Run the room engine's timers inline.
 *
 *  The turn poll sleeps `GROUP_TURN_POLL_MS` between resumes and the sync
 *  mirror debounces by 350ms; both are wall-clock waits a test must not
 *  actually serve. Resolving immediately keeps the awaits (and therefore the
 *  interleaving the race contracts depend on) while removing the delay. */
export function runTimersInline() {
  vi.stubGlobal('setTimeout', (fn: () => void) => {
    fn()

    return 0
  })
  vi.stubGlobal('clearTimeout', () => undefined)
}

/** Run the engine's timers on the next macrotask instead of inline.
 *
 *  The sync worker's retry path stores its own timer handle AFTER scheduling
 *  it (`retryTimers.set(id, setTimeout(...))`), so an inline timer deletes the
 *  handle before it is written and the entry never clears. Deferring keeps
 *  that ordering intact while still ignoring the backoff delay. */
export function deferTimers() {
  vi.stubGlobal('setTimeout', (fn: () => void) => {
    setImmediate(fn)

    return 1
  })
  vi.stubGlobal('clearTimeout', () => undefined)
}

/** Let the room engine's async loop run to completion. */
export async function drain(isRunning: () => boolean, limit = 400) {
  for (let i = 0; i < limit && isRunning(); i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  // One more flush so the `finally` bookkeeping after the last await lands.
  await new Promise(resolve => setImmediate(resolve))
}
