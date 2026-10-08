// Canonical local authority wire adapter. Remote legacy transports keep their
// existing protocol; unsupported explicit semantics fail before network admission.

export const CANONICAL_GATEWAY_PROTOCOL = 'hermes-gateway-v1'

export const canonicalProfile = (profile: unknown): string =>
  typeof profile === 'string' && profile ? profile : 'default'

export const canonicalSessionKey = (sessionId: unknown, profile?: unknown): string =>
  JSON.stringify([canonicalProfile(profile), sessionId])

// Only the authenticated adapter can establish this renderer-side provenance;
// a JSON-RPC payload's ordinary `profile` property cannot impersonate it.
const canonicalOwners = new WeakMap<object, string>()
export const canonicalOwnerProfile = (value: object): string | undefined => canonicalOwners.get(value)

export const recordCanonicalOwner = (value: object, profile: string): void => { canonicalOwners.set(value, profile) }

// Mirrors hermes_cli/gateway_mutations.slash_mutation: the typed directives
// that are canonical mutations, not gateway-executed slash commands. Model
// flags (--global/--once/--refresh) have no canonical mutation and stay on the
// exec path so the authority refuses them explicitly.
export function slashMutation(command: string): { operation: string; payload: Record<string, unknown> } | null {
  const [name, ...rest] = command.trim().replace(/^\/+/, '').split(/\s+/)
  const arg = rest.join(' ').trim()

  if (name === 'model') {
    if (!arg || arg.startsWith('-') || /(^|\s)--/.test(arg)) { return null }
    const [model, ...flags] = arg.split(/\s+/)

    return flags.length ? null : { operation: 'model', payload: { model } }
  }

  const field = ({ branch: 'title', compress: 'focus' } as Record<string, string>)[name]

  if (!field) { return null }

  return { operation: name, payload: arg ? { [field]: arg } : {} }
}

function mutationSummary(operation: string, value: Record<string, unknown>): string {
  if (operation === 'model') { return `model: ${value.model}${value.provider ? ` (${value.provider})` : ''}` }

  if (operation === 'branch') { return `branch: ${value.branched_session_id}` }

  if (operation === 'compress') {
    return value.status === 'preview' ? (value.lines as string[]).join('\n') : `compress: ${value.target_session_id ?? value.session_id}`
  }

  return `${operation}: ok`
}

// Explicit desktop methods that travel as canonical `session.mutate`.
// `session.branch_stored` / `session.branch_whole` are legacy whole-history branches keyed by a
// stored parent id; the authority has ONE branch, a `branch` mutation on the parent, whose child
// is a local route the authority can restore (a legacy-minted child has no local policy and every
// resume on it answers not_found).
// The socket is bound to a profile already; only a sibling the host multiplexes rides as `profile`.
export function siblingRoute(profile: unknown): string | null {
  return typeof profile === 'string' && profile && profile !== 'default' ? profile : null
}

const BRANCH_METHODS = new Set(['session.branch', 'session.branch_stored', 'session.branch_whole'])
const MUTATION_METHODS = new Set(['session.title', 'session.archive', 'session.compress', ...BRANCH_METHODS])

export class CanonicalDesktopProtocol {
  private creates = new Map<string, string>()
  private revisions = new Map<string, number>()
  private mutations = new Map<string, Record<string, unknown>>()
  private generations = new Map<string, number>()
  private prompts = new Map<string, Record<string, unknown>>()
  // admission_id → generation of a turn the authority recovered as `unknown`
  // (owner died mid-turn). Only these rows may be acknowledged, and only with
  // the generation the authority stamped on them, never the live one.
  private unknownAdmissions = new Map<string, { session_id: string; profile: string; generation: number }>()

  failure(params: Record<string, unknown>, error: unknown) {
    if ((error as { data?: { reason?: string } })?.data?.reason !== 'revision_conflict') { return }

    // A confirmed CAS refusal did not mutate. Ambiguous transport failures keep
    // the original revision/id so a retry cannot overwrite another user's edit.
    for (const [key, mutation] of this.mutations) { if (mutation.request_id === params.request_id) { this.mutations.delete(key) } }
  }

  // Wire method for a prepared request: composer metadata, branch and the
  // typed `/model <name>` / `/branch [title]` / `/compress [focus]` directives
  // and the dedicated compress action all travel as canonical `session.mutate`;
  // everything else keeps its name.
  wire(method: string, prepared: Record<string, unknown> = {}): string {
    if (MUTATION_METHODS.has(method)) { return 'session.mutate' }

    // The warm-cache re-attach: canonical attach is `session.resume`, which
    // rebinds the live event transport and returns the same snapshot shape.
    if (method === 'session.activate') { return 'session.resume' }

    return method === 'slash.exec' && typeof prepared.operation === 'string' ? 'session.mutate' : method
  }

  private retainedMutation(sessionId: unknown, profile: unknown, operation: string, payload: Record<string, unknown>, withGeneration: boolean): Record<string, unknown> {
    const owner = canonicalSessionKey(sessionId, profile)
    const key = JSON.stringify([owner, operation, payload])
    const retained = this.mutations.get(key)

    if (retained) { return retained }
    const revision = this.revisions.get(owner)

    if (revision === undefined) { throw new Error('Session revision unavailable; reopen the session before editing metadata') }
    const generation = this.generations.get(owner)

    if (withGeneration && generation === undefined) { throw new Error('Session execution identity unavailable; reconnect before this command') }

    const mutation: Record<string, unknown> = { session_id: sessionId, request_id: crypto.randomUUID(), expected_revision: revision,
      ...(withGeneration ? { expected_generation: generation } : {}), operation, payload }

    this.mutations.set(key, mutation)

    return mutation
  }

  prepare(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const prepared = this.preparePayload(method, params)

    // Routing belongs to the envelope, independently of the canonical payload.
    // Creation applies its own default-profile normalization.
    return method !== 'session.create' && params.profile !== undefined ? { ...prepared, profile: params.profile } : prepared
  }

  private preparePayload(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const mutation = this.prepareMutation(method, params)

    if (mutation) { return mutation }

    if (method === 'session.create') { return this.prepareCreate(params) }

    if (method === 'session.activate') {
      return { session_id: params.session_id, source: 'desktop', ...(params.profile ? { profile: params.profile } : {}) }
    }

    if (method === 'session.interrupt' || method === 'session.redirect' || method === 'session.steer') {
      const generation = params.execution_generation ?? this.generations.get(canonicalSessionKey(params.session_id, params.profile))

      if (typeof generation !== 'number') { throw new Error('Session execution identity unavailable; reconnect before controlling this turn') }

      return { ...params, session_id: params.session_id, execution_generation: generation }
    }

    if (method === 'prompt.resolve_unknown') {
      const lost = this.unknownAdmissions.get(canonicalSessionKey(params.admission_id, params.profile))

      if (!lost || lost.session_id !== params.session_id || lost.profile !== canonicalProfile(params.profile)) { throw new Error('Admission is not an unknown lost turn; reopen the session before acknowledging') }

      return { session_id: params.session_id, admission_id: params.admission_id, execution_generation: lost.generation }
    }

    if (method === 'approval.respond' || method === 'clarify.respond') { return this.preparePromptResponse(method, params) }

    return params
  }

  // Metadata, branch and typed slash directives that travel as canonical `session.mutate`; null otherwise.
  private prepareMutation(method: string, params: Record<string, unknown>): Record<string, unknown> | null {
    const field = ({ 'session.title': 'title', 'session.archive': 'archived' } as Record<string, string>)[method]

    if (field) { return this.retainedMutation(params.session_id, params.profile, field === 'title' ? 'rename' : 'archive', { [field]: params[field] }, false) }

    // Branch and compress fence the execution generation like the slash directives.
    const fenced = ({
      'session.branch': () => ({ operation: 'branch', payload: {} }),
      'session.compress': () => ({ operation: 'compress', payload: params.focus_topic ? { focus: String(params.focus_topic) } : {} })
    } as Record<string, () => { operation: string; payload: Record<string, unknown> }>)[method]?.()

    if (fenced) { return this.retainedMutation(params.session_id, params.profile, fenced.operation, fenced.payload, true) }

    if (method === 'session.branch_stored' || method === 'session.branch_whole') {
      // The stored-parent form names the parent as `parent_session_id`; the live form as `session_id`.
      const parent = params.parent_session_id ?? params.session_id
      const payload = typeof params.title === 'string' && params.title ? { title: params.title } : {}

      return this.retainedMutation(parent, params.profile, 'branch', payload, true)
    }

    if (method === 'slash.exec') {
      const directive = slashMutation(String(params.command ?? ''))

      if (directive) { return this.retainedMutation(params.session_id, params.profile, directive.operation, directive.payload, true) }
    }

    return null
  }

  private prepareCreate(params: Record<string, unknown>): Record<string, unknown> {
    const allowed = new Set(['request_id', 'source', 'cwd', 'model', 'toolsets', 'profile', 'cols', 'title', 'hidden', 'follow_profile_config'])
    const unsupported = Object.keys(params).filter(key => !allowed.has(key) && !(key === 'fast' && params[key] === false))

    if (unsupported.length) { throw new Error(`Canonical gateway does not support explicit session options: ${unsupported.join(', ')}`) }

    const result = Object.fromEntries(Object.entries(params).filter(([key, value]) =>
      ['request_id', 'cwd', 'model', 'toolsets', 'title', 'hidden'].includes(key) || (key === 'profile' && siblingRoute(value))))

    const key = JSON.stringify(result)
    const requestId = params.request_id ?? this.creates.get(key) ?? crypto.randomUUID()
    this.creates.set(key, String(requestId))

    return { ...result, request_id: requestId, source: 'gui' }
  }

  private preparePromptResponse(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const id = String(params.prompt_id ?? params.request_id ?? '')
    const prompt = this.prompts.get(canonicalSessionKey(id, params.profile))
    const sessionId = params.session_id ?? prompt?.session_id

    if (!prompt || prompt.session_id !== sessionId || prompt.profile !== canonicalProfile(params.profile) ||
        prompt.execution_generation !== this.generations.get(canonicalSessionKey(sessionId, params.profile))) {
      throw new Error('Prompt is stale or unavailable; reconnect before responding')
    }

    const field = method === 'approval.respond' ? 'choice' : 'answer'

    return { session_id: sessionId, execution_generation: prompt.execution_generation, prompt_id: id, [field]: params[field] }
  }

  private reconcileUnknownAdmissions(sid: string, profile: string, pending: Array<Record<string, unknown>>): void {
    for (const [id, lost] of this.unknownAdmissions) {
      if (lost.session_id === sid && lost.profile === profile) { this.unknownAdmissions.delete(id) }
    }

    for (const row of pending) {
      if (row?.status === 'unknown' && typeof row.admission_id === 'string' && typeof row.execution_generation === 'number') {
        this.unknownAdmissions.set(canonicalSessionKey(row.admission_id, profile), { session_id: sid, profile, generation: row.execution_generation })
      }
    }
  }

  event(event: { type: string; session_id?: string; profile?: string; execution_generation?: unknown; payload?: unknown }) {
    const payload = event.payload as Record<string, unknown> | undefined

    if (!payload || !event.session_id) { return }
    const sid = event.session_id
    const profile = canonicalProfile(event.profile)
    const owner = canonicalSessionKey(sid, profile)

    if (Array.isArray(payload.pending)) {
      payload.pending_submissions = payload.pending.map(row => ({ ...row, user: row.text }))

      this.reconcileUnknownAdmissions(sid, profile, payload.pending)
    }

    const generation = event.execution_generation ?? payload.execution_generation

    if (typeof generation === 'number') {
      const current = this.generations.get(owner) ?? -1

      if (generation < current) { return }
      this.generations.set(owner, generation)
    }

    // Turns advance the CAS revision without a session.updated event; the
    // pending fanout is where a viewer learns the value its next mutation must present.
    if (event.type === 'session.info' && typeof payload.revision === 'number') { this.revisions.set(owner, payload.revision) }

    if (typeof payload.prompt_id === 'string') {
      if (event.type.endsWith('.settled')) { this.prompts.delete(canonicalSessionKey(payload.prompt_id, profile));

 return }

      if (event.type === 'approval.request' || event.type === 'clarify.request') {
        payload.request_id = payload.prompt_id
        this.prompts.set(canonicalSessionKey(payload.prompt_id, profile), { ...payload, session_id: sid, profile })
      }
    }
  }

  result(method: string, params: Record<string, unknown>, value: any): any {
    if (!value || typeof value !== 'object') { return value }

    if (typeof value.session_id === 'string' && typeof value.revision === 'number') {
      this.revisions.set(canonicalSessionKey(value.session_id, params.profile), value.revision)
    }

    if (MUTATION_METHODS.has(method) || (method === 'slash.exec' && typeof params.operation === 'string')) {
      return this.mutationReceipt(method, params, value)
    }

    if (method === 'session.create') {
      for (const [key, id] of this.creates) { if (id === params.request_id) { this.creates.delete(key) } }
    }

    if (method === 'prompt.submit' || method === 'prompt.resolve_unknown') { return this.admissionReceipt(method, params, value) }

    if (method === 'session.resume' || method === 'session.create' || method === 'session.activate') { return this.snapshotResult(value, params.profile) }

    if (method === 'session.events.since') {
      for (const event of value.events ?? []) { this.event({ ...event, profile: canonicalProfile(params.profile) }) }
    }

    return value
  }

  private admissionReceipt(method: string, params: Record<string, unknown>, value: any): any {
    if (method === 'prompt.submit') {
      if (value.ref?.session_id !== params.session_id || typeof value.admission_id !== 'string') {
        throw new Error('Canonical admission receipt destination mismatch')
      }

      return { ...value, session_id: value.ref.session_id, submission_id: params.submission_id ?? params.input_id }
    }

    if (value.ref?.session_id !== params.session_id || value.admission_id !== params.admission_id) {
      throw new Error('Canonical admission receipt destination mismatch')
    }

    this.unknownAdmissions.delete(canonicalSessionKey(params.admission_id, params.profile))

    return { ...value, session_id: value.ref.session_id }
  }

  private mutationReceipt(method: string, params: Record<string, unknown>, value: any): any {
    if (value.session_id !== params.session_id) { throw new Error('Metadata receipt destination mismatch') }

    for (const [key, mutation] of this.mutations) { if (mutation.request_id === params.request_id) { this.mutations.delete(key) } }

    if (BRANCH_METHODS.has(method)) {
      return { ...value, session_id: value.branched_session_id, stored_session_id: value.branched_session_id, parent_session_id: params.session_id, message_count: value.copied_messages }
    }

    if (method === 'slash.exec') { return { ...value, type: 'exec', output: mutationSummary(params.operation as string, value) } }

    return { ...value, ok: true }
  }

  private snapshotResult(value: any, route: unknown): any {
    const sid = value.session_id
    const profile = canonicalProfile(route)
    this.event({ type: 'session.info', session_id: sid, profile, payload: value })

    const prompts = (value.prompts ?? []).map((prompt: Record<string, unknown>) => {
      const projected = { ...prompt, request_id: prompt.prompt_id }
      this.event({ type: `${prompt.kind}.request`, session_id: sid, profile, payload: projected })

      return projected
    })

    return { ...value, pending_approval: prompts.find((p: any) => p.kind === 'approval'), pending_clarify: prompts.find((p: any) => p.kind === 'clarify'), info: { ...value.info, stored_session_id: value.stored_session_id, pending_submissions: value.pending_submissions, execution_generation: value.execution_generation, running: value.running } }
  }

  // The canonical compress receipt carries counts, not the retained transcript;
  // the compress action repaints only from `messages`, so resume the session
  // through the normal request path (which also re-primes revision/generation).
  async settle(method: string, params: Record<string, unknown>, value: any, request: (method: string, params: Record<string, unknown>) => Promise<any>): Promise<any> {
    if (method !== 'session.compress' && !(method === 'slash.exec' && params.operation === 'compress')) { return value }

    // `--preview` in the focus argument is a read-only report: no transcript changed.
    if (value?.status === 'preview') { return { ...value, host_ack: { output: (value.lines as string[]).join('\n') } } }
    const resumed = await request('session.resume', { session_id: params.session_id, ...(params.profile !== undefined ? { profile: params.profile } : {}) })

    return { ...value, messages: resumed.messages, info: resumed.info, host_ack: { output: `compressed context: ${value.message_count} messages retained` } }
  }
}
