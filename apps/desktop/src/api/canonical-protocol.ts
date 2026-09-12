// Canonical local authority wire adapter. Remote legacy transports keep their
// existing protocol; unsupported explicit semantics fail before network admission.

export const CANONICAL_GATEWAY_PROTOCOL = 'hermes-gateway-v1'

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

  if (operation === 'compress') { return `compress: ${value.target_session_id ?? value.session_id}` }

  return `${operation}: ok`
}

export class CanonicalDesktopProtocol {
  private creates = new Map<string, string>()
  private revisions = new Map<string, number>()
  private mutations = new Map<string, Record<string, unknown>>()
  private generations = new Map<string, number>()
  private prompts = new Map<string, Record<string, unknown>>()
  // admission_id → generation of a turn the authority recovered as `unknown`
  // (owner died mid-turn). Only these rows may be acknowledged, and only with
  // the generation the authority stamped on them, never the live one.
  private unknownAdmissions = new Map<string, { session_id: string; generation: number }>()

  failure(params: Record<string, unknown>, error: unknown) {
    if ((error as { data?: { reason?: string } })?.data?.reason !== 'revision_conflict') { return }

    // A confirmed CAS refusal did not mutate. Ambiguous transport failures keep
    // the original revision/id so a retry cannot overwrite another user's edit.
    for (const [key, mutation] of this.mutations) { if (mutation.request_id === params.request_id) { this.mutations.delete(key) } }
  }

  // Wire method for a prepared request: composer metadata, branch and the
  // typed `/model <name>` / `/branch [title]` / `/compress [focus]` directives
  // all travel as canonical `session.mutate`; everything else keeps its name.
  wire(method: string, prepared: Record<string, unknown> = {}): string {
    if (method === 'session.title' || method === 'session.archive' || method === 'session.branch') { return 'session.mutate' }

    return method === 'slash.exec' && typeof prepared.operation === 'string' ? 'session.mutate' : method
  }

  private retainedMutation(sessionId: unknown, operation: string, payload: Record<string, unknown>, withGeneration: boolean): Record<string, unknown> {
    const key = JSON.stringify([sessionId, operation, payload])
    const retained = this.mutations.get(key)

    if (retained) { return retained }
    const revision = this.revisions.get(String(sessionId))

    if (revision === undefined) { throw new Error('Session revision unavailable; reopen the session before editing metadata') }
    const generation = this.generations.get(String(sessionId))

    if (withGeneration && generation === undefined) { throw new Error('Session execution identity unavailable; reconnect before this command') }

    const mutation: Record<string, unknown> = { session_id: sessionId, request_id: crypto.randomUUID(), expected_revision: revision,
      ...(withGeneration ? { expected_generation: generation } : {}), operation, payload }

    this.mutations.set(key, mutation)

    return mutation
  }

  prepare(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const field = ({ 'session.title': 'title', 'session.archive': 'archived' } as Record<string, string>)[method]

    if (field) { return this.retainedMutation(params.session_id, field === 'title' ? 'rename' : 'archive', { [field]: params[field] }, false) }

    if (method === 'session.branch') { return this.retainedMutation(params.session_id, 'branch', {}, true) }

    if (method === 'slash.exec') {
      const directive = slashMutation(String(params.command ?? ''))

      if (directive) { return this.retainedMutation(params.session_id, directive.operation, directive.payload, true) }
    }

    if (method === 'session.create') {
      const allowed = new Set(['request_id', 'source', 'cwd', 'model', 'toolsets', 'profile', 'cols'])
      const unsupported = Object.keys(params).filter(key => !allowed.has(key) && !(key === 'fast' && params[key] === false))

      if (unsupported.length) { throw new Error(`Canonical gateway does not support explicit session options: ${unsupported.join(', ')}`) }
      const result = Object.fromEntries(Object.entries(params).filter(([key]) => ['request_id', 'cwd', 'model', 'toolsets'].includes(key)))
      const key = JSON.stringify(result)
      const requestId = params.request_id ?? this.creates.get(key) ?? crypto.randomUUID()
      this.creates.set(key, String(requestId))

      return { ...result, request_id: requestId, source: 'gui' }
    }

    if (method === 'session.interrupt' || method === 'session.redirect' || method === 'session.steer') {
      const generation = params.execution_generation ?? this.generations.get(String(params.session_id))

      if (typeof generation !== 'number') { throw new Error('Session execution identity unavailable; reconnect before controlling this turn') }

      return { ...params, session_id: params.session_id, execution_generation: generation }
    }

    if (method === 'prompt.resolve_unknown') {
      const lost = this.unknownAdmissions.get(String(params.admission_id))

      if (!lost || lost.session_id !== params.session_id) { throw new Error('Admission is not an unknown lost turn; reopen the session before acknowledging') }

      return { session_id: params.session_id, admission_id: params.admission_id, execution_generation: lost.generation }
    }

    if (method === 'approval.respond' || method === 'clarify.respond') {
      const id = String(params.prompt_id ?? params.request_id ?? '')
      const prompt = this.prompts.get(id)
      const sessionId = params.session_id ?? prompt?.session_id

      if (!prompt || prompt.session_id !== sessionId || prompt.execution_generation !== this.generations.get(String(sessionId))) {
        throw new Error('Prompt is stale or unavailable; reconnect before responding')
      }

      const field = method === 'approval.respond' ? 'choice' : 'answer'

      return { session_id: sessionId, execution_generation: prompt.execution_generation, prompt_id: id, [field]: params[field] }
    }

    return params
  }

  event(event: { type: string; session_id?: string; payload?: unknown }) {
    const payload = event.payload as Record<string, unknown> | undefined

    if (!payload || !event.session_id) { return }
    const sid = event.session_id

    if (Array.isArray(payload.pending)) {
      payload.pending_submissions = payload.pending.map(row => ({ ...row, user: row.text }))

      for (const [id, lost] of this.unknownAdmissions) { if (lost.session_id === sid) { this.unknownAdmissions.delete(id) } }

      for (const row of payload.pending) {
        if (row?.status === 'unknown' && typeof row.admission_id === 'string' && typeof row.execution_generation === 'number') {
          this.unknownAdmissions.set(row.admission_id, { session_id: sid, generation: row.execution_generation })
        }
      }
    }

    if (typeof payload.execution_generation === 'number') {
      const current = this.generations.get(sid) ?? -1

      if (payload.execution_generation < current) { return }
      this.generations.set(sid, payload.execution_generation)
    }

    // Turns advance the CAS revision without a session.updated event; the
    // pending fanout is where a viewer learns the value its next mutation must present.
    if (event.type === 'session.info' && typeof payload.revision === 'number') { this.revisions.set(sid, payload.revision) }

    if (typeof payload.prompt_id === 'string') {
      if (event.type.endsWith('.settled')) { this.prompts.delete(payload.prompt_id);

 return }

      if (event.type === 'approval.request' || event.type === 'clarify.request') {
        payload.request_id = payload.prompt_id
        this.prompts.set(payload.prompt_id, { ...payload, session_id: sid })
      }
    }
  }

  result(method: string, params: Record<string, unknown>, value: any): any {
    if (!value || typeof value !== 'object') { return value }

    if (typeof value.session_id === 'string' && typeof value.revision === 'number') {
      this.revisions.set(value.session_id, value.revision)
    }

    if (method === 'session.title' || method === 'session.archive' || method === 'session.branch' || (method === 'slash.exec' && typeof params.operation === 'string')) {
      if (value.session_id !== params.session_id) { throw new Error('Metadata receipt destination mismatch') }

      for (const [key, mutation] of this.mutations) { if (mutation.request_id === params.request_id) { this.mutations.delete(key) } }

      if (method === 'session.branch') {
        return { ...value, session_id: value.branched_session_id, stored_session_id: value.branched_session_id, parent_session_id: params.session_id, message_count: value.copied_messages }
      }

      if (method === 'slash.exec') { return { type: 'exec', output: mutationSummary(params.operation as string, value) } }

      return { ...value, ok: true }
    }

    if (method === 'session.create') {
      for (const [key, id] of this.creates) { if (id === params.request_id) { this.creates.delete(key) } }
    }

    if (method === 'prompt.submit') {
      if (value.ref?.session_id !== params.session_id || typeof value.admission_id !== 'string') {
        throw new Error('Canonical admission receipt destination mismatch')
      }

      return { ...value, session_id: value.ref.session_id, submission_id: params.submission_id ?? params.input_id }
    }

    if (method === 'prompt.resolve_unknown') {
      if (value.ref?.session_id !== params.session_id || value.admission_id !== params.admission_id) {
        throw new Error('Canonical admission receipt destination mismatch')
      }

      this.unknownAdmissions.delete(String(params.admission_id))

      return { ...value, session_id: value.ref.session_id }
    }

    if (method === 'session.resume' || method === 'session.create') {
      const sid = value.session_id
      this.event({ type: 'session.info', session_id: sid, payload: value })

      const prompts = (value.prompts ?? []).map((prompt: Record<string, unknown>) => {
        const projected = { ...prompt, request_id: prompt.prompt_id }
        this.event({ type: `${prompt.kind}.request`, session_id: sid, payload: projected })

        return projected
      })

      return { ...value, pending_approval: prompts.find((p: any) => p.kind === 'approval'), pending_clarify: prompts.find((p: any) => p.kind === 'clarify'), info: { ...value.info, stored_session_id: value.stored_session_id, pending_submissions: value.pending_submissions, execution_generation: value.execution_generation, running: value.running } }
    }

    if (method === 'session.events.since') {
      for (const event of value.events ?? []) { this.event(event) }
    }

    return value
  }
}
