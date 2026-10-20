// Canonical local authority wire adapter. Remote legacy transports keep their
// existing protocol; unsupported explicit semantics fail before network admission.

export const CANONICAL_GATEWAY_PROTOCOL = 'hermes-gateway-v1'

export const canonicalProfile = (profile: unknown): string =>
  typeof profile === 'string' && profile ? profile : 'default'

export const canonicalSessionKey = (sessionId: unknown, profile?: unknown): string =>
  JSON.stringify([canonicalProfile(profile), sessionId])

const admissionObservers = new Set<(admissionId: string, status: string) => void>()

/** Every `prompt.submit` this Desktop's canonical clients send (composer, edit and regenerate,
 *  Quick Entry, bots) reports its accepted admission here, whichever surface sent it. */
export function observeCanonicalAdmissions(observer: (admissionId: string, status: string) => void): () => void {
  admissionObservers.add(observer)

  return () => admissionObservers.delete(observer)
}

/** Live events and retained snapshots must describe the same answerable card. */
export function canonicalPromptParams(
  kind: unknown,
  sessionId: string,
  prompt: Record<string, unknown>
): Record<string, unknown> {
  const choices = Array.isArray(prompt.choices) ? prompt.choices : []

  return kind === 'clarify'
    ? {
        session_id: sessionId,
        answers: prompt.answers,
        questions: Array.isArray(prompt.questions)
          ? prompt.questions
          : [{ qid: prompt.prompt_id, question: prompt.question, choices, multi_select: prompt.multi_select }]
      }
    : {
        session_id: sessionId,
        request_id: prompt.prompt_id,
        command: prompt.command,
        description: prompt.description,
        choices,
        allow_permanent: choices.includes('always'),
        edit: prompt.edit
      }
}

// Only the authenticated adapter can establish this renderer-side provenance;
// a JSON-RPC payload's ordinary `profile` property cannot impersonate it.
const canonicalOwners = new WeakMap<object, string>()
export const canonicalOwnerProfile = (value: object): string | undefined => canonicalOwners.get(value)

export const recordCanonicalOwner = (value: object, profile: string): void => {
  canonicalOwners.set(value, profile)
}

// Mirrors hermes_cli/gateway_mutations.slash_mutation: the typed directives
// that are canonical mutations, not gateway-executed slash commands. Model
// flags (--global/--once/--refresh) have no canonical mutation and stay on the
// exec path so the authority refuses them explicitly.
export function slashMutation(command: string): { operation: string; payload: Record<string, unknown> } | null {
  const [name, ...rest] = command.trim().replace(/^\/+/, '').split(/\s+/)
  const arg = rest.join(' ').trim()

  if (name === 'model') {
    if (!arg || arg.startsWith('-') || /(^|\s)--/.test(arg)) {
      return null
    }
    const [model, ...flags] = arg.split(/\s+/)

    return flags.length ? null : { operation: 'model', payload: { model } }
  }

  const field = ({ branch: 'title', compress: 'focus' } as Record<string, string>)[name]

  if (!field) {
    return null
  }

  return { operation: name, payload: arg ? { [field]: arg } : {} }
}

// The composer picker's `config.set key=model` value: `<model> --provider <id> [--session]`. The
// canonical model mutation is always session-scoped policy (the authority never rewrites
// config.yaml from it), so `--session` carries no extra meaning; any other flag has no canonical
// field and stays on config.set for the authority to refuse explicitly.
export function pickerModelMutation(value: unknown): { model: string; provider: string } | null {
  const [model, ...flags] = String(value ?? '')
    .trim()
    .split(/\s+/)
  const sessionOnly = flags.at(-1) === '--session'
  const rest = sessionOnly ? flags.slice(0, -1) : flags

  if (
    !model ||
    model.startsWith('-') ||
    rest.length !== 2 ||
    rest[0] !== '--provider' ||
    !rest[1] ||
    rest[1].startsWith('-')
  ) {
    return null
  }

  return { model, provider: rest[1] }
}

function mutationSummary(operation: string, value: Record<string, unknown>): string {
  if (operation === 'model') {
    return `model: ${value.model}${value.provider ? ` (${value.provider})` : ''}`
  }

  if (operation === 'branch') {
    return `branch: ${value.branched_session_id}`
  }

  if (operation === 'compress') {
    return value.status === 'preview'
      ? (value.lines as string[]).join('\n')
      : `compress: ${value.target_session_id ?? value.session_id}`
  }

  return `${operation}: ok`
}

// Explicit desktop methods that travel as canonical `session.mutate`.
// `session.branch_stored` / `session.branch_whole` are legacy whole-history branches keyed by a
// stored parent id; the authority has ONE branch, a `branch` mutation on the parent, whose child
// is a local route the authority can restore (a legacy-minted child has no local policy and every
// resume on it answers not_found).

// A message-level branch (`count` = the selected prefix) keeps rows through `through_message_id`.
// Without that durable id the authority would copy the whole history, so it refuses instead.
function branchBoundary(params: Record<string, unknown>): Record<string, unknown> {
  const through = params.through_message_id

  if (typeof through === 'number' && Number.isInteger(through) && through > 0) {
    return { through_message_id: through }
  }

  if (params.count !== undefined) {
    throw new Error('This message is not saved yet; wait for it to settle, then branch from it again')
  }

  return {}
}

const RELAY_HANDLE = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/
const RELAY_SENDER_STAMP = /^(Message from 🤖 [\s\S]+? \(@)[A-Za-z0-9_-]+(\): )/u

// The canonical `bot_relay.deliver` is a closed key set without the legacy `from_*` sender
// fields. Translate them as the legacy bridge did (tui_gateway/methods_bot_relay.py): a bot
// author keyed by the sender's connection (`delivery_turn_author`), and the message's
// `(@handle)` stamp qualified with that connection so a reply never lands on the recipient's
// own same-named bot (#103731; `<handle>@<connection>` always resolves there).
function canonicalBotDelivery(params: Record<string, unknown>): Record<string, unknown> {
  const { from_profile: rawProfile, from_handle: rawHandle, from_connection: rawConnection, ...rest } = params
  const [profile, connection] = [String(rawProfile ?? '').trim(), String(rawConnection ?? '').trim()]
  const handle = String(rawHandle ?? '')
    .trim()
    .replace(/^@/, '')

  if (!profile) {
    return rest
  }

  const author = {
    id: connection ? `bot:${connection}/${profile}` : `bot:${profile}`,
    name: handle || profile,
    is_bot: true
  }
  const qualify = typeof rest.message === 'string' && RELAY_HANDLE.test(handle) && RELAY_HANDLE.test(connection)
  const message = qualify
    ? String(rest.message).replace(RELAY_SENDER_STAMP, `$1${handle}@${connection}$2`)
    : rest.message

  return { ...rest, message, author }
}

// The legacy `prompt.submit` truncation keys (edit / regenerate / restore). The owner has no such
// keys: the cut is a revision-fenced `rewind` mutation on the user row, then a plain submit. The
// deep-cut confirm (#133716) guards the legacy sidecar's implicit cut; the owner's rewind names its
// row explicitly and the user already confirmed before Desktop sent it, so it is dropped here too.
const TRUNCATE_KEYS = [
  'confirm_truncate',
  'confirm_empty_truncate',
  'confirm_deep_truncate',
  'truncate_before_row_id',
  'truncate_before_message_id',
  'truncate_before_user_ordinal',
  'rebind_survivor_row_ids'
]

export function splitRewindSubmit(
  params: Record<string, unknown>
): { rewind: Record<string, unknown> | null; submit: Record<string, unknown> } | null {
  if (!TRUNCATE_KEYS.some(key => key in params)) {
    return null
  }
  const row = params.truncate_before_row_id
  const submit = Object.fromEntries(Object.entries(params).filter(([key]) => !TRUNCATE_KEYS.includes(key)))

  if (typeof row === 'number' && Number.isInteger(row) && row > 0) {
    return {
      rewind: {
        session_id: params.session_id,
        target_message_id: row,
        ...(params.profile !== undefined ? { profile: params.profile } : {})
      },
      submit
    }
  }

  // Only a durable row id names the owner's cut; a message id or ordinal would be a guess.
  if (
    ['truncate_before_row_id', 'truncate_before_message_id', 'truncate_before_user_ordinal'].some(key => key in params)
  ) {
    throw new Error('This message is not saved yet; wait for it to settle, then edit it again')
  }

  return { rewind: null, submit }
}

// Session verbs only the legacy sidecar implements: it looks the session up in its own table,
// which never holds an owner session, and answers `session not found` (read by the renderer as a
// reaped runtime). Refused by name until the owner serves them.
// `preview.restart` is served by the owner (gateway/session_tui_side.py).
const SIDECAR_SESSION_VERBS = new Set(['handoff.request', 'handoff.state', 'handoff.fail'])

// The manage_connections card's calls the owner serves for a session-owned operation (gateway/session_connections.py); any other session-owned connector call stays refused.
const OWNER_CONNECTION_VERBS = new Set([
  'connection.respond',
  'connectors.operation.status',
  'connectors.operation.wake',
  'connectors.connect'
])

// `config.set` keys the owner serves (tui_gateway/contracts/canonical_projections.py::CanonicalConfigSetParams;
// gateway/session_tui_config.py), with the keys that take a `scope` (`reasoning --global`, the
// Shift+click global yolo zap).
const CANONICAL_CONFIG_KEYS = new Set([
  'busy',
  'verbose',
  'yolo',
  'model',
  'reasoning',
  'fast',
  'personality',
  'approvals.mode',
  'voice.voice_chat_mode',
  'display.message_reactions',
  'display.in_app_tips',
  'display.in_app_tours'
])

const SCOPED_CONFIG_KEYS = new Set(['reasoning', 'yolo'])

// The one guard between a Desktop producer and the owner's closed contracts: translate a legacy
// shape the owner has a canonical form for, refuse by name what it does not serve.
function canonicalProducerFrame(method: string, params: Record<string, unknown>): Record<string, unknown> {
  // Any other key, or a scope on a key without one: refuse before the wire, like Ink, instead of a
  // bare invalid_params.
  if (
    method === 'config.set' &&
    (!CANONICAL_CONFIG_KEYS.has(String(params.key)) ||
      ('scope' in params && !SCOPED_CONFIG_KEYS.has(String(params.key))))
  ) {
    throw new Error(
      `Changing ${params.key}${'scope' in params ? ` (${params.scope})` : ''} is not available on the shared gateway yet`
    )
  }

  const sessionConnector =
    method.startsWith('connect') &&
    (params.owner as { type?: unknown } | undefined)?.type === 'session' &&
    !OWNER_CONNECTION_VERBS.has(method)

  if (SIDECAR_SESSION_VERBS.has(method) || sessionConnector) {
    throw new Error(`${method} is not available on the shared gateway yet`)
  }

  // The owner admits identified input only. An identityless producer (session tile, quick entry,
  // bot chats) is a fresh send on every call, which is what the legacy wire did with it.
  if (method === 'prompt.submit' && !params.submission_id && !params.input_id) {
    return { ...params, submission_id: crypto.randomUUID() }
  }

  return method === 'bot_relay.deliver' ? canonicalBotDelivery(params) : params
}

// The levels `hermes_constants.parse_reasoning_effort` accepts: a session `/reasoning <level>` is the
// owner's `reasoning` mutation; the display words (show/hide/full/clamp) are config.yaml writes.
const REASONING_EFFORTS = new Set([
  'none',
  'false',
  'disabled',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
  'ultra'
])

// Frozen-policy edits: a session-scope `/reasoning <level>` (`--global` stays the named refusal, no
// owner config-write verb) and Move to project, which re-homes the execution cwd and TERMINAL_CWD
// with the row (`session_key` names the sidebar row).
function policyEdit(
  method: string,
  params: Record<string, unknown>
): { session: unknown; operation: string; payload: Record<string, unknown> } | null {
  const effort = String(params.value ?? '')
    .trim()
    .toLowerCase()

  if (
    method === 'config.set' &&
    params.key === 'reasoning' &&
    params.scope !== 'global' &&
    REASONING_EFFORTS.has(effort)
  ) {
    return { session: params.session_id, operation: 'reasoning', payload: { effort } }
  }

  // The composer's folder picker on an open chat (main's session.cwd.set) is the same re-home.
  if (method === 'session.cwd.set') {
    return { session: params.session_id, operation: 'workspace', payload: { cwd: params.cwd } }
  }

  return method === 'session.workspace.move'
    ? { session: params.session_key, operation: 'workspace', payload: { cwd: params.cwd } }
    : null
}

// Desktop methods that become `session.mutate` once prepareMutation chose an operation.
const POLICY_EDIT_METHODS = new Set(['slash.exec', 'config.set', 'session.workspace.move', 'session.cwd.set'])
const BRANCH_METHODS = new Set(['session.branch', 'session.branch_stored', 'session.branch_whole'])
// Value-setting mutations: re-applying one is harmless, so a later acknowledged edit retires it.
const VALUE_OPERATIONS = new Set(['rename', 'archive', 'model', 'reasoning', 'workspace'])
// `session.model`: a whole canonical model selection (model / provider / reasoning) plus the owner's
// token, as the owner asked it (a deferred pick's confirmation): the same request the TUI resends.
const MUTATION_METHODS = new Set([
  'session.title',
  'session.archive',
  'session.compress',
  'session.rewind',
  'session.model',
  ...BRANCH_METHODS
])

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
  // [owner, model payload] → the owner's one-time token from a `confirmation_required` model
  // answer (a guarded target: cost / data policy / large context; nothing was written). The
  // dialog's resend (`confirm_expensive_model: true`, the legacy handshake every Desktop surface
  // speaks) carries it as `payload.confirm`. It is spent by the owner's ANSWER to that resend,
  // not by sending it: until an answer (applied, refused, re-refused with a fresh token) arrives,
  // `sent` keeps the token so a retry after a lost reply re-sends the exact same token-bearing
  // request (same retained request id) instead of a new unconfirmed mutation. A re-refusal
  // replaces it with an unsent token, so it is never auto-confirmed.
  private modelConfirmations = new Map<string, { token: string; sent: boolean }>()

  failure(params: Record<string, unknown>, error: unknown) {
    const reason = (error as { data?: { reason?: string } })?.data?.reason

    // The owner answered (a typed refusal): the confirmed request's outcome is known. A transport
    // failure carries no reason; its outcome is unknown and the token stays for the exact retry.
    if (typeof reason === 'string') {
      this.settleModelConfirmation(params)
    }

    // Any typed refusal of a rewind wrote nothing. A derived /undo or /retry is retained by operation
    // alone, so keeping the refused request would resend it for every later /undo instead of deriving.
    if (typeof reason === 'string' && params.operation === 'rewind') {
      this.forgetMutation(params)
    }

    // A typed busy refusal of a rewind or a move wrote nothing either; the retry is a new request at the new revision.
    if (
      reason !== 'revision_conflict' &&
      reason !== 'stale_generation' &&
      !(reason === 'session_busy' && ['rewind', 'workspace'].includes(params.operation as string))
    ) {
      return
    }

    // A confirmed fence refusal did not mutate. Ambiguous transport failures keep
    // the original revision/id so a retry cannot overwrite another user's edit.
    this.forgetMutation(params)
    // The owner names its current fences on the refusal: adopt them so the next attempt presents them.
    const data = (error as { data?: { revision?: unknown; execution_generation?: unknown } })?.data
    const owner = canonicalSessionKey(params.session_id, params.profile)

    // An older request's refusal can arrive after a newer edit committed: never move the fences backwards.
    if (typeof data?.revision === 'number' && data.revision > (this.revisions.get(owner) ?? -1)) {
      this.revisions.set(owner, data.revision)
    }

    if (
      typeof data?.execution_generation === 'number' &&
      data.execution_generation > (this.generations.get(owner) ?? -1)
    ) {
      this.generations.set(owner, data.execution_generation)
    }
  }

  // A fence refusal of a value pick (model / reasoning) that named fresh fences is retried once at them:
  // re-applying the same value is harmless, unlike a CAS edit of shared metadata. Only while it is still the
  // session's current retained request: a newer pick retired it, and re-preparing it would undo that pick.
  retryableFence(params: Record<string, unknown>, error: unknown): boolean {
    const data = (error as { data?: { reason?: unknown; revision?: unknown } })?.data
    const current = [...this.mutations.values()].some(mutation => mutation.request_id === params.request_id)

    return (
      current &&
      ['model', 'reasoning'].includes(params.operation as string) &&
      typeof data?.revision === 'number' &&
      (data.reason === 'revision_conflict' || data.reason === 'stale_generation')
    )
  }

  // Wire method for a prepared request: composer metadata, branch and the
  // typed `/model <name>` / `/branch [title]` / `/compress [focus]` directives
  // and the dedicated compress action all travel as canonical `session.mutate`;
  // everything else keeps its name.
  wire(method: string, prepared: Record<string, unknown> = {}): string {
    if (MUTATION_METHODS.has(method)) {
      return 'session.mutate'
    }

    // The warm-cache re-attach: canonical attach is `session.resume`, which
    // rebinds the live event transport and returns the same snapshot shape.
    if (method === 'session.activate') {
      return 'session.resume'
    }

    return POLICY_EDIT_METHODS.has(method) && typeof prepared.operation === 'string' ? 'session.mutate' : method
  }

  // `derived`: the payload was derived from a snapshot (a typed /undo or /retry names the Nth newest user
  // turn). Keyed by operation alone, as Ink's mutateCanonical and the classic PreparedMutations do: a
  // retry after a lost reply re-presents the original target, never one re-derived from a transcript the
  // first attempt already rewound.
  private retainedMutation(
    sessionId: unknown,
    profile: unknown,
    operation: string,
    payload: Record<string, unknown>,
    withGeneration: boolean,
    derived = false
  ): Record<string, unknown> {
    const owner = canonicalSessionKey(sessionId, profile)
    const key = JSON.stringify([owner, operation, derived ? null : payload])
    this.retireMutations(owner, key)
    const retained = this.mutations.get(key)

    if (retained) {
      return retained
    }
    const revision = this.revisions.get(owner)

    if (revision === undefined) {
      throw new Error('Session revision unavailable; reopen the session before editing metadata')
    }
    const generation = this.generations.get(owner)

    if (withGeneration && generation === undefined) {
      throw new Error('Session execution identity unavailable; reconnect before this command')
    }

    const mutation: Record<string, unknown> = {
      session_id: sessionId,
      request_id: crypto.randomUUID(),
      expected_revision: revision,
      ...(withGeneration ? { expected_generation: generation } : {}),
      operation,
      payload
    }

    this.mutations.set(key, mutation)

    return mutation
  }

  prepare(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const prepared = this.preparePayload(method, params)

    // Routing belongs to the envelope, independently of the canonical payload.
    // Creation applies its own default-profile normalization.
    return method !== 'session.create' && params.profile !== undefined
      ? { ...prepared, profile: params.profile }
      : prepared
  }

  private fencedMutation(
    method: string,
    params: Record<string, unknown>,
    fenced: { operation: string; payload: Record<string, unknown> }
  ): Record<string, unknown> {
    // An edit / regenerate / restore names its target explicitly; a typed /undo or /retry derives it.
    const derived = method === 'session.rewind' && params.derived_target === true

    // A replay-only probe only ever re-presents the retained rewind (localAnswer handles the rest).
    if (derived && params.replay_only === true && !this.retainsDerivedRewind(params.session_id, params.profile)) {
      throw new Error('No retained rewind to replay')
    }

    return this.retainedMutation(params.session_id, params.profile, fenced.operation, fenced.payload, true, derived)
  }

  // A typed /undo or /retry whose reply was lost is still retained for its exact retry.
  private retainsDerivedRewind(sessionId: unknown, profile: unknown): boolean {
    return this.mutations.has(JSON.stringify([canonicalSessionKey(sessionId, profile), 'rewind', null]))
  }

  // A request this protocol answers without the owner, or undefined. A replay-only /undo or /retry
  // probe with no retained rewind owes nothing: no request is sent, and the command derives its target.
  localAnswer(method: string, params: Record<string, unknown>): Record<string, unknown> | undefined {
    if (
      method === 'session.rewind' &&
      params.replay_only === true &&
      !this.retainsDerivedRewind(params.session_id, params.profile)
    ) {
      return { session_id: params.session_id, status: 'nothing', rewound_count: 0 }
    }

    return undefined
  }

  // Drop the retained request the owner answered: its outcome is known, so no retry may resend it.
  private forgetMutation(params: Record<string, unknown>) {
    for (const [key, mutation] of this.mutations) {
      if (mutation.request_id === params.request_id) {
        this.mutations.delete(key)
      }
    }
  }

  // Only a session's very next control may retry an ambiguous one: any other verb retires the rest.
  private retireMutations(owner: string, keep?: string) {
    for (const key of [...this.mutations.keys()]) {
      if (key !== keep && JSON.parse(key)[0] === owner) {
        this.mutations.delete(key)
      }
    }
  }

  private preparePayload(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const mutation = this.prepareMutation(method, params)

    if (mutation) {
      return mutation
    }

    if (method === 'prompt.submit') {
      this.retireMutations(canonicalSessionKey(params.session_id, params.profile))
    }

    if (method === 'session.create') {
      return this.prepareCreate(params)
    }

    if (method === 'session.activate') {
      return {
        session_id: params.session_id,
        source: 'desktop',
        ...(params.profile ? { profile: params.profile } : {})
      }
    }

    // N93: a watch window's legacy `lazy` attach is not in the owner's resume contract (the owner
    // holds the agent; nothing to defer). It skipped the REST prefetch, so it takes the transcript.
    if (method === 'session.resume' && 'lazy' in params) {
      const { lazy: _lazy, ...resume } = params

      return { ...resume, omit_messages: false }
    }

    if (method === 'session.interrupt' || method === 'session.redirect' || method === 'session.steer') {
      const generation =
        params.execution_generation ?? this.generations.get(canonicalSessionKey(params.session_id, params.profile))

      if (typeof generation !== 'number') {
        throw new Error('Session execution identity unavailable; reconnect before controlling this turn')
      }

      return { ...params, session_id: params.session_id, execution_generation: generation }
    }

    if (method === 'prompt.resolve_unknown') {
      const lost = this.unknownAdmissions.get(canonicalSessionKey(params.admission_id, params.profile))

      if (!lost || lost.session_id !== params.session_id || lost.profile !== canonicalProfile(params.profile)) {
        throw new Error('Admission is not an unknown lost turn; reopen the session before acknowledging')
      }

      return { session_id: params.session_id, admission_id: params.admission_id, execution_generation: lost.generation }
    }

    if (method === 'approval.respond' || method === 'clarify.respond') {
      return this.preparePromptResponse(method, params)
    }

    return canonicalProducerFrame(method, params)
  }

  // Metadata, branch and typed slash directives that travel as canonical `session.mutate`; null otherwise.
  private prepareMutation(method: string, params: Record<string, unknown>): Record<string, unknown> | null {
    const field = ({ 'session.title': 'title', 'session.archive': 'archived' } as Record<string, string>)[method]

    if (field) {
      return this.retainedMutation(
        params.session_id,
        params.profile,
        field === 'title' ? 'rename' : 'archive',
        { [field]: params[field] },
        false
      )
    }

    // Branch and compress fence the execution generation like the slash directives.
    const fenced = (
      {
        'session.branch': () => ({ operation: 'branch', payload: branchBoundary(params) }),
        'session.compress': () => ({
          operation: 'compress',
          payload: params.focus_topic ? { focus: String(params.focus_topic) } : {}
        }),
        'session.rewind': () => ({ operation: 'rewind', payload: { target_message_id: params.target_message_id } })
      } as Record<string, () => { operation: string; payload: Record<string, unknown> }>
    )[method]?.()

    if (fenced) {
      return this.fencedMutation(method, params, fenced)
    }

    if (method === 'session.model') {
      const payload = {
        ...(params.payload as Record<string, unknown>),
        ...(typeof params.confirm === 'string' ? { confirm: params.confirm } : {})
      }

      return this.retainedMutation(params.session_id, params.profile, 'model', payload, true)
    }

    if (method === 'session.branch_stored' || method === 'session.branch_whole') {
      // The stored-parent form names the parent as `parent_session_id`; the live form as `session_id`.
      const parent = params.parent_session_id ?? params.session_id
      const payload = typeof params.title === 'string' && params.title ? { title: params.title } : {}

      return this.retainedMutation(parent, params.profile, 'branch', payload, true)
    }

    const policy = policyEdit(method, params)

    if (policy) {
      return this.retainedMutation(policy.session, params.profile, policy.operation, policy.payload, true)
    }

    if (method === 'config.set' && params.key === 'model') {
      const pick = pickerModelMutation(params.value)

      if (pick) {
        return this.retainedMutation(
          params.session_id,
          params.profile,
          'model',
          this.confirmedModel(params, pick),
          true
        )
      }
    }

    if (method === 'slash.exec') {
      const directive = slashMutation(String(params.command ?? ''))

      if (directive) {
        const payload =
          directive.operation === 'model' ? this.confirmedModel(params, directive.payload) : directive.payload

        return this.retainedMutation(params.session_id, params.profile, directive.operation, payload, true)
      }
    }

    return null
  }

  private modelConfirmationKey(params: Record<string, unknown>, payload: Record<string, unknown>): string {
    const { confirm: _token, ...target } = payload

    return JSON.stringify([canonicalSessionKey(params.session_id, params.profile), target])
  }

  // A user-confirmed resend of a guarded model target presents the owner's token. While that
  // confirmed request's outcome is unknown (reply lost), any retry of the same target is the same
  // confirmed intent and re-sends it verbatim: the token is part of the retained mutation's key,
  // so the retry keeps its request id and the owner resolves it as an exact retry.
  private confirmedModel(params: Record<string, unknown>, payload: Record<string, unknown>): Record<string, unknown> {
    const key = this.modelConfirmationKey(params, payload)
    const confirmation = this.modelConfirmations.get(key)

    if (!confirmation || !(params.confirm_expensive_model || confirmation.sent)) {
      return payload
    }
    confirmation.sent = true

    return { ...payload, confirm: confirmation.token }
  }

  // The owner answered a token-bearing model request: that token is spent.
  private settleModelConfirmation(params: Record<string, unknown>): void {
    const payload = params.payload as Record<string, unknown> | undefined

    if (params.operation !== 'model' || typeof payload?.confirm !== 'string') {
      return
    }
    const key = this.modelConfirmationKey(params, payload)

    if (this.modelConfirmations.get(key)?.token === payload.confirm) {
      this.modelConfirmations.delete(key)
    }
  }

  // The composer's per-session create options in the owner's create contract: the effort is the
  // frozen launch `reasoning`; a manual model pick keeps its `provider`; the speed pick is `service_tier` (main: an explicit tier wins, Fast
  // pins priority, Normal pins normal over a profile tier, omitted inherits); `cwd_explicit` is the
  // provenance that lets a named profile's own workspace beat an inherited cwd.
  private prepareCreate(input: Record<string, unknown>): Record<string, unknown> {
    const { reasoning_effort: effort, fast, service_tier: picked, ...params } = input
    const tier =
      typeof picked === 'string' && picked ? picked : fast === true ? 'priority' : fast === false ? 'normal' : undefined

    if (typeof effort === 'string' && effort) {
      params.reasoning = effort
    }

    if (tier) {
      params.service_tier = tier
    }

    if (typeof params.cwd_explicit !== 'boolean') {
      delete params.cwd_explicit
    }

    const allowed = new Set([
      'request_id',
      'source',
      'cwd',
      'model',
      'provider',
      'toolsets',
      'reasoning',
      'profile',
      'cols',
      'title',
      'hidden',
      'follow_profile_config',
      'service_tier',
      'cwd_explicit'
    ])
    const unsupported = Object.keys(params).filter(key => !allowed.has(key))

    if (unsupported.length) {
      throw new Error(`Canonical gateway does not support explicit session options: ${unsupported.join(', ')}`)
    }

    // RT-26: the picked profile always rides, `default` included: a socket launched as a named
    // profile (active-profile.json) reads an omitted profile as its own launch profile.
    const result = Object.fromEntries(
      Object.entries(params).filter(
        ([key, value]) =>
          [
            'request_id',
            'cwd',
            'model',
            'provider',
            'toolsets',
            'reasoning',
            'title',
            'hidden',
            'follow_profile_config',
            'service_tier',
            'cwd_explicit'
          ].includes(key) ||
          (key === 'profile' && typeof value === 'string' && value !== '')
      )
    )

    const key = JSON.stringify(result)
    const requestId = params.request_id ?? this.creates.get(key) ?? crypto.randomUUID()
    this.creates.set(key, String(requestId))

    return { ...result, request_id: requestId, source: 'gui' }
  }

  private preparePromptResponse(method: string, params: Record<string, unknown>): Record<string, unknown> {
    const id = String(params.prompt_id ?? params.request_id ?? '')
    const prompt = this.prompts.get(canonicalSessionKey(id, params.profile))
    const sessionId = params.session_id ?? prompt?.session_id

    if (
      !prompt ||
      prompt.session_id !== sessionId ||
      prompt.profile !== canonicalProfile(params.profile) ||
      prompt.execution_generation !== this.generations.get(canonicalSessionKey(sessionId, params.profile))
    ) {
      throw new Error('Prompt is stale or unavailable; reconnect before responding')
    }

    const field = method === 'approval.respond' ? 'choice' : 'answer'

    return {
      session_id: sessionId,
      execution_generation: prompt.execution_generation,
      prompt_id: id,
      [field]: params[field]
    }
  }

  private reconcileUnknownAdmissions(sid: string, profile: string, pending: Array<Record<string, unknown>>): void {
    for (const [id, lost] of this.unknownAdmissions) {
      if (lost.session_id === sid && lost.profile === profile) {
        this.unknownAdmissions.delete(id)
      }
    }

    for (const row of pending) {
      if (
        row?.status === 'unknown' &&
        typeof row.admission_id === 'string' &&
        typeof row.execution_generation === 'number'
      ) {
        this.unknownAdmissions.set(canonicalSessionKey(row.admission_id, profile), {
          session_id: sid,
          profile,
          generation: row.execution_generation
        })
      }
    }
  }

  // Turns advance the CAS revision without a session.updated event; the pending session.info fanout
  // is where a viewer learns the value its next mutation must present. N91: another viewer's
  // committed mutation broadcasts its receipt as session.updated: adopt it monotonically (a
  // replayed older receipt never rewinds) and settle this viewer's superseded retained edits.
  private adoptEventRevision(type: string, sid: string, profile: string, payload: Record<string, unknown>): void {
    const owner = canonicalSessionKey(sid, profile)
    const revision = payload.revision

    if (typeof revision !== 'number') {
      return
    }

    if (type === 'session.info') {
      this.revisions.set(owner, revision)
    }

    if (
      type === 'session.updated' &&
      (payload.session_id ?? sid) === sid &&
      revision > (this.revisions.get(owner) ?? -1)
    ) {
      this.revisions.set(owner, revision)
      this.retireSupersededMutations({ session_id: sid, profile }, revision)
    }
  }

  event(event: {
    type: string
    session_id?: string
    profile?: string
    execution_generation?: unknown
    payload?: unknown
  }) {
    const payload = event.payload as Record<string, unknown> | undefined

    if (!payload || !event.session_id) {
      return
    }
    const sid = event.session_id
    const profile = canonicalProfile(event.profile)
    const owner = canonicalSessionKey(sid, profile)

    if (Array.isArray(payload.pending)) {
      payload.pending_submissions = payload.pending.map(row => ({ ...row, user: row.text }))
    }

    const generation = event.execution_generation ?? payload.execution_generation

    if (typeof generation === 'number') {
      const current = this.generations.get(owner) ?? -1

      if (generation < current) {
        return
      }
      this.generations.set(owner, generation)
    }

    // C20: only a frame that passed the generation fence may retire a live Discard action.
    if (Array.isArray(payload.pending)) {
      this.reconcileUnknownAdmissions(sid, profile, payload.pending)
    }

    this.adoptEventRevision(event.type, sid, profile, payload)

    // A pick deferred during a turn asked at the boundary (a guarded target): keep the owner's token, so
    // the user's confirmed resend of that target carries it exactly like an idle pick's confirmation.
    if (
      event.type === 'session.updated' &&
      payload.operation === 'model' &&
      payload.status === 'confirmation_required' &&
      typeof payload.confirm === 'string' &&
      payload.payload &&
      typeof payload.payload === 'object'
    ) {
      this.modelConfirmations.set(
        this.modelConfirmationKey({ session_id: sid, profile }, payload.payload as Record<string, unknown>),
        { token: payload.confirm, sent: false }
      )
    }

    if (typeof payload.prompt_id === 'string') {
      if (event.type.endsWith('.settled')) {
        this.prompts.delete(canonicalSessionKey(payload.prompt_id, profile))

        return
      }

      if (event.type === 'approval.request' || event.type === 'clarify.request') {
        payload.request_id = payload.prompt_id
        this.prompts.set(canonicalSessionKey(payload.prompt_id, profile), { ...payload, session_id: sid, profile })
      }
    }
  }

  result(method: string, params: Record<string, unknown>, value: any): any {
    if (!value || typeof value !== 'object') {
      return value
    }

    const mutation =
      MUTATION_METHODS.has(method) || (POLICY_EDIT_METHODS.has(method) && typeof params.operation === 'string')
    this.adoptRevision(params, value, mutation)

    if (mutation) {
      return this.mutationReceipt(method, params, value)
    }

    if (method === 'session.create') {
      for (const [key, id] of this.creates) {
        if (id === params.request_id) {
          this.creates.delete(key)
        }
      }
    }

    if (method === 'prompt.submit' || method === 'prompt.resolve_unknown') {
      return this.admissionReceipt(method, params, value)
    }

    if (method === 'session.resume' || method === 'session.create' || method === 'session.activate') {
      return this.snapshotResult(value, params.profile)
    }

    if (method === 'session.events.since') {
      for (const event of value.events ?? []) {
        this.event({ ...event, profile: canonicalProfile(params.profile) })
      }
    }

    return value
  }

  // An exact-retry mutation receipt replays its ORIGINAL revision; never move the CAS value backwards.
  private adoptRevision(params: Record<string, unknown>, value: any, mutation: boolean): void {
    if (typeof value.session_id !== 'string' || typeof value.revision !== 'number') {
      return
    }
    const owner = canonicalSessionKey(value.session_id, params.profile)

    if (!mutation || value.revision >= (this.revisions.get(owner) ?? -1)) {
      this.revisions.set(owner, value.revision)
    }
  }

  private admissionReceipt(method: string, params: Record<string, unknown>, value: any): any {
    if (method === 'prompt.submit') {
      // TUI-26: a bare stop phrase typed into a LOCAL voice chat ended that chat on the owner and
      // admitted nothing. An allowlisted no-admission variant: only these fields pass through.
      if (value.voice_stopped === true && value.admission_id === undefined && value.ref === undefined) {
        return {
          voice_stopped: true,
          session_id: params.session_id,
          submission_id: params.submission_id ?? params.input_id
        }
      }

      if (value.ref?.session_id !== params.session_id || typeof value.admission_id !== 'string') {
        throw new Error('Canonical admission receipt destination mismatch')
      }

      for (const observe of admissionObservers) {
        observe(value.admission_id, String(value.status ?? ''))
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
    if (value.session_id !== params.session_id) {
      throw new Error('Metadata receipt destination mismatch')
    }

    this.forgetMutation(params)
    this.settleModelConfirmation(params)
    this.retireSupersededMutations(params, value.revision)

    if (params.operation === 'model' && value.status === 'confirmation_required' && typeof value.confirm === 'string') {
      // The owner wrote nothing: answer in the legacy handshake (`confirm_required` +
      // `confirm_message`) so the shared confirm dialog asks, and keep the token for its resend.
      this.modelConfirmations.set(this.modelConfirmationKey(params, params.payload as Record<string, unknown>), {
        token: value.confirm,
        sent: false
      })
      const refusal = { ...value, confirm_required: true }

      return method === 'slash.exec' ? { ...refusal, type: 'exec', output: value.confirm_message } : refusal
    }

    if (BRANCH_METHODS.has(method)) {
      return {
        ...value,
        session_id: value.branched_session_id,
        stored_session_id: value.branched_session_id,
        parent_session_id: params.session_id,
        message_count: value.copied_messages
      }
    }

    if (method === 'slash.exec') {
      return { ...value, type: 'exec', output: mutationSummary(params.operation as string, value) }
    }

    // A pick on a running session is recorded for the next turn (the legacy `deferred` answer).
    if (params.operation === 'model' && value.status === 'deferred') {
      return { ...value, ok: true, deferred: true }
    }

    // The legacy config.set answer every reasoning caller reads (`result.value`).
    return params.operation === 'reasoning' ? { ...value, ok: true, value: value.effort } : { ...value, ok: true }
  }

  // An acknowledged mutation at revision R settles every older retained value-setting request of
  // the same owner: it either applied before R or can only be refused, so replaying its request
  // id would return a stale receipt for a later, equal edit. Its confirmed model token goes with
  // it. Branch/compress stay retained: a new request id there would execute a second time.
  private retireSupersededMutations(params: Record<string, unknown>, revision: unknown): void {
    if (typeof revision !== 'number') {
      return
    }
    const owner = canonicalSessionKey(params.session_id, params.profile)

    for (const [key, mutation] of this.mutations) {
      const [keyOwner, operation] = JSON.parse(key)

      if (keyOwner === owner && VALUE_OPERATIONS.has(operation) && (mutation.expected_revision as number) < revision) {
        this.mutations.delete(key)
        this.settleModelConfirmation({ ...mutation, profile: params.profile })
      }
    }
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

    const open_requests = prompts.map((prompt: Record<string, unknown>) => ({
      id: prompt.prompt_id,
      method: prompt.kind,
      params: canonicalPromptParams(prompt.kind, sid, prompt)
    }))

    return {
      ...value,
      open_requests,
      pending_approval: prompts.find((p: any) => p.kind === 'approval'),
      pending_clarify: prompts.find((p: any) => p.kind === 'clarify'),
      info: {
        ...value.info,
        stored_session_id: value.stored_session_id,
        pending_submissions: value.pending_submissions,
        replay_epoch: value.replay_epoch,
        last_sequence: value.last_sequence,
        execution_generation: value.execution_generation,
        running: value.running
      }
    }
  }

  // The canonical compress receipt carries counts, not the retained transcript;
  // the compress action repaints only from `messages`, so resume the session
  // through the normal request path (which also re-primes revision/generation).
  async settle(
    method: string,
    params: Record<string, unknown>,
    value: any,
    request: (method: string, params: Record<string, unknown>) => Promise<any>
  ): Promise<any> {
    if (method !== 'session.compress' && !(method === 'slash.exec' && params.operation === 'compress')) {
      return value
    }

    // `--preview` in the focus argument is a read-only report: no transcript changed.
    if (value?.status === 'preview') {
      return { ...value, host_ack: { output: (value.lines as string[]).join('\n') } }
    }
    const resumed = await request('session.resume', {
      session_id: params.session_id,
      ...(params.profile !== undefined ? { profile: params.profile } : {})
    })

    return {
      ...value,
      messages: resumed.messages,
      info: resumed.info,
      host_ack: { output: `compressed context: ${value.message_count} messages retained` }
    }
  }
}
