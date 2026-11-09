/** `hermes-gateway-v1` request/result translation for a client written against the stdio-era
 *  method names (Ink, and clients built on `./node/gateway-client`). Browser-safe. */
import type { CanonicalCreateParams } from './gateway-contract.generated.js'
import { type GatewayText, gatewayText } from './gateway-text.js'
import { randomUuid } from './session-http-mutations.js'

export interface SharedControl { session_id: string; execution_generation: number; prompt_id: string }
export const sharedControlParams = (prompt?: { sharedControl?: SharedControl } | null) => prompt?.sharedControl ?? {}

export interface CreationContract { sources: string[]; parameters: string[] }

/** A `session.create` policy label the gateway may offer (`runtime.describe` `session_create.sources`). */
export type LaunchSource = NonNullable<CanonicalCreateParams['source']>

/** How a client names itself on `session.create`: a label for attribution, never a grant. A gateway that takes
 *  it refuses `session.create` unless `name` is 1-64 characters of a-z, 0-9, `.`, `_` or `-` starting with a
 *  letter or digit, `version` (optional) is 1-64 printable ASCII characters without spaces, and there are no
 *  other keys. */
export interface ClientIdentity { name: string; version?: string }

/** Who is launching: the policy label (default `'tui'`, the terminal-client policy) and, optionally,
 *  the client's own name. */
export interface LaunchIdentity { source?: LaunchSource; client?: ClientIdentity }

export function canonicalRequest(
  method: string,
  original: Record<string, unknown>,
  contract?: CreationContract,
  text: GatewayText = gatewayText,
  identity: LaunchIdentity = {}
) {
  const params = { ...original }

  if (method === 'session.create') {
    const source = identity.source ?? 'tui'
    // Ink launches as 'tui' with no client name and keeps its TUI wording; any other launch reads neutral.
    const neutral = source !== 'tui' || identity.client !== undefined
    delete params.cols
    params.source = source
    params.request_id ??= randomUuid()

    if (!contract?.sources.includes(source)) {
      throw new Error(neutral ? text('canonical.launch.noSourcePolicy', source) : text('canonical.launch.noTuiPolicy'))
    }

    // The shell's environment is context, not a launch option: a gateway predating it runs the session
    // with its own environment, as before.
    if (!contract.parameters.includes('client_env')) { delete params.client_env }

    // The client's name is attribution, not a launch option: sent only to a gateway that takes it.
    if (identity.client) {
      if (contract.parameters.includes('client')) { params.client = { ...identity.client } } else { delete params.client }
    }

    const unsupported = Object.keys(params).filter(key => !contract.parameters.includes(key))

    if (unsupported.length) {
      throw new Error(text(neutral ? 'canonical.launch.unsupportedSessionOptions' : 'canonical.launch.unsupportedOptions',
        unsupported.join(', ')))
    }
  }

  if (['prompt.submit', 'session.steer', 'session.redirect'].includes(method) && params.submission_id) {
    params.input_id = params.submission_id
    delete params.submission_id
  }

  if (method === 'session.resume' || method === 'session.activate') {
    method = 'session.resume'
    // A rendering resume asks for main's display projection of the same rows (`display_messages`);
    // a metadata-only activation (`omit_messages`) never paints a transcript.
    const renders = !params.omit_messages
    delete params.cols
    delete params.omit_messages

    if (renders) { params.display_projection = true }
  }

  return { method, params }
}

export function canonicalResult(method: string, value: any, request: Record<string, unknown> = {}): any {
  if (!value || typeof value !== 'object') { return value }

  // The JSON-RPC response correlates the prepared input; admission_id is a
  // separate server-issued identity and must not be rewritten to that input ID.
  if (['prompt.submit', 'session.steer', 'session.redirect'].includes(method) && value.ref) {
    return { ...value, input_id: request.input_id, target_profile_home: value.ref.profile_id, target_session_id: value.ref.session_id }
  }

  if (['session.create', 'session.resume', 'session.activate'].includes(method)) {
    // Canonical snapshots contain stored conversation rows (`content`), not
    // the legacy TUI's display rows (`text`). Without this translation only
    // live events render, making reconnect depend on catching the final delta.
    const stored = Array.isArray(value.messages) ? value.messages.map((row: any) =>
      row && typeof row === 'object'
        ? { ...row, text: rowText(row.content), name: row.tool_name }
        : row) : value.messages

    // Renderers read `messages`: main's display projection when the owner sent it (compaction
    // handoffs stripped, /steer unwrapped, tool rows labelled), as the sidecar's resume did.
    // `stored_messages` keeps the raw rows (content + row_id) the rewind controls target.
    const { display_messages: display, ...rest } = value
    const messages = Array.isArray(display) ? display : stored

    return { ...rest, messages, stored_messages: stored, info: { ...value.info, stored_session_id: value.stored_session_id,
      execution_epoch: String(value.authority_epoch), execution_generation: value.execution_generation,
      running: value.running, pending_submissions: pendingSubmissions(value.pending) } }
  }

  return value
}

// A stored multipart row (`content` is a parts list) as display text: text parts joined, an image
// part as `[image]` (N94). `content` itself stays on the row for /undo targeting.
function rowText(content: unknown): unknown {
  if (!Array.isArray(content)) { return content }

  return content.map(part => typeof part === 'string' ? part : typeof part?.text === 'string' ? part.text
    : part?.type ? (String(part.type).includes('image') ? '[image]' : `[${part.type}]`) : '').filter(Boolean).join('\n')
}

// The authority's durable FIFO rows (`pending` on snapshots and session.info
// fanout) projected onto the client's receipt shape: one translation, so the
// queue panel, the receipt matcher and Desktop all read the same list.
export function pendingSubmissions(pending: unknown) {
  if (!Array.isArray(pending)) { return undefined }

  return pending.filter(row => row && typeof row === 'object').map(row => ({
    ...row, user: row.text, target_profile_home: row.ref?.profile_id, target_session_id: row.ref?.session_id }))
}

export function canonicalEvent<T extends { type: string; payload?: any }>(event: T): T {
  const pending = pendingSubmissions(event.payload?.pending)

  return pending ? { ...event, payload: { ...event.payload, pending_submissions: pending } } : event
}
