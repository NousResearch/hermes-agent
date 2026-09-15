import type { GatewayEventName, ServerRequestMethod } from './gateway-events'
import type { GatewayEvent } from './json-rpc-gateway'

export const VOICE_SUPERVISOR_GATEWAY_EVENTS = ['tool.start'] as const satisfies readonly GatewayEventName[]

/** Blocking prompts the backend raises as server→client requests; the voice
 *  cannot answer them, so it tells the user to look at the app. */
export const VOICE_SUPERVISOR_SERVER_REQUESTS = [
  'approval',
  'sudo',
  'secret',
  'clarify'
] as const satisfies readonly ServerRequestMethod[]

export type VoiceSupervisorSurfaceEvent =
  | { kind: 'narrate-tool'; name: string }
  | { kind: 'notify'; text: string }

const BLOCKING_REQUEST_NOTICES: Record<(typeof VOICE_SUPERVISOR_SERVER_REQUESTS)[number], string> = {
  approval: 'Hermes needs your approval in the app.',
  sudo: 'Hermes needs your administrator password in the app.',
  secret: 'Hermes needs a credential in the app.',
  clarify: 'Hermes has a question for you in the app.'
}

export function voiceSupervisorSurfaceEvent(event: GatewayEvent): VoiceSupervisorSurfaceEvent | null {
  if (event.type !== 'tool.start') {
    return null
  }

  const payload =
    event.payload && typeof event.payload === 'object' ? (event.payload as Record<string, unknown>) : {}

  const name = typeof payload.name === 'string' ? payload.name.trim() : ''

  return name ? { kind: 'narrate-tool', name } : null
}

export function voiceSupervisorSurfaceRequest(
  method: string
): Extract<VoiceSupervisorSurfaceEvent, { kind: 'notify' }> | null {
  const text = (BLOCKING_REQUEST_NOTICES as Record<string, string | undefined>)[method]

  return text ? { kind: 'notify', text } : null
}
