import type { GatewayEvent } from '@hermes/shared'

// A profile name received over the wire is descriptive, not an ownership
// claim: primary and arbitrary synthetic events can carry one. The secondary
// socket closure is the authority for profile-only pool routes, so stamp its
// outgoing copy with a non-serializable marker before it crosses registry
// fan-in. JSON-RPC payloads cannot forge a Symbol property.
const secondaryProfileOwner = Symbol('secondaryProfileOwner')

type SecondaryProfileOwnedEvent = GatewayEvent & {
  [secondaryProfileOwner]?: string
}

export function stampSecondaryProfileOwner(event: GatewayEvent, profile: string): GatewayEvent {
  const scoped = { ...event, profile } as SecondaryProfileOwnedEvent

  Object.defineProperty(scoped, secondaryProfileOwner, {
    configurable: false,
    enumerable: false,
    value: profile,
    writable: false
  })

  return scoped
}

export function secondaryProfileOwnerForEvent(event: GatewayEvent): string | undefined {
  const profile = (event as SecondaryProfileOwnedEvent)[secondaryProfileOwner]

  return typeof profile === 'string' && profile.trim() ? profile.trim() : undefined
}
