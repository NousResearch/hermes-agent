export const CLERK_SESSION_CHANGED_KIND = 'unitalk.clerk.session.changed'
export const CLERK_SESSION_SYNC_CHANNEL = 'unitalk-clerk-session'

export interface ClerkSessionChangedMessage {
  kind: typeof CLERK_SESSION_CHANGED_KIND
  organizationId: null | string
  source: string
}

export interface ClerkSessionBus {
  close: () => void
  send: (message: ClerkSessionChangedMessage) => void
  subscribe: (listener: (message: unknown) => void) => () => void
}

let activeBus: ClerkSessionBus | null = null

export function setActiveClerkSessionBus(bus: ClerkSessionBus | null): void {
  activeBus = bus
}

export function sendClerkSessionChanged(message: ClerkSessionChangedMessage): void {
  activeBus?.send(message)
}

export function isClerkSessionChangedMessage(value: unknown): value is ClerkSessionChangedMessage {
  if (!value || typeof value !== 'object') {
    return false
  }

  const message = value as Partial<ClerkSessionChangedMessage>

  return (
    message.kind === CLERK_SESSION_CHANGED_KIND &&
    typeof message.source === 'string' &&
    (message.organizationId === null || typeof message.organizationId === 'string')
  )
}

export function createClerkSessionBus(): ClerkSessionBus | null {
  const relay = typeof window !== 'undefined' ? window.hermesDesktop?.windowRelay : undefined

  if (relay?.send && relay.onMessage) {
    let unsubscribe: (() => void) | null = null

    return {
      close: () => {
        unsubscribe?.()
        unsubscribe = null
      },
      send: message => relay.send(message),
      subscribe: listener => {
        unsubscribe = relay.onMessage(listener)

        return () => {
          unsubscribe?.()
          unsubscribe = null
        }
      }
    }
  }

  if (typeof BroadcastChannel === 'undefined') {
    return null
  }

  const channel = new BroadcastChannel(CLERK_SESSION_SYNC_CHANNEL)

  return {
    close: () => channel.close(),
    send: message => channel.postMessage(message),
    subscribe: listener => {
      const onMessage = (event: MessageEvent<unknown>) => listener(event.data)

      channel.addEventListener('message', onMessage)

      return () => channel.removeEventListener('message', onMessage)
    }
  }
}
