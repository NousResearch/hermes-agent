export type DesktopStateDomain =
  | 'appearance'
  | 'config'
  | 'cron'
  | 'mcp'
  | 'menu-bar-companion'
  | 'menu-bar-transparency'
  | 'model'
  | 'pet'
  | 'skills'
  | 'translucency'

export type DesktopStateSyncMessage =
  | { profile: string; type: 'active-profile' }
  | { type: 'active-profile-request' }
  | { domain: DesktopStateDomain; profile?: string; type: 'changed'; value?: unknown }

interface DesktopStateChannel {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void
  postMessage(message: DesktopStateSyncMessage): void
  removeEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void
}

interface ChangeOptions {
  profile?: string
  value?: unknown
}

export interface NamedEnabledSyncValue {
  enabled: boolean
  name: string
}

export function parseNamedEnabledSyncValue(value: unknown): NamedEnabledSyncValue | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as Record<string, unknown>

  return typeof record.name === 'string' && typeof record.enabled === 'boolean'
    ? { enabled: record.enabled, name: record.name }
    : null
}

export function parseCronSyncValue(
  value: unknown
): (Record<string, unknown> & { enabled: boolean; id: string }) | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as Record<string, unknown>

  return typeof record.id === 'string' && typeof record.enabled === 'boolean'
    ? ({ ...record, enabled: record.enabled, id: record.id } as Record<string, unknown> & {
        enabled: boolean
        id: string
      })
    : null
}

const CHANNEL_NAME = 'hermes:desktop-state'

function isDesktopStateSyncMessage(value: unknown): value is DesktopStateSyncMessage {
  if (!value || typeof value !== 'object') {
    return false
  }

  const message = value as Record<string, unknown>

  if (message.type === 'active-profile-request') {
    return true
  }

  if (message.type === 'active-profile') {
    return typeof message.profile === 'string'
  }

  return message.type === 'changed' && typeof message.domain === 'string'
}

export function createDesktopStateSyncBus(channel: DesktopStateChannel | null) {
  return {
    broadcastActiveProfile(profile: string): void {
      channel?.postMessage({ type: 'active-profile', profile })
    },
    broadcastChange(domain: DesktopStateDomain, options: ChangeOptions = {}): void {
      const message: DesktopStateSyncMessage = {
        type: 'changed',
        domain,
        ...(options.profile ? { profile: options.profile } : {}),
        ...('value' in options ? { value: options.value } : {})
      }

      channel?.postMessage(message)
    },
    requestActiveProfile(): void {
      channel?.postMessage({ type: 'active-profile-request' })
    },
    subscribe(handler: (message: DesktopStateSyncMessage) => void): () => void {
      if (!channel) {
        return () => {}
      }

      const listener = (event: MessageEvent<unknown>) => {
        if (isDesktopStateSyncMessage(event.data)) {
          handler(event.data)
        }
      }

      channel.addEventListener('message', listener)

      return () => channel.removeEventListener('message', listener)
    }
  }
}

const channel =
  typeof window === 'undefined' || typeof window.BroadcastChannel === 'undefined'
    ? null
    : new window.BroadcastChannel(CHANNEL_NAME)

const bus = createDesktopStateSyncBus(channel)

export const broadcastActiveDesktopProfile = bus.broadcastActiveProfile
export const broadcastDesktopStateChange = bus.broadcastChange
export const onDesktopStateSync = bus.subscribe
export const requestActiveDesktopProfile = bus.requestActiveProfile
