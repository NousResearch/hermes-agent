export interface DesktopDeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

export type DesktopDeepLinkAction = { kind: 'blueprint'; command: string } | { kind: 'session'; sessionId: string }

/**
 * Convert an OS-level hermes:// payload into a narrow renderer action.
 * Unknown link kinds stay inert instead of being interpreted as commands.
 */
export function desktopDeepLinkAction(payload?: DesktopDeepLinkPayload | null): DesktopDeepLinkAction | null {
  if (!payload?.name) {
    return null
  }

  if (payload.kind === 'session') {
    return { kind: 'session', sessionId: payload.name }
  }

  if (payload.kind !== 'blueprint') {
    return null
  }

  const slots = Object.entries(payload.params || {})
    .map(([key, value]) => {
      const serialized = /\s/.test(value) ? `"${value.replace(/"/g, '\\"')}"` : value

      return `${key}=${serialized}`
    })
    .join(' ')

  return {
    kind: 'blueprint',
    command: `/blueprint ${payload.name}${slots ? ' ' + slots : ''}`
  }
}
