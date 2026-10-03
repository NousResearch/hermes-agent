export interface RemoteNative {
  remote(options: {
    sessionId: string
    action: string
    arguments: Record<string, unknown>
  }): Promise<Record<string, unknown>>
}
export interface RemoteDevice {
  id: string
  name: string
  platform: string
  role: 'origin' | 'target'
  online: boolean
}
export type DeviceAction = 'list' | 'origin' | 'target' | 'revoke'

// The caller supplies one captured socket. Never reconnect or replay enrollment.
export async function deviceCommand(
  request: (method: string, params: Record<string, unknown>, timeout: number) => Promise<unknown>,
  sessionId: string,
  action: DeviceAction
): Promise<{ devices?: RemoteDevice[]; revoked?: boolean }> {
  if (!sessionId) throw new Error('Open a chat on this gateway first.')
  const result = (await request(
    'command.dispatch',
    { name: 'desktop-devices', arg: action, session_id: sessionId },
    120000
  )) as { type?: string; output?: string }
  if (result?.type !== 'plugin' || typeof result.output !== 'string')
    throw new Error('The cross-device plugin command is unavailable.')
  const value = JSON.parse(result.output)
  if (!value || typeof value !== 'object' || value.error)
    throw new Error(value?.error || 'Invalid device enrollment result.')
  return value
}

export async function revokeRemoteAccess(native: RemoteNative, sessionId: string, cleanup: () => Promise<unknown>) {
  // Clear receiving credentials AND originating conversation grants before any network call.
  await native.remote({ sessionId, action: 'revoke', arguments: {} })
  try {
    await cleanup()
    return { gatewayConfirmed: true }
  } catch {
    return { gatewayConfirmed: false }
  }
}
