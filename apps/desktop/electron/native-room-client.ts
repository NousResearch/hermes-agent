import { mintLocalGatewayTicket, routedGatewayEndpoint } from './local-gateway'
import type { GatewayEndpoint } from './local-gateway'
import { RoomSetupError } from './room-setup-store'

/** One immutable native connection owns the complete setup exchange. Never
 * reconnect a pending mutation through a reused registry slot. */
export async function nativeRoomClient(descriptor: { baseUrl: string; gatewayEndpoint?: GatewayEndpoint }, profile: string) {
  if (!descriptor.gatewayEndpoint) {throw new RoomSetupError('canonical_connection_required')}
  const ticket = await mintLocalGatewayTicket(routedGatewayEndpoint(descriptor.gatewayEndpoint, profile,
    descriptor.gatewayEndpoint.profile_id))
  const socket = new WebSocket(descriptor.baseUrl.replace(/^http/, 'ws') + '/api/ws',
    ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket])
  let serial = 0
  const pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>()
  const fail = () => {
    for (const entry of pending.values()) {clearTimeout(entry.timer); entry.reject(new RoomSetupError('setup_connection_lost'))}
    pending.clear()
  }
  socket.addEventListener('close', fail)
  socket.addEventListener('error', fail)
  socket.addEventListener('message', event => {
    let value
    try {value = JSON.parse(String(event.data))} catch {return}
    const entry = pending.get(value.id)
    if (!entry) {return}
    pending.delete(value.id); clearTimeout(entry.timer)
    if (value.error) {
      const reason = value.error.data?.reason
      entry.reject(new RoomSetupError(typeof reason === 'string' && /^[a-z_]{1,80}$/.test(reason) ? reason : 'setup_refused'))
    } else {entry.resolve(value.result)}
  })
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {socket.close(); reject(new RoomSetupError('setup_connection_lost'))}, 15000)
      socket.addEventListener('open', () => {clearTimeout(timer); resolve()}, { once: true })
      socket.addEventListener('error', () => {clearTimeout(timer); reject(new RoomSetupError('setup_connection_lost'))}, { once: true })
    })
  } catch (error) {socket.close(); throw error}
  return {
    close: () => {socket.close(); fail()},
    async request(method: string, params: Record<string, unknown> = {}): Promise<any> {
      if (socket.readyState !== WebSocket.OPEN) {throw new RoomSetupError('setup_connection_lost')}
      const id = ++serial
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {pending.delete(id); reject(new RoomSetupError('setup_connection_lost'))}, 30000)
        pending.set(id, { resolve, reject, timer })
        socket.send(JSON.stringify({ jsonrpc: '2.0', id, method, params: { ...params, profile } }))
      })
    }
  }
}
