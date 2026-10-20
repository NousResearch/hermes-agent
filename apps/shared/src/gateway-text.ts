/** User-facing text of the shared gateway client, keyed by the Ink TUI's catalog keys. Ink passes its
 *  own `t` (so its locale packs translate these); any other client gets this English. */

const EN = {
  'canonical.launch.noTuiPolicy': () => 'gateway does not support tui session policy; update/restart the gateway',
  'canonical.launch.unsupportedOptions': (options: unknown) => `gateway does not support TUI launch options: ${options}`,
  'libText.gateway.attachUrlChanged': () => 'gateway attach url changed',
  'libText.gateway.closed': () => 'gateway closed',
  'libText.gateway.exited': () => 'gateway exited',
  'libText.gateway.exitedWithCode': (code: unknown) => `gateway exited (${code})`,
  'libText.gateway.restarting': () => 'gateway restarting',
  'libText.gateway.websocketClosed': () => 'gateway websocket closed',
  'libText.gateway.websocketClosedDuringConnect': (code: unknown) => `gateway websocket closed (${code}) during connect`,
  'libText.gateway.websocketClosedWithCode': (code: unknown) => `gateway websocket closed (${code})`,
  'libText.gateway.websocketConnectionFailed': () => 'gateway websocket connection failed',
  'libText.gateway.websocketStartupFailed': () => 'gateway websocket startup failed',
  'libText.gateway.websocketUnavailable': () => 'gateway websocket unavailable'
} satisfies Record<string, (arg: unknown) => string>

export type GatewayTextKey = keyof typeof EN

export type GatewayText = (key: GatewayTextKey, ...args: unknown[]) => string

export const gatewayText: GatewayText = (key, ...args) => EN[key](args[0])
