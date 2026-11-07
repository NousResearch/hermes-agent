/** User-facing text of the shared gateway client, in the Ink TUI catalog's shape: Ink's English catalog
 *  spreads these leaves in (`i18n/en/libText.ts`, `i18n/en/canonical.ts`) and passes its own `t`, so
 *  its locale packs translate them; any other client gets this English. Function leaves take
 *  positional args (`{0}` in a pack). */
export const gatewayTextEn = {
  canonical: {
    launch: {
      noTuiPolicy: 'gateway does not support tui session policy; update/restart the gateway',
      unsupportedOptions: (options: string) => `gateway does not support TUI launch options: ${options}`
    }
  },
  libText: {
    // Transport-drop reasons that reject in-flight RPCs and therefore surface in the transcript.
    gateway: {
      restarting: 'gateway restarting',
      exited: 'gateway exited',
      /** {0} = exit code */
      exitedWithCode: (code: string) => `gateway exited (${code})`,
      websocketUnavailable: 'gateway websocket unavailable',
      websocketConnectionFailed: 'gateway websocket connection failed',
      /** {0} = websocket close code */
      websocketClosedDuringConnect: (code: string) => `gateway websocket closed (${code}) during connect`,
      websocketClosed: 'gateway websocket closed',
      /** {0} = websocket close code */
      websocketClosedWithCode: (code: string) => `gateway websocket closed (${code})`,
      websocketStartupFailed: 'gateway websocket startup failed',
      attachUrlChanged: 'gateway attach url changed',
      closed: 'gateway closed'
    }
  }
}

export type GatewayTextKey =
  | `canonical.launch.${keyof typeof gatewayTextEn.canonical.launch}`
  | `libText.gateway.${keyof typeof gatewayTextEn.libText.gateway}`

export type GatewayText = (key: GatewayTextKey, ...args: unknown[]) => string

export const gatewayText: GatewayText = (key, ...args) => {
  const leaf = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)[part], gatewayTextEn)

  return typeof leaf === 'function' ? (leaf as (...values: unknown[]) => string)(...args) : String(leaf)
}
