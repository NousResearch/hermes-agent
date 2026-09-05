/**
 * CDP client lifecycle for the Desktop Debug MCP server.
 *
 * Thin wrapper over the shared perf-harness CDP client (scripts/perf/lib/cdp.mjs).
 * Exists so the connection lifecycle (discovery + open + console capture +
 * failure reset) is unit-testable without booting the MCP stdio server.
 *
 * Keeps the project's module-per-concern layout: server.mjs wires MCP,
 * dispatch.mjs routes tools, guard.mjs attests the target, cdp-client.mjs
 * owns the connection.
 */

import { discoverTarget, CDP } from '../scripts/perf/lib/cdp.mjs'

/**
 * Lazily connect to the dev renderer's CDP port, capturing renderer console
 * into `onConsole`. Caches the handle; `invalidate()` must be called on
 * socket close so the next call rediscovers cleanly.
 *
 * NOTE (BUG #2, unfixed in this first cut): if CDP.connect throws, `handle`
 * is left in a half-state and the next call re-runs discovery (3s timeout)
 * instead of retrying quickly; a partially-opened socket may also leak.
 *
 * @param {object} opts
 * @param {number} opts.port
 * @param {string} opts.match
 * @param {(entry: {level:string,text:string,t:number}) => void} opts.onConsole
 * @param {typeof discoverTarget} [opts.discoverTargetImpl]
 * @param {typeof CDP} [opts.CDPImpl]
 */
export const isConnectablePage = (t) =>
  t && t.type === 'page' && typeof t.webSocketDebuggerUrl === 'string'

/**
 * Same URL filter discoverTarget applies when picking "our" page — one source
 * of truth so status() can never report a target the client would refuse
 * (Bugbot P2: status counted any connectable page, client required the
 * --match substring, so preflight said alive while the next tool failed).
 * match === undefined/empty keeps discovery's permissive fallback semantics.
 */
export const matchesTarget = (url, match) =>
  !match || String(url).includes(match)

export function createCdpClient({ port, match, onConsole, discoverTargetImpl = discoverTarget, CDPImpl = CDP }) {
  let handle = null

  async function connect() {
    if (handle) return handle

    try {
      await discoverTargetImpl({ port, match, timeoutMs: 3000 })
    } catch (e) {
      handle = null
      throw new Error(
        `No CDP target on :${port}. The debug port only exists for DEV runs ` +
          '(packaged builds never open it). Either ask the user to start the dev ' +
          "server (`cd apps/desktop && npm run dev`) or launch an isolated probe " +
          'instance (see apps/desktop/mcp/README.md).'
      )
    }

    try {
      handle = await CDPImpl.connect({ port, match })
    } catch (e) {
      // BUG #2 fix: reset to null so the next call rediscovers cleanly
      // instead of caching a half-open socket or re-hitting a 3s timeout.
      handle = null
      throw e
    }

    // A1: when the socket closes (renderer reload / window close), drop the
    // cached handle so the next tool call rediscovers instead of failing on a
    // dead socket until the server restarts. Identity guard: a stale socket's
    // late close event must never clear a NEWER handle.
    const thisHandle = handle
    handle.ws?.addEventListener?.('close', () => {
      if (handle === thisHandle) handle = null
    })

    handle.on('Runtime.consoleAPICalled', (p) => {
      onConsole({
        level: p.type,
        text: (p.args || []).map((a) => a.value ?? a.description ?? '').join(' ').slice(0, 200),
        t: Date.now()
      })
    })

    return handle
  }

  function invalidate() {
    handle = null
  }

  return { connect, invalidate, get handle() { return handle } }
}
