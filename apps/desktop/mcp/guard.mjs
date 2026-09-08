/**
 * Isolation guard for the Desktop Debug MCP server.
 *
 * A debug MCP run must target an isolated sandbox (its own HERMES_HOME), never
 * the operator's real data. We do NOT trust the caller's declaration alone:
 * the connected renderer exposes a per-instance descriptor
 * (`__DEBUG_MCP_INSTANCE__`, emitted by the Electron main process that opened
 * the CDP port) carrying the *realized* data root. `assertTargetAttested`
 * reads that from the target and refuses unless it matches EXPECTED_HOME.
 *
 * If the target proves a different home (or exposes no descriptor at all), the
 * tool is refused. This is target-derived authority, not a second
 * caller-supplied coordinate — so a real dev Desktop on ~/.hermes cannot be
 * reached by declaring a fake EXPECTED_HOME.
 */

import path from 'node:path'
import os from 'node:os'

export function canon(p) {
  try {
    return path.resolve(p)
  } catch {
    return p
  }
}

/**
 * @param {object} cdp - connected CDP handle with an `eval(expr)` method
 * @param {{ expectedHome?: string, defaultHome?: string }} opts
 * @returns {Promise<void>} resolves when the target is proven to be the sandbox
 * @throws {Error} with a REFUSED message when the boundary is not satisfied
 */
export async function assertTargetAttested(cdp, opts) {
  const EXPECTED_HOME = opts.expectedHome || ''
  const DEFAULT_HOME = opts.defaultHome || ''

  if (!EXPECTED_HOME) {
    throw new Error(
      'REFUSED: DESKTOP_DEBUG_MCP_EXPECTED_HOME is not set. The debug MCP ' +
        'server will not touch a desktop instance unless you declare which ' +
        'isolated HERMES_HOME it is running against, AND the connected target ' +
        'proves it runs against that same home. Launch with ' +
        'DESKTOP_DEBUG_MCP_EXPECTED_HOME=/tmp/your-sandbox-home and start the ' +
        'desktop instance with the same HERMES_HOME. Never point this at your ' +
        'real ~/.hermes.'
    )
  }

  // Read the REALIZED target, not a second declaration.
  let realized
  try {
    const d = await cdp.eval(
      'globalThis.__DEBUG_MCP_INSTANCE__ ? globalThis.__DEBUG_MCP_INSTANCE__.dataRoot : null'
    )
    realized = typeof d === 'string' ? d : d?.dataRoot ?? null
  } catch {
    realized = null
  }

  if (!realized) {
    throw new Error(
      'REFUSED: the connected target exposes no debug-instance descriptor ' +
        '(__DEBUG_MCP_INSTANCE__). Cannot prove it is the isolated sandbox you ' +
        'declared. Check that the desktop instance was launched in dev mode with ' +
        'the CDP port open and DESKTOP_DEBUG_MCP_EXPECTED_HOME matching its home.'
    )
  }

  // Isolation policy, separate from identity: even when caller and target
  // agree, the protected operator home is never an acceptable debug target.
  // Protected = the server's own default home (HERMES_HOME env or ~/.hermes)
  // plus the literal ~/.hermes of the OS user (covers a containerized server
  // with no HERMES_HOME env).
  const PROTECTED = new Set(
    [opts.defaultHome, path.join(os.homedir(), '.hermes')].filter(Boolean).map(canon)
  )
  if (PROTECTED.has(canon(realized))) {
    throw new Error(
      `REFUSED: the target's realized home (${realized}) is the protected ` +
        "operator home. The debug MCP server never acts on a desktop instance " +
        'running against the operator\'s real profile — launch an isolated ' +
        'sandbox instead (e.g. HERMES_HOME=/tmp/your-sandbox-home).'
    )
  }

  if (canon(realized) !== canon(EXPECTED_HOME)) {
    throw new Error(
      `REFUSED: connected target home (${realized}) does not match declared ` +
        `EXPECTED_HOME (${EXPECTED_HOME}). The MCP server will not act on a ` +
        'target it cannot prove is your isolated sandbox.'
    )
  }
}
