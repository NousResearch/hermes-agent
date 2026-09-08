/**
 * Tool dispatch for the Desktop Debug MCP server.
 *
 * Extracted from server.mjs so the dispatch logic is unit-testable without
 * booting the MCP stdio server (which has a top-level connect). Follows the
 * module-per-concern layout already used by guard.mjs / tools/act.mjs /
 * tools/flows.mjs: server.mjs stays a thin MCP-wiring entry point and calls
 * dispatchTool() with its concrete dependencies.
 *
 * All side-effecting helpers (connect, status, inspect, query, consoleLog,
 * screenshot, handleAct, handleFlow) are injected via `deps` so tests can
 * pass mocks.
 */

/**
 * Run one tool by name. Returns the RAW tool output (string | object |
 * { content: [...] }); the caller wraps it for MCP via wrapResult().
 * @param {string} name
 * @param {object} args
 * @param {object} deps injected dependencies (see server.mjs for the real ones)
 */
export async function dispatchTool(name, args = {}, deps) {
  const {
    EXPECTED_HOME,
    DEFAULT_HOME,
    connect,
    assertTargetAttested,
    handleAct,
    handleFlow,
    status,
    inspect,
    query,
    consoleLog,
    screenshot,
    readTools,
    actTools,
    flowTools,
    CFG
  } = deps

  let out

  // Preflight diagnostic: deliberately NO connect()/attestation — its whole
  // job is reporting unavailability (cdpAlive:false must be reachable). This
  // is the ONLY tool that bypasses the gate; everything else stays attested.
  if (name === 'desktop_ui_status') {
    return wrapResult(await status())
  }

  if (readTools.some((t) => t.name === name)) {
    const live = await connect()
    await assertTargetAttested(live, { expectedHome: EXPECTED_HOME, defaultHome: DEFAULT_HOME })
    if (name === 'desktop_ui_status') out = await status()
    else if (name === 'ui_inspect') out = await inspect(args)
    else if (name === 'ui_query') out = await query(args)
    else if (name === 'ui_console') out = await consoleLog(args)
    else if (name === 'ui_screenshot') out = await screenshot(args)
  } else if (actTools.some((t) => t.name === name)) {
    if (!CFG.allowAct) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              error: 'mutating tools are disabled',
              hint: "set DESKTOP_DEBUG_MCP_ALLOW_ACT=1 in this MCP server's env to enable ui_click/ui_type/ui_press/ui_eval"
            })
          }
        ]
      }
    }
    const live = await connect()
    await assertTargetAttested(live, { expectedHome: EXPECTED_HOME, defaultHome: DEFAULT_HOME })
    out = await handleAct(name, args, { evalBounded: deps.evalBounded, resolveSelector: deps.resolveSelector, cdp: live })
  } else if (flowTools.some((t) => t.name === name)) {
    if (!CFG.allowAct) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ error: 'flows mutate the UI — disabled without DESKTOP_DEBUG_MCP_ALLOW_ACT=1' }) }]
      }
    }
    const live = await connect()
    await assertTargetAttested(live, { expectedHome: EXPECTED_HOME, defaultHome: DEFAULT_HOME })
    out = await handleFlow(name, args, { cdp: live })
  } else {
    throw new Error(`unknown tool: ${name}`)
  }

  return out
}

/**
 * Wrap a raw tool result for the MCP CallTool response.
 *
 * If the result already carries a `content` array (e.g. ui_screenshot returns
 * MCP image content), return it verbatim — do NOT JSON-stringify it, or the
 * image block is lost (BUG #1). Otherwise stringify the value into a text
 * block. `isError` is preserved from the caller.
 */
export function wrapResult(out, isError = false) {
  if (out && Array.isArray(out.content)) {
    return { content: out.content, isError }
  }
  const text = typeof out === 'string' ? out : JSON.stringify(out, null, 1)
  return { content: [{ type: 'text', text }], isError }
}
