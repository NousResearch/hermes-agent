#!/usr/bin/env node
/**
 * Hermes Desktop Debug MCP server.
 *
 * Gives LLM agents native tools to inspect (and, gated, drive) the live
 * renderer of `apps/desktop` over the dev-only CDP port. Wraps the existing
 * perf-harness client (`scripts/perf/lib/cdp.mjs`) so protocol fixes stay in
 * one place.
 *
 * Read-only by default. Mutating tools require DESKTOP_DEBUG_MCP_ALLOW_ACT=1
 * in the server's environment.
 *
 * Run:  node server.mjs [--port 9222] [--match 5174]
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema
} from '@modelcontextprotocol/sdk/types.js'

import { SELECTORS } from '../scripts/perf/lib/cdp.mjs'
import os from 'node:os'
import path from 'node:path'
import { actTools, handleAct } from './tools/act.mjs'
import { flowTools, handleFlow } from './tools/flows.mjs'
import { assertTargetAttested, canon } from './guard.mjs'
import { dispatchTool, wrapResult } from './dispatch.mjs'
import { createCdpClient } from './cdp-client.mjs'
import { createReadTools } from './tools/read.mjs'

// ---------------------------------------------------------------------------
// Output bounds — never dump the whole DOM into an agent's context.
const MAX_TEXT = 80 // per-node text snippet length
const MAX_NODES = 20 // ui_query row cap
const MAX_EVAL = 4000 // ui_eval output cap (chars)
const MAX_CONSOLE = 50

const args = process.argv.slice(2)
const argOf = (name, fallback) => {
  const i = args.indexOf(name)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}

const CFG = {
  port: Number(argOf('--port', process.env.DESKTOP_DEBUG_MCP_PORT || '9222')),
  match: argOf('--match', '5174'),
  allowAct: process.env.DESKTOP_DEBUG_MCP_ALLOW_ACT === '1'
}

// The Hermes home the operator DECLARES this desktop instance runs against.
// The guard (guard.mjs) does NOT trust this alone — it reads the *realized*
// data root from the connected target and refuses unless they match.
const EXPECTED_HOME = process.env.DESKTOP_DEBUG_MCP_EXPECTED_HOME || ''
const DEFAULT_HOME = process.env.HERMES_HOME || path.join(os.homedir(), '.hermes')

const consoleRing = [] // renderer console capture (bounded)

// Lazily connected CDP client. Owns the connection lifecycle (discovery + open
// + console capture + failure reset) so server.mjs stays a thin wiring layer.
const cdpClient = createCdpClient({
  port: CFG.port,
  match: CFG.match,
  onConsole: (entry) => {
    consoleRing.push(entry)
    if (consoleRing.length > 200) consoleRing.shift()
  }
})
const connect = () => cdpClient.connect()

// Read-only tool implementations live in tools/read.mjs (module-per-concern,
// same layout as tools/act.mjs / tools/flows.mjs). server.mjs only wires them.
const readDeps = createReadTools({
  connect,
  MAX_TEXT,
  MAX_NODES,
  MAX_EVAL,
  MAX_CONSOLE,
  SELECTORS,
  consoleRing,
  port: CFG.port,
  match: CFG.match,
  allowAct: CFG.allowAct
})
const { evalBounded, status, inspect, query, consoleLog, screenshot } = readDeps
const resolveSelector = (sel) => readDeps.resolveSelector(sel)


// ---------------------------------------------------------------------------

const readTools = [
  {
    name: 'desktop_ui_status',
    description:
      'Check whether the Hermes desktop dev app has its CDP debug port alive, and which page targets/selectors are available. Call this FIRST before any other desktop UI tool.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'ui_inspect',
    description:
      'Inspect ONE element in the Hermes desktop renderer: tag, classes, bounding box, visibility, key computed styles, plus an inheritance hint (own classes vs inherited rule). Selector may be a stable key (composer, threadViewport, assistantMessage, turnPair, profileRail) or any CSS selector.',
    inputSchema: {
      type: 'object',
      properties: { selector: { type: 'string', description: 'SELECTORS key or CSS selector' } },
      required: ['selector']
    }
  },
  {
    name: 'ui_query',
    description:
      'List up to 20 elements matching a selector with bounded text snippets and visibility. Good for "what messages exist in the thread right now".',
    inputSchema: {
      type: 'object',
      properties: {
        selector: { type: 'string' },
        limit: { type: 'number', description: 'max nodes (hard cap 20)' }
      },
      required: ['selector']
    }
  },
  {
    name: 'ui_console',
    description: 'Recent renderer console output captured while connected.',
    inputSchema: {
      type: 'object',
      properties: {
        level: { type: 'string', description: 'error|warning|log|info|debug' },
        sinceMs: { type: 'number' }
      }
    }
  },
  {
    name: 'ui_screenshot',
    description: 'Capture the current window as a PNG and return it as image content. Does NOT write to disk (the tool is read-only; persist the returned bytes yourself if you need a file).',
    inputSchema: { type: 'object', properties: {} }
  }
]

const allTools = [...readTools, ...actTools, ...flowTools]

const server = new Server(
  { name: 'hermes-desktop-debug', version: '0.1.0' },
  { capabilities: { tools: {} } }
)

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: allTools.map(t => ({
    name: t.name,
    description:
      t.gated && !CFG.allowAct
        ? `${t.description} [DISABLED: set DESKTOP_DEBUG_MCP_ALLOW_ACT=1 to enable]`
        : t.description,
    inputSchema: t.inputSchema
  }))
}))


server.setRequestHandler(CallToolRequestSchema, async req => {
  const { name, arguments: a = {} } = req.params

  try {
    const out = await dispatchTool(name, a, {
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
      CFG,
      evalBounded,
      resolveSelector
    })
    return wrapResult(out)
  } catch (err) {
    return wrapResult({ error: String(err.message || err) }, true)
  }
})

const transport = new StdioServerTransport()
await server.connect(transport)
console.error(`[desktop-debug-mcp] ready on :${CFG.port} allowAct=${CFG.allowAct}`)
