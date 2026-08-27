import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Newer desktop builds export the WHOLE core Capabilities surface (SkillsView,
// hermes-agent#87317). The bot editor's Advanced section must render it pinned
// to the bot's profile — in Edit Profile directly, and in New Agent behind a
// Capabilities tab that materializes the profile first. Feature-detected so
// the plugin still loads (and keeps its checklist UI) on older desktops.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadAdvanced(SkillsView, sdkComponents = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => values.set(slot, value),
      listen: () => undefined
    }
    values.set(slot, initial)
    return slot
  }
  const jsx = (type, props = {}) => ({ type, props })
  const stateValues = [true, false, '']
  const routed = []
  let gatewayReads = 0
  let query
  const context = {
    atom,
    Checkbox: 'Checkbox',
    GlyphSpinner: 'GlyphSpinner',
    Input: 'Input',
    ScrollArea: 'ScrollArea',
    Textarea: 'Textarea',
    document: { createElement: () => ({}), getElementById: () => null, head: { appendChild: () => undefined } },
    host: {
      getGateway: () => {
        gatewayReads += 1
        return 'ambient-gateway'
      },
      request: async (...args) => routed.push(['ambient', ...args]),
      requestProfile: async (...args) => routed.push(['profile', ...args]),
      state: {
        gateway: { get: () => 'open', listen: () => undefined },
        profile: { get: () => 'default', listen: () => undefined }
      }
    },
    jsx,
    jsxs: jsx,
    queryClient: { invalidateQueries: () => undefined },
    sdk: { SkillsView, ...sdkComponents },
    useQuery: options => {
      query = options
      return { data: { providers: [] }, isLoading: false, error: null }
    },
    useState: initial => [stateValues.length ? stateValues.shift() : initial, () => undefined],
    window: { setTimeout, clearTimeout }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__AdvancedProfileConfig = AdvancedProfileConfig; globalThis.__useModelOptions = useModelOptions;')
  vm.runInNewContext(code, context, { filename: 'plugin.js' })

  return { context, gatewayReads: () => gatewayReads, query: () => query, routed }
}

function descendants(node) {
  if (!node || typeof node !== 'object') return []
  const children = Array.isArray(node.props?.children) ? node.props.children : [node.props?.children]
  return [node, ...children.flatMap(descendants)]
}

const remoteBot = {
  connectionId: 'remote-a',
  name: 'default',
  remoteSource: true,
  route: {
    connectionId: 'remote-a',
    mode: 'remote',
    profile: 'default',
    targetProfile: 'backend-default'
  },
  sourceScoped: true,
  targetProfile: 'backend-default'
}

const localBot = {
  connectionId: 'local',
  name: 'default',
  remoteSource: false,
  route: {
    connectionId: 'local',
    mode: 'local',
    profile: 'default',
    targetProfile: 'default'
  },
  sourceScoped: true,
  targetProfile: 'default'
}

const advancedState = {
  loaded: true,
  provider: '',
  model: '',
  soul: '',
  skills: [],
  toolsets: [],
  mcp: [{ name: 'remote-mcp', enabled: false, fromCatalog: true, installed: false, requires: ['TOKEN'] }]
}

test('resolves SkillsView as an optional SDK namespace export', () => {
  assert.match(source, /import \* as sdk from '@hermes\/plugin-sdk'/)
  assert.match(source, /const SkillsView = typeof sdk === 'undefined' \? undefined : sdk\.SkillsView/)
})

test('Edit Profile renders the pinned Capabilities surface when the export exists', () => {
  assert.match(source, /if \(SkillsView && \(!botRoute \|\| skillsViewRoutesConnections\)\) \{/)
  assert.match(source, /fixedProfile: backendProfile/)
  assert.match(source, /fixedConnection: botRoute\.connectionId/)
})

test('New Agent gains a Capabilities tab that materializes the profile first', () => {
  // Tab list swaps to General + Capabilities on newer builds…
  assert.match(source, /\['capabilities', 'Capabilities'\]/)
  // …and opening it creates the profile through the same lazy door MCP setup uses.
  assert.match(source, /id === 'capabilities'/)
  assert.match(source, /ensureAgentCreated\(\)\s*\n?\s*\.then\(created => created && setCreatedForCaps\(created\)\)/)
  assert.match(source, /jsx\(SkillsView, \{\s*embedded: true,\s*fixedProfile: createdForCaps,/)
})

test('remote-target drafts pin the live surface to the target connection', () => {
  // Builds whose SkillsView routes fixedConnection get the live Capabilities
  // tab for remote targets too — pinned to the target machine's backend.
  assert.match(source, /skillsViewRoutesConnections = Boolean\(SkillsView && SkillsView\.supportsFixedConnection\)/)
  assert.match(source, /SkillsView && \(!remoteTarget \|\| skillsViewRoutesConnections\)/)
  assert.match(source, /\.\.\.\(remoteTarget \? \{ fixedConnection: targetConnection \} : \{\}\)/)
})

test('older-build fallback keeps the checklist UI intact', () => {
  // The staged CheckList sections and the hub search section must survive for
  // desktops without the SkillsView export.
  assert.match(source, /jsx\(CheckList, \{ items: visibleSkills/)
  assert.match(source, /jsx\(HubSkillsSection, \{/)
})

test('older SkillsView builds fail closed for remote Edit Profile', () => {
  function OldSkillsView() {}
  function McpTab() {}
  function ToolsetConfigPanel() {}
  const runtime = loadAdvanced(OldSkillsView, { McpTab, ToolsetConfigPanel })
  const tree = runtime.context.__AdvancedProfileConfig({
    bot: remoteBot,
    state: advancedState,
    setState: () => undefined
  })
  const nodes = descendants(tree)

  assert.equal(nodes.some(node => node.type === OldSkillsView), false)
  assert.equal(nodes.some(node => node.type?.name === 'ModelPicker'), true)
  assert.equal(nodes.some(node => node.type === 'Textarea'), true)
  assert.equal(nodes.some(node => node.type?.name === 'HubSkillsSection'), false)
  assert.equal(nodes.some(node => node.type?.name === 'McpSetupButton'), false)
  assert.equal(nodes.some(node => node.type === McpTab), false)
  assert.equal(nodes.some(node => node.type === ToolsetConfigPanel), false)
  assert.equal(runtime.gatewayReads(), 0)

  for (const node of nodes) {
    node.props?.onClick?.()
    node.props?.onCheckedChange?.(true)
  }

  assert.equal(runtime.routed.some(call => call[0] === 'ambient'), false)
  assert.equal(runtime.gatewayReads(), 0)
})

test('older SkillsView builds keep local capability fallback unchanged', () => {
  function OldSkillsView() {}
  function McpTab() {}
  function ToolsetConfigPanel() {}
  const runtime = loadAdvanced(OldSkillsView, { McpTab, ToolsetConfigPanel })
  const tree = runtime.context.__AdvancedProfileConfig({
    bot: localBot,
    state: { ...advancedState, toolsets: [{ name: 'local-tools', enabled: true }] },
    setState: () => undefined
  })
  const nodes = descendants(tree)

  assert.equal(nodes.some(node => node.type?.name === 'HubSkillsSection'), true)
  assert.equal(nodes.some(node => node.type === McpTab), true)
  assert.equal(nodes.some(node => node.type === ToolsetConfigPanel), true)
  assert.equal(runtime.gatewayReads(), 1)
})

test('connection-aware SkillsView receives separate connection and backend profile', () => {
  function RoutedSkillsView() {}
  RoutedSkillsView.supportsFixedConnection = true
  const runtime = loadAdvanced(RoutedSkillsView)
  const tree = runtime.context.__AdvancedProfileConfig({
    bot: remoteBot,
    state: advancedState,
    setState: () => undefined
  })
  const view = descendants(tree).find(node => node.type === RoutedSkillsView)

  assert.equal(view.props.fixedConnection, 'remote-a')
  assert.equal(view.props.fixedProfile, 'backend-default')
})

test('Edit Profile model options use the captured non-identity Bot route', async () => {
  const runtime = loadAdvanced(undefined)
  runtime.context.__useModelOptions(remoteBot)

  await runtime.query().queryFn()

  assert.deepEqual(JSON.parse(JSON.stringify(runtime.routed)), [[
    'profile',
    remoteBot.route,
    'model.options',
    // No `refresh`: the picker's catalog read participates in the staleTime
    // cache — a forced network read on every mount re-entered the spinner and
    // wiped staged selections on Bots view remounts (#95279).
    { include_unconfigured: true, explicit_only: false }
  ]])
})
