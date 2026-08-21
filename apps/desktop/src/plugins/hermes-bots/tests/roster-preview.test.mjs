import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function runtime() {
  const atom = value => ({ get: () => value, set: () => undefined })
  const jsx = (type, props = {}) => ({ type, props })
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: value => (value?.get ? value.get() : value),
    useState: value => [value, () => undefined],
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state: { profile: { get: () => 'ops', listen: () => undefined } }, request: () => undefined }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__previewKind = previewKind;\nglobalThis.__generatedSessionTitle = generatedSessionTitle;\nglobalThis.__isGenericTitle = isGenericTitle;'
    )
  vm.runInNewContext(code, context)
  return context
}

// NOTE: objects created inside the vm realm carry that realm's Object
// prototype, so assert.deepEqual against host-realm literals fails on
// reference-equality. Compare fields explicitly.
function fromBotOf(preview) {
  return runtime().__previewKind(preview).fromBot
}

test('previewKind: a plain chat preview is a human exchange, not a DM', () => {
  assert.equal(fromBotOf('Can you check the vault sync?'), null)
})

test('previewKind: parses the current 🤖 delivery prefix and sender handle', () => {
  assert.equal(fromBotOf('Message from 🤖 manager (@manager): Learn-share: skill installed'), 'manager')
})

test('previewKind: parses the legacy agent-prefix shape', () => {
  assert.equal(fromBotOf("Message from agent 'researcher': here is the paper"), 'researcher')
})

test('previewKind: empty or absent preview is not a DM', () => {
  assert.equal(fromBotOf(''), null)
  assert.equal(fromBotOf(undefined), null)
})

test('isGenericTitle: auto-assigned titles are generic', () => {
  const r = runtime()
  assert.equal(r.__isGenericTitle('Bot Chat'), true)
  assert.equal(r.__isGenericTitle('New chat'), true)
  assert.equal(r.__isGenericTitle(''), true)
  assert.equal(r.__isGenericTitle('Weekly review planning'), false)
})

test('generatedSessionTitle: keeps a meaningful stored title', () => {
  const r = runtime()
  assert.equal(r.__generatedSessionTitle({ title: 'Weekly review' }, 'some preview'), 'Weekly review')
})

test('generatedSessionTitle: invents a label from the preview for generic titles', () => {
  const r = runtime()
  assert.equal(r.__generatedSessionTitle({ title: 'Bot Chat' }, 'The tailnet proxy binds 100.64.0.1'), 'The tailnet proxy binds 100.64.0.1')
})

test('generatedSessionTitle: strips the bot-to-bot prefix before generating', () => {
  const r = runtime()
  const out = r.__generatedSessionTitle({ title: '' }, 'Message from 🤖 manager (@manager): Learn-share: skill installed')
  assert.match(out, /Learn-share/)
  assert.doesNotMatch(out, /Message from/)
})

test('generatedSessionTitle: caps the generated label length', () => {
  const r = runtime()
  const out = r.__generatedSessionTitle({ title: '' }, 'this is a very long preview that goes on and on and on about something or other entirely')
  assert.ok(out.length <= 34, `expected <= 34 chars, got ${out.length}: ${out}`)
})

// ── render smoke: BotRow must paint the new row furniture without throwing ──

function renderRuntime() {
  const atom = value => ({ get: () => value, set: () => undefined })
  const jsx = (type, props = {}) => ({ type, props })
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    cn: (...args) => args.filter(Boolean).join(' '),
    Button: 'Button',
    BotFace: 'BotFace',
    Codicon: 'Codicon',
    ContextMenu: 'ContextMenu',
    ContextMenuContent: 'ContextMenuContent',
    ContextMenuItem: 'ContextMenuItem',
    ContextMenuSeparator: 'ContextMenuSeparator',
    ContextMenuTrigger: 'ContextMenuTrigger',
    haptic: () => undefined,
    host: {
      state: {
        profile: { get: () => 'scribe', listen: () => undefined },
        gateway: { get: () => 'idle', listen: () => undefined }
      },
      request: () => Promise.resolve({ sessions: [] }),
      openSession: () => undefined,
      newChat: () => undefined,
      navigate: () => undefined
    },
    profileColor: () => '#8b5cf6',
    queryClient: { invalidateQueries: () => undefined },
    relativeTime: () => 'now',
    useQuery: () => ({}),
    useValue: value => (value?.get ? value.get() : value),
    useState: value => [value, () => undefined],
    useEffect: () => undefined,
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__BotRow = BotRow;')
  vm.runInNewContext(code, context)
  return context
}

function textOf(node) {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join(' ')
  if (typeof node === 'object') {
    if (node.props) return textOf(node.props.children ?? '')
    return Object.values(node).map(textOf).join(' ')
  }
  return ''
}

function findNode(node, predicate) {
  if (node == null || typeof node !== 'object') return null
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findNode(child, predicate)
      if (found) return found
    }
    return null
  }
  if (predicate(node)) return node
  return findNode(node.props?.children, predicate)
}

const DM_BOT = {
  name: 'scribe',
  title: 'Scribe',
  description: '',
  last_session: {
    id: 's1',
    title: 'Bot Chat',
    preview: 'Message from 🤖 manager (@manager): Learn-share: skill installed in your profile',
    last_active: Math.floor(Date.now() / 1000) - 5
  }
}

test('render: BotRow shows the sender badge and stripped DM preview', () => {
  const r = renderRuntime()
  const tree = r.__BotRow({ bot: DM_BOT, onEdit: () => undefined })
  const text = textOf(tree)
  assert.match(text, /@manager/)
  assert.match(text, /Learn-share/)
  assert.doesNotMatch(text, /Message from/)
})

test('render: BotRow renders plain previews without a badge', () => {
  const r = renderRuntime()
  const tree = r.__BotRow({
    bot: { name: 'ops', title: 'Ops', description: '', last_session: { id: 's2', title: 'Weekly review', preview: 'All hosts are healthy', last_active: 1_700_000_000 } },
    onEdit: () => undefined
  })
  const text = textOf(tree)
  assert.match(text, /All hosts are healthy/)
  assert.doesNotMatch(text, /@manager/)
  // The inline session-history chip is gone — the conversation lives in the
  // bot's one canonical chat, so the title no longer renders inline.
  assert.doesNotMatch(text, /Weekly review/)
})

test('render: BotRow tolerates a fresh bot with no sessions yet', () => {
  const r = renderRuntime()
  const tree = r.__BotRow({ bot: { name: 'newbie', title: '', description: 'Fresh bot' }, onEdit: () => undefined })
  const text = textOf(tree)
  assert.match(text, /Fresh bot/)
})

test('render: a remote gateway name is not squeezed out by its handle', () => {
  const r = renderRuntime()
  const tree = r.__BotRow({
    bot: {
      connectionId: 'studio-over-ssh',
      connectionLabel: 'Studio over SSH',
      handle: 'default-studio-over-ssh',
      name: 'default',
      remoteSource: true
    },
    onEdit: () => undefined
  })
  const name = findNode(tree, node => node.type === 'span' && textOf(node) === 'Studio over SSH')
  const handle = findNode(tree, node => node.type === 'span' && textOf(node) === '@default-studio-over-ssh')

  assert.ok(name)
  assert.match(name.props.className, /shrink-0/)
  assert.ok(handle)
  assert.match(handle.props.className, /min-w-0/)
  assert.match(handle.props.className, /truncate/)
  assert.doesNotMatch(handle.props.className, /shrink-0/)
})

test('render: BotRow previews the pinned canonical chat, not an unrelated latest session', () => {
  // hermes-agent#88200: the row opens the pinned chat on click, so the
  // preview must describe that same session — not the profile's most recent
  // (but unrelated) activity.
  const r = renderRuntime()
  const tree = r.__BotRow({
    bot: {
      name: 'ops',
      title: 'Ops',
      description: '',
      last_session: { id: 'scratch9', title: 'Scratch', preview: 'unrelated scratch content', last_active: 1_800_000_000 },
      preferred_session: { id: 'pinned1', resolved_id: 'pinned1', title: 'Bot Chat', preview: 'pinned chat content', started_at: 1, last_active: 1_700_000_000, message_count: 5 }
    },
    onEdit: () => undefined
  })
  const text = textOf(tree)
  assert.match(text, /pinned chat content/)
  assert.doesNotMatch(text, /unrelated scratch content/)
})

test('previewKind: the primary profile surfaces as @hermes, never @default', () => {
  // botHandle() exists so "the word 'default' never surfaces in the UI"; the
  // bot-to-bot badge was rendering the raw captured profile name (#89484).
  assert.equal(fromBotOf("Message from agent 'default': deploy is green"), 'hermes')
})

test('previewKind: a named profile keeps its own handle', () => {
  assert.equal(fromBotOf("Message from agent 'ops': deploy is green"), 'ops')
})
