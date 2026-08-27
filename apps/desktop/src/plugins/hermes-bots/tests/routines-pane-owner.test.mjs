import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// The Cronjobs pane read the shared roster with a bare `$lastRoster.get()`
// while rendering. BotsHomeView owns the roster fetch, so whenever this pane
// mounted before the roster hydrated (fresh boot ordering, renderer reload
// resetting the atoms) it captured an empty snapshot forever: the pane stayed
// pinned on "Cronjobs are unavailable until this agent appears in the
// roster." even for a focused bot chat whose exact roster row existed, and
// Create Cronjob silently no-oped (#94483).
//
// Contract:
// 1. RoutinesPane must SUBSCRIBE to the roster (like every other consumer
//    via useValue), so hydration/roster changes can re-render the pane.
// 2. Once the owner resolves, the pane actually renders its content and a
//    working create affordance instead of staying on the placeholder.
// 3. A complete (authoritative) focused owner with no exact roster row still
//    fails closed — the subscription fix must not loosen identity matching.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

const NL = String.fromCharCode(10)

function stripImports(source) {
  const out = []
  let inImportBlock = false
  for (const line of source.split(NL)) {
    if (inImportBlock) {
      if (line.includes(' from ')) inImportBlock = false
      continue
    }
    if (line.startsWith('import ')) {
      if (!line.includes(' from ')) inImportBlock = true
      continue
    }
    out.push(line)
  }
  return out.join(NL)
}

function makeAtom(initial) {
  const slot = {
    value: initial,
    listeners: new Set(),
    get: () => slot.value,
    set: next => {
      slot.value = next
      for (const fn of [...slot.listeners]) fn()
    },
    listen: fn => {
      slot.listeners.add(fn)
      return () => slot.listeners.delete(fn)
    }
  }
  return slot
}

const UNAVAILABLE = 'Cronjobs are unavailable until this agent appears in the roster.'

// Function components are lazily expanded (React would call them during
// render); a component that explodes under the stubs contributes an opaque
// subtree rather than failing the whole traversal.
function expandNode(node, depth) {
  if (typeof node.type !== 'function' || depth > 12) return null
  try {
    return node.type(node.props)
  } catch {
    return null
  }
}

function collectStrings(node, out = [], depth = 0) {
  if (node == null || depth > 24) return out
  if (Array.isArray(node)) {
    for (const child of node) collectStrings(child, out, depth + 1)
    return out
  }
  if (typeof node === 'object') {
    if (!node.props) return out
    if (typeof node.type === 'function') {
      collectStrings(expandNode(node, depth), out, depth + 1)
      return out
    }
    collectStrings(node.props.children, out, depth + 1)
    return out
  }
  if (typeof node === 'string' || typeof node === 'number') out.push(String(node))
  return out
}

/** Depth-first search over the stubbed element tree. */
function findNode(node, match, depth = 0) {
  if (node == null || depth > 24) return null
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = findNode(child, match, depth + 1)
      if (hit) return hit
    }
    return null
  }
  if (!node || typeof node !== 'object' || !node.props) return null
  if (match(node)) return node
  if (typeof node.type === 'function') return findNode(expandNode(node, depth), match, depth + 1)
  return findNode(node.props.children, match, depth + 1)
}

function runtime({ focusedOwner, jobs = [] } = {}) {
  const subscribed = new Set()

  // Minimal React hook memory: one slot list per runtime, indexed by call
  // order within a render pass. `render()` starts a fresh pass so repeated
  // calls model re-renders while setters keep their values across them.
  const hooks = []
  let hookIndex = 0

  const context = {
    atom: initial => makeAtom(initial),
    sdk: new Proxy({}, { get: () => undefined }),
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    profileColor: () => '#64748b',
    haptic: () => undefined,
    queryClient: {},
    relativeTime: () => '',
    cn: (...parts) => parts.filter(Boolean).join(' '),
    Button: 'Button',
    Checkbox: 'Checkbox',
    Codicon: 'Codicon',
    ContextMenu: 'ContextMenu',
    ContextMenuContent: 'ContextMenuContent',
    ContextMenuItem: 'ContextMenuItem',
    ContextMenuSeparator: 'ContextMenuSeparator',
    ContextMenuTrigger: 'ContextMenuTrigger',
    ConfirmDialog: 'ConfirmDialog',
    CopyButton: 'CopyButton',
    Dialog: 'Dialog',
    DialogContent: 'DialogContent',
    DialogDescription: 'DialogDescription',
    DialogFooter: 'DialogFooter',
    DialogHeader: 'DialogTitle-header',
    DialogTitle: 'DialogTitle',
    DropdownMenu: 'DropdownMenu',
    DropdownMenuContent: 'DropdownMenuContent',
    DropdownMenuItem: 'DropdownMenuItem',
    DropdownMenuSeparator: 'DropdownMenuSeparator',
    DropdownMenuTrigger: 'DropdownMenuTrigger',
    EmptyState: 'EmptyState',
    GlyphSpinner: 'GlyphSpinner',
    Input: 'Input',
    ScrollArea: 'ScrollArea',
    SearchField: 'SearchField',
    Select: 'Select',
    SelectContent: 'SelectContent',
    SelectItem: 'SelectItem',
    SelectTrigger: 'SelectTrigger',
    SelectValue: 'SelectValue',
    Switch: 'Switch',
    Textarea: 'Textarea',
    Tip: 'Tip',
    jsx: (type, props = {}) => ({ type, props }),
    jsxs: (type, props = {}) => ({ type, props }),
    // Minimal React stand-ins. useValue mirrors the SDK contract: registering
    // the dependency is exactly what lets a later store write re-render us.
    useState: initial => {
      const i = hookIndex++
      if (!(i in hooks)) hooks[i] = typeof initial === 'function' ? initial() : initial
      const setValue = value => {
        hooks[i] = typeof value === 'function' ? value(hooks[i]) : value
      }
      return [hooks[i], setValue]
    },
    useEffect: () => undefined,
    useMemo: fn => fn(),
    useRef: value => ({ current: value }),
    useCallback: fn => fn,
    useQuery: () => ({
      data: { jobs, scoped: 'research' },
      error: null,
      isLoading: false,
      refetch: () => {}
    }),
    useValue: value => {
      if (value && typeof value.listen === 'function') {
        value.listen(() => {})
        subscribed.add(value)
      }
      return value?.get ? value.get() : value
    },
    host: {
      request: async () => ({}),
      state: {
        connectionId: { get: () => 'local', listen: () => undefined },
        profile: { get: () => 'research', listen: () => undefined },
        ...(focusedOwner ? { focusedSessionOwner: focusedOwner } : {})
      }
    }
  }

  const marker = [NL + 'globalThis.__pane = { RoutinesPane, $lastRoster };'].join('')
  const source = stripImports(pluginSource)
    .replace('export default {', 'globalThis.plugin = {')
    .concat(marker)

  vm.runInNewContext(source, context, { filename: 'plugin.js' })

  const render = () => {
    hookIndex = 0
    return context.__pane.RoutinesPane()
  }

  return { render, $lastRoster: context.__pane.$lastRoster, subscribed }
}

test('regression: the pane follows the roster hydrating after mount', () => {
  const focusedOwner = makeAtom({ connectionId: 'local', profile: 'research' })
  const { render, $lastRoster, subscribed } = runtime({
    focusedOwner,
    jobs: [{ job_id: 'j-1', name: 'Report', schedule: 'every 1h', state: 'scheduled', enabled: true }]
  })

  // Mount order: pane first, roster fetch lands afterwards (BotsHomeView owns
  // the fetch). First paint must be the fail-closed placeholder...
  const before = collectStrings(render()).join(' ')
  assert.ok(before.includes(UNAVAILABLE))

  // ...but the pane must be wired to the store so the write can re-render it.
  assert.ok(subscribed.has($lastRoster), 'RoutinesPane must subscribe to $lastRoster (bare .get() captured an empty roster forever)')
  assert.ok($lastRoster.listeners.size >= 1, 'hydration must have a live path to re-render the pane')

  // Roster arrives: the same focused bot chat now resolves to its exact row
  // and the pane paints real content without any unrelated state change.
  $lastRoster.set([{ name: 'research', connectionId: 'local' }])
  const after = collectStrings(render()).join(' ')
  assert.ok(!after.includes(UNAVAILABLE))
  assert.ok(after.includes('Cronjobs'), 'pane should render its header once the owner resolves')
  assert.ok(after.includes('Report'), 'the scheduled job row should paint after the roster hydrates')
})

test('create affordance appears once the owner resolves', () => {
  const focusedOwner = makeAtom({ connectionId: 'local', profile: 'research' })
  const { render, $lastRoster } = runtime({ focusedOwner, jobs: [] })

  $lastRoster.set([{ name: 'research', connectionId: 'local' }])
  const tree = render()

  // The reported "Create Cronjob silently no-ops" symptom starts here: while
  // the owner is stuck unresolved the pane never offers any create control.
  // (Dialog internals are owned by the routine-create flow tests.)
  const create = findNode(tree, node => {
    if (node.type !== 'Button') return false
    return collectStrings(node).join(' ') === 'Create Cronjob'
  })
  assert.ok(create, 'the empty-state Create Cronjob button should exist once the owner resolves')
  assert.equal(typeof create.props.onClick, 'function')

  const headerTip = findNode(tree, node => node.type === 'Tip' && node.props?.label === 'New Cronjob')
  assert.ok(headerTip, 'the header New Cronjob affordance should exist once the owner resolves')
  assert.equal(typeof headerTip.props.children?.props?.onClick, 'function')
})

test('a complete focused owner without an exact roster row still fails closed', () => {
  const focusedOwner = makeAtom({ connectionId: 'local', profile: 'ghost-profile' })
  const { render, $lastRoster } = runtime({ focusedOwner })

  $lastRoster.set([{ name: 'research', connectionId: 'local' }])
  const text = collectStrings(render()).join(' ')
  assert.ok(text.includes(UNAVAILABLE))
})
