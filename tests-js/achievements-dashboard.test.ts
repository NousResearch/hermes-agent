import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const require = createRequire(import.meta.url)
const BUNDLE_PATH = path.join(REPO_ROOT, 'plugins/hermes-achievements/dashboard/dist/index.js')

interface EffectSlot {
  cleanup?: () => void
  dependencies?: unknown[]
}

interface Element {
  props: { className?: string; onClick?: () => void } | null
  children: unknown[]
}

interface Payload {
  achievements: Array<{ id: string; name: string; category: string; state: string }>
  scan_meta: { mode: string }
  total_count: number
  unlocked_count: number
}

interface PendingRequest {
  url: string
  resolve: (payload: Payload) => void
  reject: (error: Error) => void
}

function dependenciesChanged(previous: unknown[] | undefined, next: unknown[]): boolean {
  return !previous || previous.length !== next.length || previous.some((value, index) => !Object.is(value, next[index]))
}

function payload(name: string, mode = 'full'): Payload {
  return {
    achievements: [{ id: 'let_him_cook', name, category: 'Autonomy', state: 'discovered' }],
    scan_meta: { mode },
    total_count: 1,
    unlocked_count: 0,
  }
}

async function flush(): Promise<void> {
  for (let index = 0; index < 8; index++) {await Promise.resolve()}
}

afterEach(() => {
  vi.unstubAllGlobals()
})

test('only mounted current-locale loads and polls may update dashboard state', async () => {
  let locale = 'en'
  let page: (() => unknown) | undefined
  let stateCursor = 0
  let effectCursor = 0
  let refCursor = 0
  let stateWrites = 0
  let intervalId = 0
  const state: unknown[] = []
  const refs: Array<{ current: unknown }> = []
  const effects: EffectSlot[] = []
  const pendingEffects: Array<() => void> = []
  const requests: PendingRequest[] = []
  const clearedIntervals: number[] = []
  const intervals = new Map<number, () => void>()

  const hooks = {
    useState<T>(initial: T): [T, (value: T | ((current: T) => T)) => void] {
      const index = stateCursor++

      if (index >= state.length) {state[index] = initial}

      return [
        state[index] as T,
        value => {
          stateWrites++
          state[index] = typeof value === 'function'
            ? (value as (current: T) => T)(state[index] as T)
            : value
        },
      ]
    },
    useRef<T>(initial: T): { current: T } {
      const index = refCursor++

      if (index >= refs.length) {refs[index] = { current: initial }}

      return refs[index] as { current: T }
    },
    useEffect(effect: () => void | (() => void), dependencies: unknown[]): void {
      const index = effectCursor++
      const previous = effects[index]

      if (!dependenciesChanged(previous?.dependencies, dependencies)) {return}

      pendingEffects.push(() => {
        previous?.cleanup?.()
        const cleanup = effect()

        effects[index] = {
          cleanup: typeof cleanup === 'function' ? cleanup : undefined,
          dependencies: [...dependencies],
        }
      })
    },
  }

  const sdk = {
    React: {
      createElement: (type: unknown, props: unknown, ...children: unknown[]) => ({ type, props, children }),
      useEffect: hooks.useEffect,
      useRef: hooks.useRef,
    },
    components: { Button: 'button', Card: 'card', CardContent: 'card-content' },
    fetchJSON: (url: string) => new Promise<Payload>((resolve, reject) => {requests.push({ url, resolve, reject })}),
    hooks,
    useI18n: () => ({ locale, t: { achievements: null } }),
    utils: { cn: (...values: unknown[]) => values.filter(Boolean).join(' ') },
  }

  vi.stubGlobal('clearInterval', (id: number) => {
    clearedIntervals.push(id)
    intervals.delete(id)
  })
  vi.stubGlobal('setInterval', (callback: () => void) => {
    const id = ++intervalId

    intervals.set(id, callback)

    return id
  })
  vi.stubGlobal('fetch', () => {throw new Error('requests must use the authenticated host SDK')})
  vi.stubGlobal('window', {
    __HERMES_PLUGINS__: {
      register: (_name: string, component: () => unknown) => {page = component},
    },
    __HERMES_PLUGIN_SDK__: sdk,
  })

  // Execute the shipped artifact, with refs retained across renders as in React.
  delete require.cache[require.resolve(BUNDLE_PATH)]
  require(BUNDLE_PATH)
  assert.ok(page, 'dashboard bundle did not register its page component')

  function render(): Element {
    stateCursor = effectCursor = refCursor = 0
    const tree = page!() as Element

    pendingEffects.splice(0).forEach(effect => {effect()})

    return tree
  }

  function rescanButton(tree: Element): (() => void) | undefined {
    if (tree.props?.className === 'ha-refresh') {return tree.props.onClick}

    for (const child of tree.children) {
      if (!child || typeof child !== 'object' || !('children' in child)) {continue}
      const onClick = rescanButton(child as Element)

      if (onClick) {return onClick}
    }

    return undefined
  }

  function switchLocale(next: string): PendingRequest {
    locale = next
    render()
    const request = requests.at(-1)!

    assert.equal(request.url, '/api/plugins/hermes-achievements/achievements?locale=' + next)

    return request
  }

  function poll(): PendingRequest {
    assert.equal(intervals.size, 1)
    Array.from(intervals.values())[0]()

    return requests.at(-1)!
  }

  async function complete(request: PendingRequest, name: string, mode = 'full'): Promise<void> {
    request.resolve(payload(name, mode))
    await flush()
    render()
  }

  function assertCurrent(name: string): void {
    assert.equal((state[0] as Payload).achievements[0].name, name)
    assert.equal(state[1], false)
    assert.equal(state[2], null)
  }

  render()
  const initialEnglish = requests[0]

  assert.equal(initialEnglish.url, '/api/plugins/hermes-achievements/achievements?locale=en')
  await complete(switchLocale('zh'), '放手一搏')
  await complete(initialEnglish, 'Let Him Cook')
  assertCurrent('放手一搏')
  assert.equal(intervals.size, 0, 'a full snapshot has no later poll to repair stale data')

  const oldLoad = switchLocale('en')
  const chineseLoad = switchLocale('zh')

  oldLoad.reject(new Error('obsolete English load failed'))
  await flush()
  assert.equal(state[1], true, 'an obsolete finally must not finish the current load')
  assert.equal(state[2], null, 'an obsolete failure must not replace the current error')
  await complete(chineseLoad, '放手一搏', 'pending')

  const oldPoll = poll()
  const englishLoad = switchLocale('en')

  assert.deepEqual(clearedIntervals, [1], 'locale changes must clean up the old poller')
  assert.equal(intervals.size, 1, 'the poller must be recreated for the new locale')
  assert.equal(poll().url, '/api/plugins/hermes-achievements/achievements?locale=en')
  await complete(englishLoad, 'Let Him Cook')
  await complete(oldPoll, '放手一搏')
  assertCurrent('Let Him Cook')
  assert.equal(intervals.size, 0)

  await complete(switchLocale('zh'), '放手一搏', 'pending')
  const failedPoll = poll()
  const currentLoad = switchLocale('en')

  failedPoll.reject(new Error('obsolete Chinese poll failed'))
  await flush()
  assert.equal(state[1], true)
  assert.equal(state[2], null)
  await complete(currentLoad, 'Let Him Cook')
  assertCurrent('Let Him Cook')

  await complete(switchLocale('zh'), '放手一搏', 'pending')
  const unmountedPoll = poll()
  const unmountedFailure = poll()
  const onClick = rescanButton(render())

  assert.ok(onClick, 'the page must expose its load action')
  onClick()
  const unmountedLoad = requests.at(-1)!

  effects.forEach(effect => {effect.cleanup?.()})
  assert.equal(intervals.size, 0)
  const writesBeforeCompletion = stateWrites

  unmountedPoll.resolve(payload('obsolete poll'))
  unmountedFailure.reject(new Error('unmounted poll failed'))
  unmountedLoad.resolve(payload('obsolete load'))
  await flush()
  assert.equal(stateWrites, writesBeforeCompletion, 'unmounted requests must not update data, error or loading')
})
