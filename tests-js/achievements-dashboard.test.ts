import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterEach, test, vi } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')

const BUNDLE_PATH = path.join(
  REPO_ROOT,
  'plugins',
  'hermes-achievements',
  'dashboard',
  'dist',
  'index.js'
)

interface EffectSlot {
  cleanup?: () => void
  dependencies?: unknown[]
}

function dependenciesChanged(previous: unknown[] | undefined, next: unknown[]): boolean {
  return !previous || previous.length !== next.length || previous.some((value, index) => !Object.is(value, next[index]))
}

afterEach(() => {
  vi.unstubAllGlobals()
})

test('in-flight scan polling is recreated with the current locale', async () => {
  let locale = 'en'
  let page: (() => unknown) | undefined
  let stateCursor = 0
  let effectCursor = 0
  const state: unknown[] = []
  const effects: EffectSlot[] = []
  const requests: string[] = []
  const clearedIntervals: number[] = []
  const intervalCallbacks: Array<() => void> = []

  const hooks = {
    useState<T>(initial: T): [T, (value: T | ((current: T) => T)) => void] {
      const index = stateCursor++

      if (index >= state.length) {state[index] = initial}

      return [
        state[index] as T,
        value => {
          state[index] = typeof value === 'function'
            ? (value as (current: T) => T)(state[index] as T)
            : value
        },
      ]
    },
    useEffect(effect: () => void | (() => void), dependencies: unknown[]): void {
      const index = effectCursor++
      const previous = effects[index]

      if (!dependenciesChanged(previous?.dependencies, dependencies)) {return}

      previous?.cleanup?.()
      const cleanup = effect()

      effects[index] = {
        cleanup: typeof cleanup === 'function' ? cleanup : undefined,
        dependencies: [...dependencies],
      }
    },
  }

  const sdk = {
    React: {
      createElement: (type: unknown, props: unknown, ...children: unknown[]) => ({ type, props, children }),
      useEffect: hooks.useEffect,
      useRef: () => ({ current: null }),
    },
    components: {
      Button: 'button',
      Card: 'card',
      CardContent: 'card-content',
    },
    fetchJSON: (url: string) => {
      requests.push(url)

      return Promise.resolve({
        achievements: [],
        scan_meta: { mode: 'pending' },
        total_count: 0,
        unlocked_count: 0,
      })
    },
    hooks,
    useI18n: () => ({ locale, t: { achievements: null } }),
    utils: { cn: (...values: unknown[]) => values.filter(Boolean).join(' ') },
  }

  const windowObject = {
    __HERMES_PLUGINS__: {
      register: (_name: string, component: () => unknown) => {page = component},
    },
    __HERMES_PLUGIN_SDK__: sdk,
  }

  vi.stubGlobal('clearInterval', (id: number) => {clearedIntervals.push(id)})
  vi.stubGlobal('setInterval', (callback: () => void) => {
    intervalCallbacks.push(callback)

    return intervalCallbacks.length
  })
  vi.stubGlobal('window', windowObject)

  // Import the shipped dashboard artifact as the host does; assertions below
  // exercise hook/network/timer behavior rather than reading its source text.
  await import(pathToFileURL(BUNDLE_PATH).href)
  assert.ok(page, 'dashboard bundle did not register its page component')

  function render(): void {
    stateCursor = 0
    effectCursor = 0
    page!()
  }

  render()
  await Promise.resolve()
  await Promise.resolve()
  render()

  assert.equal(intervalCallbacks.length, 1)
  assert.equal(requests.at(-1), '/api/plugins/hermes-achievements/achievements?locale=en')

  locale = 'zh-CN'
  render()
  await Promise.resolve()
  await Promise.resolve()

  assert.deepEqual(clearedIntervals, [1], 'the old-locale polling interval must be cleaned up')
  assert.equal(intervalCallbacks.length, 2, 'locale changes must create a fresh polling interval')

  intervalCallbacks.at(-1)!()
  await Promise.resolve()

  assert.equal(requests.at(-1), '/api/plugins/hermes-achievements/achievements?locale=zh-CN')
})
