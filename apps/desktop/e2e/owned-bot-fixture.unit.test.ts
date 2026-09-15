import { EventEmitter } from 'node:events'

import type { ElectronApplication } from '@playwright/test'
import { expect, test } from 'vitest'

import type { MockBackendFixture } from './fixtures'
import { setupOwnedBotFixture, withOwnedSeeder } from './owned-bot-fixture'

function scenario() {
  const evidence = new Map([['desktop.log', 'Synthetic evidence']])
  const steps: string[] = []

  const child = Object.assign(new EventEmitter(), {
    exitCode: null as number | null,
    signalCode: null as NodeJS.Signals | null,
    kill: () => { child.exitCode = 0; child.emit('exit'); steps.push('owned kill');

 return true }
  })

  const app = {
    process: () => child,
    close: async () => { child.exitCode = 0; child.emit('exit'); steps.push('app close') }
  } as unknown as ElectronApplication

  const mock = { url: 'http://127.0.0.1:1', close: async () => { steps.push('mock close') } } as MockBackendFixture['mock']

  const options = {
    sandbox: { root: '/synthetic', hermesHome: '/synthetic/home', userDataDir: '/synthetic/data', cleanup: () => evidence.clear() },
    startMock: async () => mock,
    seed: async (_url: string, childAttempted: () => void) => { childAttempted() },
    launch: async (onLaunched: (owned: ElectronApplication) => void) => {
      onLaunched(app)

      return { app, page: {} as MockBackendFixture['page'] }
    },
    ready: async () => {},
    onRetained: async (root: string) => { steps.push(`retained ${root}`) }
  }

  return { evidence, steps, app, child, mock, options }
}

test('seeding failure retains the whole sandbox and closes the mock before returning the original error', async () => {
  const setup = scenario()
  const original = new Error('Synthetic seed failure')

  setup.options.seed = async (_url, attempted) => { attempted(); throw original }
  const error = await setupOwnedBotFixture(setup.options).catch(failure => failure)

  expect(setup.evidence.has('desktop.log')).toBe(true)
  expect(setup.steps).toContain('mock close')
  expect(setup.steps).toContain('retained /synthetic')
  expect(error.errors[0]).toBe(original)
})

test('early launch ownership closes Electron even if firstWindow fails and preserves mock-close errors', async () => {
  const setup = scenario()
  const original = new Error('Synthetic firstWindow failure')
  const closeError = new Error('Synthetic app close failure')
  const mockError = new Error('Synthetic mock close failure')

  setup.app.close = async () => { throw closeError }

  setup.mock.close = async () => { setup.steps.push('mock close'); throw mockError }

  setup.options.launch = async onLaunched => { onLaunched(setup.app); throw original }
  const error = await setupOwnedBotFixture(setup.options).catch(failure => failure)

  expect(setup.steps).toContain('owned kill')
  expect(setup.steps).toContain('mock close')
  expect(setup.evidence.has('desktop.log')).toBe(true)
  expect(error.errors[0]).toBe(original)
  expect(error.errors[1].errors[0].errors).toEqual([closeError])
  expect(error.errors[1].errors[1]).toBe(mockError)
})

test('READY failure closes the owned app and mock while preserving seeding evidence', async () => {
  const setup = scenario()
  const original = new Error('Synthetic app READY failure')

  setup.options.ready = async () => { throw original }
  const error = await setupOwnedBotFixture(setup.options).catch(failure => failure)

  expect(setup.steps).toEqual(['app close', 'mock close', 'retained /synthetic'])
  expect(setup.evidence.has('desktop.log')).toBe(true)
  expect(error.errors[0]).toBe(original)
})

test('successful fixture cleanup retains the sandbox after seeding and does not repeat owned cleanup', async () => {
  const setup = scenario()
  const fixture = await setupOwnedBotFixture(setup.options)
  await fixture.cleanup()
  await fixture.cleanup()

  expect(setup.evidence.has('desktop.log')).toBe(true)
  expect(setup.steps).toEqual(['app close', 'mock close', 'retained /synthetic'])
})

test('setup failure before any child attempt can remove its synthetic sandbox', async () => {
  const setup = scenario()
  const original = new Error('Synthetic mock startup failure')

  setup.options.startMock = async () => { throw original }
  const error = await setupOwnedBotFixture(setup.options).catch(failure => failure)

  expect(setup.evidence.has('desktop.log')).toBe(false)
  expect(error.errors[0]).toBe(original)
})

test('seeding and builder-close errors both survive with the original seeding error first', async () => {
  const original = new Error('Synthetic createSession failure')
  const closeError = new Error('Synthetic seeder close failure')

  const error = await withOwnedSeeder({ close: async () => { throw closeError } }, async () => { throw original })
    .catch(failure => failure)

  expect(error).toBeInstanceOf(AggregateError)
  expect(error.errors).toEqual([original, closeError])
})
