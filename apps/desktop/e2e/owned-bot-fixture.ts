import type { ChildProcess } from 'node:child_process'

import type { ElectronApplication } from '@playwright/test'

import type { MockBackendFixture, Sandbox } from './fixtures'
import { cleanupAfterOwnedElectron, finishOwnedElectronShutdown } from './owned-electron-cleanup'

interface OwnedBotOptions {
  sandbox: Sandbox
  startMock: () => Promise<MockBackendFixture['mock']>
  seed: (mockUrl: string, childAttempted: () => void) => Promise<void>
  launch: (onLaunched: (app: ElectronApplication) => void) => Promise<Pick<MockBackendFixture, 'app' | 'page'>>
  ready: (fixture: MockBackendFixture) => Promise<void>
  onRetained: (root: string) => Promise<void>
}

export async function withOwnedSeeder<T>(builder: { close: () => Promise<void> }, seed: () => Promise<T>): Promise<void> {
  const failures: unknown[] = []

  try {
    await seed()
  } catch (error) {
    failures.push(error)
  } finally {
    await builder.close().catch(error => failures.push(error))
  }

  if (failures.length > 0) {
    throw new AggregateError(failures, 'Bot seeding or owned seeder cleanup failed')
  }
}

function waitForOwnedExit(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) {return Promise.resolve()}

  return new Promise<void>((resolve, reject) => {
    const onExit = () => {
      clearTimeout(timer)
      resolve()
    }

    const timer = setTimeout(() => {
      child.off('exit', onExit)
      reject(new Error('Owned Bot Electron exit unconfirmed after 5 seconds'))
    }, 5_000)

    child.once('exit', onExit)
  })
}

export async function setupOwnedBotFixture(options: OwnedBotOptions): Promise<MockBackendFixture> {
  let mock: MockBackendFixture['mock'] | undefined
  let app: ElectronApplication | undefined
  let child: ChildProcess | undefined
  let childAttempted = false
  let cleanupPromise: Promise<void> | undefined

  const cleanup = (): Promise<void> => {
    cleanupPromise ??= (async () => {
      const failures: unknown[] = []

      try {
        if (app && child) {
          const ownedApp = app
          const ownedChild = child
          await finishOwnedElectronShutdown({
            close: () => ownedApp.close(),
            killIfRunning: () => {
              if (ownedChild.exitCode === null && ownedChild.signalCode === null && !ownedChild.kill()) {
                throw new Error('Could not signal the owned Bot Electron process')
              }
            },
            waitForExit: () => waitForOwnedExit(ownedChild)
          }).catch(error => failures.push(error))
        }
      } finally {
        try {
          await mock?.close().catch(error => failures.push(error))
        } finally {
          try {
            if (!cleanupAfterOwnedElectron({ launchAttempted: childAttempted }, options.sandbox.cleanup)) {
              await options.onRetained(options.sandbox.root)
            }
          } catch (error) {
            failures.push(error)
          }
        }
      }

      if (failures.length > 0) {
        throw new AggregateError(failures, 'Owned Bot fixture cleanup failed')
      }
    })()

    return cleanupPromise
  }

  try {
    mock = await options.startMock()
    await options.seed(mock.url, () => { childAttempted = true })
    childAttempted = true

    const launched = await options.launch(ownedApp => {
      app = ownedApp
      child = ownedApp.process()
    })

    const fixture: MockBackendFixture = {
      ...launched, mock, mockUrl: mock.url, sandbox: options.sandbox, cleanup
    }

    await options.ready(fixture)

    return fixture
  } catch (error) {
    const failures = [error]
    await cleanup().catch(cleanupError => failures.push(cleanupError))

    throw new AggregateError(failures, 'Bot fixture setup failed with owned cleanup attempted')
  }
}
