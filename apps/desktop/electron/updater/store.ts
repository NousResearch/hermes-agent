import type { RelaunchRegistration } from './relaunch'

import type { UpdaterApplyResultWire, UpdaterStatusWire, UpdaterStrategy } from './index'

export type StoreMode = 'check' | 'download' | 'install'

export interface StoreResult {
  available: boolean | null
  packages: string[]
  ok: boolean
  error?: string
}

export interface StoreStrategyDeps {
  run: (mode: StoreMode) => Promise<StoreResult>
  appVersion: string
  registerPendingRelaunch: (fromVersion: string) => Promise<RelaunchRegistration>
  teardown: () => Promise<void>
  restore: () => Promise<void>
  quit: () => void
  emitProgress: (progress: { stage: string; message: string; percent: number | null }) => void
}

export class StoreStrategy implements UpdaterStrategy {
  readonly mechanism = 'microsoft-store' as const

  constructor(private readonly deps: StoreStrategyDeps) {}

  async check(): Promise<UpdaterStatusWire> {
    const result = await this.deps.run('check')

    return {
      supported: true,
      mechanism: this.mechanism,
      currentVersion: this.deps.appVersion,
      updateAvailable: result.ok && result.available === true,
      error: result.ok && result.available !== null ? undefined : result.error || 'Microsoft Store check unavailable',
      fetchedAt: Date.now()
    }
  }

  async apply(): Promise<UpdaterApplyResultWire> {
    let registration: RelaunchRegistration | undefined
    let stopped = false

    try {
      this.deps.emitProgress({ stage: 'fetch', message: 'Downloading the update from Microsoft Store.', percent: null })
      const downloaded = await this.deps.run('download')

      if (!downloaded.ok || downloaded.available === null) {
        throw new Error(downloaded.error || 'Microsoft Store download did not complete')
      }

      if (!downloaded.available) {
        return { ok: true, updateAvailable: false, mechanism: this.mechanism }
      }

      registration = await this.deps.registerPendingRelaunch(this.deps.appVersion)

      if (!registration.automatic) {
        throw new Error('Could not register automatic relaunch for the Store update')
      }

      stopped = true
      await this.deps.teardown()
      this.deps.emitProgress({
        stage: 'restart',
        message: 'Microsoft Store is installing the update. Hermes will reopen.',
        percent: null
      })
      const installed = await this.deps.run('install')

      if (!installed.ok || installed.available !== true) {
        throw new Error(installed.error || 'Microsoft Store did not confirm installation')
      }

      this.deps.quit()

      return { ok: true, bundled: true, handedOff: true, mechanism: this.mechanism }
    } catch (error) {
      const errors: unknown[] = [error]

      try {
        await registration?.cancel()
      } catch (cancelError) {
        errors.push(cancelError)
      }

      if (stopped) {
        try {
          await this.deps.restore()
        } catch (restoreError) {
          errors.push(restoreError)
        }
      }

      const message = errors.map(item => (item instanceof Error ? item.message : String(item))).join('; ')
      this.deps.emitProgress({ stage: 'error', message, percent: null })
      throw errors.length > 1 ? new AggregateError(errors, message, { cause: error }) : error
    }
  }
}
