import { applyPackagedHandoff } from './packaged-handoff'
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
    return applyPackagedHandoff(
      {
        teardown: this.deps.teardown,
        restore: this.deps.restore,
        emitProgress: this.deps.emitProgress,
        relaunch: {
          register: (): Promise<RelaunchRegistration> => this.deps.registerPendingRelaunch(this.deps.appVersion),
          onManual: (): never => {
            throw new Error('Could not register automatic relaunch for the Store update')
          }
        }
      },
      async (stop: () => Promise<void>): Promise<UpdaterApplyResultWire> => {
        this.deps.emitProgress({
          stage: 'fetch',
          message: 'Downloading the update from Microsoft Store.',
          percent: null
        })
        const downloaded = await this.deps.run('download')

        if (!downloaded.ok || downloaded.available === null) {
          throw new Error(downloaded.error || 'Microsoft Store download did not complete')
        }

        if (!downloaded.available) {
          return { ok: true, updateAvailable: false, mechanism: this.mechanism }
        }

        await stop()
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
      }
    )
  }
}
