import path from 'node:path'

import { launchRecovery, prepareRecovery, readRecovery } from './update-recovery'

interface RecoveryOptions {
  root: string
  home: string
  executable: string
  version: string
  supported: boolean
  stop: () => Promise<{ unlocked: boolean }>
  restart: () => Promise<unknown>
  quit: () => void
}

export function createRecoveryController(options: RecoveryOptions) {
  let gate: Promise<void> | null = null
  const snapshot = () => prepareRecovery({ ...options, installRoot: path.dirname(options.executable) })

  const run = async (restore: boolean) => {
    if (!options.supported) {
      throw new Error('Kopia aplikacji jest dostępna w pełnym pakiecie Windows.')
    }

    if (gate) {
      throw new Error('Trwa już operacja przywracania lub kopii.')
    }

    let release = () => {}
    gate = new Promise<void>(resolve => {
      release = resolve
    })
    let quitting = false

    try {
      if (!(await options.stop()).unlocked) {
        throw new Error('Nie udało się zatrzymać backendu. Nie zmieniono danych.')
      }

      if (restore) {
        await launchRecovery(options.root, process.pid, path.dirname(options.executable))
        quitting = true
        options.quit()

        return { ok: true }
      }

      await snapshot()

      return { ok: true }
    } finally {
      release()
      gate = null

      if (!quitting) {
        await options.restart()
      }
    }
  }

  return {
    wait: async () => {
      await gate
    },
    status: () => {
      const record = readRecovery(options.root)

      return {
        supported: options.supported,
        available: Boolean(record),
        createdAt: record?.createdAt,
        version: record?.version,
        status: record?.status
      }
    },
    prepare: () => run(false),
    restore: () => run(true),
    // The updater has already stopped the backends and owns its startup gate.
    checkpoint: snapshot
  }
}
