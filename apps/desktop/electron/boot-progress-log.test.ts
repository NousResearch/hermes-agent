import { describe, expect, it } from 'vitest'

import { createBootMessageLogger } from './boot-progress-log'

describe('createBootMessageLogger', () => {
  it('logs a message repeated on every progress tick once, and again after it changes back', () => {
    const lines: string[] = []
    const logBoot = createBootMessageLogger(line => lines.push(line))
    const parked = 'An update is finishing — Hermes will start automatically when it completes…'

    for (let tick = 0; tick < 600; tick += 1) {
      logBoot(parked)
    }

    logBoot('Starting Hermes backend')
    logBoot('Hermes backend is ready. Finalizing desktop startup')
    logBoot(parked)
    logBoot(undefined)

    expect(lines).toEqual([
      `[boot] ${parked}`,
      '[boot] Starting Hermes backend',
      '[boot] Hermes backend is ready. Finalizing desktop startup',
      `[boot] ${parked}`
    ])
  })
})
