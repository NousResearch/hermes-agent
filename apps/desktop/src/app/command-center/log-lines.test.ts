import { describe, expect, it } from 'vitest'

import { commandCenterLogSeverity } from './log-lines'

describe('command center log lines', () => {
  it('recognizes explicit severity without depending on color', () => {
    expect(commandCenterLogSeverity('2026-09-23 21:15:07 WARNING gateway.run: slow')).toBe('WARNING')
    expect(commandCenterLogSeverity('2026-09-23 21:15:07 WARN gateway.run: slow')).toBe('WARNING')
    expect(commandCenterLogSeverity('2026-09-23 21:15:08 ERROR gateway.run: failed')).toBe('ERROR')
    expect(commandCenterLogSeverity('Traceback (most recent call last):')).toBeNull()
  })
})
