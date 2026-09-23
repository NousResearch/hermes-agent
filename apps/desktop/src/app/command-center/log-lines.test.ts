import { describe, expect, it } from 'vitest'

import { commandCenterLogSeverity, highlightLogSegments } from './log-lines'

describe('command center log lines', () => {
  it('recognizes explicit severity without depending on color', () => {
    expect(commandCenterLogSeverity('2026-09-23 21:15:07 WARNING gateway.run: slow')).toBe('WARNING')
    expect(commandCenterLogSeverity('2026-09-23 21:15:07 WARN gateway.run: slow')).toBe('WARNING')
    expect(commandCenterLogSeverity('2026-09-23 21:15:08 ERROR gateway.run: failed')).toBe('ERROR')
    expect(commandCenterLogSeverity('Traceback (most recent call last):')).toBeNull()
  })

  it('splits every case-insensitive search match while preserving original text', () => {
    expect(highlightLogSegments('Docker docker DOCKER', 'docker')).toEqual([
      { match: true, text: 'Docker' },
      { match: false, text: ' ' },
      { match: true, text: 'docker' },
      { match: false, text: ' ' },
      { match: true, text: 'DOCKER' }
    ])
  })
})
