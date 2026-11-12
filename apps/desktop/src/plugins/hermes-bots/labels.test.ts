import { describe, expect, it } from 'vitest'

import { connectionDisplayLabel } from './labels'

describe('connectionDisplayLabel', () => {
  it("names the shell's unrenamed local connection in the reader's language", () => {
    expect(connectionDisplayLabel('local', 'This device', 'Это устройство')).toBe('Это устройство')
    expect(connectionDisplayLabel('local', '', 'Это устройство')).toBe('Это устройство')
  })

  it('keeps a renamed or remote connection as the registry names it', () => {
    expect(connectionDisplayLabel('local', 'Studio Mac', 'Это устройство')).toBe('Studio Mac')
    expect(connectionDisplayLabel('remote-1', 'This device', 'Это устройство')).toBe('This device')
    expect(connectionDisplayLabel('remote-1', undefined, 'Это устройство')).toBe('')
  })
})
