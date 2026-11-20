import { describe, expect, it } from 'vitest'

import { readDesktopSshBudgets } from './desktop-ssh-budgets'

function read(lines: string[]) {
  const warned: string[] = []
  const logged: string[] = []

  const budgets = readDesktopSshBudgets(lines.join('\n'), {
    log: (msg: string) => void logged.push(msg),
    warn: (msg: string) => void warned.push(msg)
  })

  return { budgets, logged, warned }
}

describe('readDesktopSshBudgets (#132508)', () => {
  it('reads both keys in seconds from the desktop block only', () => {
    const { budgets, logged, warned } = read([
      'terminal:',
      '  ssh_connect_timeout: 5',
      'desktop:',
      '  ssh_connect_timeout: 45   # slow VPN',
      "  ssh_forward_timeout: '60'",
      'display:',
      '  ssh_forward_timeout: 9'
    ])

    expect(budgets).toEqual({ connectTimeoutMs: 45_000, forwardTimeoutMs: 60_000 })
    expect(warned).toEqual([])
    expect(logged).toHaveLength(2)
  })

  it('reads past a column-0 comment inside the desktop block', () => {
    const { budgets } = read(['desktop:', '  ssh_connect_timeout: 45', '# slow link', '  ssh_forward_timeout: 60'])

    expect(budgets).toEqual({ connectTimeoutMs: 45_000, forwardTimeoutMs: 60_000 })
  })

  it('keeps the default and warns once per key when a value is not whole seconds', () => {
    const { budgets, warned } = read(['desktop:', '  ssh_connect_timeout: 45s', '  ssh_forward_timeout: 1.5'])

    expect(budgets).toEqual({})
    expect(warned).toHaveLength(2)
    expect(warned[0]).toContain('desktop.ssh_connect_timeout was ignored')
    expect(warned[1]).toContain('desktop.ssh_forward_timeout was ignored')
  })

  it('caps a huge value and treats an empty or absent key as unset without warning', () => {
    expect(read(['desktop:', '  ssh_connect_timeout: 999999']).budgets).toEqual({ connectTimeoutMs: 120_000 })

    for (const lines of [['desktop:', '  ssh_connect_timeout:'], ['desktop:', '  ssh_path: x'], ['model: x'], []]) {
      const { budgets, warned } = read(lines)

      expect(budgets).toEqual({})
      expect(warned).toEqual([])
    }
  })
})
