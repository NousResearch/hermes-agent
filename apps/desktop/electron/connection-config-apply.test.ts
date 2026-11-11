import { describe, expect, it, vi } from 'vitest'

import { applyConnectionConfigAtomically } from './connection-config-apply'

describe('applyConnectionConfigAtomically', () => {
  it('commits legacy and registry state before activation', async () => {
    const events: string[] = []

    await applyConnectionConfigAtomically({
      previousConfig: 'old-config',
      previousRegistry: 'old-registry',
      nextConfig: 'remote-config',
      nextRegistry: 'remote-registry',
      writeConfig: value => events.push(`config:${value}`),
      writeRegistry: value => events.push(`registry:${value}`),
      apply: async () => {
        events.push('activate')
      }
    })

    expect(events).toEqual(['config:remote-config', 'registry:remote-registry', 'activate'])
  })

  it('rolls both stores back when activation fails', async () => {
    const writeConfig = vi.fn()
    const writeRegistry = vi.fn()

    await expect(
      applyConnectionConfigAtomically({
        previousConfig: 'local-config',
        previousRegistry: 'local-registry',
        nextConfig: 'remote-config',
        nextRegistry: 'remote-registry',
        writeConfig,
        writeRegistry,
        apply: async () => {
          throw new Error('activation failed')
        }
      })
    ).rejects.toThrow('activation failed')

    expect(writeConfig.mock.calls).toEqual([['remote-config'], ['local-config']])
    expect(writeRegistry.mock.calls).toEqual([['remote-registry'], ['local-registry']])
  })

  it('rolls legacy state back when the registry write fails', async () => {
    const writes: string[] = []
    let registryWrites = 0

    await expect(
      applyConnectionConfigAtomically({
        previousConfig: 'local-config',
        previousRegistry: 'local-registry',
        nextConfig: 'remote-config',
        nextRegistry: 'remote-registry',
        writeConfig: value => writes.push(`config:${value}`),
        writeRegistry: value => {
          registryWrites += 1

          if (registryWrites === 1) {
            throw new Error('disk full')
          }

          writes.push(`registry:${value}`)
        },
        apply: vi.fn()
      })
    ).rejects.toThrow('disk full')

    expect(writes).toEqual(['config:remote-config', 'config:local-config', 'registry:local-registry'])
  })

  it('preflights before writing either store', async () => {
    const events: string[] = []

    await applyConnectionConfigAtomically({
      previousConfig: 'local-config',
      previousRegistry: 'local-registry',
      nextConfig: 'remote-config',
      nextRegistry: 'remote-registry',
      preflight: async () => {
        events.push('preflight')
      },
      writeConfig: value => events.push(`config:${value}`),
      writeRegistry: value => events.push(`registry:${value}`),
      apply: async () => {
        events.push('activate')
      }
    })

    expect(events).toEqual(['preflight', 'config:remote-config', 'registry:remote-registry', 'activate'])
  })

  it('leaves both stores untouched when the preflight rejects', async () => {
    const writeConfig = vi.fn()
    const writeRegistry = vi.fn()
    const apply = vi.fn()

    await expect(
      applyConnectionConfigAtomically({
        previousConfig: 'local-config',
        previousRegistry: 'local-registry',
        nextConfig: 'remote-config',
        nextRegistry: 'remote-registry',
        preflight: async () => {
          throw new Error('gateway unreachable')
        },
        writeConfig,
        writeRegistry,
        apply
      })
    ).rejects.toThrow('gateway unreachable')

    expect(writeConfig).not.toHaveBeenCalled()
    expect(writeRegistry).not.toHaveBeenCalled()
    expect(apply).not.toHaveBeenCalled()
  })
})
