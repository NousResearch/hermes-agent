import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionKind } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import * as notifications from '@/store/notifications'

import { applySimpleAgentDefaults, SIMPLE_DENY_GLOBS } from './simple-agent-defaults'

function registerConnection(kind: DesktopConnectionKind) {
  $connectionsRegistry.set({
    version: 1,
    primary: 'fixture',
    secureTokenStorage: true,
    connections: [{ id: 'fixture', kind, label: 'Fixture', tokenSet: false, tokenPreview: null }]
  })
}

function installApi(api: (request: Record<string, unknown>) => Promise<unknown>) {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
}

// Answers the user-config read with `userConfig`, every write with ok.
function configApi(userConfig: Record<string, unknown>) {
  return vi.fn(async (request: Record<string, unknown>) => (request.method === 'PUT' ? { ok: true } : userConfig))
}

afterEach(() => {
  $connectionsRegistry.set(null)
  vi.restoreAllMocks()
})

describe('applySimpleAgentDefaults', () => {
  it('reads the user config, then writes the preset to the same connection and profile', async () => {
    registerConnection('local')
    const api = configApi({})
    installApi(api)

    await applySimpleAgentDefaults({ connectionId: 'fixture', profile: 'work' })

    expect(api).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ connectionId: 'fixture', profile: 'work', path: '/api/config?include_defaults=false' })
    )
    expect(api).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        connectionId: 'fixture',
        profile: 'work',
        method: 'PUT',
        path: '/api/config',
        body: {
          config: {
            checkpoints: { enabled: true },
            agent: { coding_context: 'off' },
            approvals: { deny: SIMPLE_DENY_GLOBS }
          }
        }
      })
    )
  })

  it("keeps the user's deny globs and never overrides a value they set", async () => {
    registerConnection('local')

    const api = configApi({
      approvals: { deny: ['npm publish*', 'git *reset*--hard*'] },
      checkpoints: { enabled: false },
      agent: { coding_context: 'auto' }
    })

    installApi(api)

    await applySimpleAgentDefaults({ connectionId: 'fixture', profile: 'work' })

    expect(api).toHaveBeenLastCalledWith(
      expect.objectContaining({
        method: 'PUT',
        body: {
          config: {
            approvals: {
              deny: ['npm publish*', 'git *reset*--hard*', ...SIMPLE_DENY_GLOBS.filter(g => g !== 'git *reset*--hard*')]
            }
          }
        }
      })
    )
  })

  it.each<DesktopConnectionKind>(['remote', 'ssh', 'cloud'])(
    'never touches a %s connection and says so once',
    async kind => {
      registerConnection(kind)
      const api = configApi({})
      installApi(api)
      const notify = vi.spyOn(notifications, 'notify').mockReturnValue('toast')

      await applySimpleAgentDefaults({ connectionId: 'fixture', profile: 'work' })

      expect(api).not.toHaveBeenCalled()
      expect(notify).toHaveBeenCalledOnce()
      expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'info' }))
    }
  )

  it('stays quiet when the preset is written', async () => {
    registerConnection('local')
    installApi(configApi({}))
    const notify = vi.spyOn(notifications, 'notify').mockReturnValue('toast')

    await applySimpleAgentDefaults({ connectionId: 'fixture', profile: 'work' })

    expect(notify).not.toHaveBeenCalled()
  })

  it('turns a failed write into a toast instead of throwing', async () => {
    registerConnection('local')
    installApi(async () => {
      throw new Error('disk full')
    })
    const notifyError = vi.spyOn(notifications, 'notifyError').mockReturnValue('toast')

    await expect(applySimpleAgentDefaults({ connectionId: 'fixture', profile: 'work' })).resolves.toBeUndefined()

    expect(notifyError).toHaveBeenCalledOnce()
  })
})
