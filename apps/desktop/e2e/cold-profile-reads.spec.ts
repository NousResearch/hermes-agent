import * as fs from 'node:fs'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { createSandbox } from './fixtures'
import { launchSettingsDesktop, seedSettingsProfile, type SettingsDesktopWindow } from './settings-fixtures'
import { expect, test } from './test'

test('registry reads cold local profiles without consuming background pool slots', async () => {
  test.skip(
    process.platform === 'win32',
    'The real-backend fixture uses a POSIX virtual environment and HOME isolation'
  )
  test.setTimeout(180_000)
  const sandbox = createSandbox('cold-profile-reads')
  sandbox.hermesHome = path.join(sandbox.root, '.hermes')
  const mock = await startMockServer()
  let app: Awaited<ReturnType<typeof launchSettingsDesktop>>['app'] | undefined

  try {
    seedSettingsProfile(sandbox.hermesHome, mock.url, 25)

    for (const [index, name] of ['warm-a', 'warm-b', 'reader-a', 'reader-b'].entries()) {
      seedSettingsProfile(path.join(sandbox.hermesHome, 'profiles', name), mock.url, 31 + index)
    }

    fs.writeFileSync(
      path.join(sandbox.userDataDir, 'connections.json'),
      JSON.stringify({
        version: 2,
        primary: 'local',
        launchMode: 'primary',
        lastUsed: 'local',
        connections: [{ id: 'local', kind: 'local', label: 'This device' }]
      })
    )
    fs.writeFileSync(
      path.join(sandbox.userDataDir, 'pool-limits.json'),
      JSON.stringify({ maxBackends: 3, idleMs: 600_000 })
    )
    const launched = await launchSettingsDesktop(sandbox)
    app = launched.app
    const page = launched.page
    await expect(page.locator('[data-slot="composer-rich-input"]')).toBeVisible({ timeout: 90_000 })

    // Occupy both background leases through the real preload/main-process bridge.
    await page.evaluate(async () => {
      for (const profile of ['warm-a', 'warm-b']) {
        await (window as unknown as SettingsDesktopWindow).hermesDesktop.getConnectionFor({
          connectionId: 'local',
          profile,
          priority: 'background'
        })
      }
    })

    // A → B → A must keep the explicit profile query even though the process is shared.
    for (const profile of ['reader-a', 'reader-b', 'reader-a']) {
      const observed = await page.evaluate(async profile => {
        const api = (window as unknown as SettingsDesktopWindow).hermesDesktop.api
        const owner = { connectionId: 'local', profile }

        const config = await api<{ agent: { max_turns: number } }>({
          ...owner,
          path: '/api/config?include_defaults=false'
        })

        const accounts = await api<{ providers: unknown[] }>({ ...owner, path: '/api/providers/oauth' })

        const endpoints = await api<{ endpoints: { id: string }[] }>({
          ...owner,
          path: '/api/providers/custom-endpoints'
        })

        return {
          maxTurns: config.agent.max_turns,
          accounts: Array.isArray(accounts.providers),
          endpointIds: endpoints.endpoints.map(item => item.id)
        }
      }, profile)

      expect(observed.maxTurns).toBe(profile === 'reader-a' ? 33 : 34)
      expect(observed.accounts).toBe(true)
      expect(observed.endpointIds).toContain('mock')
    }
  } finally {
    await app?.close()
    await mock.close()
    sandbox.cleanup()
  }
})
