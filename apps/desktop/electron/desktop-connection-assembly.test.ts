import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { createDesktopConnectionAssembly } from './desktop-connection-assembly'

test('connection assembly keeps forward backend, terminal, route, and SSH callbacks lazy', () => {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-connection-assembly-'))
  const premature = vi.fn(() => { throw new Error('deferred callback ran during assembly') })

  try {
    const connections = createDesktopConnectionAssembly({
      app: { getPath: () => userData, isReady: () => false },
      BrowserWindow: { getAllWindows: () => [] },
      electronNet: {},
      session: {},
      safeStorage: {},
      DESKTOP_CONNECTION_CONFIG_PATH: path.join(userData, 'connection.json'),
      DESKTOP_CONNECTIONS_REGISTRY_PATH: path.join(userData, 'connections.json'),
      DESKTOP_INSTALLATION_PATH: path.join(userData, 'desktop-installation.json'),
      DESKTOP_MANAGED_SSH_RECOVERY_PATH: path.join(userData, 'managed-ssh-update-recovery.json'),
      DESKTOP_PROFILE_CONFIG_PATH: path.join(userData, 'active-profile.json'),
      HERMES_HOME: userData,
      PROFILE_NAME_RE: /^[a-z][a-z0-9-]*$/,
      GUEST_ONBOARDING: false,
      fetchJson: premature,
      fetchPublicJson: premature,
      rememberLog: vi.fn(),
      writeFileAtomic: premature,
      ensureBackend: premature,
      ensureRegistryBackend: premature,
      stopRegistryConnectionBackends: premature,
      primaryProfileKey: premature,
      getIsolatedBackend: premature,
      backendDialClaims: {},
      waitForHermes: premature,
      getWindowConnectionRoute: premature,
      disposeTerminalSessionsForSshScope: premature,
      managedSshConfig: premature,
      getMainWindow: premature,
      startHermes: premature
    })

    expect(premature).not.toHaveBeenCalled()
    expect(connections.managedUpdateQuitState.wait).toBeNull()
    expect(connections.managedUpdateQuitState.done).toBe(false)
    expect(typeof connections.bootstrapSshConnection).toBe('function')
    expect(typeof connections.readDesktopConnectionsRegistry).toBe('function')
    expect(typeof connections.ensureNativeAccessToken).toBe('function')
  } finally {
    fs.rmSync(userData, { recursive: true, force: true })
  }
})
