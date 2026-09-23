import { spawn } from 'node:child_process'
import path from 'node:path'

import { evictConnectionCaches, sshInventoryAttemptedAt, sshRosterCache } from './connection-caches'
import { normalizeRemoteBaseUrl, normAuthMode, resolveTestWsUrl } from './connection-config'
import { removeConnection, setConnectionLaunchMode, setLastUsedConnection, setPrimaryConnection } from './connection-registry'
import { probeGatewayWebSocket } from './gateway-ws-probe'
import { collectSshConfigHosts, parseSshGOutput } from './ssh-config'
import { hiddenWindowsChildOptions } from './windows-child-options'

interface DesktopConnectionRegistryIpcDeps {
  ipcMain: any
  testDesktopConnectionConfig: any
  secretStoragePolicy: any
  applySecretStorageEncryption: any
  sanitizeConnectionsRegistry: any
  saveRegistryConnection: any
  managedConnectionUpdateGate: any
  readDesktopConnectionsRegistry: any
  writeDesktopConnectionsRegistry: any
  stopRegistryConnectionBackends: any
  broadcastConnectionsChanged: any
  desktopProfilePreferences: any
  assertCanMutateManagedPrimaryRouting: any
  probeSshProfileInventory: any
  startHermes: any
  decryptRemoteHeaders: any
  decryptDesktopSecret: any
  fetchConnectionStatus: any
  rememberConnectionInstallId: any
  mintGatewayWsTicket: any
}

export function registerDesktopConnectionRegistryIpc(deps: DesktopConnectionRegistryIpcDeps) {
  const {
    ipcMain,
    testDesktopConnectionConfig,
    secretStoragePolicy,
    applySecretStorageEncryption,
    sanitizeConnectionsRegistry,
    saveRegistryConnection,
    managedConnectionUpdateGate,
    readDesktopConnectionsRegistry,
    writeDesktopConnectionsRegistry,
    stopRegistryConnectionBackends,
    broadcastConnectionsChanged,
    desktopProfilePreferences,
    assertCanMutateManagedPrimaryRouting,
    probeSshProfileInventory,
    startHermes,
    decryptRemoteHeaders,
    decryptDesktopSecret,
    fetchConnectionStatus,
    rememberConnectionInstallId,
    mintGatewayWsTicket
  } = deps

ipcMain.handle('hermes:ssh-config:hosts', async () => ({ hosts: collectSshConfigHosts() }))
ipcMain.handle('hermes:ssh-config:resolve', async (_event, host) => {
  const value = String(host || '').trim()

  if (!value) {
    throw new Error('SSH host is required.')
  }

  const ssh =
    process.platform === 'win32'
      ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
      : 'ssh'

  return new Promise((resolve, reject) => {
    const child = spawn(ssh, ['-G', '--', value], hiddenWindowsChildOptions({ stdio: ['ignore', 'pipe', 'pipe'] }))
    let stdout = ''
    let stderr = ''

    const timer = setTimeout(() => {
      child.kill()
      reject(new Error('SSH config resolution timed out.'))
    }, 10_000)

    child.stdout.on('data', chunk => {
      stdout += String(chunk)
    })
    child.stderr.on('data', chunk => {
      stderr += String(chunk)
    })
    child.once('error', error => {
      clearTimeout(timer)
      reject(error)
    })
    child.once('close', code => {
      clearTimeout(timer)

      if (code !== 0) {
        reject(new Error(stderr.trim() || 'Could not resolve SSH host.'))
      } else {
        resolve(parseSshGOutput(stdout))
      }
    })
  })
})
ipcMain.handle('hermes:connection-config:test', async (_event, payload) => testDesktopConnectionConfig(payload))

// ── Opt-in keychain encryption for stored secrets ───────────────────────────
// get returns the current policy without touching safeStorage; set flips it
// and re-encodes every stored secret (see applySecretStorageEncryption).
ipcMain.handle('hermes:secret-storage:get', async () => ({ on: secretStoragePolicy().on }))
ipcMain.handle('hermes:secret-storage:set', async (_event: any, on: any) => applySecretStorageEncryption(on === true))

// ── v2 connection registry IPC (multi-source) ───────────────────────────────
// Storage-level CRUD for named agent sources. Routing/pooling consumption of
// the registry lands separately; these handlers only manage the persisted
// list, so they are safe to ship ahead of the switchover.
ipcMain.handle('hermes:connections:list', async () => sanitizeConnectionsRegistry())
ipcMain.handle('hermes:connections:save', async (_event, payload) => {
  const saved = await saveRegistryConnection(payload)

  return { ok: true, connection: saved, registry: sanitizeConnectionsRegistry() }
})
ipcMain.handle('hermes:connections:remove', async (_event, id) => {
  const key = String(id || '')
  managedConnectionUpdateGate.assertCanMutate(key)
  const registry = removeConnection(readDesktopConnectionsRegistry(), key)
  writeDesktopConnectionsRegistry(registry)
  // Tear down anything the removed connection still had running: pooled
  // backends under its composite keys and any ssh tunnel scopes it owned.
  await stopRegistryConnectionBackends(key)
  // …and everything cached ABOUT it. Ids are recycled label slugs, so re-adding "Mac mini"
  // gets `mac-mini` back — with the removed machine's profile list still cached under it.
  evictConnectionCaches(key)
  // And the renderer side: without this push, secondaries scoped to the
  // removed connection keep their WebSocket open (remote/cloud have no local
  // process to kill) and stream ghost events until page reload.
  broadcastConnectionsChanged({ connectionId: key, reason: 'removed' })
  desktopProfilePreferences.connectionRemoved(key)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-primary', async (_event, id) => {
  assertCanMutateManagedPrimaryRouting()
  const registry = setPrimaryConnection(readDesktopConnectionsRegistry(), String(id || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-launch-mode', async (_event, mode) => {
  assertCanMutateManagedPrimaryRouting()
  const registry = setConnectionLaunchMode(readDesktopConnectionsRegistry(), String(mode || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-last-used', async (_event, id) => {
  const registry = setLastUsedConnection(readDesktopConnectionsRegistry(), String(id || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:test', async (_event, id) => {
  const registry = readDesktopConnectionsRegistry()
  const entry = registry.connections.find(c => c.id === String(id || ''))

  if (!entry) {
    throw new Error(`No connection with id "${String(id || '')}".`)
  }

  // The ssh probe path in testDesktopConnectionConfig never consults v1
  // connection state, so mapping the entry onto it is safe.
  if (entry.kind === 'ssh') {
    const result = await testDesktopConnectionConfig({
      mode: 'ssh',
      sshHost: entry.host,
      sshUser: entry.user,
      sshPort: entry.port,
      sshKeyPath: entry.keyPath,
      sshRemoteHermesPath: entry.remoteHermesPath
    })

    if (result?.reachable) {
      sshInventoryAttemptedAt.delete(entry.id)
      sshRosterCache.delete(entry.id)
      await probeSshProfileInventory(entry)
    }

    return result
  }

  // Remote/cloud/local probe built DIRECTLY from the registry entry. Routing
  // through coerceDesktopConnectionConfig would use v1 connection.json as the
  // `existing` base: an entry with a broken/absent token would inherit the v1
  // global remote's token and send it to THIS entry's URL (cross-host
  // credential transmission + a false "reachable"), and testing the local
  // entry would probe whatever v1's global mode points at instead of the
  // app-managed local backend.
  let baseUrl
  let token = null
  let authMode = 'token'
  let testHeaders = {}

  if (entry.kind === 'local') {
    const local = await startHermes()
    baseUrl = local.baseUrl
    token = local.token
    authMode = normAuthMode(local.authMode)
  } else {
    baseUrl = normalizeRemoteBaseUrl(entry.url)
    authMode = normAuthMode(entry.authMode)
    testHeaders = decryptRemoteHeaders(entry.headers)

    if (authMode !== 'oauth') {
      token = decryptDesktopSecret(entry.token)

      if (!token) {
        throw new Error('This connection has no saved session token. Edit the connection and paste one.')
      }
    }
  }

  const status = (await fetchConnectionStatus(baseUrl, authMode, token, testHeaders)) as any

  // The Test button is the cheapest moment to (re)learn this backend's stable
  // identity for the same-backend roster collapse + Settings hint.
  rememberConnectionInstallId(entry.id, status)

  // Same HTTP+WS two-leg check as testDesktopConnectionConfig: HTTP alone is
  // a false positive when the WebSocket leg is blocked.
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, {
    mintTicket: url => mintGatewayWsTicket(url, testHeaders)
  })

  if (wsUrl && typeof globalThis.WebSocket === 'function') {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket, headers: testHeaders })

    if (!probe.ok) {
      throw new Error(
        `Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ${probe.reason} ` +
          'The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or gateway auth/origin guard.'
      )
    }
  }

  return { ok: true, baseUrl, version: status?.version || null }
})

}
