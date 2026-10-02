import { freeConnectorAllowed } from '@/app/capabilities/free-tier'
import { profileScoped } from '@/api/client'
import { addMcpServer, listMcpServers, setMcpServerEnabled } from '@/api/mcp'
import { setMcpBearerToken } from '@/app/capabilities/connectors/data/rpc'
import type { DesktopBrandConnector } from '@/global'
import { $brandSession, capabilitiesUnlocked } from '@/store/brand-session'

/** The shared test server. A signed-in brand does not use this key. */
const SHARED_TEST_MCP_SERVER = 'firecrawl'

let reloadWhenOpen: (() => void) | null = null

/** Config writes do not reach a chat that already started. Reload connects the new servers. */
async function reloadConnectedBrandMcp(): Promise<void> {
  const { $gateway } = await import('@/store/gateway')
  const { $activeSessionId } = await import('@/store/session')

  const send = (): boolean => {
    const gateway = $gateway.get()

    if (!gateway || gateway.connectionState !== 'open') {
      return false
    }

    void gateway
      .request('reload.mcp', { confirm: true, session_id: $activeSessionId.get() ?? undefined })
      .catch(() => undefined)

    return true
  }

  if (send()) {
    reloadWhenOpen?.()
    reloadWhenOpen = null
    return
  }

  if (reloadWhenOpen) {
    return
  }

  reloadWhenOpen = $gateway.subscribe(() => {
    if (!send()) {
      return
    }

    reloadWhenOpen?.()
    reloadWhenOpen = null
  })
}

let mcpWrites: Promise<void> = Promise.resolve()

function writeMcp(task: () => Promise<void>): Promise<void> {
  const run = mcpWrites.then(task, task)
  mcpWrites = run.then(
    () => undefined,
    () => undefined
  )
  return run
}

async function applyBrandMcp(connectors: DesktopBrandConnector[]): Promise<void> {
  const listed = await listMcpServers()
  const byName = new Map(listed.servers.map(server => [server.name, server]))

  if (byName.has(SHARED_TEST_MCP_SERVER)) {
    await setMcpServerEnabled(SHARED_TEST_MCP_SERVER, false)
  }

  const wanted = new Set<string>()

  const unlocked = capabilitiesUnlocked($brandSession.get())

  for (const row of connectors) {
    if (!row.hermesName || !row.mcpUrl || !row.credential) {
      continue
    }

    if (!freeConnectorAllowed(row.connectorId, unlocked)) {
      continue
    }

    wanted.add(row.hermesName)
    const existing = byName.get(row.hermesName)
    // Firecrawl's hosted server ignores a non-fc key and returns the 3 keyless tools.
    const sendKey = row.connectorId !== 'firecrawl' || row.credential.startsWith('fc-')

    if (!existing) {
      if (!sendKey) {
        continue
      }

      await addMcpServer({
        name: row.hermesName,
        url: row.mcpUrl,
        auth: 'header',
        bearer_token: row.credential
      })
    } else {
      if (!existing.enabled) {
        await setMcpServerEnabled(row.hermesName, true)
      }

      if (sendKey) {
        await setMcpBearerToken(profileScoped(), row.hermesName, row.credential).catch(() => undefined)
      }
    }
  }

  for (const server of listed.servers) {
    if (server.name.startsWith('ivx-') && !wanted.has(server.name) && server.enabled) {
      await setMcpServerEnabled(server.name, false)
    }
  }

  await reloadConnectedBrandMcp()
}

export function syncBrandMcp(connectors: DesktopBrandConnector[]): Promise<void> {
  return writeMcp(() => applyBrandMcp(connectors))
}

export function releaseBrandMcp(): Promise<void> {
  return writeMcp(async () => {
    const listed = await listMcpServers()
    const names = new Set(listed.servers.map(server => server.name))

    if (names.has(SHARED_TEST_MCP_SERVER)) {
      await setMcpServerEnabled(SHARED_TEST_MCP_SERVER, true)
    }

    for (const server of listed.servers) {
      if (server.name.startsWith('ivx-') && server.enabled) {
        await setMcpServerEnabled(server.name, false)
      }
    }

    await reloadConnectedBrandMcp()
  })
}
