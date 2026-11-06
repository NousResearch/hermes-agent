import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { type McpTestResult, type ProfileScope, testMcpServer } from '@/hermes'
import { PROBE_TTL_MS, probeCache, probeKey, serverFingerprint } from '@/lib/mcp-probe-cache'
import { type McpServerEntry, type McpServers, serverEnabled } from '@/lib/mcp-servers'
import { countEnabledTools } from '@/lib/mcp-tool-filter'

import { loadMcpUsage, okProbe, type Probe, serverCost, type ServerCost } from './mcp-status'

export interface McpProbes {
  costFor: (name: string, entry: McpServerEntry) => ServerCost
  probes: Record<string, Probe>
  publishAuthorized: (name: string, tools: McpTestResult['tools'], server: McpServerEntry) => void
  resetForProfileSwitch: () => void
  retainProbes: (entries: McpServers, prevServers: McpServers) => void
  runProbe: (name: string) => Promise<void>
  setProbe: (name: string, value: Probe) => void
  toolCounts: Record<string, { on: number; total: number }>
  usageByServer: Record<string, number>
}

export interface UseMcpProbesOptions {
  appProfile: ProfileScope
  profile?: ProfileScope
  profileEpoch: RefObject<number>
  scopeProfileKey: string
  servers: McpServers
}

export function useMcpProbes({
  appProfile,
  profile,
  profileEpoch,
  scopeProfileKey,
  servers
}: UseMcpProbesOptions): McpProbes {
  const [probes, setProbes] = useState<Record<string, Probe>>({})
  const probesRef = useRef(probes)
  probesRef.current = probes
  const probeRevision = useRef<Record<string, number>>({})

  const [toolCalls30d, setToolCalls30d] = useState<null | Record<string, number>>(null)

  const setProbe = (serverName: string, value: Probe) => {
    probeRevision.current[serverName] = (probeRevision.current[serverName] ?? 0) + 1
    setProbes(current => ({ ...current, [serverName]: value }))
  }

  const publishAuthorized = (serverName: string, tools: McpTestResult['tools'], server: McpServerEntry) => {
    const result: McpTestResult = { ok: true, tools }
    setProbe(serverName, result)
    probeCache.set(probeKey(serverName, server, scopeProfileKey), { at: Date.now(), result })
  }

  const runProbe = async (serverName: string) => {
    const epoch = profileEpoch.current
    const revision = (probeRevision.current[serverName] ?? 0) + 1
    probeRevision.current[serverName] = revision
    const key = probeKey(serverName, servers[serverName], scopeProfileKey)
    setProbes(current => ({ ...current, [serverName]: 'probing' }))

    try {
      const result = await testMcpServer(serverName, profile ?? undefined)

      if (profileEpoch.current !== epoch || probeRevision.current[serverName] !== revision) {
        return
      }

      probeCache.set(key, { at: Date.now(), result })
      setProbes(current => ({ ...current, [serverName]: result }))
    } catch (err) {
      if (profileEpoch.current !== epoch || probeRevision.current[serverName] !== revision) {
        return
      }

      const result = { ok: false, error: err instanceof Error ? err.message : String(err), tools: [] }
      probeCache.set(key, { at: Date.now(), result })
      setProbes(current => ({ ...current, [serverName]: result }))
    }
  }

  useEffect(() => {
    for (const [serverName, server] of Object.entries(servers)) {
      if (!serverEnabled(server) || probesRef.current[serverName] !== undefined) {
        continue
      }

      const cached = probeCache.get(probeKey(serverName, server, scopeProfileKey))

      if (cached && Date.now() - cached.at < PROBE_TTL_MS) {
        setProbes(current => ({ ...current, [serverName]: cached.result }))
      } else {
        void runProbe(serverName)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers])

  useEffect(() => {
    const epoch = profileEpoch.current

    void loadMcpUsage(scopeProfileKey, profile ?? appProfile ?? null).then(value => {
      if (profileEpoch.current === epoch) {
        setToolCalls30d(value)
      }
    })
  }, [scopeProfileKey, profile, appProfile, profileEpoch])

  const costFor = (serverName: string, server: McpServerEntry): ServerCost =>
    serverCost(server, probes[serverName], serverName, toolCalls30d)

  const toolCounts = useMemo(() => {
    const counts: Record<string, { on: number; total: number }> = {}

    for (const [serverName, server] of Object.entries(servers)) {
      const probe = okProbe(probes[serverName])

      if (probe) {
        counts[serverName] = {
          on: countEnabledTools(
            server,
            probe.tools.map(tool => tool.name)
          ),
          total: probe.tools.length
        }
      }
    }

    return counts
  }, [probes, servers])

  const usageByServer = useMemo(() => {
    const uses: Record<string, number> = {}

    if (!toolCalls30d) {
      return uses
    }

    for (const serverName of Object.keys(servers)) {
      uses[serverName] = serverCost(servers[serverName], probes[serverName], serverName, toolCalls30d).uses ?? 0
    }

    return uses
  }, [probes, servers, toolCalls30d])

  const retainProbes = (entries: McpServers, prevServers: McpServers) => {
    setProbes(current =>
      Object.fromEntries(
        Object.entries(current).filter(
          ([name]) => name in entries && serverFingerprint(entries[name]) === serverFingerprint(prevServers[name] ?? {})
        )
      )
    )
  }

  const resetForProfileSwitch = () => {
    probeRevision.current = {}
    setProbes({})
    setToolCalls30d(null)
  }

  return {
    costFor,
    probes,
    publishAuthorized,
    resetForProfileSwitch,
    retainProbes,
    runProbe,
    setProbe,
    toolCounts,
    usageByServer
  }
}
