import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { type ProfileScope, testMcpServer } from '@/hermes'
import { PROBE_TTL_MS, probeCache, probeKey, serverFingerprint } from '@/lib/mcp-probe-cache'
import { type McpServerEntry, type McpServers, serverEnabled } from '@/lib/mcp-servers'
import { countEnabledTools } from '@/lib/mcp-tool-filter'

import { loadMcpUsage, okProbe, type Probe, serverCost, type ServerCost } from './mcp-status'

interface MountedProbe {
  at: number
  epoch: number
  key: string
  value: Probe
}

export interface McpProbes {
  costFor: (name: string, entry: McpServerEntry) => ServerCost
  probes: Record<string, Probe>
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
  const [mountedProbes, setMountedProbes] = useState<Record<string, MountedProbe>>({})
  const probesRef = useRef(mountedProbes)
  const scopeRef = useRef({ servers, scopeProfileKey })
  scopeRef.current = { servers, scopeProfileKey }

  const [toolCalls30d, setToolCalls30d] = useState<null | Record<string, number>>(null)

  const replaceProbes = (next: Record<string, MountedProbe>) => {
    // Update synchronously so superseding calls and effect replays see the new owner.
    probesRef.current = next
    setMountedProbes(next)
  }

  const putProbe = (serverName: string, key: string, value: Probe, at = Date.now()): MountedProbe => {
    const mounted = { key, value, at, epoch: profileEpoch.current }
    replaceProbes({ ...probesRef.current, [serverName]: mounted })

    return mounted
  }

  const setProbe = (serverName: string, value: Probe) => {
    // Manual authentication owns the row too, superseding any pending health probe.
    putProbe(serverName, probeKey(serverName, servers[serverName], scopeProfileKey), value)
  }

  const runProbe = async (serverName: string) => {
    const epoch = profileEpoch.current
    const key = probeKey(serverName, servers[serverName], scopeProfileKey)
    const pending = putProbe(serverName, key, 'probing')

    const isCurrent = () => {
      const scope = scopeRef.current
      const server = scope.servers[serverName]

      return (
        profileEpoch.current === epoch &&
        probesRef.current[serverName] === pending &&
        server !== undefined &&
        serverEnabled(server) &&
        probeKey(serverName, server, scope.scopeProfileKey) === key
      )
    }

    let result

    try {
      result = await testMcpServer(serverName, profile ?? undefined)
    } catch (err) {
      result = { ok: false, error: err instanceof Error ? err.message : String(err), tools: [] }
    }

    if (!isCurrent()) {
      return
    }

    const at = Date.now()
    probeCache.set(key, { at, result })
    putProbe(serverName, key, result, at)
  }

  const probes = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(mountedProbes)
          .filter(([name, mounted]) => servers[name] && mounted.key === probeKey(name, servers[name], scopeProfileKey))
          .map(([name, mounted]) => [name, mounted.value])
      ),
    [mountedProbes, servers, scopeProfileKey]
  )

  useEffect(() => {
    // Retire removed/changed rows and pending requests invalidated by effect
    // cleanup; a late request must not regain ownership when its row is reused.
    const retained = Object.fromEntries(
      Object.entries(probesRef.current).filter(
        ([name, mounted]) =>
          servers[name] &&
          serverEnabled(servers[name]) &&
          (mounted.value !== 'probing' || mounted.epoch === profileEpoch.current) &&
          mounted.key === probeKey(name, servers[name], scopeProfileKey)
      )
    )

    if (Object.keys(retained).length !== Object.keys(probesRef.current).length) {
      replaceProbes(retained)
    }

    for (const [serverName, server] of Object.entries(servers)) {
      if (!serverEnabled(server)) {
        continue
      }

      const key = probeKey(serverName, server, scopeProfileKey)
      const mounted = probesRef.current[serverName]

      if (mounted?.key === key && (mounted.value === 'probing' || Date.now() - mounted.at < PROBE_TTL_MS)) {
        continue
      }

      const cached = probeCache.get(key)

      if (cached && Date.now() - cached.at < PROBE_TTL_MS) {
        putProbe(serverName, key, cached.result, cached.at)
      } else {
        void runProbe(serverName)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers, scopeProfileKey])

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
    replaceProbes(
      Object.fromEntries(
        Object.entries(probesRef.current).filter(
          ([name]) => name in entries && serverFingerprint(entries[name]) === serverFingerprint(prevServers[name] ?? {})
        )
      )
    )
  }

  const resetForProfileSwitch = () => {
    replaceProbes({})
    setToolCalls30d(null)
  }

  return {
    costFor,
    probes,
    resetForProfileSwitch,
    retainProbes,
    runProbe,
    setProbe,
    toolCounts,
    usageByServer
  }
}
