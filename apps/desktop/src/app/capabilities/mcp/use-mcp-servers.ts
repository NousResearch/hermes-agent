import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  getMcpCatalog,
  type HermesGateway,
  type McpCatalogEntry,
  type McpCatalogResponse,
  type McpTestResult,
  type ProfileScope,
  profileScopeKey,
  saveMcpServers
} from '@/hermes'
import { useI18n } from '@/i18n'
import { completeMcpDesktopOAuth } from '@/lib/mcp-dashboard-oauth'
import { probeCache, probeKey } from '@/lib/mcp-probe-cache'
import { getServers, type McpServerEntry, type McpServers } from '@/lib/mcp-servers'
import { setDisabledTools, toggleToolInServer } from '@/lib/mcp-tool-filter'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $activeSessionId } from '@/store/session'
import type { HermesConfigRecord } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../../hooks/use-config-record'
import { useOnProfileSwitch } from '../../hooks/use-on-profile-switch'
import { useProfileSwitchLatch } from '../../hooks/use-profile-switch-latch'
import { seedOptions } from '../connectors/data/persist'

import { parseServersDoc, withEnabled } from './mcp-doc'
import { MCP_CATALOG_KEY, type ServerStatus, statusOf } from './mcp-status'
import { type McpDraft, useMcpDraft } from './use-mcp-draft'
import { type McpProbes, useMcpProbes } from './use-mcp-probes'

type PublishedProbes = Pick<McpProbes, 'costFor' | 'probes' | 'runProbe' | 'toolCounts' | 'usageByServer'>

export interface McpServersController extends McpDraft, PublishedProbes {
  addServerEntry: (name: string, entry: McpServerEntry) => Promise<boolean>
  authing: null | string
  authenticate: (name: string) => Promise<void>
  availableCatalog: McpCatalogEntry[]
  catalog: McpCatalogEntry[]
  catalogLoading: boolean
  config: HermesConfigRecord | null
  configError: unknown
  configFailed: boolean
  configLoading: boolean
  descriptionFor: (name: string, entry: McpServerEntry) => null | string
  names: string[]
  onCatalogInstalled: () => Promise<void>
  profilePending: boolean
  refetchConfig: () => void
  removeServer: (name: string) => Promise<boolean>
  saveDoc: () => Promise<void>
  saving: boolean
  servers: McpServers
  setServerEnabled: (name: string, enabled: boolean) => Promise<void>
  setServerTools: (name: string, disabled: string[], discovered: string[]) => Promise<boolean>
  statuses: Record<string, ServerStatus>
  toggleTool: (name: string, toolName: string) => Promise<void>
}

export interface UseMcpServersOptions {
  gateway: HermesGateway | null
  profile?: ProfileScope
}

export function useMcpServers({ gateway, profile }: UseMcpServersOptions): McpServersController {
  const { t } = useI18n()
  const m = t.settings.mcp
  const activeSessionId = useStore($activeSessionId)

  const appProfile = useStore($activeGatewayProfile)
  const scopeProfileKey = profile != null ? profileScopeKey(profile) : normalizeProfileKey(appProfile)

  const {
    data: config,
    isLoading: configLoading,
    isError: configFailed,
    error: configError,
    refetch: refetchConfigQuery,
    dataUpdatedAt: configUpdatedAt,
    errorUpdatedAt: configErroredAt
  } = useHermesConfigRecord(profile)

  const setConfig = hermesConfigCacheWriter(profile)

  const { arm: armProfileLatch, pending: profilePending } = useProfileSwitchLatch({
    dataUpdatedAt: configUpdatedAt,
    errorUpdatedAt: configErroredAt
  })

  const [saving, setSaving] = useState(false)

  const [authing, setAuthing] = useState<null | string>(null)

  const servers = useMemo(() => getServers(config ?? null), [config])
  const names = useMemo(() => Object.keys(servers), [servers])

  const draft = useMcpDraft({ config, names, profilePending, servers, writable: !profilePending })

  const catalogQuery = useQuery({
    ...seedOptions<McpCatalogResponse>(scopeProfileKey, 'bundled'),
    queryKey: [...MCP_CATALOG_KEY, scopeProfileKey],
    queryFn: () => getMcpCatalog(profile ?? undefined),
    staleTime: 5 * 60_000
  })

  const catalog = useMemo(() => catalogQuery.data?.entries ?? [], [catalogQuery.data])

  const availableCatalog = useMemo(
    () => catalog.filter((entry: McpCatalogEntry) => !entry.installed && !(entry.name in servers)),
    [catalog, servers]
  )

  const descriptionFor = (serverName: string, server: McpServerEntry): null | string => {
    const lower = serverName.toLowerCase()

    const match = catalog.find(
      entry =>
        entry.name.toLowerCase() === lower ||
        (entry.url && entry.url === server.url) ||
        (entry.command && entry.command === server.command)
    )

    return match?.description ?? null
  }

  const profileEpoch = useRef(0)

  useEffect(
    () => () => {
      profileEpoch.current += 1
    },
    [scopeProfileKey]
  )

  const fleet = useMcpProbes({ appProfile, profile, profileEpoch, scopeProfileKey, servers })

  useOnProfileSwitch(() => {
    profileEpoch.current += 1
    fleet.resetForProfileSwitch()
    setAuthing(null)
    draft.reset()
    armProfileLatch()
  })

  const silentReload = async () => {
    if (!gateway) {
      return
    }

    try {
      await gateway.request('reload.mcp', { confirm: true, session_id: activeSessionId ?? undefined })
    } catch (err) {
      notifyError(err, m.reloadFailed)
    }
  }

  const authenticate = async (serverName: string) => {
    const epoch = profileEpoch.current
    setAuthing(serverName)
    fleet.setProbe(serverName, 'probing')

    try {
      const flow = await completeMcpDesktopOAuth({
        serverName,
        profile,
        cancelled: () => profileEpoch.current !== epoch
      })

      const result: McpTestResult = { ok: true, tools: flow.tools ?? [] }

      if (profileEpoch.current !== epoch) {
        return
      }

      fleet.setProbe(serverName, result)
      const probedConfig = result.ok ? { ...servers[serverName], auth: 'oauth' } : servers[serverName]
      probeCache.set(probeKey(serverName, probedConfig, scopeProfileKey), { at: Date.now(), result })

      if (result.ok) {
        const nextServers = { ...servers, [serverName]: { ...servers[serverName], auth: 'oauth' } }
        setConfig(current => (current ? { ...current, mcp_servers: nextServers } : current))

        if (draft.dirty) {
          draft.patchDraft(doc =>
            doc[serverName] ? { ...doc, [serverName]: { ...doc[serverName], auth: 'oauth' } } : doc
          )
        } else {
          draft.resetDraft(nextServers)
        }

        notify({
          kind: 'success',
          title: m.authenticatedTitle,
          message: m.authenticatedMessage(serverName, result.tools.length)
        })
        void silentReload()
      } else if (result.error) {
        notifyError(new Error(result.error), serverName)
      }
    } catch (err) {
      if (profileEpoch.current !== epoch) {
        return
      }

      fleet.setProbe(serverName, { ok: false, error: err instanceof Error ? err.message : String(err), tools: [] })
      notifyError(err, serverName)
    } finally {
      if (profileEpoch.current === epoch) {
        setAuthing(null)
      }
    }
  }

  const statuses = useMemo(() => {
    const table: Record<string, ServerStatus> = {}

    for (const [serverName, server] of Object.entries(servers)) {
      table[serverName] = statusOf(server, fleet.probes[serverName])
    }

    return table
  }, [fleet.probes, servers])

  const persist = async (nextServers: McpServers): Promise<boolean> => {
    const epoch = profileEpoch.current
    await saveMcpServers(nextServers, profile ?? undefined)

    if (profileEpoch.current !== epoch) {
      return false
    }

    setConfig(current => ({ ...current, mcp_servers: nextServers }))
    void silentReload()

    return true
  }

  const mirror = (nextServers: McpServers, patch: (doc: McpServers) => McpServers) => {
    if (draft.dirty) {
      draft.patchDraft(patch)
    } else {
      draft.resetDraft(nextServers)
    }
  }

  const onCatalogInstalled = async () => {
    void catalogQuery.refetch()
    const { data } = await refetchConfigQuery()
    const nextServers = getServers(data ?? null)

    mirror(nextServers, doc => ({ ...nextServers, ...doc }))

    void silentReload()
  }

  const writeEntry = async (
    serverName: string,
    transform: (entry: McpServerEntry) => McpServerEntry
  ): Promise<boolean> => {
    const base = servers[serverName]

    if (!base || profilePending) {
      return false
    }

    const next = transform(base)

    try {
      if (!(await persist({ ...servers, [serverName]: next }))) {
        return false
      }

      mirror({ ...servers, [serverName]: next }, doc =>
        doc[serverName] ? { ...doc, [serverName]: transform(doc[serverName]) } : doc
      )

      return true
    } catch (err) {
      notifyError(err, m.saveFailed)

      return false
    }
  }

  const addServerEntry = async (serverName: string, entry: McpServerEntry): Promise<boolean> => {
    if (profilePending || serverName in servers) {
      return false
    }

    const next = { ...servers, [serverName]: entry }

    setSaving(true)

    try {
      if (!(await persist(next))) {
        return false
      }

      mirror(next, doc => ({ ...doc, [serverName]: entry }))
      void fleet.runProbe(serverName)

      return true
    } catch (err) {
      notifyError(err, m.saveFailed)

      return false
    } finally {
      setSaving(false)
    }
  }

  const setServerEnabled = async (serverName: string, enabled: boolean) => {
    if ((await writeEntry(serverName, entry => withEnabled(entry, enabled))) && enabled) {
      void fleet.runProbe(serverName)
    }
  }

  const toggleTool = async (serverName: string, toolName: string) => {
    await writeEntry(serverName, entry => toggleToolInServer(entry, toolName))
  }

  const setServerTools = (serverName: string, disabled: string[], discovered: string[]): Promise<boolean> =>
    writeEntry(serverName, entry => setDisabledTools(entry, disabled, discovered))

  const removeServer = async (serverName: string): Promise<boolean> => {
    if (profilePending) {
      return false
    }

    setSaving(true)

    try {
      const next = { ...servers }
      delete next[serverName]

      if (!(await persist(next))) {
        return false
      }

      mirror(next, doc => {
        const patched = { ...doc }
        delete patched[serverName]

        return patched
      })

      draft.setCursor(0)
      void catalogQuery.refetch()

      return true
    } catch (err) {
      notifyError(err, m.removeFailed)

      return false
    } finally {
      setSaving(false)
    }
  }

  const saveDoc = async () => {
    if (profilePending) {
      return
    }

    let entries: McpServers

    try {
      entries = parseServersDoc(draft.draft)
    } catch (err) {
      notifyError(err, m.invalidJson)

      return
    }

    setSaving(true)

    const prevServers = servers

    try {
      if (!(await persist(entries))) {
        return
      }

      draft.resetDraft(entries)
      fleet.retainProbes(entries, prevServers)
      notify({ kind: 'success', title: m.savedTitle, message: m.savedMessage('mcp.json') })
    } catch (err) {
      notifyError(err, m.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  return {
    ...draft,
    addServerEntry,
    authenticate,
    authing,
    availableCatalog,
    catalog,
    catalogLoading: catalogQuery.isLoading,
    config: config ?? null,
    configError,
    configFailed,
    configLoading,
    costFor: fleet.costFor,
    descriptionFor,
    names,
    onCatalogInstalled,
    probes: fleet.probes,
    profilePending,
    refetchConfig: () => void refetchConfigQuery(),
    removeServer,
    runProbe: fleet.runProbe,
    saveDoc,
    saving,
    servers,
    setServerEnabled,
    setServerTools,
    statuses,
    toggleTool,
    toolCounts: fleet.toolCounts,
    usageByServer: fleet.usageByServer
  }
}
