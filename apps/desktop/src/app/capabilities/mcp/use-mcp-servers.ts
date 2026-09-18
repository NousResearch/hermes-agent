// The servers-on-this-Mac engine, headless.
//
// Everything `McpTab` used to hold in its own body: the config record, the probe
// fleet, the 30-day usage overlay, the catalog, and every write (enable,
// per-tool gating, remove, save, OAuth). The document half lives in
// `use-mcp-draft.ts`; this hook composes it and re-exports it, so one call gets
// the whole surface.
//
// `McpTab` renders it; the Connectors page renders the same controller for its
// "On this Mac" group, its `Add your own` dialog and the `Advanced` section of a
// local server. One engine, two surfaces, so a server can never be on in one and
// off in the other.
//
// The hook owns no route and no layout. Deep links belong to whichever page is
// mounted, so the caller drives `focusServer` itself.

import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  getMcpCatalog,
  type HermesGateway,
  type McpCatalogEntry,
  type McpTestResult,
  type ProfileScope,
  profileScopeKey,
  saveMcpServers,
  testMcpServer
} from '@/hermes'
import { useI18n } from '@/i18n'
import { completeMcpDesktopOAuth } from '@/lib/mcp-dashboard-oauth'
import { PROBE_TTL_MS, probeCache, probeKey, serverFingerprint } from '@/lib/mcp-probe-cache'
import { getServers, type McpServers } from '@/lib/mcp-servers'
import { setDisabledTools, toggleToolInServer } from '@/lib/mcp-tool-filter'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $activeSessionId } from '@/store/session'
import type { HermesConfigRecord } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../../hooks/use-config-record'
import { useOnProfileSwitch } from '../../hooks/use-on-profile-switch'
import { useProfileSwitchLatch } from '../../hooks/use-profile-switch-latch'

import { parseServersDoc, serverEnabled, withEnabled } from './mcp-doc'
import {
  enabledToolCount,
  loadMcpUsage,
  MCP_CATALOG_KEY,
  type Probe,
  serverCost,
  type ServerCost,
  type ServerStatus,
  statusOf
} from './mcp-status'
import { type McpDraft, useMcpDraft } from './use-mcp-draft'

export interface McpServersController extends McpDraft {
  /** The server whose OAuth flow is blocking the browser, if any. */
  authing: null | string
  authenticate: (name: string) => Promise<void>
  /** Catalog entries that are not installed on this profile yet. */
  availableCatalog: McpCatalogEntry[]
  catalog: McpCatalogEntry[]
  catalogLoading: boolean
  config: HermesConfigRecord | null
  configError: unknown
  configFailed: boolean
  configLoading: boolean
  costFor: (name: string, entry: Record<string, unknown>) => ServerCost
  descriptionFor: (name: string, entry: Record<string, unknown>) => null | string
  /** Config/document order, not alphabetical — the list mirrors mcp.json. */
  names: string[]
  onCatalogInstalled: () => Promise<void>
  probes: Record<string, Probe>
  /** True from a profile switch until the config query resettles: writes are
   *  refused until then, because `servers` still holds the other profile's map. */
  profilePending: boolean
  refetchConfig: () => void
  /** Drop one server from mcp.json. `false` means the write was refused or
   *  failed, so a confirm dialog can raise it inline instead of closing over a
   *  server that is still configured. */
  removeServer: (name: string) => Promise<boolean>
  runProbe: (name: string) => Promise<void>
  saveDoc: () => Promise<void>
  saving: boolean
  servers: McpServers
  setServerEnabled: (name: string, enabled: boolean) => Promise<void>
  /** Write one server's whole off-list at once. `false` means the write was
   *  refused or failed, which is what the dialog's Save reports. */
  setServerTools: (name: string, disabled: string[], discovered: string[]) => Promise<boolean>
  statusFor: (name: string) => ServerStatus
  toggleTool: (name: string, toolName: string) => Promise<void>
  /** 30-day call counts per registry tool name; null when analytics are off. */
  toolCalls30d: null | Record<string, number>
  /** Registered/discovered tool counts per server. Absent until probed. */
  toolCounts: Record<string, { on: number; total: number }>
  /** 30-day call counts per SERVER, for the page's "unused" pill. */
  usageByServer: Record<string, number>
}

export interface UseMcpServersOptions {
  /** Only for the live `reload.mcp` RPC. Null withholds it (cross-backend scope). */
  gateway: HermesGateway | null
  profile?: ProfileScope
}

export function useMcpServers({ gateway, profile }: UseMcpServersOptions): McpServersController {
  const { t } = useI18n()
  const m = t.settings.mcp
  const activeSessionId = useStore($activeSessionId)

  // The profile this controller configures: the Capabilities scope selector's
  // choice (`profile`) when set, otherwise the app-wide active profile. Every
  // fetch/save below is scoped to it, and it keys the config/catalog/probe
  // caches so switching the selector refetches and never shows another
  // profile's servers (AGENTS.md scope-in-key).
  const appProfile = useStore($activeGatewayProfile)
  const scopeProfileKey = profile != null ? profileScopeKey(profile) : normalizeProfileKey(appProfile)

  // Shared config cache (see use-config-record): revisiting the page paints the
  // cached record instantly; mutations write through `setConfig` and stay
  // visible to the other settings surfaces.
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

  // True from a profile switch until the config query resettles for the new
  // profile. Until then `config` (and thus `servers`) still holds profile A's
  // data, so any persist would write A's server list into B — block mutations.
  const { arm: armProfileLatch, pending: profilePending } = useProfileSwitchLatch({
    dataUpdatedAt: configUpdatedAt,
    errorUpdatedAt: configErroredAt
  })

  const [saving, setSaving] = useState(false)
  const [probes, setProbes] = useState<Record<string, Probe>>({})
  const probesRef = useRef(probes)
  probesRef.current = probes

  // 30-day per-tool call counts (registry names). null = analytics unavailable
  // or not loaded yet — the cost overlay then omits usage entirely.
  const [toolCalls30d, setToolCalls30d] = useState<null | Record<string, number>>(null)

  // Blocks the browser until an OAuth flow lands a token; also reset on profile
  // switch, so declared up here alongside the other per-profile view state.
  const [authing, setAuthing] = useState<null | string>(null)

  const servers = useMemo(() => getServers(config ?? null), [config])
  const names = useMemo(() => Object.keys(servers), [servers])

  const draft = useMcpDraft({ config, names, profilePending, servers, writable: !profilePending })

  // Key by the SCOPED profile — installed/enabled badges are per-profile, so
  // sharing one cache across profiles would flash the previous profile's state
  // on switch.
  const catalogQuery = useQuery({
    queryKey: [...MCP_CATALOG_KEY, scopeProfileKey],
    queryFn: () => getMcpCatalog(profile ?? undefined),
    staleTime: 5 * 60_000
  })

  const catalog = useMemo(() => catalogQuery.data?.entries ?? [], [catalogQuery.data])

  // The catalog SECTION of the unified list only offers entries that aren't
  // already configured — installed servers appear once, in the fleet list
  // above, with live status. Match by catalog `installed` flag or a config
  // entry under the same name (covers a just-saved doc the catalog refetch
  // hasn't caught up with yet).
  const availableCatalog = useMemo(
    () => catalog.filter((entry: McpCatalogEntry) => !entry.installed && !(entry.name in servers)),
    [catalog, servers]
  )

  const descriptionFor = (serverName: string, server: Record<string, unknown>): null | string => {
    const lower = serverName.toLowerCase()

    const match = catalog.find(
      entry =>
        entry.name.toLowerCase() === lower ||
        (entry.url && entry.url === server.url) ||
        (entry.command && entry.command === server.command)
    )

    return match?.description ?? null
  }

  // Bumped on every profile switch. Async probe/auth completions capture the
  // epoch at call time and bail if it changed, so a slow profile-A request can't
  // write its result into profile B's state after the user switched.
  const profileEpoch = useRef(0)

  // Scoped tabs remount when their owner changes; stop the old native OAuth
  // waiter even when no app-wide profile-switch event is emitted.
  useEffect(
    () => () => {
      profileEpoch.current += 1
    },
    [scopeProfileKey]
  )

  // A profile switch invalidates the config query (see store/profile.ts), which
  // refetches the new backend's mcp.json. Reset ALL per-profile view state — the
  // draft (incl. a dirty one, so profile A's edits can't be saved into B), its
  // seed latch, probes, and cursor — so everything reseeds for the new profile.
  // The probe cache is already profile-keyed, so this just forces a re-probe.
  useOnProfileSwitch(() => {
    profileEpoch.current += 1
    setProbes({})
    setToolCalls30d(null)
    setAuthing(null)
    draft.reset()
    // Mark stale until the config query replaces profile A's data — guards
    // sidebar mutations from persisting A's server list into B mid-refetch.
    // The latch releases on a fresh success OR a fresh failure, so a failed
    // refetch surfaces the retry UI instead of leaving mutations no-op forever.
    armProfileLatch()
  })

  const runProbe = async (serverName: string) => {
    const epoch = profileEpoch.current
    const key = probeKey(serverName, servers[serverName], scopeProfileKey)
    setProbes(current => ({ ...current, [serverName]: 'probing' }))

    try {
      const result = await testMcpServer(serverName, profile ?? undefined)

      // Drop the result if the profile changed mid-probe — it belongs to A.
      if (profileEpoch.current !== epoch) {
        return
      }

      probeCache.set(key, { at: Date.now(), result })
      setProbes(current => ({ ...current, [serverName]: result }))
    } catch (err) {
      if (profileEpoch.current !== epoch) {
        return
      }

      const result = { ok: false, error: err instanceof Error ? err.message : String(err), tools: [] }
      probeCache.set(key, { at: Date.now(), result })
      setProbes(current => ({ ...current, [serverName]: result }))
    }
  }

  // Config writes reach live sessions immediately — no manual "Reload MCP".
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

  // First-class OAuth: opens the system browser, blocks until the flow lands a
  // token (verified on disk — a friendly tools/list is not proof), then the
  // auth result doubles as the probe (it carries the tool list).
  const authenticate = async (serverName: string) => {
    const epoch = profileEpoch.current
    setAuthing(serverName)
    setProbes(current => ({ ...current, [serverName]: 'probing' }))

    try {
      const flow = await completeMcpDesktopOAuth({
        serverName,
        profile,
        cancelled: () => profileEpoch.current !== epoch
      })

      const result: McpTestResult = { ok: true, tools: flow.tools ?? [] }

      // Bail if the user switched profiles mid-flow — this result is profile A's.
      if (profileEpoch.current !== epoch) {
        return
      }

      setProbes(current => ({ ...current, [serverName]: result }))
      // Cache under the POST-auth fingerprint (auth: oauth) on success — that's
      // the config the mount effect will read back, so it hits this entry.
      const probedConfig = result.ok ? { ...servers[serverName], auth: 'oauth' } : servers[serverName]
      probeCache.set(probeKey(serverName, probedConfig, scopeProfileKey), { at: Date.now(), result })

      if (result.ok) {
        // The endpoint persisted `auth: oauth` — mirror it locally.
        const nextServers = { ...servers, [serverName]: { ...servers[serverName], auth: 'oauth' } }
        setConfig(current => (current ? { ...current, mcp_servers: nextServers } : current))

        // Mirror `auth: oauth` into the editor too. If we only reset a clean
        // draft, a dirty draft keeps the pre-auth text and the next Save would
        // drop the freshly-persisted auth field — so patch the dirty draft in
        // place instead of clobbering the user's other edits.
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

      setProbes(current => ({
        ...current,
        [serverName]: { ok: false, error: err instanceof Error ? err.message : String(err), tools: [] }
      }))
      notifyError(err, serverName)
    } finally {
      if (profileEpoch.current === epoch) {
        setAuthing(null)
      }
    }
  }

  // It should just know: probe enabled servers as config arrives — but through
  // the cache, so revisiting the page doesn't respawn/reconnect the fleet.
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
    // Re-run only when the server set changes; runProbe is recreated every
    // render and adding it would re-probe the fleet on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers])

  // Cosmetic 30-day usage counts for the cost overlay — cached module-wide per
  // scope profile, epoch-guarded like the probes so a slow profile-A fetch
  // can't paint into profile B.
  useEffect(() => {
    const epoch = profileEpoch.current

    void loadMcpUsage(scopeProfileKey, profile ?? appProfile ?? null).then(value => {
      if (profileEpoch.current === epoch) {
        setToolCalls30d(value)
      }
    })
  }, [scopeProfileKey, profile, appProfile])

  const costFor = (serverName: string, server: Record<string, unknown>): ServerCost =>
    serverCost(server, probes[serverName], serverName, toolCalls30d)

  const statusFor = (serverName: string): ServerStatus => statusOf(servers[serverName] ?? {}, probes[serverName])

  // The registered/discovered pair per server, which is what a card's "31 tools,
  // 29 on" line says. A server with no successful probe has no entry at all —
  // an absent count is honest, a zero would read as "this server has no tools".
  const toolCounts = useMemo(() => {
    const counts: Record<string, { on: number; total: number }> = {}

    for (const [serverName, server] of Object.entries(servers)) {
      const probe = probes[serverName]
      const on = enabledToolCount(server, probe)

      if (on !== null && probe && probe !== 'probing' && probe.ok) {
        counts[serverName] = { on, total: probe.tools.length }
      }
    }

    return counts
  }, [probes, servers])

  // The same 30-day analytics the cost overlay reads, summed per server rather
  // than per tool, because a card says "unused", not "which tool went unused".
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

  // Whole-map replace (NOT saveHermesConfig, which deep-merges and so can never
  // delete a server, drop `enabled: false`, or remove a nested field). Only
  // after the replace lands do we write the cache through + reload live sessions.
  // Returns false when the profile switched mid-save: the write hit profile A's
  // backend (correct), but the client-side cache/editor now belong to B, so the
  // caller must skip its post-await writes.
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

  /** A write landed on the backend: mirror it into the document, either by
   *  redrawing it or — when the person is mid-edit — by patching that one
   *  entry and leaving every other word of theirs alone. */
  const mirror = (nextServers: McpServers, patch: (doc: McpServers) => McpServers) => {
    if (draft.dirty) {
      draft.patchDraft(patch)
    } else {
      draft.resetDraft(nextServers)
    }
  }

  // A catalog install wrote a new server into config.yaml on the backend —
  // refresh the catalog (installed state) and the config, then RECONCILE THE
  // EDITOR DRAFT with the fresh servers. Without this a dirty draft (or even a
  // clean one the seed never refreshes) would omit the new server, and the next
  // whole-map Save would silently drop it.
  const onCatalogInstalled = async () => {
    void catalogQuery.refetch()
    const { data } = await refetchConfigQuery()
    const nextServers = getServers(data ?? null)

    // Keep the user's in-progress edits (doc wins), add any server the install
    // introduced that the draft doesn't have yet.
    mirror(nextServers, doc => ({ ...nextServers, ...doc }))

    void silentReload()
  }

  const setServerEnabled = async (serverName: string, enabled: boolean) => {
    if (profilePending) {
      return
    }

    const next = withEnabled(servers[serverName], enabled)

    try {
      if (!(await persist({ ...servers, [serverName]: next }))) {
        return
      }

      mirror({ ...servers, [serverName]: next }, doc =>
        doc[serverName] ? { ...doc, [serverName]: withEnabled(doc[serverName], enabled) } : doc
      )

      if (enabled) {
        void runProbe(serverName)
      }
    } catch (err) {
      notifyError(err, m.saveFailed)
    }
  }

  // Per-tool gating writes the server's `tools.include`/`tools.exclude` and
  // persists like any other config change (immediate reload of live sessions).
  // The probe still lists every discovered tool; the filter decides which ones
  // the agent actually registers.
  const toggleTool = async (serverName: string, toolName: string) => {
    const base = servers[serverName]

    if (!base || profilePending) {
      return
    }

    const next = toggleToolInServer(base, toolName)

    try {
      if (!(await persist({ ...servers, [serverName]: next }))) {
        return
      }

      mirror({ ...servers, [serverName]: next }, doc =>
        doc[serverName] ? { ...doc, [serverName]: toggleToolInServer(doc[serverName], toolName) } : doc
      )
    } catch (err) {
      notifyError(err, m.saveFailed)
    }
  }

  // The Connectors dialog edits every switch of one server and then saves, so it
  // writes the whole off-list in ONE config write rather than one per switch.
  const setServerTools = async (serverName: string, disabled: string[], discovered: string[]): Promise<boolean> => {
    const base = servers[serverName]

    if (!base || profilePending) {
      return false
    }

    const next = setDisabledTools(base, disabled, discovered)

    try {
      if (!(await persist({ ...servers, [serverName]: next }))) {
        return false
      }

      mirror({ ...servers, [serverName]: next }, doc =>
        doc[serverName] ? { ...doc, [serverName]: setDisabledTools(doc[serverName], disabled, discovered) } : doc
      )

      return true
    } catch (err) {
      notifyError(err, m.saveFailed)

      return false
    }
  }

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
      // Keep only probes for servers that survived AND kept the same config;
      // removed OR edited entries drop their probe so the mount effect re-probes
      // the new shape (the cache also misses on the changed fingerprint).
      setProbes(current =>
        Object.fromEntries(
          Object.entries(current).filter(
            ([name]) =>
              name in entries && serverFingerprint(entries[name]) === serverFingerprint(prevServers[name] ?? {})
          )
        )
      )
      notify({ kind: 'success', title: m.savedTitle, message: m.savedMessage('mcp.json') })
    } catch (err) {
      notifyError(err, m.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  return {
    ...draft,
    authenticate,
    authing,
    availableCatalog,
    catalog,
    catalogLoading: catalogQuery.isLoading,
    config: config ?? null,
    configError,
    configFailed,
    configLoading,
    costFor,
    descriptionFor,
    names,
    onCatalogInstalled,
    probes,
    profilePending,
    refetchConfig: () => void refetchConfigQuery(),
    removeServer,
    runProbe,
    saveDoc,
    saving,
    servers,
    setServerEnabled,
    setServerTools,
    statusFor,
    toggleTool,
    toolCalls30d,
    toolCounts,
    usageByServer
  }
}
