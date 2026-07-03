import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import {
  getHermesConfigRecord,
  getMcpCatalog,
  type HermesGateway,
  installMcpCatalogEntry,
  type McpCatalogEntry,
  saveHermesConfig,
  setMcpServerEnabled,
  testMcpServer
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Wrench } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $activeSessionId } from '@/store/session'
import type { HermesConfigRecord, McpServerTestResponse } from '@/types/hermes'

import { EmptyState, LoadingState, Pill, SettingsContent } from './primitives'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

interface McpSettingsProps {
  gateway?: HermesGateway | null
  onConfigSaved?: () => void
}

type McpServers = Record<string, Record<string, unknown>>
type McpView = 'catalog' | 'servers'

const EMPTY_SERVER = {
  command: '',
  args: [],
  env: {}
}

function getServers(config: HermesConfigRecord | null): McpServers {
  const raw = config?.mcp_servers

  return raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as McpServers) : {}
}

const transportLabel = (server: Record<string, unknown>) =>
  typeof server.transport === 'string'
    ? server.transport
    : typeof server.url === 'string'
      ? 'http'
      : typeof server.command === 'string'
        ? 'stdio'
        : 'custom'

export function McpSettings({ gateway, onConfigSaved }: McpSettingsProps) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const activeSessionId = useStore($activeSessionId)
  const [view, setView] = useState<McpView>('servers')
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [body, setBody] = useState('')
  const [saving, setSaving] = useState(false)
  const [reloading, setReloading] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<McpServerTestResponse | null>(null)
  const [togglingEnabled, setTogglingEnabled] = useState(false)

  useEffect(() => {
    let cancelled = false

    getHermesConfigRecord()
      .then(next => {
        if (cancelled) {
          return
        }

        setConfig(next)
        const first = Object.keys(getServers(next)).sort()[0] ?? null
        setSelected(first)
      })
      .catch(err => notifyError(err, m.failedLoad))

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once on mount; copy is stable
  }, [])

  const servers = useMemo(() => getServers(config), [config])
  const names = useMemo(() => Object.keys(servers).sort(), [servers])

  useDeepLinkHighlight({
    block: 'nearest',
    elementId: serverName => `mcp-server-${serverName}`,
    onResolve: setSelected,
    param: 'server',
    ready: serverName => Boolean(config) && serverName in servers
  })

  useEffect(() => {
    const server = selected ? servers[selected] : null

    setName(selected ?? '')
    setBody(JSON.stringify(server ?? EMPTY_SERVER, null, 2))
    setTestResult(null)
  }, [selected, servers])

  const refreshConfig = useCallback(async () => {
    try {
      const next = await getHermesConfigRecord()
      setConfig(next)
    } catch (err) {
      notifyError(err, m.failedLoad)
    }
  }, [m.failedLoad])

  if (!config) {
    return <LoadingState label={m.loading} />
  }

  const saveServer = async () => {
    const nextName = name.trim()

    if (!nextName) {
      notify({ kind: 'error', title: m.nameRequiredTitle, message: m.nameRequiredMessage })

      return
    }

    let parsed: Record<string, unknown>

    try {
      const raw = JSON.parse(body)

      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error(m.objectRequired)
      }

      parsed = raw as Record<string, unknown>
    } catch (err) {
      notifyError(err, m.invalidJson)

      return
    }

    setSaving(true)

    try {
      const nextServers = { ...servers }

      if (selected && selected !== nextName) {
        delete nextServers[selected]
      }

      nextServers[nextName] = parsed

      const nextConfig = { ...config, mcp_servers: nextServers }
      await saveHermesConfig(nextConfig)
      setConfig(nextConfig)
      setSelected(nextName)
      onConfigSaved?.()
      notify({ kind: 'success', title: m.savedTitle, message: m.savedMessage(nextName) })
    } catch (err) {
      notifyError(err, m.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  const removeServer = async (serverName: string) => {
    setSaving(true)

    try {
      const nextServers = { ...servers }
      delete nextServers[serverName]

      const nextConfig = { ...config, mcp_servers: nextServers }
      await saveHermesConfig(nextConfig)
      setConfig(nextConfig)
      setSelected(Object.keys(nextServers).sort()[0] ?? null)
      onConfigSaved?.()
    } catch (err) {
      notifyError(err, m.removeFailed)
    } finally {
      setSaving(false)
    }
  }

  const reloadMcp = async () => {
    if (!gateway) {
      notify({ kind: 'warning', title: m.gatewayUnavailableTitle, message: m.gatewayUnavailableMessage })

      return
    }

    setReloading(true)

    try {
      await gateway.request('reload.mcp', {
        confirm: true,
        session_id: activeSessionId ?? undefined
      })
      notify({ kind: 'success', title: m.reloadedTitle, message: m.reloadedMessage })
    } catch (err) {
      notifyError(err, m.reloadFailed)
    } finally {
      setReloading(false)
    }
  }

  const runTest = async (serverName: string) => {
    setTesting(true)
    setTestResult(null)

    try {
      const result = await testMcpServer(serverName)
      setTestResult(result)
    } catch (err) {
      setTestResult({ ok: false, error: err instanceof Error ? err.message : String(err), tools: [] })
    } finally {
      setTesting(false)
    }
  }

  const toggleEnabled = async (serverName: string, enabled: boolean) => {
    setTogglingEnabled(true)

    try {
      await setMcpServerEnabled(serverName, enabled)
      // Mirror the change locally so the editor and list stay in sync.
      const nextServers = { ...servers, [serverName]: { ...servers[serverName], enabled } }
      setConfig({ ...config, mcp_servers: nextServers })
      notify({
        kind: 'success',
        title: enabled ? m.serverEnabled(serverName) : m.serverDisabled(serverName),
        message: ''
      })
    } catch (err) {
      notifyError(err, m.toggleFailed(serverName))
    } finally {
      setTogglingEnabled(false)
    }
  }

  const selectedEnabled = selected ? servers[selected]?.enabled !== false : true

  return (
    <SettingsContent>
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <TabButton active={view === 'servers'} label={m.tabServers} onClick={() => setView('servers')} />
          <TabButton active={view === 'catalog'} label={m.tabCatalog} onClick={() => setView('catalog')} />
        </div>
        {view === 'servers' && (
          <div className="flex items-center gap-4">
            <Button onClick={() => setSelected(null)} size="xs" variant="text">
              {m.newServer}
            </Button>
            <Button disabled={reloading} onClick={() => void reloadMcp()} size="xs" variant="text">
              {reloading ? m.reloading : m.reload}
            </Button>
          </div>
        )}
      </div>

      {view === 'catalog' ? (
        <McpCatalogBrowser onInstalled={() => void refreshConfig()} />
      ) : (
        <div className="grid min-h-0 gap-6 lg:grid-cols-[16rem_minmax(0,1fr)]">
          <div className="min-h-64">
            {names.length === 0 ? (
              <EmptyState description={m.emptyDesc} title={m.emptyTitle} />
            ) : (
              <div className="grid gap-0.5">
                {names.map(serverName => {
                  const server = servers[serverName]
                  const active = selected === serverName

                  return (
                    <button
                      className={cn(
                        'scroll-mt-2 rounded-md px-2 py-2 text-left transition-colors hover:bg-(--chrome-action-hover)',
                        active ? 'bg-(--ui-bg-tertiary) text-foreground' : 'text-muted-foreground'
                      )}
                      id={`mcp-server-${serverName}`}
                      key={serverName}
                      onClick={() => setSelected(serverName)}
                      type="button"
                    >
                      <div className="truncate text-sm font-medium">{serverName}</div>
                      <div className="mt-1 flex items-center gap-1.5">
                        <Pill>{transportLabel(server)}</Pill>
                        {(server.enabled === false || server.disabled === true) && <Pill>{m.disabled}</Pill>}
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          <div className="grid content-start gap-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Wrench className="size-4 text-muted-foreground" />
                {selected ? m.editServer : m.newServer}
              </div>
              {selected && (
                <div className="flex items-center gap-2">
                  <Button disabled={testing} onClick={() => void runTest(selected)} size="xs" variant="text">
                    {testing ? m.testing : m.test}
                  </Button>
                  <Switch
                    aria-label={selectedEnabled ? m.disableServer(selected) : m.enableServer(selected)}
                    checked={selectedEnabled}
                    disabled={togglingEnabled}
                    onCheckedChange={checked => void toggleEnabled(selected, checked)}
                  />
                </div>
              )}
            </div>
            {testResult && (
              <div
                className={cn(
                  'rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 text-xs',
                  testResult.ok ? 'text-emerald-400' : 'text-destructive'
                )}
              >
                {testResult.ok ? m.testOk(testResult.tools.length) : `${m.testFailed}: ${testResult.error ?? ''}`}
                {testResult.ok && testResult.tools.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {testResult.tools.map(tool => (
                      <span
                        className="rounded-md bg-(--ui-bg-quinary) px-1.5 py-0.5 font-mono text-[0.65rem] text-(--ui-text-tertiary)"
                        key={tool.name}
                        title={tool.description}
                      >
                        {tool.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            <label className="grid gap-1.5">
              <span className="text-xs text-muted-foreground">{m.name}</span>
              <Input onChange={event => setName(event.currentTarget.value)} placeholder="filesystem" value={name} />
            </label>
            <label className="grid gap-1.5">
              <span className="text-xs text-muted-foreground">{m.serverJson}</span>
              <Textarea
                className="min-h-80 font-mono text-xs"
                onChange={event => setBody(event.currentTarget.value)}
                spellCheck={false}
                value={body}
              />
            </label>
            <div className="flex items-center justify-between">
              {selected ? (
                <Button
                  className="text-destructive hover:text-destructive"
                  disabled={saving}
                  onClick={() => void removeServer(selected)}
                  size="xs"
                  variant="text"
                >
                  {m.remove}
                </Button>
              ) : (
                <span />
              )}
              <Button disabled={saving} onClick={() => void saveServer()} size="sm">
                {saving ? t.common.saving : m.saveServer}
              </Button>
            </div>
          </div>
        </div>
      )}
    </SettingsContent>
  )
}

function TabButton({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      className={cn(
        'cursor-pointer text-sm font-medium transition-colors',
        active ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
      )}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  )
}

/** Nous-approved MCP catalog browser — the desktop counterpart of
 *  `hermes mcp catalog` / `hermes mcp install` and the dashboard MCP page. */
function McpCatalogBrowser({ onInstalled }: { onInstalled: () => void }) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const [entries, setEntries] = useState<McpCatalogEntry[] | null>(null)
  const [installing, setInstalling] = useState<null | string>(null)
  // Per-entry env var drafts for catalog entries that need credentials.
  const [envDrafts, setEnvDrafts] = useState<Record<string, Record<string, string>>>({})
  const [envOpenFor, setEnvOpenFor] = useState<null | string>(null)

  useEffect(() => {
    let cancelled = false

    getMcpCatalog()
      .then(response => {
        if (!cancelled) {
          setEntries(response.entries)
        }
      })
      .catch(err => {
        if (!cancelled) {
          notifyError(err, m.catalogLoadFailed)
          setEntries([])
        }
      })

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once on mount
  }, [])

  const install = async (entry: McpCatalogEntry) => {
    const required = entry.required_env.filter(env => env.required)
    const draft = envDrafts[entry.name] ?? {}

    if (required.some(env => !draft[env.name]?.trim())) {
      if (envOpenFor !== entry.name) {
        setEnvOpenFor(entry.name)

        return
      }

      notify({ kind: 'error', title: m.catalogEnvPrompt(entry.name), message: m.catalogEnvRequired })

      return
    }

    setInstalling(entry.name)

    try {
      await installMcpCatalogEntry(entry.name, draft)
      notify({ kind: 'success', title: m.catalogInstallStarted(entry.name), message: '' })
      setEntries(
        current =>
          current?.map(row => (row.name === entry.name ? { ...row, installed: true, enabled: true } : row)) ?? current
      )
      setEnvOpenFor(null)
      onInstalled()
    } catch (err) {
      notifyError(err, m.catalogInstallFailed(entry.name))
    } finally {
      setInstalling(null)
    }
  }

  if (entries === null) {
    return <LoadingState label={m.catalogLoading} />
  }

  if (entries.length === 0) {
    return <EmptyState description={m.catalogEmpty} title={m.tabCatalog} />
  }

  return (
    <div>
      {entries.map(entry => {
        const envOpen = envOpenFor === entry.name
        const draft = envDrafts[entry.name] ?? {}

        return (
          <div className="px-0 py-2.5" key={entry.name}>
            <div className="flex items-center justify-between gap-2">
              <div className="flex min-w-0 items-center gap-2">
                <span className="truncate text-sm font-medium">{entry.name}</span>
                <Pill>{entry.transport}</Pill>
                {entry.installed && (
                  <Badge className="bg-emerald-500/15 text-emerald-400">
                    {entry.enabled ? m.catalogEnabled : m.catalogInstalled}
                  </Badge>
                )}
                {entry.needs_install && !entry.installed && <Pill>{m.catalogNeedsInstall}</Pill>}
              </div>
              <Button
                disabled={entry.installed || installing !== null}
                onClick={() => void install(entry)}
                size="xs"
                variant="textStrong"
              >
                {installing === entry.name
                  ? m.catalogInstalling
                  : entry.installed
                    ? m.catalogInstalled
                    : m.catalogInstall}
              </Button>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{entry.description}</p>
            {envOpen && entry.required_env.length > 0 && (
              <div className="mt-2 grid max-w-md gap-2">
                {entry.required_env.map(env => (
                  <label className="grid gap-1" key={env.name}>
                    <span className="text-xs text-muted-foreground">
                      {env.prompt || env.name}
                      {env.required ? ' *' : ''}
                    </span>
                    <Input
                      onChange={event =>
                        setEnvDrafts(prev => ({
                          ...prev,
                          [entry.name]: { ...prev[entry.name], [env.name]: event.currentTarget.value }
                        }))
                      }
                      type="password"
                      value={draft[env.name] ?? ''}
                    />
                  </label>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
