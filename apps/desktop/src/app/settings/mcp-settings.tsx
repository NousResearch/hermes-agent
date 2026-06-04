import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { getHermesConfigRecord, type HermesGateway, saveHermesConfig } from '@/hermes'
import { Wrench } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $activeSessionId } from '@/store/session'
import type { HermesConfigRecord } from '@/types/hermes'

import { EmptyState, LoadingState, Pill, SettingsContent } from './primitives'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

interface McpSettingsProps {
  gateway?: HermesGateway | null
  onConfigSaved?: () => void
}

type McpServers = Record<string, Record<string, unknown>>

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
  const activeSessionId = useStore($activeSessionId)
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [body, setBody] = useState('')
  const [saving, setSaving] = useState(false)
  const [reloading, setReloading] = useState(false)

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
      .catch(err => notifyError(err, 'MCP config failed to load'))

    return () => void (cancelled = true)
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
  }, [selected, servers])

  if (!config) {
    return <LoadingState label="Loading MCP servers..." />
  }

  const saveServer = async () => {
    const nextName = name.trim()

    if (!nextName) {
      notify({ kind: 'error', title: 'Name required', message: 'Give this MCP server a config key.' })

      return
    }

    let parsed: Record<string, unknown>

    try {
      const raw = JSON.parse(body)

      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error('Server config must be a JSON object')
      }

      parsed = raw as Record<string, unknown>
    } catch (err) {
      notifyError(err, 'Invalid MCP JSON')

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
      notify({ kind: 'success', title: 'MCP server saved', message: `${nextName} applies after MCP reload.` })
    } catch (err) {
      notifyError(err, 'Save failed')
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
      notifyError(err, 'Remove failed')
    } finally {
      setSaving(false)
    }
  }

  const reloadMcp = async () => {
    if (!gateway) {
      notify({ kind: 'warning', title: 'Gateway unavailable', message: 'Reconnect the gateway before reloading MCP.' })

      return
    }

    setReloading(true)

    try {
      await gateway.request('reload.mcp', {
        confirm: true,
        session_id: activeSessionId ?? undefined
      })
      notify({ kind: 'success', title: 'MCP tools reloaded', message: 'New tool schemas apply to fresh turns.' })
    } catch (err) {
      notifyError(err, 'MCP reload failed')
    } finally {
      setReloading(false)
    }
  }

  return (
    <SettingsContent>
      <div className="mb-4 flex items-center justify-end gap-4">
        <Button onClick={() => setSelected(null)} size="xs" variant="text">
          New server
        </Button>
        <Button disabled={reloading} onClick={() => void reloadMcp()} size="xs" variant="text">
          {reloading ? 'Reloading...' : 'Reload MCP'}
        </Button>
      </div>

      <div className="grid min-h-0 gap-6 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <div className="min-h-64">
          {names.length === 0 ? (
            <EmptyState description="Add a stdio or HTTP server to expose MCP tools." title="No MCP servers" />
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
                      {server.disabled === true && <Pill>disabled</Pill>}
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        <div className="grid content-start gap-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Wrench className="size-4 text-muted-foreground" />
            {selected ? 'Edit server' : 'New server'}
          </div>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted-foreground">Name</span>
            <Input onChange={event => setName(event.currentTarget.value)} placeholder="filesystem" value={name} />
          </label>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted-foreground">Server JSON</span>
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
                Remove
              </Button>
            ) : (
              <span />
            )}
            <Button disabled={saving} onClick={() => void saveServer()} size="sm">
              {saving ? 'Saving...' : 'Save server'}
            </Button>
          </div>
        </div>
      </div>
    </SettingsContent>
  )
}
