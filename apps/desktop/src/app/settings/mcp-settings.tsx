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
      .catch(err => notifyError(err, 'MCP 설정을 로드하지 못했습니다'))

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
    return <LoadingState label="MCP 서버 로드 중..." />
  }

  const saveServer = async () => {
    const nextName = name.trim()

    if (!nextName) {
      notify({ kind: 'error', title: '이름 필요', message: '이 MCP 서버에 설정 키를 지정하세요.' })

      return
    }

    let parsed: Record<string, unknown>

    try {
      const raw = JSON.parse(body)

      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error('서버 설정은 JSON 객체여야 합니다')
      }

      parsed = raw as Record<string, unknown>
    } catch (err) {
      notifyError(err, '유효하지 않은 MCP JSON')

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
      notify({ kind: 'success', title: 'MCP 서버 저장됨', message: `${nextName} 서버는 MCP 새로고침 후 적용됩니다.` })
    } catch (err) {
      notifyError(err, '저장 실패')
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
      notifyError(err, '삭제 실패')
    } finally {
      setSaving(false)
    }
  }

  const reloadMcp = async () => {
    if (!gateway) {
      notify({ kind: 'warning', title: '게이트웨이 사용 불가', message: 'MCP를 새로고침하기 전에 게이트웨이를 다시 연결하세요.' })

      return
    }

    setReloading(true)

    try {
      await gateway.request('reload.mcp', {
        confirm: true,
        session_id: activeSessionId ?? undefined
      })
      notify({ kind: 'success', title: 'MCP 도구 새로고침 됨', message: '새 턴부터 새로운 도구 스키마가 적용됩니다.' })
    } catch (err) {
      notifyError(err, 'MCP 새로고침 실패')
    } finally {
      setReloading(false)
    }
  }

  return (
    <SettingsContent>
      <div className="mb-4 flex items-center justify-end gap-4">
        <Button onClick={() => setSelected(null)} size="xs" variant="text">
          새 서버
        </Button>
        <Button disabled={reloading} onClick={() => void reloadMcp()} size="xs" variant="text">
          {reloading ? '새로고침 중...' : 'MCP 새로고침'}
        </Button>
      </div>

      <div className="grid min-h-0 gap-6 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <div className="min-h-64">
          {names.length === 0 ? (
            <EmptyState description="MCP 도구를 사용하려면 stdio 또는 HTTP 서버를 추가하세요." title="MCP 서버 없음" />
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
                      {server.disabled === true && <Pill>비활성화됨</Pill>}
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
            {selected ? '서버 편집' : '새 서버'}
          </div>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted-foreground">이름</span>
            <Input onChange={event => setName(event.currentTarget.value)} placeholder="filesystem" value={name} />
          </label>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted-foreground">서버 JSON</span>
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
                삭제
              </Button>
            ) : (
              <span />
            )}
            <Button disabled={saving} onClick={() => void saveServer()} size="sm">
              {saving ? '저장 중...' : '서버 저장'}
            </Button>
          </div>
        </div>
      </div>
    </SettingsContent>
  )
}
