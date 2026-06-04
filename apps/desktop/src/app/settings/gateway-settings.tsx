import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AlertCircle, Check, FileText, Globe, Loader2, Monitor } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { CONTROL_TEXT } from './constants'
import { EmptyState, ListRow, LoadingState, Pill, SettingsContent } from './primitives'

type Mode = 'local' | 'remote'

interface GatewaySettingsState {
  envOverride: boolean
  mode: Mode
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  remoteUrl: string
}

const EMPTY_STATE: GatewaySettingsState = {
  envOverride: false,
  mode: 'local',
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

function ModeCard({
  active,
  description,
  disabled,
  icon: Icon,
  onSelect,
  title
}: {
  active: boolean
  description: string
  disabled?: boolean
  icon: typeof Monitor
  onSelect: () => void
  title: string
}) {
  return (
    <button
      className={cn(
        'rounded-xl border p-3 text-left transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) hover:bg-(--chrome-action-hover)',
        disabled && 'cursor-not-allowed opacity-50'
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
        <Icon className="size-4 text-muted-foreground" />
        <span>{title}</span>
        {active ? <Check className="ml-auto size-4 text-primary" /> : null}
      </div>
      <p className="mt-1.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </button>
  )
}

export function GatewaySettings() {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [state, setState] = useState<GatewaySettingsState>(EMPTY_STATE)
  const [remoteToken, setRemoteToken] = useState('')
  const [lastTest, setLastTest] = useState<null | string>(null)

  useEffect(() => {
    let cancelled = false
    const desktop = window.hermesDesktop

    if (!desktop?.getConnectionConfig) {
      setLoading(false)

      return () => void (cancelled = true)
    }

    desktop
      .getConnectionConfig()
      .then(config => {
        if (cancelled) {
          return
        }

        setState(config)
      })
      .catch(err => notifyError(err, 'Gateway settings failed to load'))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
  }, [])

  const canUseRemote = useMemo(
    () => Boolean(state.remoteUrl.trim()) && (Boolean(remoteToken.trim()) || state.remoteTokenSet),
    [remoteToken, state.remoteTokenSet, state.remoteUrl]
  )

  const payload = () => ({
    mode: state.mode,
    remoteToken: remoteToken.trim() || undefined,
    remoteUrl: state.remoteUrl.trim()
  })

  const save = async (apply: boolean) => {
    if (state.mode === 'remote' && !canUseRemote) {
      notify({
        kind: 'warning',
        title: 'Remote gateway incomplete',
        message: 'Enter a remote URL and session token before switching to remote.'
      })

      return
    }

    setSaving(true)

    try {
      const next = apply
        ? await window.hermesDesktop.applyConnectionConfig(payload())
        : await window.hermesDesktop.saveConnectionConfig(payload())

      setState(next)
      setRemoteToken('')
      notify({
        kind: 'success',
        title: apply ? '网关连接正在重启' : '网关设置已保存',
        message: apply ? 'Hermes 桌面版将使用保存的设置重新连接。' : '已保存，下次重启生效。'
      })
    } catch (err) {
      notifyError(err, apply ? '无法应用网关设置' : '无法保存网关设置')
    } finally {
      setSaving(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: '远程网关不完整',
        message: '测试前请输入远程 URL 和会话令牌。'
      })

      return
    }

    setTesting(true)
    setLastTest(null)

    try {
      const result = await window.hermesDesktop.testConnectionConfig({
        mode: 'remote',
        remoteToken: remoteToken.trim() || undefined,
        remoteUrl: state.remoteUrl.trim()
      })

      const message = `已连接到 ${result.baseUrl}${result.version ? ` · Hermes ${result.version}` : ''}`
      setLastTest(message)
      notify({ kind: 'success', title: '远程网关可连接', message })
    } catch (err) {
      notifyError(err, '远程网关测试失败')
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label="正在加载网关设置..." />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return (
      <EmptyState
        description="桌面 IPC 桥接不暴露网关设置。"
        title="网关设置不可用"
      />
    )
  }

  return (
    <SettingsContent>
      <div className="mb-5">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Globe className="size-4 text-muted-foreground" />
          Gateway Connection
          {state.envOverride ? <Pill tone="primary">环境变量覆盖</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          Hermes 桌面版默认启动自己的本地网关。当您希望此应用控制在另一台机器或受信任代理上已经运行的 Hermes 后端时，请使用远程网关。
        </p>
      </div>

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">环境变量正在控制此桌面会话。</div>
            <div className="mt-1 leading-5">
              取消设置 <code>HERMES_DESKTOP_REMOTE_URL</code> 和 <code>HERMES_DESKTOP_REMOTE_TOKEN</code> 以使用下方保存的设置。
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ModeCard
          active={state.mode === 'local'}
          description="在 localhost 上启动私有的 Hermes 后端。这是默认设置，可离线工作。"
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title="本地网关"
        />
        <ModeCard
          active={state.mode === 'remote'}
          description="将此桌面 Shell 连接到远程 Hermes 后端，使用其会话令牌。"
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title="远程网关"
        />
      </div>

      <div className="mt-5 divide-y divide-border/40">
        <ListRow
          action={
            <Input
              className={cn('h-8', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
              placeholder="https://gateway.example.com/hermes"
              value={state.remoteUrl}
            />
          }
          description="远程仪表板后端的基础 URL。支持路径前缀，例如 /hermes。"
          title="远程 URL"
        />
        <ListRow
          action={
            <Input
              autoComplete="off"
              className={cn('h-8 font-mono', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setRemoteToken(event.target.value)}
              placeholder={
                state.remoteTokenSet ? `现有令牌 ${state.remoteTokenPreview ?? '已保存'}` : '粘贴会话令牌'
              }
              type="password"
              value={remoteToken}
            />
          }
          description="用于 REST 和 WebSocket 访问的仪表板会话令牌。留空以保留已保存的令牌。"
          title="会话令牌"
        />
      </div>

      {lastTest ? <div className="mt-4 text-xs text-primary">{lastTest}</div> : null}

      <div className="mt-6 flex flex-wrap justify-end gap-3">
        <Button
          disabled={state.envOverride || testing || !canUseRemote}
          onClick={() => void testRemote()}
          variant="outline"
        >
          {testing ? <Loader2 className="size-4 animate-spin" /> : null}
          测试远程
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(false)} variant="outline">
          保存，下次重启生效
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(true)}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          保存并重新连接
        </Button>
      </div>

      <div className="mt-6 divide-y divide-border/40">
        <ListRow
          action={
            <Button onClick={() => void window.hermesDesktop?.revealLogs()} variant="outline">
              <FileText className="size-4" />
              打开日志
            </Button>
          }
          description="在文件管理器中显示 desktop.log — 当网关启动失败时很有用。"
          title="诊断信息"
        />
      </div>
    </SettingsContent>
  )
}
