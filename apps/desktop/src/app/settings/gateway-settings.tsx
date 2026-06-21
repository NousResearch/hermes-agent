import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AlertCircle, Check, Globe, Loader2, Monitor } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopLanguage } from '@/store/language'
import { notify, notifyError } from '@/store/notifications'
import { useStore } from '@nanostores/react'

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
        'rounded-2xl border p-4 text-left transition',
        active ? 'border-primary bg-primary/10 ring-2 ring-primary/15' : 'border-border bg-background/60 hover:bg-muted/40',
        disabled && 'cursor-not-allowed opacity-50'
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center gap-2 text-sm font-medium">
        <Icon className="size-4 text-muted-foreground" />
        <span>{title}</span>
        {active ? <Check className="ml-auto size-4 text-primary" /> : null}
      </div>
      <p className="mt-2 text-xs leading-5 text-muted-foreground">{description}</p>
    </button>
  )
}

export function GatewaySettings() {
  const language = useStore($desktopLanguage)
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
      .catch(err => notifyError(err, language === 'zh' ? '网关设置加载失败' : 'Gateway settings failed to load'))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
  }, [language])

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
        title: language === 'zh' ? '远程网关信息不完整' : 'Remote gateway incomplete',
        message:
          language === 'zh'
            ? '切换到远程网关前，请填写远程 URL 和会话令牌。'
            : 'Enter a remote URL and session token before switching to remote.'
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
        title: apply
          ? language === 'zh'
            ? '网关连接正在重启'
            : 'Gateway connection restarting'
          : language === 'zh'
            ? '网关设置已保存'
            : 'Gateway settings saved',
        message: apply
          ? language === 'zh'
            ? '元话 Agent Desktop 将使用已保存设置重新连接。'
            : '元话 Agent Desktop will reconnect using the saved settings.'
          : language === 'zh'
            ? '已保存，将在下次重启时生效。'
            : 'Saved for the next restart.'
      })
    } catch (err) {
      notifyError(
        err,
        language === 'zh'
          ? apply
            ? '无法应用网关设置'
            : '无法保存网关设置'
          : apply
            ? 'Could not apply gateway settings'
            : 'Could not save gateway settings'
      )
    } finally {
      setSaving(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: language === 'zh' ? '远程网关信息不完整' : 'Remote gateway incomplete',
        message:
          language === 'zh'
            ? '测试前请填写远程 URL 和会话令牌。'
            : 'Enter a remote URL and session token before testing.'
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

      const message = `Connected to ${result.baseUrl}${result.version ? ` · metakina-agent runtime ${result.version}` : ''}`
      setLastTest(message)
      notify({ kind: 'success', title: language === 'zh' ? '远程网关可连接' : 'Remote gateway reachable', message })
    } catch (err) {
      notifyError(err, language === 'zh' ? '远程网关测试失败' : 'Remote gateway test failed')
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label={language === 'zh' ? '正在加载网关设置...' : 'Loading gateway settings...'} />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return (
      <EmptyState
        description={
          language === 'zh'
            ? '桌面 IPC 桥未暴露网关设置。'
            : 'The desktop IPC bridge does not expose gateway settings.'
        }
        title={language === 'zh' ? '网关设置不可用' : 'Gateway settings unavailable'}
      />
    )
  }

  return (
    <SettingsContent>
      <div className="mb-6">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Globe className="size-4 text-muted-foreground" />
          {language === 'zh' ? '网关连接' : 'Gateway Connection'}
          {state.envOverride ? <Pill tone="primary">{language === 'zh' ? '环境变量覆盖' : 'env override'}</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-xs leading-5 text-muted-foreground">
          {language === 'zh'
            ? '元话 Agent Desktop 默认启动本地网关。需要控制另一台机器或可信代理后已经运行的 metakina-agent runtime 时，可以使用远程网关。'
            : '元话 Agent Desktop starts its own local gateway by default. Use a remote gateway when you want this app to control an already-running metakina-agent runtime on another machine or behind a trusted proxy.'}
        </p>
      </div>

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-xs text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">
              {language === 'zh'
                ? '环境变量正在控制当前桌面会话。'
                : 'Environment variables are controlling this desktop session.'}
            </div>
            <div className="mt-1 leading-5">
              {language === 'zh' ? '取消设置 ' : 'Unset '}
              <code>HERMES_DESKTOP_REMOTE_URL</code>
              {language === 'zh' ? ' 和 ' : ' and '}
              <code>HERMES_DESKTOP_REMOTE_TOKEN</code>
              {language === 'zh' ? ' 后，才能使用下方保存的设置。' : ' to use the saved setting below.'}
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ModeCard
          active={state.mode === 'local'}
          description={
            language === 'zh'
              ? '在本机启动私有 metakina-agent dashboard 后端。这是默认模式，可离线工作。'
              : 'Start a private metakina-agent dashboard backend on localhost. This is the default and works offline.'
          }
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title={language === 'zh' ? '本地网关' : 'Local gateway'}
        />
        <ModeCard
          active={state.mode === 'remote'}
          description={
            language === 'zh'
              ? '使用会话令牌将当前桌面壳连接到远程 metakina-agent dashboard 后端。'
              : 'Connect this desktop shell to a remote metakina-agent dashboard backend using its session token.'
          }
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title={language === 'zh' ? '远程网关' : 'Remote gateway'}
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
          description={
            language === 'zh'
              ? '远程 dashboard 后端的基础 URL。支持路径前缀，例如 /hermes。'
              : 'Base URL for the remote dashboard backend. Path prefixes are supported, for example /hermes.'
          }
          title={language === 'zh' ? '远程 URL' : 'Remote URL'}
        />
        <ListRow
          action={
            <Input
              autoComplete="off"
              className={cn('h-8 font-mono', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setRemoteToken(event.target.value)}
              placeholder={
                state.remoteTokenSet
                  ? language === 'zh'
                    ? `已有令牌 ${state.remoteTokenPreview ?? '已保存'}`
                    : `Existing token ${state.remoteTokenPreview ?? 'saved'}`
                  : language === 'zh'
                    ? '粘贴会话令牌'
                    : 'Paste session token'
              }
              type="password"
              value={remoteToken}
            />
          }
          description={
            language === 'zh'
              ? '用于 REST 和 WebSocket 访问的 dashboard 会话令牌。留空则保留已保存令牌。'
              : 'The dashboard session token used for REST and WebSocket access. Leave blank to keep the saved token.'
          }
          title={language === 'zh' ? '会话令牌' : 'Session token'}
        />
      </div>

      {lastTest ? <div className="mt-4 text-xs text-primary">{lastTest}</div> : null}

      <div className="mt-6 flex flex-wrap justify-end gap-3">
        <Button disabled={state.envOverride || testing || !canUseRemote} onClick={() => void testRemote()} variant="outline">
          {testing ? <Loader2 className="size-4 animate-spin" /> : null}
          {language === 'zh' ? '测试远程' : 'Test remote'}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(false)} variant="outline">
          {language === 'zh' ? '保存到下次重启' : 'Save for next restart'}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(true)}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          {language === 'zh' ? '保存并重新连接' : 'Save and reconnect'}
        </Button>
      </div>
    </SettingsContent>
  )
}
