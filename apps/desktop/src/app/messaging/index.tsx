import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '@nanostores/react'

import { PageLoader } from '@/components/page-loader'
import { StatusDot, type StatusTone } from '@/components/status-dot'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  getMessagingPlatforms,
  type MessagingEnvVarInfo,
  type MessagingPlatformInfo,
  updateMessagingPlatform
} from '@/hermes'
import { AlertTriangle, ChevronDown, ExternalLink, RefreshCw, Save, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopLanguage } from '@/store/language'
import { notify, notifyError } from '@/store/notifications'

import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'
import { titlebarHeaderBaseClass } from '../shell/titlebar'
import type { SetTitlebarToolGroup } from '../shell/titlebar-controls'

interface MessagingViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
  setTitlebarToolGroup?: SetTitlebarToolGroup
}

type EditMap = Record<string, Record<string, string>>

const STATE_LABELS: Record<string, string> = {
  connected: 'Connected',
  connecting: 'Connecting',
  disabled: 'Disabled',
  fatal: 'Error',
  gateway_stopped: 'Gateway stopped',
  not_configured: 'Needs setup',
  pending_restart: 'Restart needed',
  retrying: 'Retrying',
  startup_failed: 'Startup failed'
}

const STATE_LABELS_ZH: Record<string, string> = {
  connected: '已连接',
  connecting: '连接中',
  disabled: '已禁用',
  fatal: '错误',
  gateway_stopped: '网关已停止',
  not_configured: '需要配置',
  pending_restart: '需要重启',
  retrying: '重试中',
  startup_failed: '启动失败'
}

const PLATFORM_TINTS: Record<string, string> = {
  telegram: 'bg-sky-500/15 text-sky-600 dark:text-sky-300',
  discord: 'bg-indigo-500/15 text-indigo-600 dark:text-indigo-300',
  slack: 'bg-violet-500/15 text-violet-600 dark:text-violet-300',
  mattermost: 'bg-blue-500/15 text-blue-600 dark:text-blue-300',
  matrix: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300',
  signal: 'bg-cyan-500/15 text-cyan-600 dark:text-cyan-300',
  whatsapp: 'bg-green-500/15 text-green-600 dark:text-green-300',
  bluebubbles: 'bg-blue-500/15 text-blue-600 dark:text-blue-300',
  homeassistant: 'bg-teal-500/15 text-teal-600 dark:text-teal-300',
  email: 'bg-amber-500/15 text-amber-600 dark:text-amber-300',
  sms: 'bg-rose-500/15 text-rose-600 dark:text-rose-300',
  dingtalk: 'bg-blue-500/15 text-blue-600 dark:text-blue-300',
  feishu: 'bg-cyan-500/15 text-cyan-600 dark:text-cyan-300',
  wecom: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300',
  wecom_callback: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300',
  weixin: 'bg-green-500/15 text-green-600 dark:text-green-300',
  qqbot: 'bg-amber-500/15 text-amber-600 dark:text-amber-300',
  yuanbao: 'bg-orange-500/15 text-orange-600 dark:text-orange-300',
  api_server: 'bg-slate-500/15 text-slate-600 dark:text-slate-300',
  webhook: 'bg-zinc-500/15 text-zinc-600 dark:text-zinc-300'
}

const PILL_TONE: Record<StatusTone, string> = {
  good: 'bg-primary/10 text-primary',
  muted: 'bg-muted text-muted-foreground',
  warn: 'bg-amber-500/10 text-amber-600 dark:text-amber-300',
  bad: 'bg-destructive/10 text-destructive'
}

const HINT_BY_STATE: Record<string, string> = {
  pending_restart: 'Restart the gateway from the status bar to apply this change.',
  gateway_stopped: 'Start the gateway from the status bar to connect.'
}

const HINT_BY_STATE_ZH: Record<string, string> = {
  pending_restart: '请从状态栏重启网关以应用此更改。',
  gateway_stopped: '请从状态栏启动网关后再连接。'
}

const stateLabel = (state: null | string | undefined, language: 'zh' | 'en') =>
  state
    ? language === 'zh'
      ? STATE_LABELS_ZH[state] || state.replace(/_/g, ' ')
      : STATE_LABELS[state] || state.replace(/_/g, ' ')
    : language === 'zh'
      ? '未知'
      : 'Unknown'

function stateTone({ enabled, state }: MessagingPlatformInfo): StatusTone {
  if (!enabled) {return 'muted'}

  if (state === 'connected') {return 'good'}

  if (state === 'fatal' || state === 'startup_failed') {return 'bad'}

  return 'warn'
}

const trimEdits = (edits: Record<string, string>): Record<string, string> =>
  Object.fromEntries(
    Object.entries(edits)
      .map(([k, v]) => [k, v.trim()])
      .filter(([, v]) => v)
  )

const FIELD_COPY: Record<string, { advanced?: boolean; help?: string; label: string; placeholder?: string }> = {
  TELEGRAM_BOT_TOKEN: {
    label: 'Bot token',
    help: 'Create a bot with @BotFather, then paste the token it gives you.',
    placeholder: '123456:ABC...'
  },
  TELEGRAM_ALLOWED_USERS: {
    label: 'Allowed Telegram user IDs',
    help: 'Recommended. Comma-separated numeric IDs from @userinfobot. Without this, anyone can DM your bot.'
  },
  TELEGRAM_PROXY: {
    label: 'Proxy URL',
    help: 'Only needed on networks where Telegram is blocked.',
    advanced: true
  },
  DISCORD_BOT_TOKEN: {
    label: 'Bot token',
    help: 'Create an application in the Discord Developer Portal, add a bot, then paste its token.'
  },
  DISCORD_ALLOWED_USERS: {
    label: 'Allowed Discord user IDs',
    help: 'Recommended. Comma-separated Discord user IDs.'
  },
  DISCORD_REPLY_TO_MODE: {
    label: 'Reply style',
    help: 'first, all, or off.',
    advanced: true
  },
  SLACK_BOT_TOKEN: {
    label: 'Slack bot token',
    help: 'Starts with xoxb-. Found under OAuth & Permissions after installing your Slack app.',
    placeholder: 'xoxb-...'
  },
  SLACK_APP_TOKEN: {
    label: 'Slack app token',
    help: 'Starts with xapp-. Required for Socket Mode.',
    placeholder: 'xapp-...'
  },
  SLACK_ALLOWED_USERS: {
    label: 'Allowed Slack user IDs',
    help: 'Recommended. Comma-separated Slack user IDs.'
  },
  MATTERMOST_URL: {
    label: 'Server URL',
    placeholder: 'https://mattermost.example.com'
  },
  MATTERMOST_TOKEN: {
    label: 'Bot token'
  },
  MATTERMOST_ALLOWED_USERS: {
    label: 'Allowed user IDs',
    help: 'Recommended. Comma-separated Mattermost user IDs.'
  },
  MATRIX_HOMESERVER: {
    label: 'Homeserver URL',
    placeholder: 'https://matrix.org'
  },
  MATRIX_ACCESS_TOKEN: {
    label: 'Access token'
  },
  MATRIX_USER_ID: {
    label: 'Bot user ID',
    placeholder: '@hermes:example.org'
  },
  MATRIX_ALLOWED_USERS: {
    label: 'Allowed Matrix user IDs',
    help: 'Recommended. Comma-separated user IDs in @user:server format.'
  },
  SIGNAL_HTTP_URL: {
    label: 'Signal bridge URL',
    placeholder: 'http://127.0.0.1:8080',
    help: 'URL of a running signal-cli REST bridge.'
  },
  SIGNAL_ACCOUNT: {
    label: 'Phone number',
    help: 'The number registered with your signal-cli bridge.'
  },
  SIGNAL_ALLOWED_USERS: {
    label: 'Allowed Signal users',
    help: 'Recommended. Comma-separated Signal identifiers.'
  },
  WHATSAPP_ENABLED: {
    label: 'Enable WhatsApp bridge',
    help: 'Set automatically by the toggle below. Leave alone unless you know you need it.',
    advanced: true
  },
  WHATSAPP_MODE: {
    label: 'Bridge mode',
    advanced: true
  },
  WHATSAPP_ALLOWED_USERS: {
    label: 'Allowed WhatsApp users',
    help: 'Recommended. Comma-separated phone numbers or WhatsApp IDs.'
  }
}

function fieldCopy(field: MessagingEnvVarInfo) {
  const copy = FIELD_COPY[field.key] || {}

  return {
    label: copy.label || field.prompt || field.key,
    help: copy.help || field.description,
    placeholder: copy.placeholder || field.prompt,
    advanced: Boolean(copy.advanced || field.advanced)
  }
}

export function MessagingView({
  setStatusbarItemGroup: _setStatusbarItemGroup,
  setTitlebarToolGroup,
  ...props
}: MessagingViewProps) {
  const language = useStore($desktopLanguage)
  const [platforms, setPlatforms] = useState<MessagingPlatformInfo[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [edits, setEdits] = useState<EditMap>({})
  const [refreshing, setRefreshing] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)

  const refreshPlatforms = useCallback(async (silent = false) => {
    if (!silent) {
      setRefreshing(true)
    }

    try {
      const result = await getMessagingPlatforms()
      setPlatforms(result.platforms)
      setSelectedId(current => current || result.platforms[0]?.id || null)
    } catch (err) {
      if (!silent) {
        notifyError(err, language === 'zh' ? '消息平台加载失败' : 'Messaging platforms failed to load')
      }
    } finally {
      if (!silent) {
        setRefreshing(false)
      }
    }
  }, [language])

  useEffect(() => {
    void refreshPlatforms()
  }, [refreshPlatforms])

  // Auto-poll while the user is on the messaging page so connection status
  // updates without a manual "check" click. Pause when the tab is hidden.
  useEffect(() => {
    let cancelled = false

    function tick() {
      if (cancelled || document.hidden) {
        return
      }

      void refreshPlatforms(true)
    }

    const id = window.setInterval(tick, 6000)

    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [refreshPlatforms])

  useEffect(() => {
    if (!setTitlebarToolGroup) {
      return
    }

    setTitlebarToolGroup('messaging', [
      {
        disabled: refreshing,
        icon: <RefreshCw className={cn(refreshing && 'animate-spin')} />,
        id: 'refresh-messaging',
        label: language === 'zh' ? (refreshing ? '正在刷新消息平台' : '刷新消息平台') : refreshing ? 'Refreshing messaging' : 'Refresh messaging',
        onSelect: () => void refreshPlatforms()
      }
    ])

    return () => setTitlebarToolGroup('messaging', [])
  }, [language, refreshPlatforms, refreshing, setTitlebarToolGroup])

  const selected = useMemo(() => {
    if (!platforms) {
      return null
    }

    return platforms.find(platform => platform.id === selectedId) || platforms[0] || null
  }, [platforms, selectedId])

  const enabledCount = platforms?.filter(platform => platform.enabled).length || 0

  async function handleToggle(platform: MessagingPlatformInfo, enabled: boolean) {
    setSaving(`enabled:${platform.id}`)

    try {
      await updateMessagingPlatform(platform.id, { enabled })
      setPlatforms(
        current =>
          current?.map(row =>
            row.id === platform.id
              ? {
                  ...row,
                  enabled,
                  state: enabled ? (row.configured ? 'pending_restart' : 'not_configured') : 'disabled'
                }
              : row
          ) ?? current
      )
      notify({
        kind: 'success',
        title: language === 'zh' ? `${platform.name} 已${enabled ? '启用' : '禁用'}` : enabled ? `${platform.name} enabled` : `${platform.name} disabled`,
        message: language === 'zh' ? '请重启网关以使此更改生效。' : 'Restart the gateway for this change to take effect.'
      })
    } catch (err) {
      notifyError(err, language === 'zh' ? `更新 ${platform.name} 失败` : `Failed to update ${platform.name}`)
    } finally {
      setSaving(null)
    }
  }

  async function handleSave(platform: MessagingPlatformInfo) {
    const env = trimEdits(edits[platform.id] || {})

    if (Object.keys(env).length === 0) {
      return
    }

    setSaving(`env:${platform.id}`)

    try {
      await updateMessagingPlatform(platform.id, { env })
      setEdits(current => ({ ...current, [platform.id]: {} }))
      await refreshPlatforms()
      notify({
        kind: 'success',
        title: language === 'zh' ? `${platform.name} 配置已保存` : `${platform.name} setup saved`,
        message: language === 'zh' ? '请重启网关以使用新凭据重新连接。' : 'Restart the gateway to reconnect with the new credentials.'
      })
    } catch (err) {
      notifyError(err, language === 'zh' ? `保存 ${platform.name} 失败` : `Failed to save ${platform.name}`)
    } finally {
      setSaving(null)
    }
  }

  async function handleClear(platform: MessagingPlatformInfo, key: string) {
    setSaving(`clear:${key}`)

    try {
      await updateMessagingPlatform(platform.id, { clear_env: [key] })
      setEdits(current => ({
        ...current,
        [platform.id]: {
          ...(current[platform.id] || {}),
          [key]: ''
        }
      }))
      await refreshPlatforms()
      notify({
        kind: 'success',
        title: language === 'zh' ? `${key} 已清除` : `${key} cleared`,
        message: language === 'zh' ? `${platform.name} 配置已更新。` : `${platform.name} setup was updated.`
      })
    } catch (err) {
      notifyError(err, language === 'zh' ? `清除 ${key} 失败` : `Failed to clear ${key}`)
    } finally {
      setSaving(null)
    }
  }

  return (
    <section {...props} className="flex h-full min-w-0 flex-col overflow-hidden rounded-b-[0.9375rem] bg-background">
      <header className={titlebarHeaderBaseClass}>
        <h2 className="pointer-events-auto text-base font-semibold leading-none tracking-tight">
          {language === 'zh' ? '消息平台' : 'Messaging'}
        </h2>
        <span className="pointer-events-auto text-xs text-muted-foreground">
          {language === 'zh'
            ? enabledCount === 0
              ? '未启用平台'
              : `已启用 ${enabledCount} 个`
            : enabledCount === 0
              ? 'No platforms enabled'
              : `${enabledCount} enabled`}
        </span>
      </header>

      <div className="min-h-0 flex-1 overflow-hidden rounded-b-[1.0625rem] border border-border/50 bg-background/85">
        {!platforms ? (
          <PageLoader label={language === 'zh' ? '正在加载消息平台...' : 'Loading messaging platforms...'} />
        ) : (
          <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[16rem_minmax(0,1fr)]">
            <aside className="min-h-0 overflow-y-auto border-b border-border/50 p-2 lg:border-b-0 lg:border-r">
              <ul className="space-y-1">
                {platforms.map(platform => (
                  <li key={platform.id}>
                    <PlatformRow
                      active={selected?.id === platform.id}
                      onSelect={() => setSelectedId(platform.id)}
                      platform={platform}
                    />
                  </li>
                ))}
              </ul>
            </aside>

            <main className="min-h-0 overflow-hidden">
              {selected && (
                <PlatformDetail
                  edits={edits[selected.id] || {}}
                  onClear={key => void handleClear(selected, key)}
                  onEdit={(key, value) =>
                    setEdits(current => ({
                      ...current,
                      [selected.id]: {
                        ...(current[selected.id] || {}),
                        [key]: value
                      }
                    }))
                  }
                  onSave={() => void handleSave(selected)}
                  onToggle={enabled => void handleToggle(selected, enabled)}
                  platform={selected}
                  saving={saving}
                  language={language}
                />
              )}
            </main>
          </div>
        )}
      </div>
    </section>
  )
}

function PlatformRow({
  active,
  onSelect,
  platform
}: {
  active: boolean
  onSelect: () => void
  platform: MessagingPlatformInfo
}) {
  return (
    <button
      className={cn(
        'flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition-colors',
        active ? 'bg-accent text-foreground' : 'text-foreground/85 hover:bg-accent/60'
      )}
      onClick={onSelect}
      type="button"
    >
      <PlatformAvatar platformId={platform.id} platformName={platform.name} />
      <span className="flex min-w-0 flex-1 items-center justify-between gap-2">
        <span className="truncate text-sm font-medium">{platform.name}</span>
        <StatusDot tone={stateTone(platform)} />
      </span>
    </button>
  )
}

function PlatformAvatar({ platformId, platformName }: { platformId: string; platformName: string }) {
  return (
    <span
      className={cn(
        'inline-flex size-7 shrink-0 items-center justify-center rounded-md text-sm font-semibold',
        PLATFORM_TINTS[platformId] || 'bg-muted text-muted-foreground'
      )}
    >
      {platformName.charAt(0).toUpperCase()}
    </span>
  )
}

function PlatformDetail({
  edits,
  onClear,
  onEdit,
  onSave,
  onToggle,
  platform,
  saving,
  language
}: {
  edits: Record<string, string>
  onClear: (key: string) => void
  onEdit: (key: string, value: string) => void
  onSave: () => void
  onToggle: (enabled: boolean) => void
  platform: MessagingPlatformInfo
  saving: string | null
  language: 'zh' | 'en'
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const hasEdits = Object.keys(trimEdits(edits)).length > 0
  const requiredFields = platform.env_vars.filter(field => field.required)
  const optionalFields = platform.env_vars.filter(field => !field.required && !fieldCopy(field).advanced)
  const advancedFields = platform.env_vars.filter(field => !field.required && fieldCopy(field).advanced)
  const hiddenCount = advancedFields.length
  const isSavingEnv = saving === `env:${platform.id}`

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-7 px-6 py-6">
          <header className="flex items-start gap-4">
            <PlatformAvatar platformId={platform.id} platformName={platform.name} />
            <div className="min-w-0 flex-1">
              <h3 className="text-xl font-semibold tracking-tight">{platform.name}</h3>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">{platform.description}</p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <StatePill tone={stateTone(platform)}>{stateLabel(platform.state, language)}</StatePill>
                <SetupPill active={platform.configured}>
                  {language === 'zh' ? (platform.configured ? '凭据已设置' : '需要配置') : platform.configured ? 'Credentials set' : 'Needs setup'}
                </SetupPill>
                {!platform.gateway_running && <SetupPill active={false}>{language === 'zh' ? '网关已停止' : 'Gateway stopped'}</SetupPill>}
              </div>
              <PlatformHint language={language} platform={platform} />
            </div>
          </header>

          {platform.error_message && (
            <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-xs leading-5 text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{platform.error_message}</span>
            </div>
          )}

          <section>
            <SectionTitle>{language === 'zh' ? '获取凭据' : 'Get your credentials'}</SectionTitle>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">
              {introCopy(platform)}
            </p>
            <div className="mt-3">
              <Button asChild size="sm" variant="outline">
                <a href={platform.docs_url} rel="noreferrer" target="_blank">
                  {language === 'zh' ? '打开配置指南' : 'Open setup guide'}
                  <ExternalLink className="size-3.5" />
                </a>
              </Button>
            </div>
          </section>

          <section>
            <SectionTitle>{language === 'zh' ? '必填' : 'Required'}</SectionTitle>
            <div className="mt-3 space-y-4">
              {requiredFields.length > 0 ? (
                requiredFields.map(field => (
                  <MessagingField
                    edits={edits}
                    field={field}
                    key={field.key}
                    onClear={onClear}
                    onEdit={onEdit}
                    saving={saving}
                    language={language}
                  />
                ))
              ) : (
                <p className="text-sm leading-6 text-muted-foreground">
                  {language === 'zh'
                    ? '此平台不需要在这里填写令牌。请先使用上方配置指南，然后在下方启用。'
                    : 'This platform does not need a token here. Use the setup guide above, then enable it below.'}
                </p>
              )}
            </div>
          </section>

          {optionalFields.length > 0 && (
            <section>
              <SectionTitle>{language === 'zh' ? '推荐' : 'Recommended'}</SectionTitle>
              <div className="mt-3 space-y-4">
                {optionalFields.map(field => (
                  <MessagingField
                    edits={edits}
                    field={field}
                    key={field.key}
                    onClear={onClear}
                    onEdit={onEdit}
                    saving={saving}
                    language={language}
                  />
                ))}
              </div>
            </section>
          )}

          {hiddenCount > 0 && (
            <section>
              <button
                className="flex w-full items-center justify-between gap-2 rounded-lg px-1 py-1 text-left text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground"
                onClick={() => setShowAdvanced(value => !value)}
                type="button"
              >
                <span>{language === 'zh' ? '高级' : 'Advanced'} ({hiddenCount})</span>
                <ChevronDown
                  className={cn('size-3.5 transition-transform', !showAdvanced && '-rotate-90')}
                />
              </button>
              {showAdvanced && (
                <div className="mt-3 space-y-4">
                  {advancedFields.map(field => (
                    <MessagingField
                      edits={edits}
                      field={field}
                      key={field.key}
                      onClear={onClear}
                      onEdit={onEdit}
                      saving={saving}
                      language={language}
                    />
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      </div>

      <footer className="border-t border-border/50 bg-background/95 px-6 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-2xl flex-wrap items-center gap-2">
          <label className="flex shrink-0 items-center gap-2 rounded-lg border border-border/50 bg-muted/25 px-3 py-1.5 text-sm">
            <Switch
              aria-label={
                language === 'zh'
                  ? `${platform.enabled ? '禁用' : '启用'} ${platform.name}`
                  : platform.enabled
                    ? `Disable ${platform.name}`
                    : `Enable ${platform.name}`
              }
              checked={platform.enabled}
              disabled={saving === `enabled:${platform.id}`}
              onCheckedChange={onToggle}
            />
            <span className="text-xs font-medium text-muted-foreground">
              {language === 'zh' ? (platform.enabled ? '已启用' : '已禁用') : platform.enabled ? 'Enabled' : 'Disabled'}
            </span>
          </label>

          <div className="ml-auto flex items-center gap-2">
            {hasEdits && <span className="text-xs text-muted-foreground">{language === 'zh' ? '有未保存更改' : 'Unsaved changes'}</span>}
            <Button disabled={!hasEdits || isSavingEnv} onClick={onSave} size="sm">
              <Save />
              {language === 'zh' ? (isSavingEnv ? '正在保存...' : '保存更改') : isSavingEnv ? 'Saving...' : 'Save changes'}
            </Button>
          </div>
        </div>
      </footer>
    </div>
  )
}

const PLATFORM_INTRO: Record<string, string> = {
  telegram:
    'In Telegram, talk to @BotFather, run /newbot, and copy the token it gives you. Then grab your numeric user ID from @userinfobot.',
  discord:
    'Open the Discord Developer Portal, create an application, add a Bot, then copy its token. Invite the bot to your server with the right scopes.',
  slack:
    'Create a Slack app, enable Socket Mode, install it to your workspace, then copy the Bot token (xoxb-) and App-level token (xapp-).',
  mattermost:
    'On your Mattermost server, create a bot account or personal access token, then paste the server URL and token here.',
  matrix: 'Sign in to your homeserver with the bot account, then copy the access token, user ID, and homeserver URL.',
  signal: 'Run a signal-cli REST bridge somewhere reachable, then point Hermes at the URL and the registered phone number.',
  whatsapp:
    'Start the WhatsApp bridge that ships with Hermes, scan the QR code on first run, then enable the platform.',
  bluebubbles:
    'Run BlueBubbles Server on a Mac with iMessage, expose its API, then point Hermes at the URL with the server password.',
  homeassistant:
    'In Home Assistant, open your profile and create a long-lived access token. Paste it here along with your HA URL.',
  email:
    'Use a dedicated mailbox. For Gmail/Workspace, create an app password and use imap.gmail.com / smtp.gmail.com.',
  sms: 'Get your Twilio Account SID and Auth Token from the Twilio console, plus a phone number that can send SMS.',
  dingtalk:
    'Create a DingTalk app in the developer console, then copy the Client ID (App key) and Client Secret here.',
  feishu:
    'Create a Feishu / Lark app, configure the bot capability, and copy the App ID, App secret, and event encryption keys.',
  wecom:
    'Add a group robot in WeCom and copy its webhook key as WECOM_BOT_ID. Send-only — use the WeCom (app) option for two-way.',
  wecom_callback:
    'Set up a WeCom self-built app, expose its callback URL, and provide the corp ID, secret, agent ID, and AES key.',
  weixin:
    'Sign in to the WeChat Official Account platform, copy the AppID and Token, and point the message callback URL at Hermes.',
  qqbot: 'Register an app on the QQ Open Platform (q.qq.com) and copy the App ID and Client Secret.',
  api_server:
    'Expose Hermes as an OpenAI-compatible API. Set an auth key, then point Open WebUI / LobeChat / etc. at the host:port.',
  webhook: 'Run an HTTP server that other tools (GitHub, GitLab, custom apps) can POST to. Use the secret to verify signatures.'
}

const introCopy = (platform: MessagingPlatformInfo) => PLATFORM_INTRO[platform.id] || platform.description

function MessagingField({
  edits,
  field,
  language,
  onClear,
  onEdit,
  saving
}: {
  edits: Record<string, string>
  field: MessagingEnvVarInfo
  language: 'zh' | 'en'
  onClear: (key: string) => void
  onEdit: (key: string, value: string) => void
  saving: string | null
}) {
  const copy = fieldCopy(field)

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-baseline gap-2">
        <label className="text-sm font-medium text-foreground" htmlFor={`messaging-field-${field.key}`}>
          {copy.label}
        </label>
        {field.is_set && <span className="text-[0.66rem] font-medium text-primary">{language === 'zh' ? '已保存' : 'Saved'}</span>}
      </div>
      <div className="flex items-center gap-2">
        <Input
          className="h-9 rounded-lg font-mono text-sm"
          id={`messaging-field-${field.key}`}
          onChange={event => onEdit(field.key, event.target.value)}
          placeholder={field.is_set ? field.redacted_value || (language === 'zh' ? '替换当前值' : 'Replace current value') : copy.placeholder}
          type={field.is_password ? 'password' : 'text'}
          value={edits[field.key] || ''}
        />
        {field.url && (
          <Button asChild size="icon-sm" title={language === 'zh' ? '打开文档' : 'Open docs'} variant="ghost">
            <a href={field.url} rel="noreferrer" target="_blank">
              <ExternalLink className="size-3.5" />
            </a>
          </Button>
        )}
        {field.is_set && (
          <Button
            disabled={saving === `clear:${field.key}`}
            onClick={() => onClear(field.key)}
            size="icon-sm"
            title={language === 'zh' ? `清除 ${field.key}` : `Clear ${field.key}`}
            variant="ghost"
          >
            <Trash2 className="size-3.5" />
          </Button>
        )}
      </div>
      {copy.help && <p className="text-xs leading-5 text-muted-foreground">{copy.help}</p>}
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{children}</h4>
  )
}

function PlatformHint({ platform, language }: { platform: MessagingPlatformInfo; language: 'zh' | 'en' }) {
  if (!platform.enabled || platform.state === 'connected') {return null}

  const hint =
    (language === 'zh' ? HINT_BY_STATE_ZH[platform.state || ''] : HINT_BY_STATE[platform.state || '']) ||
    (platform.gateway_running ? null : language === 'zh' ? HINT_BY_STATE_ZH.gateway_stopped : HINT_BY_STATE.gateway_stopped)

  return hint ? <p className="mt-2 text-xs leading-5 text-muted-foreground">{hint}</p> : null
}

function StatePill({ children, tone }: { children: string; tone: StatusTone }) {
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 text-[0.66rem] font-medium',
        PILL_TONE[tone]
      )}
    >
      <StatusDot tone={tone} />
      {children}
    </span>
  )
}

function SetupPill({ active, children }: { active: boolean; children: string }) {
  return (
    <span
      className={cn('inline-flex items-center rounded-full px-2 py-0.5 text-[0.66rem] font-medium', PILL_TONE[active ? 'good' : 'muted'])}
    >
      {children}
    </span>
  )
}
