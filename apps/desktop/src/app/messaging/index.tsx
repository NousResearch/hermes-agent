import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { StatusDot, type StatusTone } from '@/components/status-dot'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  getMessagingPlatforms,
  type MessagingEnvVarInfo,
  type MessagingPlatformInfo,
  updateMessagingPlatform
} from '@/hermes'
import { AlertTriangle, ExternalLink, Save, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PageSearchShell } from '../page-search-shell'
import { CREDENTIAL_CONTROL_CLASS } from '../settings/credential-key-ui'
import { ListRow } from '../settings/primitives'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { PlatformAvatar } from './platform-icon'

interface MessagingViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

type EditMap = Record<string, Record<string, string>>

const STATE_LABELS: Record<string, string> = {
  connected: 'Connected',
  connecting: 'Connecting',
  disabled: 'Disabled',
  fatal: 'Error',
  gateway_stopped: 'Messaging gateway stopped',
  not_configured: 'Needs setup',
  pending_restart: 'Restart needed',
  retrying: 'Retrying',
  startup_failed: 'Startup failed'
}

const TONE_VARIANT: Record<StatusTone, BadgeProps['variant']> = {
  good: 'default',
  muted: 'muted',
  warn: 'warn',
  bad: 'destructive'
}

const HINT_BY_STATE: Record<string, string> = {
  pending_restart: 'Restart the gateway from the status bar to apply this change.',
  gateway_stopped: 'Start the gateway from the status bar to connect.'
}

const stateLabel = (state?: null | string) => (state ? STATE_LABELS[state] || state.replace(/_/g, ' ') : 'Unknown')

function stateTone({ enabled, state }: MessagingPlatformInfo): StatusTone {
  if (!enabled) {
    return 'muted'
  }

  if (state === 'connected') {
    return 'good'
  }

  if (state === 'fatal' || state === 'startup_failed') {
    return 'bad'
  }

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
  DISCORD_ALLOW_ALL_USERS: {
    label: 'Allow all Discord users',
    help: 'Development only. When true, anyone can DM the bot without an allowlist.',
    advanced: true
  },
  DISCORD_HOME_CHANNEL: {
    label: 'Home channel ID',
    help: 'Channel where the bot sends proactive messages (cron output, reminders).',
    advanced: true
  },
  DISCORD_HOME_CHANNEL_NAME: {
    label: 'Home channel name',
    help: 'Display name for the home channel in logs and status output.',
    advanced: true
  },
  BLUEBUBBLES_ALLOW_ALL_USERS: {
    label: 'Allow all iMessage users',
    help: 'When true, skip the BlueBubbles allowlist.',
    advanced: true
  },
  MATTERMOST_ALLOW_ALL_USERS: {
    label: 'Allow all Mattermost users',
    advanced: true
  },
  MATTERMOST_HOME_CHANNEL: {
    label: 'Home channel',
    advanced: true
  },
  QQ_ALLOW_ALL_USERS: {
    label: 'Allow all QQ users',
    advanced: true
  },
  QQBOT_HOME_CHANNEL: {
    label: 'QQ home channel',
    help: 'Default channel or group for cron delivery.',
    advanced: true
  },
  QQBOT_HOME_CHANNEL_NAME: {
    label: 'QQ home channel name',
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

export function MessagingView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: MessagingViewProps) {
  const [platforms, setPlatforms] = useState<MessagingPlatformInfo[] | null>(null)
  const [edits, setEdits] = useState<EditMap>({})
  const [query, setQuery] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)
  const platformIds = useMemo(() => platforms?.map(p => p.id) ?? [], [platforms])
  const [selectedId, setSelectedId] = useRouteEnumParam('platform', platformIds, platformIds[0] ?? '')

  const refreshPlatforms = useCallback(async (silent = false) => {
    if (!silent) {
      setRefreshing(true)
    }

    try {
      const result = await getMessagingPlatforms()
      setPlatforms(result.platforms)
    } catch (err) {
      if (!silent) {
        notifyError(err, 'Messaging platforms failed to load')
      }
    } finally {
      if (!silent) {
        setRefreshing(false)
      }
    }
  }, [])

  useRefreshHotkey(() => void refreshPlatforms())

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

  const selected = useMemo(() => {
    if (!platforms) {
      return null
    }

    return platforms.find(platform => platform.id === selectedId) || platforms[0] || null
  }, [platforms, selectedId])

  const visiblePlatforms = useMemo(() => {
    if (!platforms) {
      return []
    }

    const q = query.trim().toLowerCase()

    if (!q) {
      return platforms
    }

    return platforms.filter(platform =>
      [platform.id, platform.name, platform.description, platform.state]
        .filter(Boolean)
        .some(value => String(value).toLowerCase().includes(q))
    )
  }, [platforms, query])

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
        title: enabled ? `${platform.name} enabled` : `${platform.name} disabled`,
        message: 'Restart the gateway for this change to take effect.'
      })
    } catch (err) {
      notifyError(err, `Failed to update ${platform.name}`)
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
        title: `${platform.name} setup saved`,
        message: 'Restart the gateway to reconnect with the new credentials.'
      })
    } catch (err) {
      notifyError(err, `Failed to save ${platform.name}`)
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
      notify({ kind: 'success', title: `${key} cleared`, message: `${platform.name} setup was updated.` })
    } catch (err) {
      notifyError(err, `Failed to clear ${key}`)
    } finally {
      setSaving(null)
    }
  }

  return (
    <PageSearchShell
      {...props}
      onSearchChange={setQuery}
      searchHidden={(platforms?.length ?? 0) === 0}
      searchPlaceholder="Search messaging..."
      searchValue={query}
    >
      {!platforms ? (
        <PageLoader label="Loading messaging platforms..." />
      ) : (
        <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[14rem_minmax(0,1fr)]">
          <aside className="min-h-0 overflow-y-auto p-2">
            <ul className="space-y-1">
              {visiblePlatforms.map(platform => (
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
              />
            )}
          </main>
        </div>
      )}
    </PageSearchShell>
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
        'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors',
        active
          ? 'bg-(--ui-row-active-background) text-foreground'
          : 'text-(--ui-text-secondary) hover:bg-(--ui-row-hover-background) hover:text-foreground'
      )}
      onClick={onSelect}
      type="button"
    >
      <PlatformAvatar platformId={platform.id} platformName={platform.name} />
      <span className="flex min-w-0 flex-1 items-center justify-between gap-2">
        <span className="truncate text-[length:var(--conversation-text-font-size)] font-normal">{platform.name}</span>
        <StatusDot tone={stateTone(platform)} />
      </span>
    </button>
  )
}

function PlatformDetail({
  edits,
  onClear,
  onEdit,
  onSave,
  onToggle,
  platform,
  saving
}: {
  edits: Record<string, string>
  onClear: (key: string) => void
  onEdit: (key: string, value: string) => void
  onSave: () => void
  onToggle: (enabled: boolean) => void
  platform: MessagingPlatformInfo
  saving: string | null
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
        <div className="mx-auto max-w-2xl space-y-5 px-5 py-4">
          <header className="flex items-start gap-3">
            <PlatformAvatar platformId={platform.id} platformName={platform.name} />
            <div className="min-w-0 flex-1">
              <h3 className="text-[0.9375rem] font-semibold tracking-tight">{platform.name}</h3>
              <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {platform.description}
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <StatePill tone={stateTone(platform)}>{stateLabel(platform.state)}</StatePill>
                <SetupPill active={platform.configured}>
                  {platform.configured ? 'Credentials set' : 'Needs setup'}
                </SetupPill>
                {!platform.gateway_running && <SetupPill active={false}>Messaging gateway stopped</SetupPill>}
              </div>
              <PlatformHint platform={platform} />
            </div>
          </header>

          {platform.error_message && (
            <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{platform.error_message}</span>
            </div>
          )}

          <section>
            <SectionTitle>Get your credentials</SectionTitle>
            <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
              {introCopy(platform)}
            </p>
            <div className="mt-3">
              <Button asChild size="sm" variant="textStrong">
                <a href={platform.docs_url} rel="noreferrer" target="_blank">
                  Open setup guide
                  <ExternalLink className="size-3.5" />
                </a>
              </Button>
            </div>
          </section>

          <section>
            <SectionTitle>Required</SectionTitle>
            <div className="mt-3 grid gap-1">
              {requiredFields.length > 0 ? (
                requiredFields.map(field => (
                  <MessagingField
                    edits={edits}
                    field={field}
                    key={field.key}
                    onClear={onClear}
                    onEdit={onEdit}
                    saving={saving}
                  />
                ))
              ) : (
                <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                  This platform does not need a token here. Use the setup guide above, then enable it below.
                </p>
              )}
            </div>
          </section>

          {optionalFields.length > 0 && (
            <section>
              <SectionTitle>Recommended</SectionTitle>
              <div className="mt-3 grid gap-1">
                {optionalFields.map(field => (
                  <MessagingField
                    edits={edits}
                    field={field}
                    key={field.key}
                    onClear={onClear}
                    onEdit={onEdit}
                    saving={saving}
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
                <span>Advanced ({hiddenCount})</span>
                <DisclosureCaret open={showAdvanced} size="0.875rem" />
              </button>
              {showAdvanced && (
                <div className="mt-3 grid gap-1">
                  {advancedFields.map(field => (
                    <MessagingField
                      edits={edits}
                      field={field}
                      key={field.key}
                      onClear={onClear}
                      onEdit={onEdit}
                      saving={saving}
                    />
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      </div>

      <footer className="bg-(--ui-chat-surface-background) px-5 py-2.5">
        <div className="mx-auto flex max-w-2xl flex-wrap items-center gap-2">
          <Switch
            aria-label={platform.enabled ? `Disable ${platform.name}` : `Enable ${platform.name}`}
            checked={platform.enabled}
            disabled={saving === `enabled:${platform.id}`}
            onCheckedChange={onToggle}
            size="xs"
          />

          <div className="ml-auto flex items-center gap-2">
            {hasEdits && <span className="text-xs text-muted-foreground">Unsaved changes</span>}
            <Button disabled={!hasEdits || isSavingEnv} onClick={onSave} size="sm">
              <Save />
              {isSavingEnv ? 'Saving...' : 'Save changes'}
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
  signal:
    'Run a signal-cli REST bridge somewhere reachable, then point Hermes at the URL and the registered phone number.',
  whatsapp:
    'Start the WhatsApp bridge that ships with Hermes, scan the QR code on first run, then enable the platform.',
  bluebubbles:
    'Run BlueBubbles Server on a Mac with iMessage, expose its API, then point Hermes at the URL with the server password.',
  homeassistant:
    'In Home Assistant, open your profile and create a long-lived access token. Paste it here along with your HA URL.',
  email:
    'Use a dedicated mailbox. For Gmail/Workspace, create an app password and use imap.gmail.com / smtp.gmail.com.',
  sms: 'Get your Twilio Account SID and Auth Token from the Twilio console, plus a phone number that can send SMS.',
  dingtalk: 'Create a DingTalk app in the developer console, then copy the Client ID (App key) and Client Secret here.',
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
  webhook:
    'Run an HTTP server that other tools (GitHub, GitLab, custom apps) can POST to. Use the secret to verify signatures.'
}

const introCopy = (platform: MessagingPlatformInfo) => PLATFORM_INTRO[platform.id] || platform.description

function MessagingField({
  edits,
  field,
  onClear,
  onEdit,
  saving
}: {
  edits: Record<string, string>
  field: MessagingEnvVarInfo
  onClear: (key: string) => void
  onEdit: (key: string, value: string) => void
  saving: string | null
}) {
  const copy = fieldCopy(field)
  const fieldId = `messaging-field-${field.key}`

  return (
    <ListRow
      action={
        <div className="flex items-center gap-2">
          <Input
            className={CREDENTIAL_CONTROL_CLASS}
            id={fieldId}
            onChange={event => onEdit(field.key, event.target.value)}
            placeholder={field.is_set ? field.redacted_value || 'Replace current value' : copy.placeholder}
            type={field.is_password ? 'password' : 'text'}
            value={edits[field.key] || ''}
          />
          {field.url && (
            <Button asChild className="size-8 shrink-0" title="Open docs" variant="ghost">
              <a href={field.url} rel="noreferrer" target="_blank">
                <ExternalLink className="size-3.5" />
              </a>
            </Button>
          )}
          {field.is_set && (
            <Button
              className="size-8 shrink-0"
              disabled={saving === `clear:${field.key}`}
              onClick={() => onClear(field.key)}
              title={`Clear ${field.key}`}
              variant="ghost"
            >
              <Trash2 className="size-3.5" />
            </Button>
          )}
        </div>
      }
      description={copy.help}
      title={
        <span className="flex flex-wrap items-center gap-2">
          <label htmlFor={fieldId}>{copy.label}</label>
          {field.is_set && <span className="text-[0.66rem] font-medium text-primary">Saved</span>}
        </span>
      }
    />
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{children}</h4>
}

function PlatformHint({ platform }: { platform: MessagingPlatformInfo }) {
  if (!platform.enabled || platform.state === 'connected') {
    return null
  }

  const hint = HINT_BY_STATE[platform.state || ''] || (platform.gateway_running ? null : HINT_BY_STATE.gateway_stopped)

  return hint ? <p className="mt-2 text-xs leading-5 text-muted-foreground">{hint}</p> : null
}

function StatePill({ children, tone }: { children: string; tone: StatusTone }) {
  return (
    <Badge variant={TONE_VARIANT[tone]}>
      <StatusDot tone={tone} />
      {children}
    </Badge>
  )
}

function SetupPill({ active, children }: { active: boolean; children: string }) {
  return <Badge variant={active ? 'default' : 'muted'}>{children}</Badge>
}
