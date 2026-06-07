import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AlertCircle, Check, FileText, Globe, Loader2, Monitor } from '@/lib/icons'
import { useTranslation } from '@/hooks/use-translation'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $profiles, refreshActiveProfile } from '@/store/profile'

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

function ScopeChip({ active, label, onSelect }: { active: boolean; label: string; onSelect: () => void }) {
  return (
    <button
      className={cn(
        'rounded-full border px-3 py-1 text-[length:var(--conversation-caption-font-size)] transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) text-(--ui-text-primary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover)'
      )}
      onClick={onSelect}
      type="button"
    >
      {label}
    </button>
  )
}

export function GatewaySettings() {
  const { t } = useTranslation()
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

    setLoading(true)
    // Clear scope-local entry state so a token from one scope can't leak into
    // the next when switching profiles.
    setRemoteToken('')
    setLastTest(null)

    desktop
      .getConnectionConfig(scope)
      .then(config => {
        if (cancelled) {
          return
        }

        setState(config)
      })
      .catch(err => notifyError(err, g.failedLoad))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
  }, [scope])

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
        title: apply ? g.restartingTitle : g.savedTitle,
        message: apply ? g.restartingMessage : g.savedMessage
      })
    } catch (err) {
      notifyError(err, apply ? g.applyFailed : g.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: 'Remote gateway incomplete',
        message: 'Enter a remote URL and session token before testing.'
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

      const message = g.connectedTo(result.baseUrl, result.version ?? undefined)
      setLastTest(message)
      notify({ kind: 'success', title: g.reachableTitle, message })
    } catch (err) {
      notifyError(err, g.testFailed)
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label={t('gateway.loading')} />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return (
      <EmptyState
        description={t('gateway.unavailableDesc')}
        title={t('gateway.unavailable')}
      />
    )
  }

  return (
    <SettingsContent>
      <div className="mb-5">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Globe className="size-4 text-muted-foreground" />
          {t('gateway.connection')}
          {state.envOverride ? <Pill tone="primary">{t('gateway.envOverride')}</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {t('gateway.desc')}
        </p>
      </div>

      {namedProfiles.length > 0 ? (
        <div className="mb-5 grid gap-2">
          <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
            {g.appliesTo}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <ScopeChip active={scope === null} label={g.allProfiles} onSelect={() => setScope(null)} />
            {namedProfiles.map(profile => (
              <ScopeChip
                active={scope === profile.name}
                key={profile.name}
                label={profile.name}
                onSelect={() => setScope(profile.name)}
              />
            ))}
          </div>
          <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {scope === null ? g.defaultConnection : g.profileConnection(scope)}
          </p>
        </div>
      ) : null}

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">{t('gateway.envOverrideTitle')}</div>
            <div className="mt-1 leading-5">
              {t('gateway.envOverrideDesc', { urlVar: 'HERMES_DESKTOP_REMOTE_URL', tokenVar: 'HERMES_DESKTOP_REMOTE_TOKEN' })}
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ModeCard
          active={state.mode === 'local'}
          description={t('gateway.localDesc')}
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title={t('gateway.localTitle')}
        />
        <ModeCard
          active={state.mode === 'remote'}
          description={t('gateway.remoteDesc')}
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title={t('gateway.remoteTitle')}
        />
      </div>

      <div className="mt-5 divide-y divide-border/40">
        <ListRow
          action={
            <Input
              className={cn('h-8', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
              placeholder={t('gateway.remoteUrlPlaceholder')}
              value={state.remoteUrl}
            />
          }
          description={t('gateway.remoteUrlDesc')}
          title={t('gateway.remoteUrl')}
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
                  ? t('gateway.existingToken', { preview: state.remoteTokenPreview ?? t('gateway.tokenSaved') })
                  : t('gateway.sessionTokenPlaceholder')
              }
              type="password"
              value={remoteToken}
            />
          }
          description={t('gateway.sessionTokenDesc')}
          title={t('gateway.sessionToken')}
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
          {t('gateway.testRemote')}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(false)} variant="outline">
          {t('gateway.saveRestart')}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(true)}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          {t('gateway.saveReconnect')}
        </Button>
      </div>

      <div className="mt-6 divide-y divide-border/40">
        <ListRow
          action={
            <Button onClick={() => void window.hermesDesktop?.revealLogs()} variant="outline">
              <FileText className="size-4" />
              {t('gateway.openLogs')}
            </Button>
          }
          description={t('gateway.diagnosticsDesc')}
          title={t('gateway.diagnostics')}
        />
      </div>
    </SettingsContent>
  )
}
