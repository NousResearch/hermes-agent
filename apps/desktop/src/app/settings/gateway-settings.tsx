import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

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

    desktop
      .getConnectionConfig()
      .then(config => {
        if (cancelled) {
          return
        }

        setState(config)
      })
      .catch(err => notifyError(err, t('settings:gateway.loadFailed')))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
  }, [t])

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
        title: t('settings:gateway.incomplete.title'),
        message: t('settings:gateway.incomplete.connect')
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
        title: apply ? t('settings:gateway.restarting.title') : t('settings:gateway.saved.title'),
        message: apply ? t('settings:gateway.restarting.message') : t('settings:gateway.saved.message')
      })
    } catch (err) {
      notifyError(err, apply ? t('settings:gateway.applyFailed') : t('settings:gateway.saveFailed'))
    } finally {
      setSaving(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: t('settings:gateway.incomplete.title'),
        message: t('settings:gateway.incomplete.test')
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

      const message = result.version
        ? t('settings:gateway.testSuccess.version', { url: result.baseUrl, version: result.version })
        : t('settings:gateway.testSuccess.connected', { url: result.baseUrl })

      setLastTest(message)
      notify({ kind: 'success', title: t('settings:gateway.testSuccess.title'), message })
    } catch (err) {
      notifyError(err, t('settings:gateway.testFailed'))
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label={t('settings:gateway.loading')} />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return (
      <EmptyState
        description={t('settings:gateway.unavailable.description')}
        title={t('settings:gateway.unavailable.title')}
      />
    )
  }

  return (
    <SettingsContent>
      <div className="mb-5">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Globe className="size-4 text-muted-foreground" />
          {t('settings:gateway.title')}
          {state.envOverride ? <Pill tone="primary">{t('settings:gateway.envOverride')}</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {t('settings:gateway.description')}
        </p>
      </div>

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">{t('settings:gateway.envOverrideWarning.title')}</div>
            <div className="mt-1 leading-5">{t('settings:gateway.envOverrideWarning.description')}</div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ModeCard
          active={state.mode === 'local'}
          description={t('settings:gateway.localMode.description')}
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title={t('settings:gateway.localMode.title')}
        />
        <ModeCard
          active={state.mode === 'remote'}
          description={t('settings:gateway.remoteMode.description')}
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title={t('settings:gateway.remoteMode.title')}
        />
      </div>

      <div className="mt-5 divide-y divide-border/40">
        <ListRow
          action={
            <Input
              className={cn('h-8', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
              placeholder={t('settings:gateway.remoteUrl.placeholder')}
              value={state.remoteUrl}
            />
          }
          description={t('settings:gateway.remoteUrl.description')}
          title={t('settings:gateway.remoteUrl')}
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
                  ? t('settings:gateway.sessionToken.existing', {
                      preview: state.remoteTokenPreview ?? t('common:saved')
                    })
                  : t('settings:gateway.sessionToken.placeholder')
              }
              type="password"
              value={remoteToken}
            />
          }
          description={t('settings:gateway.sessionToken.description')}
          title={t('settings:gateway.sessionToken')}
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
          {t('settings:gateway.testRemote')}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(false)} variant="outline">
          {t('settings:gateway.saveForRestart')}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(true)}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          {t('settings:gateway.saveAndReconnect')}
        </Button>
      </div>

      <div className="mt-6 divide-y divide-border/40">
        <ListRow
          action={
            <Button onClick={() => void window.hermesDesktop?.revealLogs()} variant="outline">
              <FileText className="size-4" />
              {t('settings:gateway.openLogs')}
            </Button>
          }
          description={t('settings:gateway.logsDescription')}
          title={t('settings:gateway.diagnostics')}
        />
      </div>
    </SettingsContent>
  )
}
