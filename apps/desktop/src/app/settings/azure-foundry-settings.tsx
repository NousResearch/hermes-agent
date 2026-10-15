import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  deleteEnvVar,
  discoverFoundryDeployments,
  type FoundrySettings,
  type FoundryUpdate,
  getFoundrySettings,
  type ProfileScope,
  saveFoundrySettings,
  validateFoundrySettings
} from '@/hermes'
import { useI18n } from '@/i18n'
import { isMissingRestEndpoint } from '@/lib/gateway-rpc'
import { ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { queryClient } from '@/lib/query-client'
import { confirm } from '@/store/confirm'
import { $activeConnectionId } from '@/store/connections'
import { notify, notifyError } from '@/store/notifications'
import { $settingsRequestProfile } from '@/store/settings-scope'

import { FoundryFields } from './azure-foundry-fields'

interface FoundryProps {
  expanded: boolean
  fallback: ReactNode
  onToggle: () => void
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
}

export function AzureFoundrySettings(props: FoundryProps) {
  const profile = useStore($settingsRequestProfile)
  const connectionId = useStore($activeConnectionId)

  const scope = useMemo(() => ({ connectionId: connectionId ?? undefined, profile }), [connectionId, profile])

  return <FoundryCard {...props} key={`${connectionId}:${profile}`} scope={scope} />
}

function formFromSettings({ base_url, model, api_mode, auth }: FoundrySettings): FoundryUpdate {
  return { base_url, model, api_mode, auth }
}

function FoundryCard({
  expanded,
  fallback,
  onToggle,
  onConfigSaved,
  onMainModelChanged,
  scope
}: FoundryProps & { scope: ProfileScope }) {
  const { t } = useI18n()
  const copy = t.settings.foundry
  const ce = t.settings.customEndpoints
  const [settings, setSettings] = useState<FoundrySettings | null>(null)
  const [form, setForm] = useState<FoundryUpdate | null>(null)
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error' | 'unsupported'>('loading')
  const [busy, setBusy] = useState<'test' | 'save' | 'remove' | 'discover' | null>(null)
  const [discovery, setDiscovery] = useState<{ connection: string; deployments: string[] } | null>(null)
  const [attempt, setAttempt] = useState(0)
  const mounted = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- each keyed instance owns one connection/profile; late responses are discarded
  useEffect(() => {
    mounted.current = true
    let cancelled = false
    setLoadState('loading')
    void getFoundrySettings(scope)
      .then(data => {
        if (!cancelled) {
          setSettings(data)
          setForm(formFromSettings(data))
          setLoadState('ready')
        }
      })
      .catch(error => {
        if (!cancelled) {
          setLoadState(isMissingRestEndpoint(error) ? 'unsupported' : 'error')
        }
      })

    return () => {
      cancelled = true
      mounted.current = false
    }
  }, [scope, attempt])

  if (loadState === 'unsupported') {
    return fallback
  }

  async function submit(action: 'test' | 'save') {
    if (!form) {
      return
    }

    setBusy(action)

    try {
      // Blank input keeps the stored key; GET never returns a secret to the renderer.
      const payload: FoundryUpdate =
        form.auth.kind === 'api_key'
          ? { ...form, auth: { kind: 'api_key', api_key: form.auth.api_key?.trim() || undefined } }
          : form

      if (action === 'save') {
        const saved = await saveFoundrySettings(payload, scope)

        if (!mounted.current) {
          return
        }

        setSettings(saved)
        setForm(formFromSettings(saved))
        void queryClient.invalidateQueries({ queryKey: ['model-options'] })
        onMainModelChanged?.('azure-foundry', saved.model)
        onConfigSaved?.()
        notify({ kind: 'success', message: copy.saved })
      } else {
        const result = await validateFoundrySettings(payload, scope)

        if (!mounted.current) {
          return
        }

        notify({
          kind: result.ok ? 'success' : 'warning',
          message: result.message || (result.ok ? ce.endpointReachable : ce.endpointValidationFailed)
        })
      }
    } catch (error) {
      if (mounted.current) {
        notifyError(error, action === 'save' ? ce.saveFailed : ce.validationFailed)
      }
    } finally {
      if (mounted.current) {
        setBusy(null)
      }
    }
  }

  const discoveryKey = form && JSON.stringify({ base_url: form.base_url, api_mode: form.api_mode, auth: form.auth })
  const deployments = discovery?.connection === discoveryKey ? discovery.deployments : []

  async function discover() {
    if (!form || !discoveryKey) {
      return
    }
    setBusy('discover')
    try {
      const result = await discoverFoundryDeployments(form, scope)
      if (!mounted.current) {
        return
      }
      setDiscovery({ connection: discoveryKey, deployments: result.deployments })
      if (!result.deployments.length) {
        notify({ kind: 'warning', message: copy.deploymentsUnavailable })
      }
    } catch (error) {
      if (mounted.current) {
        notifyError(error, copy.deploymentsUnavailable)
      }
    } finally {
      if (mounted.current) {
        setBusy(null)
      }
    }
  }

  async function removeKey() {
    const key = 'AZURE_FOUNDRY_API_KEY'
    if (!(await confirm({ destructive: true, title: t.settings.toolsets.removeConfirm(key) })) || !mounted.current) {
      return
    }
    setBusy('remove')
    try {
      await deleteEnvVar(key, scope)
      if (!mounted.current) {
        return
      }
      setDiscovery(null)
      setSettings(current => current && { ...current, api_key_set: false })
      setForm(current => (current?.auth.kind === 'api_key' ? { ...current, auth: { kind: 'api_key' } } : current))
      void queryClient.invalidateQueries({ queryKey: ['model-options'] })
      onConfigSaved?.()
      notify({
        kind: 'success',
        title: t.settings.toolsets.removedTitle,
        message: t.settings.toolsets.removedMessage(key)
      })
    } catch (error) {
      if (mounted.current) {
        notifyError(error, t.settings.toolsets.failedRemove(key))
      }
    } finally {
      if (mounted.current) {
        setBusy(null)
      }
    }
  }

  return (
    <section
      className={cn(
        'grid gap-3 rounded-[6px] p-3',
        expanded && 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)'
      )}
    >
      <Button aria-expanded={expanded} className="justify-start" onClick={onToggle} size="inline" variant="text">
        <span
          className={cn(
            'size-2 shrink-0 rounded-full',
            settings?.is_current || settings?.api_key_set ? 'bg-primary' : 'bg-(--ui-stroke-secondary)'
          )}
        />
        Microsoft Foundry
        <ChevronDown className={cn('size-3.5', expanded && 'rotate-180')} />
      </Button>
      {expanded && loadState === 'loading' && <p className="text-xs text-muted-foreground">{t.common.loading}</p>}
      {expanded && loadState === 'error' && (
        <div className="grid gap-2 text-xs">
          <p>{copy.loadFailed}</p>
          <Button onClick={() => setAttempt(n => n + 1)} size="sm" variant="outline">
            {t.common.retry}
          </Button>
        </div>
      )}
      {expanded && loadState === 'ready' && form && (
        <FoundryFields
          deployments={deployments}
          discover={() => void discover()}
          removeKey={() => void removeKey()}
          busy={busy}
          form={form}
          setForm={setForm}
          settings={settings}
          submit={action => void submit(action)}
        />
      )}
    </section>
  )
}
