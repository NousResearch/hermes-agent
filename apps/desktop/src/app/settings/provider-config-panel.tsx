import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  getMemoryProviderConfig,
  getMemoryProviderOAuthStatus,
  saveMemoryProviderConfig,
  startMemoryProviderOAuth
} from '@/hermes'
import { Check, Loader2, LogIn, Save } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { LoadingState, Pill } from './primitives'

/** Seed editable values from the schema: non-secret fields keep their current
 *  value, secret fields start blank (their value is never returned). */
function seedValues(config: MemoryProviderConfig): Record<string, string> {
  return Object.fromEntries(
    config.fields.map(field => [field.key, field.kind === 'secret' ? '' : field.value])
  )
}

function FieldControl({
  field,
  value,
  onChange
}: {
  field: MemoryProviderField
  value: string
  onChange: (value: string) => void
}) {
  if (field.kind === 'select') {
    const selected = field.options.find(option => option.value === value)

    return (
      <>
        <Select onValueChange={onChange} value={value}>
          <SelectTrigger className={CONTROL_TEXT}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {field.options.map(option => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {(selected?.description || field.description) && (
          <span className="text-xs text-muted-foreground">{selected?.description || field.description}</span>
        )}
      </>
    )
  }

  if (field.kind === 'secret') {
    return (
      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="min-w-64 flex-1 font-mono"
          onChange={event => onChange(event.target.value)}
          placeholder={field.is_set ? 'Leave blank to keep current value' : field.placeholder}
          type="password"
          value={value}
        />
        {field.is_set && (
          <Pill tone="primary">
            <Check className="size-3" />
            Set
          </Pill>
        )}
      </div>
    )
  }

  return (
    <Input
      className="font-mono"
      onChange={event => onChange(event.target.value)}
      placeholder={field.placeholder}
      value={value}
    />
  )
}

export function ProviderConfigPanel({ provider }: { provider: string }) {
  const [config, setConfig] = useState<MemoryProviderConfig | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)
  const [signingIn, setSigningIn] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const next = await getMemoryProviderConfig(provider)
      setConfig(next)
      setValues(seedValues(next))
    } catch (err) {
      notifyError(err, 'Memory provider settings failed to load')
      setConfig(null)
    }
  }, [provider])

  useEffect(() => {
    setConfig(null)
    void refresh()
  }, [refresh])

  const save = useCallback(async () => {
    if (!config) {
      return
    }

    setSaving(true)

    try {
      await saveMemoryProviderConfig(provider, values)
      notify({ kind: 'success', title: `${config.label} saved`, message: 'Memory provider configuration updated.' })
      await refresh()
    } catch (err) {
      notifyError(err, `Failed to save ${config.label} settings`)
    } finally {
      setSaving(false)
    }
  }, [config, provider, refresh, values])

  const signIn = useCallback(async () => {
    if (!config) {
      return
    }

    setSigningIn(true)

    try {
      // Kick off the browser flow; the backend runs it in the background.
      await startMemoryProviderOAuth(provider)
      notify({
        kind: 'info',
        title: `Signing in to ${config.label}`,
        message: 'Approve the request in your browser to finish connecting.'
      })

      // Poll until the background sign-in completes (it's interactive, so this
      // can take a while) or fails.
      const deadline = Date.now() + 180_000
      // eslint-disable-next-line no-constant-condition
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        const status = await getMemoryProviderOAuthStatus(provider)
        if (status.flow === 'error') {
          throw new Error(status.error || 'Sign-in failed')
        }
        // Gate on THIS sign-in completing (flow === 'done'), not on
        // `authenticated` — a pre-existing token would otherwise report success
        // before the user has approved the new request.
        if (status.flow === 'done') {
          const orgLabel = status.org_name || (status.org_id ? `org ${status.org_id}` : '')
          const orgNote = orgLabel ? ` (${orgLabel})` : ''
          notify({ kind: 'success', title: `Connected to ${config.label}`, message: `Signed in via browser${orgNote}.` })
          await refresh()
          return
        }
        if (Date.now() > deadline) {
          throw new Error('Timed out waiting for the browser sign-in to complete.')
        }
      }
    } catch (err) {
      notifyError(err, `Failed to sign in to ${config.label}`)
    } finally {
      setSigningIn(false)
    }
  }, [config, provider, refresh])

  // Providers without a declared config surface (e.g. builtin) render nothing.
  if (config && config.fields.length === 0) {
    return null
  }

  if (!config) {
    return <LoadingState label="Loading memory provider settings..." />
  }

  const secretFields = config.fields.filter(field => field.kind === 'secret')

  return (
    <section className="py-3">
      <button
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 rounded-lg bg-background/60 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setExpanded(open => !open)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <DisclosureCaret open={expanded} />
          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {config.label} settings
          </span>
          {secretFields.map(field => (
            <Pill key={field.key}>{field.is_set ? `${field.label} set` : `${field.label} not set`}</Pill>
          ))}
        </span>
      </button>

      {expanded && (
        <div className="mt-3 grid gap-4 rounded-xl bg-background/60 p-4">
          {config.oauth?.supported && values.mode === 'cloud' && (
            <div className="grid gap-1.5">
              <span className="text-xs font-medium text-muted-foreground">Sign in</span>
              <div className="flex flex-wrap items-center gap-2">
                <Button disabled={signingIn} onClick={() => void signIn()} size="sm">
                  {signingIn ? <Loader2 className="size-3.5 animate-spin" /> : <LogIn className="size-3.5" />}
                  {config.oauth.authenticated ? 'Re-authenticate' : 'Sign in with browser'}
                </Button>
                {config.oauth.authenticated && (
                  <Pill tone="primary">
                    <Check className="size-3" />
                    {config.oauth.org_name
                      ? `Connected — ${config.oauth.org_name}`
                      : config.oauth.org_id
                        ? `Connected — org ${config.oauth.org_id}`
                        : 'Connected'}
                  </Pill>
                )}
              </div>
              <span className="text-xs text-muted-foreground">
                Opens your browser to authorize Hermes — no API key needed. Or paste a key below.
              </span>
            </div>
          )}

          {config.fields.map(field => (
            <label className="grid gap-1.5" key={field.key}>
              <span className="text-xs font-medium text-muted-foreground">{field.label}</span>
              <FieldControl
                field={field}
                onChange={value => setValues(current => ({ ...current, [field.key]: value }))}
                value={values[field.key] ?? ''}
              />
              {field.kind !== 'select' && field.description && (
                <span className="text-xs text-muted-foreground">{field.description}</span>
              )}
            </label>
          ))}

          <div className="flex justify-end">
            <Button disabled={saving} onClick={() => void save()} size="sm">
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
              Save
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
