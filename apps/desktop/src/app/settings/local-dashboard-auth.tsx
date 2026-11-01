import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertCircle, Check, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { activeGatewayProfileKey, requestGatewayForProfile } from '@/store/gateway'

import { CONTROL_TEXT } from './constants'
import { ListRow } from './primitives'

/**
 * Local dashboard sign-in, for Settings → Gateways in local mode only.
 *
 * The backend already lets Desktop *sign in* to a backend that has dashboard
 * auth; it had no way to *set* those credentials from the app. This panel
 * closes that gap for the local gateway only: it sends the username and the
 * new password to the gateway's `config.set` `dashboard.auth` setter, which
 * hashes the password server-side and writes only the hash to the profile's
 * config.yaml. The renderer never computes a hash, never sees a password or
 * hash back (the read returns only a configured username, or ""), and never
 * creates credentials on a remote backend — the remote panel keeps its
 * existing sign-in-only flow.
 *
 * Auth takes effect on gateway restart: the basic-auth provider reads its
 * settings at registration, not per request.
 */
export function LocalDashboardAuth() {
  const { t } = useI18n()
  const g = t.settings.gateway

  const [configuredUsername, setConfiguredUsername] = useState<string | null>(null)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const result = await requestGatewayForProfile<{ value?: string }>(activeGatewayProfileKey(), 'config.get', {
        key: 'dashboard.auth'
      })
      const current = typeof result?.value === 'string' ? result.value : ''
      setConfiguredUsername(current)
      setUsername(current)
    } catch {
      setError(g.localDashboardAuthLoadFailed)
      setConfiguredUsername(null)
    } finally {
      setLoading(false)
    }
  }, [g.localDashboardAuthLoadFailed])

  useEffect(() => {
    void load()
  }, [load])

  const resetForm = useCallback(() => {
    setPassword('')
    setConfirm('')
    setShowPassword(false)
    setError('')
    setSaved(false)
    setUsername(configuredUsername ?? '')
  }, [configuredUsername])

  const save = useCallback(async () => {
    const nextUsername = username.trim()

    if (!nextUsername) {
      setError(g.localDashboardAuthUsernameRequired)
      return
    }

    if (password.length < 8) {
      setError(g.localDashboardAuthPasswordTooShort)
      return
    }

    if (password !== confirm) {
      setError(g.localDashboardAuthPasswordMismatch)
      return
    }

    setSaving(true)
    setError('')
    setSaved(false)

    try {
      // No password_hash here by design: the server hashes. A client-side
      // hash would still be a password-equivalent the server would trust.
      await requestGatewayForProfile(activeGatewayProfileKey(), 'config.set', {
        key: 'dashboard.auth',
        value: '',
        username: nextUsername,
        password
      })
      setConfiguredUsername(nextUsername)
      setPassword('')
      setConfirm('')
      setSaved(true)
    } catch {
      setError(g.localDashboardAuthSaveFailed)
    } finally {
      setSaving(false)
    }
  }, [
    confirm,
    g.localDashboardAuthPasswordMismatch,
    g.localDashboardAuthPasswordTooShort,
    g.localDashboardAuthSaveFailed,
    g.localDashboardAuthUsernameRequired,
    password,
    username
  ])

  return (
    <div className="mt-5 grid gap-2">
      <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
        {g.localDashboardAuthTitle}
      </div>
      <p className="text-xs text-muted-foreground">{g.localDashboardAuthDesc}</p>

      {loading ? (
        <div className="flex items-center gap-2 py-2 text-xs text-(--ui-text-tertiary)">
          <Loader2 className="size-4 animate-spin" />
        </div>
      ) : (
        <>
          <p className="text-xs text-muted-foreground">
            {configuredUsername
              ? g.localDashboardAuthConfigured(configuredUsername)
              : g.localDashboardAuthNotConfigured}
          </p>

          <ListRow
            action={
              <Input
                aria-label={g.localDashboardAuthUsername}
                className={cn('h-8', CONTROL_TEXT)}
                onChange={event => setUsername(event.target.value)}
                value={username}
              />
            }
            title={g.localDashboardAuthUsername}
          />
          <ListRow
            action={
              <div className="flex items-center gap-2">
                <Input
                  aria-label={g.localDashboardAuthPassword}
                  className={cn('h-8', CONTROL_TEXT)}
                  onChange={event => setPassword(event.target.value)}
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                />
                <Button onClick={() => setShowPassword(current => !current)} size="sm" variant="outline">
                  {showPassword ? g.localDashboardAuthHidePassword : g.localDashboardAuthShowPassword}
                </Button>
              </div>
            }
            title={g.localDashboardAuthPassword}
          />
          <ListRow
            action={
              <Input
                aria-label={g.localDashboardAuthConfirm}
                className={cn('h-8', CONTROL_TEXT)}
                onChange={event => setConfirm(event.target.value)}
                type={showPassword ? 'text' : 'password'}
                value={confirm}
              />
            }
            title={g.localDashboardAuthConfirm}
          />

          <p className="text-xs text-muted-foreground">{g.localDashboardAuthTrustedNote}</p>

          {error ? (
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              <span>{error}</span>
            </div>
          ) : null}

          {saved ? (
            <div className="flex items-start gap-2 text-sm text-primary">
              <Check className="mt-0.5 size-4 shrink-0" />
              <span>
                {g.localDashboardAuthSaved} {g.localDashboardAuthRestartNote}
              </span>
            </div>
          ) : null}

          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button disabled={saving} onClick={resetForm} size="sm" variant="outline">
              {g.localDashboardAuthResetForm}
            </Button>
            <Button disabled={saving} onClick={() => void save()} size="sm">
              {saving ? <Loader2 className="animate-spin" /> : null}
              {g.localDashboardAuthSave}
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
