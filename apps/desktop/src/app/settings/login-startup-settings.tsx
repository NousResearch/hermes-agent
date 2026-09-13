import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'

import { ToggleRow } from './primitives'

type StartupStatus = { supported: boolean; openAtLogin: boolean }

/** Windows, not a profile config file, owns the authoritative login preference. */
export function LoginStartupSettings() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const api = window.hermesDesktop?.loginStartup
  const [status, setStatus] = useState<StartupStatus | null>(null)
  const [saving, setSaving] = useState(false)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let mounted = true

    void api?.getSettings().then(
      next => {
        if (mounted) {
          setStatus(next)
        }
      },
      () => {
        if (mounted) {
          setFailed(true)
        }
      }
    )

    return () => {
      mounted = false
    }
  }, [api])

  if (!api || status?.supported === false) {
    return null
  }

  const save = async (enabled: boolean) => {
    setSaving(true)
    setFailed(false)

    try {
      const next = await api.setSettings(enabled)
      setStatus(next)
      setFailed(!next.supported || next.openAtLogin !== enabled)
    } catch {
      // Keep the last confirmed value; a rejected OS write is never shown as saved.
      setFailed(true)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <ToggleRow
        checked={status?.openAtLogin === true}
        description={a.loginStartupDesc}
        disabled={status === null || saving}
        label={a.loginStartupTitle}
        onChange={enabled => void save(enabled)}
      />
      {failed && (
        <p className="text-sm text-destructive" role="alert">
          {a.loginStartupFailed}
        </p>
      )}
    </div>
  )
}
