import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import { ToggleRow } from './primitives'

export function MinimizeToTraySetting() {
  const { t } = useI18n()
  const c = t.settings.config
  const bridge = window.hermesDesktop?.minimizeToTray
  const [status, setStatus] = useState<{ enabled: boolean; available: boolean } | null>(null)
  const [saving, setSaving] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const revision = useRef(0)

  const load = useCallback(async () => {
    if (!bridge) {
      return
    }

    const version = ++revision.current
    setLoadFailed(false)

    try {
      const next = await bridge.get()

      if (revision.current === version) {
        setStatus(next)
      }
    } catch (error) {
      if (revision.current === version) {
        setLoadFailed(true)
        notifyError(error, c.failedLoad)
      }
    }
  }, [bridge, c.failedLoad])

  useEffect(() => {
    if (!bridge) {
      return
    }

    const unsubscribe = bridge.onChanged(next => {
      revision.current++
      setStatus(next)
      setLoadFailed(false)
    })

    void load()

    return () => {
      revision.current++
      unsubscribe()
    }
  }, [bridge, load])

  if (!bridge) {
    return null
  }

  const save = async (enabled: boolean) => {
    const previous = status
    const version = ++revision.current
    setSaving(true)
    setStatus({ enabled, available: status?.available ?? false })

    try {
      const next = await bridge.set(enabled)

      if (revision.current === version) {
        setStatus(next)
      }
    } catch (error) {
      if (revision.current === version) {
        setStatus(previous)
      }

      notifyError(error, c.autosaveFailed)
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <ToggleRow
        checked={status?.enabled ?? false}
        description={
          status?.enabled && !status.available && !saving ? c.minimizeToTrayUnavailable : c.minimizeToTrayDesc
        }
        disabled={!status || saving}
        label={c.minimizeToTrayTitle}
        onChange={enabled => void save(enabled)}
      />
      {loadFailed && (
        <Button onClick={() => void load()} size="sm" variant="secondary">
          {t.settings.screenshot.retry}
        </Button>
      )}
    </>
  )
}
