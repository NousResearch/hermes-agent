import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import { ToggleRow } from './primitives'

/**
 * Device-local toggle (#135441): downloads skip the OS save dialog and land
 * straight in Downloads. The value lives in main (JSON beside the app's
 * userData), so this row only reads/writes the bridge and mirrors pushes.
 */
export function DownloadsSaveDirectSetting({ id }: { id: string }) {
  const { t } = useI18n()
  const c = t.settings.config
  const bridge = window.hermesDesktop?.downloadSaveDirect
  const [enabled, setEnabled] = useState<boolean | null>(null)
  const [saving, setSaving] = useState(false)
  const revision = useRef(0)

  const load = useCallback(async () => {
    if (!bridge) {
      return
    }

    const version = ++revision.current

    try {
      const next = await bridge.get()

      if (revision.current === version) {
        setEnabled(next)
      }
    } catch (error) {
      // Leave the row disabled rather than guessing a device-local value.
      if (revision.current === version) {
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
      setEnabled(next)
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

  const save = async (on: boolean) => {
    const previous = enabled
    const version = ++revision.current
    setSaving(true)
    setEnabled(on)

    try {
      const next = await bridge.set(on)

      if (revision.current === version) {
        setEnabled(next)
      }
    } catch (error) {
      if (revision.current === version) {
        setEnabled(previous)
      }

      notifyError(error, c.autosaveFailed)
    } finally {
      setSaving(false)
    }
  }

  return (
    <ToggleRow
      checked={enabled ?? false}
      description={c.downloadsSaveDirectDesc}
      disabled={enabled === null || saving}
      id={id}
      label={c.downloadsSaveDirectTitle}
      onChange={on => void save(on)}
    />
  )
}
