import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { type ProfileScope, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import { hermesConfigKey, useHermesConfigRecord } from '../hooks/use-config-record'

import { getNested, setNested } from './helpers'
import { ToggleRow } from './primitives'

export function CodingWorkspaceSetting({ profile }: { profile: ProfileScope }) {
  const { t } = useI18n()
  const copy = t.codingWorkspace
  const { data: config } = useHermesConfigRecord(profile)
  const client = useQueryClient()
  const [busy, setBusy] = useState(false)
  const enabled = getNested(config ?? {}, 'desktop.coding.show_controls') === true

  const toggle = async (on: boolean) => {
    if (!config || busy) {return}
    const key = hermesConfigKey(profile)
    setBusy(true)
    await client.cancelQueries({ queryKey: key, exact: true })
    client.setQueryData(key, setNested(config, 'desktop.coding.show_controls', on))

    try {
      const result = await saveHermesConfigRecord({ desktop: { coding: { show_controls: on } } }, profile)

      if (!result.ok) {throw new Error(copy.saveFailed)}
      await client.invalidateQueries({ queryKey: key, exact: true })
    } catch (error) {
      client.setQueryData(key, config)
      notifyError(error, copy.saveFailed)
    } finally {
      setBusy(false)
    }
  }

  return <ToggleRow checked={enabled} description={copy.showControlsDescription} disabled={busy || !config}
    label={copy.showControls} onChange={on => void toggle(on)} />
}
