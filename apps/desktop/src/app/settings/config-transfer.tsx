import type { RefObject } from 'react'

import { Button } from '@/components/ui/button'
import { getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { confirm } from '@/store/confirm'
import { notifyError } from '@/store/notifications'
import { usePortalShell } from '@/store/shell-variant'

import { ListRow } from './primitives'
import { SettingsCard, SettingsGroup } from './settings-card'
import { useSettingsPagesCopy } from './settings-pages-copy'

/** Export, import and reset of the profile's config: one set of actions, drawn as classic's rail
 *  icons or as the redesign's labelled rows on Advanced. */
export function useConfigTransfer(importInputRef: RefObject<HTMLInputElement | null>, onConfigSaved?: () => void) {
  const { t } = useI18n()

  const exportConfig = async () => {
    try {
      const cfg = await getHermesConfigRecord()
      const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'hermes-config.json'
      a.click()
      URL.revokeObjectURL(url)
      triggerHaptic('success')
    } catch (err) {
      notifyError(err, t.settings.exportFailed)
    }
  }

  const importConfig = () => {
    triggerHaptic('open')
    importInputRef.current?.click()
  }

  const resetConfig = async () => {
    triggerHaptic('warning')

    const ok = await confirm({
      confirmLabel: t.settings.resetToDefaults,
      destructive: true,
      title: t.settings.resetConfirm
    })

    if (!ok) {
      return
    }

    try {
      await saveHermesConfig(await getHermesConfigDefaults())
      triggerHaptic('success')
      onConfigSaved?.()
    } catch (err) {
      notifyError(err, t.settings.resetFailed)
    }
  }

  return { exportConfig, importConfig, resetConfig }
}

interface ConfigTransferGroupProps {
  importInputRef: RefObject<HTMLInputElement | null>
  onConfigSaved?: () => void
}

/** Export / import / reset as labelled rows on Advanced › Desktop & startup. Only the redesign:
 *  classic keeps them as the icons under its Settings rail. */
export function ConfigTransferGroup(props: ConfigTransferGroupProps) {
  return usePortalShell() ? <ConfigTransferRows {...props} /> : null
}

function ConfigTransferRows({ importInputRef, onConfigSaved }: ConfigTransferGroupProps) {
  const copy = useSettingsPagesCopy().configuration
  const { t } = useI18n()
  const { exportConfig, importConfig, resetConfig } = useConfigTransfer(importInputRef, onConfigSaved)

  return (
    <SettingsGroup title={copy.title}>
      <SettingsCard>
        <ListRow
          action={
            <Button onClick={() => void exportConfig()} variant="secondary">
              {copy.export}
            </Button>
          }
          description={copy.exportDesc}
          title={t.settings.exportConfig}
        />
        <ListRow
          action={
            <Button onClick={importConfig} variant="secondary">
              {copy.import}
            </Button>
          }
          description={copy.importDesc}
          title={t.settings.importConfig}
        />
        <ListRow
          action={
            <Button onClick={() => void resetConfig()} variant="destructiveOutline">
              {copy.reset}
            </Button>
          }
          description={copy.resetDesc}
          title={t.settings.resetToDefaults}
        />
      </SettingsCard>
    </SettingsGroup>
  )
}
