import type { Translations } from '@/i18n'
import { transferAvailable } from '@/lib/hermes-cloud-host'
import { Archive, CloudUpload, type IconComponent } from '@/lib/icons'

export const BACKUP_SETTINGS_HREF = '/settings?tab=backup'

/** The query param naming the cloud agent to preselect in the transfer list. */
export const CLOUD_AGENT_PARAM = 'cloudAgent'

/** The old `/move-to-cloud?agent=<name>` route (the portal's `hermes://open/move-to-cloud` link)
 *  → Settings ▸ Backup & Transfer, keeping the agent it names. */
export function backupSettingsHrefForLegacy(search: string): string {
  const agent = (new URLSearchParams(search).get('agent') ?? '').trim()

  return agent ? `${BACKUP_SETTINGS_HREF}&${new URLSearchParams({ [CLOUD_AGENT_PARAM]: agent })}` : BACKUP_SETTINGS_HREF
}

export interface BackupPaletteEntry {
  icon: IconComponent
  id: string
  keywords: string[]
  label: string
}

/** The ⌘K rows that open Settings ▸ Backup & Transfer: one per task. Where Transfer cannot run
 *  (`transferAvailable`) only the backup row shows. */
export function backupPaletteEntries(t: Translations): BackupPaletteEntry[] {
  const copy = t.settings.backup

  return [
    {
      icon: Archive,
      id: 'backup-agent',
      keywords: ['backup', 'export', 'archive', 'save', 'download', 'settings'],
      label: copy.backupCommand
    },
    ...(transferAvailable()
      ? [
          {
            icon: CloudUpload,
            id: 'transfer-to-cloud',
            keywords: ['move', 'migrate', 'cloud', 'hermes cloud', 'upload', 'transfer', 'replace', 'settings'],
            label: copy.transferCommand
          }
        ]
      : [])
  ]
}
