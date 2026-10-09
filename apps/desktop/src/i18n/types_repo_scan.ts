// The repo-scan settings section's copy shape; `Translations.settings.repoScan`.
// It also carries the adjacent uninstall-section shape, so the section costs one
// intersection line rather than two references.
import type { UninstallSectionTranslations } from './types_uninstall_section'
export interface RepoScanSettingsTranslations {
  uninstallSection: UninstallSectionTranslations
  repoScan: {
    title: string
    hint: string
    scanNow: string
    scanning: string
    found: (count: number) => string
    foundOne: string
    disabled: string
    noRoots: string
    noBridge: string
    rejected: string
    failed: string
    failedWith: (detail: string) => string
  }
}
