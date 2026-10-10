// The settings-section repo-scan copy, composed by en.ts. It also carries the
// adjacent uninstall-section reference, so the section costs one spread line.
import { enUninstallSection } from './en_uninstall_section'
import type { Translations } from './types'
export const enSettingsRepoScan: Pick<Translations['settings'], 'repoScan' | 'uninstallSection'> = {
  uninstallSection: enUninstallSection,
  repoScan: {
    title: 'Repository Discovery Scan',
    hint: 'The same scan the sidebar runs when the Projects view opens or the window regains focus.',
    scanNow: 'Scan now',
    scanning: 'Scanning...',
    found: (count: number) => `Found ${count} repositories.`,
    foundOne: 'Found 1 repository.',
    disabled: 'Turn on Automatic Repository Discovery to scan.',
    noRoots: 'Nothing to scan: set a Working Directory above, or add a discovery root.',
    noBridge: 'This build cannot read the local disk, so there is nothing to scan.',
    rejected: 'The backend refused the scan: its discovery settings differ from this page.',
    failed: 'The scan failed. Try again.',
    failedWith: (detail: string) => `The scan failed: ${detail}`
  }
}
