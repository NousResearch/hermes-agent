import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { $reposScanning, type RepoScanOutcome, scanAndRecordRepos } from '@/store/projects'

import { useHermesConfigRecord } from '../hooks/use-config-record'

import { getNested } from './helpers'
import { ListRow } from './primitives'

/**
 * Run the repository discovery scan now, rather than waiting for one of its own triggers — the
 * Projects view's first paint, the window regaining focus, or a saved discovery setting.
 *
 * The scan's quiet endings are the whole reason this reports anything: nothing configured to walk,
 * a policy the backend refused, or a build with no local git bridge all leave the sidebar exactly as
 * empty as a workspace that really holds no repositories.
 */
export function RepoScanSetting() {
  const { t } = useI18n()
  const copy = t.settings.repoScan
  const scanning = useStore($reposScanning)
  const { data: loadedConfig } = useHermesConfigRecord()
  const [outcome, setOutcome] = useState<null | RepoScanOutcome>(null)

  // The scan's own switch decides whether there is a scan — the same gate the nested-discovery
  // toggle sits behind. Not yet loaded counts as on: the button is honest about the outcome anyway.
  const enabled = !loadedConfig || getNested(loadedConfig, 'desktop.repo_scan_enabled') !== false

  const run = () => {
    setOutcome(null)

    void scanAndRecordRepos(true).then(setOutcome)
  }

  const status = (() => {
    if (!outcome) {
      return copy.hint
    }

    switch (outcome.reason) {
      case 'ok':
        return outcome.found === 1 ? copy.foundOne : copy.found(outcome.found)

      case 'disabled':
        return copy.disabled

      case 'no-bridge':
        return copy.noBridge

      case 'no-roots':
        return copy.noRoots

      case 'rejected':
        return copy.rejected

      case 'failed':
        // The gateway's message names the exact key it refused; "the scan failed" alone sends the
        // user (and whoever reads the report) nowhere.
        return outcome.detail ? copy.failedWith(outcome.detail) : copy.failed

      default:
        return copy.hint
    }
  })()

  return (
    <ListRow
      action={
        <Button disabled={!enabled || scanning} onClick={run} size="sm">
          {scanning ? copy.scanning : copy.scanNow}
        </Button>
      }
      description={enabled ? status : copy.disabled}
      title={copy.title}
    />
  )
}
