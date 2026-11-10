import type { Translations } from '@/i18n/types'
import type { AgentPluginScanFinding } from '@/store/agent-plugins'
import { confirm } from '@/store/confirm'
import { type FleetInstall, fleetInstallKey, updateFleetInstall } from '@/store/plugin-fleet-updates'

import { scanFindingLines } from './plugin-update'

type PluginsCopy = Translations['skills']['plugins']

export type FleetRowState = 'pending' | 'running' | 'done' | 'failed' | 'declined'

interface FleetOptions {
  acceptCapabilities?: boolean
  force?: boolean
}

interface FleetJob {
  install: FleetInstall
  options: FleetOptions
}

interface Gated<T> {
  job: FleetJob
  detail: T
}

const UPDATE_CONCURRENCY = 3

const names = (gated: Gated<unknown>[]) => [...new Set(gated.map(entry => entry.job.install.row.name))].join(', ')

async function drain(jobs: FleetJob[], apply: (job: FleetJob) => Promise<void>) {
  const work = [...jobs]

  await Promise.all(
    Array.from({ length: Math.min(UPDATE_CONCURRENCY, work.length) }, async () => {
      for (let next = work.shift(); next; next = work.shift()) {
        await apply(next)
      }
    })
  )
}

/**
 * Apply a fleet update batch with the same overrides as the single-profile
 * update (`runPluginUpdate`): a catalog pin that widens the plugin asks for
 * consent, a caution security-scan verdict asks for an override, a dangerous
 * verdict fails. Each question is asked ONCE for the whole batch, not per
 * profile; a declined question marks its rows `declined`.
 */
export async function runFleetBatch(
  installs: FleetInstall[],
  p: PluginsCopy,
  setState: (install: FleetInstall, state: FleetRowState, error?: string) => void
): Promise<{ applied: number; settled: Set<string> }> {
  const settled = new Set<string>()
  let applied = 0
  let pending: FleetJob[] = installs.map(install => ({ install, options: {} }))

  while (pending.length > 0) {
    const consent: Gated<string[]>[] = []
    const scan: Gated<AgentPluginScanFinding[]>[] = []

    await drain(pending, async job => {
      const { install, options } = job
      setState(install, 'running')
      const outcome = await updateFleetInstall(install, options)

      if (outcome.kind === 'consent' && !options.acceptCapabilities) {
        consent.push({ job, detail: outcome.deltaLines })
        setState(install, 'pending')
      } else if (outcome.kind === 'scan' && outcome.verdict === 'caution' && !options.force) {
        scan.push({ job, detail: outcome.findings })
        setState(install, 'pending')
      } else if (outcome.kind === 'scan') {
        setState(install, 'failed', p.updateScanBlocked(install.row.name))
      } else if (outcome.kind === 'failed' || outcome.kind === 'consent') {
        setState(install, 'failed', outcome.kind === 'failed' ? outcome.error : p.updateFailed(install.row.name))
      } else {
        applied += outcome.kind === 'applied' ? 1 : 0
        settled.add(fleetInstallKey(install))
        setState(install, 'done')
      }
    })

    pending = []

    if (consent.length > 0) {
      const ok = await confirm({
        confirmLabel: p.updateConsentConfirm,
        description: [
          p.fleet.consentBody(names(consent), consent.length),
          ...new Set(consent.flatMap(e => e.detail))
        ].join('\n'),
        title: p.updateConsentTitle(names(consent))
      })

      pending.push(...retry(consent, ok, { acceptCapabilities: true }, setState))
    }

    if (scan.length > 0) {
      // The same plugin reports the same findings in every profile; list each once.
      const findings = [
        ...new Map(
          scan.flatMap(entry => entry.detail).map(f => [`${f.file}:${f.line}:${f.category}:${f.description}`, f])
        ).values()
      ]

      const ok = await confirm({
        confirmLabel: p.updateScanConfirm,
        description: [
          p.updateScanBody(names(scan), findings.length),
          ...scanFindingLines(findings, p.updateScanMore)
        ].join('\n'),
        destructive: true,
        title: p.updateScanTitle(names(scan))
      })

      pending.push(...retry(scan, ok, { force: true }, setState))
    }
  }

  return { applied, settled }
}

/** The approved rows go round again with the new override; a declined question marks them. */
function retry(
  gated: Gated<unknown>[],
  ok: boolean,
  grant: FleetOptions,
  setState: (install: FleetInstall, state: FleetRowState) => void
): FleetJob[] {
  if (!ok) {
    gated.forEach(entry => setState(entry.job.install, 'declined'))

    return []
  }

  return gated.map(entry => ({ install: entry.job.install, options: { ...entry.job.options, ...grant } }))
}
