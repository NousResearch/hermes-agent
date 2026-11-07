import type { Translations } from '@/i18n/types'
import {
  type AgentPluginRow,
  type AgentPluginScanFinding,
  type AgentPluginUpdateOptions,
  type AgentPluginUpdateOutcome,
  type GatewayRequest,
  updateAgentPlugin
} from '@/store/agent-plugins'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'

type PluginsCopy = Translations['skills']['plugins']

/** What the Update button moves to: the catalog's version label or short pin, else a git install's target. */
export const updateTargetLabel = (agent: Pick<AgentPluginRow, 'catalog_sha' | 'catalog_version' | 'update_target'>) =>
  agent.catalog_version ?? agent.catalog_sha?.slice(0, 8) ?? shortTarget(agent.update_target)

/** A full commit shortens to 8 characters; a feed version stays as it is. */
const shortTarget = (target?: string) =>
  target && /^[0-9a-f]{40}$/i.test(target) ? target.slice(0, 8) : (target ?? '')

const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low']

const severityRank = (severity: string) => {
  const rank = SEVERITY_ORDER.indexOf(severity)

  return rank === -1 ? SEVERITY_ORDER.length : rank
}

/** The findings a caution scan blocked an update on, one line each, worst first; capped so the dialog stays readable. */
export function scanFindingLines(findings: AgentPluginScanFinding[], more: (n: number) => string, cap = 8): string[] {
  const sorted = [...findings].sort((a, b) => severityRank(a.severity) - severityRank(b.severity))

  const lines = sorted
    .slice(0, cap)
    .map(f => `${f.severity.toUpperCase()}  ${f.file}:${f.line}  ${f.description || f.category}`)

  return sorted.length > cap ? [...lines, more(sorted.length - cap)] : lines
}

/** Update one plugin, asking the user before each override the backend asks for: a catalog pin that widens
 *  the plugin, or new code the security scan flagged as caution. Each question is asked at most once; a
 *  dangerous verdict is reported and never offered. Resolves with the final outcome. */
export async function runPluginUpdate(
  request: GatewayRequest,
  row: Pick<AgentPluginRow, 'catalog_name' | 'name'>,
  scope: null | string,
  p: PluginsCopy
): Promise<AgentPluginUpdateOutcome> {
  const { name } = row
  const gitInstall = !row.catalog_name
  const options: AgentPluginUpdateOptions = {}

  for (;;) {
    const outcome = await updateAgentPlugin(request, name, p.updateFailed(name), scope, options)

    if (outcome.kind === 'consent' && !options.acceptCapabilities) {
      // The new pin widens the plugin (tools, hooks, deps, capabilities, a Desktop half); the backend
      // changed nothing until the user confirms the delta.
      const ok = await confirm({
        confirmLabel: p.updateConsentConfirm,
        description: [p.updateConsentBody(name, outcome.sha), ...outcome.deltaLines].join('\n'),
        title: p.updateConsentTitle(name)
      })

      if (!ok) {
        return outcome
      }

      options.acceptCapabilities = true

      continue
    }

    if (outcome.kind === 'scan' && outcome.verdict === 'caution' && !options.force) {
      const ok = await confirm({
        confirmLabel: p.updateScanConfirm,
        description: [
          p.updateScanBody(name, outcome.findings.length),
          ...scanFindingLines(outcome.findings, p.updateScanMore)
        ].join('\n'),
        destructive: true,
        title: p.updateScanTitle(name)
      })

      if (!ok) {
        return outcome
      }

      options.force = true

      continue
    }

    if (outcome.kind === 'scan') {
      notifyError(new Error(p.updateScanBlocked(name)), p.updateFailed(name))
    } else if (outcome.kind === 'applied') {
      notify({ kind: 'success', message: gitInstall ? p.updatedGit(name) : p.updated(name) })
    }

    return outcome
  }
}
