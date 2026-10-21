import { enablePackageDesktopHalf } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import type { useI18n } from '@/i18n'
import {
  type AgentPluginInstallResult,
  type AgentPluginLiveNow,
  type GatewayRequest,
  installAgentPlugin,
  loadAgentPlugins
} from '@/store/agent-plugins'
import { notify } from '@/store/notifications'
import type { PluginInstallRequest } from '@/store/plugin-install-request'

export type ProbeResult = Awaited<ReturnType<NonNullable<NonNullable<Window['hermesDesktop']>['probePluginRepo']>>>

export type InstallModalCopy = ReturnType<typeof useI18n>['t']['settings']['plugins']['installModal']

export interface InstallComponents {
  agent: boolean
  desktop: boolean
}

const HINT_COMPONENTS = {
  agent: probe => ({ agent: probe.agent, desktop: false }),
  both: probe => ({ agent: probe.agent, desktop: probe.desktop }),
  desktop: probe => ({ agent: false, desktop: probe.desktop })
} satisfies Record<'agent' | 'both' | 'desktop', (probe: ProbeResult) => InstallComponents>

/** What an install puts down: every half the repository has, narrowed by a legacy deep link or by
 *  "install the agent half here" (whose desktop half is already loaded). */
export function installComponents(request: PluginInstallRequest, probe: ProbeResult): InstallComponents {
  return HINT_COMPONENTS[request.legacyHint ?? 'both'](probe)
}

/** A unified package installed into a local backend carries its own desktop half; the app copies that half
 *  out of the package folder. Only a remote backend (whose plugins/ folder this machine cannot read) or a
 *  desktop-only repo needs a separate desktop clone. */
export function desktopHalfFromPackage(components: InstallComponents, remote: boolean): boolean {
  return components.agent && !remote
}

export interface InstallPlan {
  request: PluginInstallRequest
  probe: ProbeResult
  profile: string
  enable: boolean
  force: boolean
  ref?: string
  remote: boolean
}

export interface InstallDeps {
  copy: InstallModalCopy
  navigate: (to: string) => void
  requestGateway: GatewayRequest
}

export type InstallOutcome = { error: string; status: 'failed' } | { status: 'done' } | { status: 'uncertain' }

interface Report {
  errors: string[]
  successes: string[]
}

/** What an agent-plugin install made usable, as toast fragments ("12 tools connected", ...). */
function installOutcome(m: InstallModalCopy, live: AgentPluginLiveNow, nextChat: boolean): string[] {
  const tools = live.mcpServers.reduce((n, server) => n + (server.connected ? server.tools.length : 0), 0)

  return [
    ...(tools > 0 ? [m.toolsConnected(tools)] : []),
    ...(live.skills.length > 0 ? [m.skillsReady(live.skills)] : []),
    ...(nextChat ? [m.nextChat] : [])
  ]
}

function notifyAgentFollowUps(result: AgentPluginInstallResult, name: string, deps: InstallDeps): void {
  const firstVar = result.missingEnv?.[0]

  if (firstVar) {
    notify({
      kind: 'warning',
      message: deps.copy.missingEnv(name, result.missingEnv!.join(', ')),
      // Deep-link straight to the credential card instead of leaving the user to hunt through Settings.
      action: {
        label: deps.copy.missingEnvAction,
        onClick: () => deps.navigate(`/settings?tab=keys&key=${encodeURIComponent(firstVar)}`)
      }
    })
  }

  for (const warning of result.warnings ?? []) {
    notify({ kind: 'warning', message: warning })
  }
}

async function installAgentHalf(plan: InstallPlan, deps: InstallDeps, report: Report) {
  const { request } = plan

  const result = await installAgentPlugin(deps.requestGateway, {
    identifier: request.repo,
    force: plan.force,
    enable: plan.enable,
    catalogName: request.catalogName,
    ref: plan.ref,
    profile: plan.profile
  })

  if (!result.ok) {
    report.errors.push(result.error || deps.copy.agentFailed)

    return result
  }

  const name = result.pluginName ?? request.repo

  report.successes.push(
    [deps.copy.agentSuccess(name), ...installOutcome(deps.copy, result.live, result.nextChat)].join(' · ')
  )
  notifyAgentFollowUps(result, name, deps)

  return result
}

async function cloneDesktopHalf(plan: InstallPlan, copy: InstallModalCopy, report: Report): Promise<void> {
  const installFn = window.hermesDesktop?.installDesktopPlugin

  if (!installFn) {
    report.errors.push(copy.desktopUnavailable)

    return
  }

  const result = await installFn({ identifier: plan.request.repo, force: plan.force })

  if (!result.ok) {
    report.errors.push(result.error || copy.desktopFailed)

    return
  }

  report.successes.push(copy.desktopSuccess(result.pluginName ?? plan.request.repo))
  await discoverRuntimePlugins()
}

async function installDesktopHalf(
  plan: InstallPlan,
  components: InstallComponents,
  copy: InstallModalCopy,
  report: Report,
  agentResult?: AgentPluginInstallResult
): Promise<void> {
  if (desktopHalfFromPackage(components, plan.remote)) {
    // The desktop half ships inside the package folder: materialise it from there (one source of truth,
    // follows updates/uninstall) whether or not the agent install succeeded. A package already on disk
    // answers "already exists" without Force, and cloning would land desktop-plugins/<git-name>/ beside
    // the package copy (#100412).
    const touched = (await window.hermesDesktop?.reconcileDesktopPlugins?.()) ?? []

    if (agentResult?.ok || touched.length > 0) {
      report.successes.push(copy.desktopSuccess(plan.probe.agentName ?? plan.request.repo))
    }

    if (touched.length > 0) {
      await discoverRuntimePlugins()
    }
  } else {
    await cloneDesktopHalf(plan, copy, report)
  }

  // The desktop half of a unified package lands opt-in; an install that enabled the agent half turns it on too.
  if (agentResult?.enabled && agentResult.pluginName) {
    await enablePackageDesktopHalf(agentResult.pluginName)
  }
}

/** Install every half `installComponents` picks, then refresh the inventory. A client timeout on the agent
 *  half is `uncertain`: the backend may still be installing, so nothing else starts. */
export async function runPluginInstall(plan: InstallPlan, deps: InstallDeps): Promise<InstallOutcome> {
  const components = installComponents(plan.request, plan.probe)

  if (!components.agent && !components.desktop) {
    return { status: 'failed', error: deps.copy.nothingToInstall }
  }

  const report: Report = { errors: [], successes: [] }
  const agentResult = components.agent ? await installAgentHalf(plan, deps, report) : undefined

  if (agentResult?.timedOut) {
    void loadAgentPlugins(deps.requestGateway, plan.profile)

    return { status: 'uncertain' }
  }

  if (components.desktop) {
    await installDesktopHalf(plan, components, deps.copy, report, agentResult)
  }

  await loadAgentPlugins(deps.requestGateway, plan.profile)

  for (const message of report.successes) {
    notify({ kind: 'success', message })
  }

  if (report.errors.length > 0) {
    return { status: 'failed', error: report.errors.join(' ') }
  }

  // Open chats of the profile already have the plugin's MCP tools and skills; say which servers did not connect.
  if (agentResult?.ok && plan.enable) {
    for (const server of agentResult.live.mcpServers.filter(s => !s.connected)) {
      notify({ kind: 'warning', message: deps.copy.serverNotConnected(server.name, server.error || '') })
    }
  }

  return { status: 'done' }
}
