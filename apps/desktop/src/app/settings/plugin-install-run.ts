import { enablePackageDesktopHalf } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import type { useI18n } from '@/i18n'
import { isBrowserHostedDesktop } from '@/lib/platform'
import {
  type AgentPluginInstallResult,
  type AgentPluginLiveNow,
  type GatewayRequest,
  installAgentPlugin,
  installDesktopPluginHalf,
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

/** Whether this app can put a desktop half down. Native Desktop clones or copies one in Electron; the
 *  browser-hosted app has no disk of its own, so the host that serves it clones one there
 *  (`plugins.manage install_desktop`) and the Webapp's loader picks it up. */
export function desktopInstallSupported(): boolean {
  return Boolean(window.hermesDesktop?.installDesktopPlugin) || isBrowserHostedDesktop()
}

/** What an install puts down: every half the repository has, narrowed by a legacy deep link or by
 *  "install the agent half here" (whose desktop half is already loaded), and by what this app can install. */
export function installComponents(
  request: PluginInstallRequest,
  probe: ProbeResult,
  desktopSupported = desktopInstallSupported()
): InstallComponents {
  const wanted = HINT_COMPONENTS[request.legacyHint ?? 'both'](probe)

  return { agent: wanted.agent, desktop: wanted.desktop && desktopSupported }
}

/** A desktop half the request wants that this app cannot install: skipped and said so, never reported done. */
export function skipsDesktopHalf(request: PluginInstallRequest, probe: ProbeResult): boolean {
  return installComponents(request, probe, true).desktop && !installComponents(request, probe).desktop
}

/** A unified package installed into a local backend carries its own desktop half; the app copies that half
 *  out of the package folder. A browser tab's backend is the host that serves it, so the package lands on
 *  the disk the tab's plugins load from and that host copies the half the same way: one checkout, never a
 *  second clone racing its scan. Only native Desktop on a remote backend (whose plugins/ folder this
 *  machine cannot read) or a desktop-only repo needs a separate desktop clone. */
export function desktopHalfFromPackage(components: InstallComponents, remote: boolean): boolean {
  return components.agent && (!remote || isBrowserHostedDesktop())
}

export interface InstallPlan {
  request: PluginInstallRequest
  probe: ProbeResult
  profile: string
  enable: boolean
  force: boolean
  ref?: string
  remote: boolean
  /** The user chose "Install anyway" on a backend that cannot hold the agent half to the reviewed commit. */
  withoutReviewedCommit?: boolean
}

export interface InstallDeps {
  copy: InstallModalCopy
  navigate: (to: string) => void
  requestGateway: GatewayRequest
}

/** `stale`: a half was refused because the source left the commit the dialog showed. Retrying would resend
 *  that commit and fail forever, so the dialog reads the source again and shows the new commit first.
 *  `partial`: the agent half installed at the reviewed commit, then the desktop half was refused because the
 *  source left it; only a replacement of both halves may follow. `needsConfirm`: the backend cannot hold the
 *  install to the reviewed commit and installed nothing; the user decides whether to install anyway. */
export type InstallOutcome =
  | { error: string; status: 'failed' }
  | { error: string; status: 'partial' }
  | { error: string; status: 'stale' }
  | { status: 'done' }
  | { status: 'needsConfirm' }
  | { status: 'uncertain' }

interface Report {
  errors: string[]
  successes: string[]
  /** A half was refused on the backend's or Electron's structured source-moved marker. */
  sourceMoved: boolean
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

/** The commit the dialog showed, which both halves are held to: a branch that moved, or a catalog entry
 *  re-pinned, since must not install a tree nobody reviewed. A catalog pick shows its reviewed pin
 *  (`request.sha`), unless the backend read the entry at that pin (`probe.catalogPin`: the browser-hosted
 *  app), whose commit a re-inspection re-reads. Electron's probe reads the branch tip, which a catalog
 *  install never checks out. */
export function reviewedCommit({ probe, request }: Pick<InstallPlan, 'probe' | 'request'>): string | undefined {
  return !request.catalogName || probe.catalogPin ? probe.sha : request.sha
}

async function installAgentHalf(plan: InstallPlan, deps: InstallDeps, report: Report) {
  const { request } = plan

  const result = await installAgentPlugin(deps.requestGateway, {
    identifier: request.repo,
    force: plan.force,
    enable: plan.enable,
    catalogName: request.catalogName,
    ref: plan.ref,
    inspectedSha: plan.withoutReviewedCommit ? undefined : reviewedCommit(plan),
    profile: plan.profile
  })

  if (!result.ok) {
    report.errors.push(result.error || deps.copy.agentFailed)
    report.sourceMoved ||= Boolean(result.sourceMoved)

    return result
  }

  const name = result.pluginName ?? request.repo

  report.successes.push(
    [deps.copy.agentSuccess(name), ...installOutcome(deps.copy, result.live, result.nextChat)].join(' · ')
  )
  notifyAgentFollowUps(result, name, deps)

  return result
}

async function cloneDesktopHalf(
  plan: InstallPlan,
  copy: InstallModalCopy,
  report: Report,
  deps: InstallDeps
): Promise<void> {
  // Native Desktop clones it in Electron; a browser-hosted tab asks the host that serves it, and the
  // plugin lands in the folder this Webapp already loads. A tab installing a package's agent half never
  // clones the desktop half here: the host copies it out of that one checkout (`desktopHalfFromPackage`).
  const installFn = window.hermesDesktop?.installDesktopPlugin

  const result: { error?: string; ok: boolean; pluginName?: string; sourceMoved?: boolean; unsupported?: boolean } =
    installFn
      ? await installFn({ identifier: plan.request.repo, force: plan.force, sha: plan.ref ?? reviewedCommit(plan) })
      : await installDesktopPluginHalf(deps.requestGateway, {
          identifier: plan.request.repo,
          catalogName: plan.request.catalogName,
          force: plan.force,
          profile: plan.profile,
          // The pin the user typed, else the commit the dialog showed: the host installs exactly that tree
          // or refuses.
          ref: plan.ref ?? reviewedCommit(plan)
        })

  if (!result.ok) {
    // A backend older than this build has no place to put the half: say what this environment can do,
    // which is what an install without Electron always said.
    report.errors.push(result.unsupported ? copy.desktopUnavailable : result.error || copy.desktopFailed)
    // A pin the user typed that the source lacks is their pin to fix; a fresh inspection cannot change it.
    report.sourceMoved ||= !plan.ref && Boolean(result.sourceMoved)

    return
  }

  report.successes.push(copy.desktopSuccess(result.pluginName ?? plan.request.repo))
  await discoverRuntimePlugins()
}

/** Whether the app's desktop-plugins root holds `<name>/plugin.js` after a reconcile, read from the folder
 *  itself (Electron's fs bridge, or the host's /api/desktop-plugins on a browser tab). `touched` names
 *  folders the pass DELETED as well as copied (a force reinstall whose new tree dropped its desktop entry
 *  prunes the old copy), and one that found the copy current touches nothing. Only a bridge with no
 *  folder reader falls back to the pass's own list. */
async function desktopHalfLanded(name: string, touched: readonly string[]): Promise<boolean> {
  const desktop = window.hermesDesktop

  if (!desktop?.desktopPluginsRoot || !desktop.readDir) {
    return touched.some(folder => folder.split(/[\\/]/).pop() === name)
  }

  try {
    const root = await desktop.desktopPluginsRoot()
    const folder = (await desktop.readDir(root)).entries.find(entry => entry.isDirectory && entry.name === name)

    return Boolean(folder && (await desktop.readDir(folder.path)).entries.some(entry => entry.name === 'plugin.js'))
  } catch {
    return false
  }
}

async function installDesktopHalf(
  plan: InstallPlan,
  components: InstallComponents,
  deps: InstallDeps,
  report: Report,
  agentResult?: AgentPluginInstallResult
): Promise<void> {
  if (desktopHalfFromPackage(components, plan.remote)) {
    // The desktop half ships inside the package folder: materialise it from there (one source of truth,
    // follows updates/uninstall) whether or not the agent install succeeded. A package already on disk
    // answers "already exists" without Force, and cloning would land desktop-plugins/<git-name>/ beside
    // the package copy (#100412).
    const touched = (await window.hermesDesktop?.reconcileDesktopPlugins?.()) ?? []
    const name = agentResult?.pluginName ?? plan.probe.agentName ?? plan.request.repo

    if (await desktopHalfLanded(name, touched)) {
      report.successes.push(deps.copy.desktopSuccess(name))
    } else if (agentResult?.ok) {
      // The agent half is in, but the host copied no desktop half out of it: never report one installed.
      report.errors.push(deps.copy.desktopFailed)
    }

    if (touched.length > 0) {
      await discoverRuntimePlugins()
    }
  } else {
    await cloneDesktopHalf(plan, deps.copy, report, deps)
  }

  // The desktop half of a unified package lands opt-in; an install that enabled the agent half turns it on too.
  if (agentResult?.enabled && agentResult.pluginName) {
    await enablePackageDesktopHalf(agentResult.pluginName)
  }
}

/** An agent-half answer after which nothing else may install. A client timeout is `uncertain`: the backend
 *  may still be installing. A refusal because the source moved means nothing from that review installs, not
 *  the other half either. An older backend refusing the reviewed commit at admission installed nothing:
 *  the user decides. */
function agentStopsInstall(
  result: AgentPluginInstallResult,
  plan: InstallPlan,
  deps: InstallDeps,
  report: Report
): InstallOutcome | null {
  if (result.timedOut) {
    void loadAgentPlugins(deps.requestGateway, plan.profile)

    return { status: 'uncertain' }
  }

  if (result.sourceMoved) {
    return { status: 'stale', error: report.errors.join(' ') }
  }

  return result.inspectedShaUnsupported ? { status: 'needsConfirm' } : null
}

/** The outcome of an install that reported errors. The agent half in at the reviewed commit while the desktop
 *  half could not follow it is `partial`: a fresh review would lead the next Install to put the desktop half
 *  down at another commit beside it. */
function failedOutcome(plan: InstallPlan, deps: InstallDeps, report: Report, agentInstalled: boolean): InstallOutcome {
  if (agentInstalled && report.sourceMoved) {
    const sha = plan.withoutReviewedCommit ? '' : (reviewedCommit(plan)?.slice(0, 12) ?? '')

    return { status: 'partial', error: deps.copy.partialInstall(sha) }
  }

  return { status: report.sourceMoved ? 'stale' : 'failed', error: report.errors.join(' ') }
}

/** Install every half `installComponents` picks, then refresh the inventory. A client timeout on the agent
 *  half is `uncertain`: the backend may still be installing, so nothing else starts. */
export async function runPluginInstall(plan: InstallPlan, deps: InstallDeps): Promise<InstallOutcome> {
  const components = installComponents(plan.request, plan.probe)

  if (!components.agent && !components.desktop) {
    return { status: 'failed', error: deps.copy.nothingToInstall }
  }

  const report: Report = { errors: [], successes: [], sourceMoved: false }
  const agentResult = components.agent ? await installAgentHalf(plan, deps, report) : undefined
  const stopped = agentResult && agentStopsInstall(agentResult, plan, deps, report)

  if (stopped) {
    return stopped
  }

  if (components.desktop) {
    await installDesktopHalf(plan, components, deps, report, agentResult)
  }

  await loadAgentPlugins(deps.requestGateway, plan.profile)

  for (const message of report.successes) {
    notify({ kind: 'success', message })
  }

  if (agentResult?.ok && skipsDesktopHalf(plan.request, plan.probe)) {
    notify({ kind: 'warning', message: deps.copy.desktopUnavailable })
  }

  if (report.errors.length > 0) {
    return failedOutcome(plan, deps, report, Boolean(agentResult?.ok))
  }

  // Open chats of the profile already have the plugin's MCP tools and skills; say which servers did not connect.
  if (agentResult?.ok && plan.enable) {
    for (const server of agentResult.live.mcpServers.filter(s => !s.connected)) {
      notify({ kind: 'warning', message: deps.copy.serverNotConnected(server.name, server.error || '') })
    }
  }

  return { status: 'done' }
}
