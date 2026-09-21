import { atom, map } from 'nanostores'

// The Capabilities Skills-list query key, imported rather than re-spelled so a
// hub (un)install updates the Skills tab, not just the hub.
import { SKILLS_QUERY_KEY } from '@/app/capabilities/skills/skills-data'
import {
  getActionStatus,
  installSkillFromHub,
  type ProfileScope,
  scanSkillHub,
  uninstallSkillFromHub,
  updateSkillsFromHub
} from '@/hermes'
import { translateNow } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { upsertDesktopActionTask } from '@/store/activity'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

const POLL_MS = 1200

// Shared with hub.tsx's sources useQuery so a finished action refreshes the
// installed map.
export const HUB_SOURCES_KEY = ['skill-hub-sources'] as const
// The built-in optional-skills catalog rows in the Skills tab: an install
// flips one of them to an installed (toggle) row, so the catalog's
// installed-flags must refetch alongside the skills list.
export const OFFICIAL_SKILLS_KEY = ['official-skills'] as const
// Non-identifier key for the fleet-wide "Update installed" action.
export const UPDATE_ALL_KEY = '__update_all__'

export type HubActionKind = 'install' | 'uninstall' | 'update'

export interface HubAction {
  kind: HubActionKind
  running: boolean
  lines: string[]
}

// Per-item action status, keyed by skill identifier (or UPDATE_ALL_KEY). Each
// row drives its own button off ITS entry — one install never touches another.
export const $hubActions = map<Record<string, HubAction | undefined>>({})

// Optimistic installed overrides so a row flips to its resolved state the instant
// its own action finishes, instead of waiting on (and racing) the sources
// refetch. install/update → true, uninstall → false; sources reconciles after.
export const $hubInstalledOverride = map<Record<string, boolean | undefined>>({})

// The key whose log the bottom pane currently tails (the latest-started action).
export const $hubActiveLog = atom<null | string>(null)

// Hub action state is per-profile: a profile switch must drop every in-flight
// entry, optimistic override, and active log so profile A's install/uninstall
// state can never render (or be polled) in profile B. Cleared at the source so
// it holds regardless of whether the Hub view is mounted. The epoch bumps on
// every switch; a runHubAction() started before the switch captures it and bails
// before any store write once it no longer matches (so an A action finishing
// after the clear can't repopulate B).
let _hubProfile: null | string = null
let _hubEpoch = 0

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)

  if (_hubProfile !== null && _hubProfile !== key) {
    _hubEpoch += 1
    $hubActions.set({})
    $hubInstalledOverride.set({})
    $hubActiveLog.set(null)
  }

  _hubProfile = key
})

// One self-contained task: spawn → tail its own action log into the store →
// mark resolved. Concurrency-safe: state is per-key, so parallel installs never
// stomp each other, and the sources query is invalidated once at the end.
// `profile` is the Capabilities profile-scope override — the action (and its
// status polling) runs against THAT profile's backend; undefined keeps the
// app-wide active profile (unchanged behavior).
async function runHubAction(
  key: string,
  kind: HubActionKind,
  spawn: () => Promise<{ name: string }>,
  profile?: ProfileScope
): Promise<void> {
  const epoch = _hubEpoch
  const switched = () => _hubEpoch !== epoch

  $hubActions.setKey(key, { kind, running: true, lines: [] })
  $hubActiveLog.set(key)

  try {
    const started = await spawn()
    let exitCode: number | null = null

    for (;;) {
      const status = await getActionStatus(started.name, 200, profile)

      // Profile switched mid-flight: the store was cleared for the new profile,
      // so drop this A-profile result instead of writing it back into B.
      if (switched()) {
        return
      }

      upsertDesktopActionTask(status)
      $hubActions.setKey(key, { kind, running: status.running, lines: status.lines })

      if (!status.running) {
        exitCode = status.exit_code

        break
      }

      await new Promise(resolve => setTimeout(resolve, POLL_MS))
    }

    // Only flip the row on a clean exit — a failed install/uninstall must not
    // render as installed/removed.
    if (key !== UPDATE_ALL_KEY && exitCode === 0) {
      $hubInstalledOverride.setKey(key, kind !== 'uninstall')
    }

    // Refresh the hub's installed map AND the Capabilities Skills list — a hub
    // (un)install adds/removes a skill, so its count/rows must update too.
    void queryClient.invalidateQueries({ queryKey: HUB_SOURCES_KEY })
    void queryClient.invalidateQueries({ queryKey: SKILLS_QUERY_KEY })
    void queryClient.invalidateQueries({ queryKey: OFFICIAL_SKILLS_KEY })
    // …and the composer's `/` list, which caches the command catalog for an
    // hour and would otherwise keep offering the skill we just removed.
    invalidateSlashCompletions()

    // A non-zero exit is a real failure — throw so the caller's catch toasts
    // it. Before this, a failed subprocess (scan gate, network, bad
    // identifier) just stopped silently: no flip, no toast, and the user read
    // the unchanged skills list as "install did nothing" (Aug 2026 report).
    // The last log lines carry the subprocess's actual error.
    if (exitCode !== null && exitCode !== 0) {
      const lines = $hubActions.get()[key]?.lines ?? []
      const blocked = parseInstallBlocked(lines)

      if (blocked) {
        throw new HubInstallBlockedError(key, blocked.findings, blocked.unverified, lines.slice(-3).join('\n').trim())
      }

      const detail = lines.slice(-3).join('\n').trim()

      throw new Error(detail || `Action exited with code ${exitCode}`)
    }
  } catch (err) {
    // A profile switch points the next poll at the new backend, which 404s the
    // old action name — that's an abandonment, not a failure, so swallow it
    // instead of letting the caller toast a phantom error. Real (same-profile)
    // failures still propagate.
    if (switched()) {
      return
    }

    throw err
  } finally {
    // Skip the running=false write after a switch — it would re-add the key the
    // profile-switch clear just dropped.
    const current = $hubActions.get()[key]

    if (current && !switched()) {
      $hubActions.setKey(key, { ...current, running: false })
    }
  }
}

export function installHubSkill(identifier: string, profile?: ProfileScope): Promise<void> {
  return runHubAction(identifier, 'install', () => installSkillFromHub(identifier, profile), profile)
}

export function uninstallHubSkill(identifier: string, name: string, profile?: ProfileScope): Promise<void> {
  return runHubAction(identifier, 'uninstall', () => uninstallSkillFromHub(name, profile), profile)
}

export function updateHubSkills(profile?: ProfileScope): Promise<void> {
  return runHubAction(UPDATE_ALL_KEY, 'update', () => updateSkillsFromHub(profile), profile)
}

export function closeHubLog(): void {
  $hubActiveLog.set(null)
}

// `hermes skills install` exits non-zero when the security scan gate refuses,
// and its printed tail is the only signal the Desktop gets, so parse it into a
// structured failure the toast can explain (tools-runtime-21). `--force` has
// no Desktop route, so the remedy offered is reading the scan, not overriding
// it. Two CLI shapes exist (`hermes_cli/skills_hub.py::_scan_block_message`):
//   current: "Not installed: the security scan found 2 high-risk pattern(s) in
//            'org/skill' (listed above). Hermes never installs unverified
//            skills with high-risk findings, even with --force. ..."
//            (the "never installs unverified" sentence only appears for a
//            non-official source; otherwise it says "Re-run with --force").
//   legacy:  "Installation blocked: Blocked (community source + caution
//            verdict, 2 findings). Use --force to override."
// The console may wrap the sentence, so whitespace between words is `\s+`.
const INSTALL_BLOCKED_CURRENT_RE =
  /Not installed:\s+the security scan found\s+(?:(?<findings>\d+)\s+)?high-risk\s+pattern/i

const INSTALL_BLOCKED_UNVERIFIED_RE = /never installs\s+unverified/i

const INSTALL_BLOCKED_LEGACY_RE =
  /Installation blocked:.*?\((?<source>[a-z_-]+) source \+ (?<verdict>[a-z_]+) verdict, (?<findings>\d+) findings?\)/i

export function parseInstallBlocked(lines: readonly string[]): { findings: number; unverified: boolean } | null {
  const text = lines.join('\n')
  const current = text.match(INSTALL_BLOCKED_CURRENT_RE)

  if (current?.groups) {
    return {
      findings: current.groups.findings ? Number(current.groups.findings) : 0,
      unverified: INSTALL_BLOCKED_UNVERIFIED_RE.test(text)
    }
  }

  const legacy = text.match(INSTALL_BLOCKED_LEGACY_RE)

  if (!legacy?.groups) {
    return null
  }

  return { findings: Number(legacy.groups.findings), unverified: legacy.groups.source !== 'official' }
}

export class HubInstallBlockedError extends Error {
  constructor(
    readonly identifier: string,
    readonly findings: number,
    readonly unverified: boolean,
    detail: string
  ) {
    super(detail)
    this.name = 'HubInstallBlockedError'
  }
}

/** Toast for a failed hub action: a blocked install explains the scan gate and
 *  offers "View scan"; anything else keeps the generic summary + raw tail. */
export function notifyHubActionFailed(err: unknown, fallbackTitle: string, skillName?: string, profile?: ProfileScope): void {
  if (!(err instanceof HubInstallBlockedError)) {
    notifyError(err, fallbackTitle)

    return
  }

  const name = skillName || err.identifier

  notify({
    kind: 'error',
    title: translateNow('skills.hub.installBlockedTitle', name),
    message: translateNow('skills.hub.installBlockedMessage', err.findings, err.unverified),
    detail: err.message || undefined,
    action: {
      label: translateNow('skills.hub.viewScan'),
      onClick: () =>
        void scanSkillHub(err.identifier, typeof profile === 'object' ? profile?.profile : profile)
          .then(scan =>
            notify({
              kind: 'warning',
              title: translateNow('skills.hub.installBlockedTitle', name),
              message: scan.summary,
              detail: scan.findings.map(f => `${f.severity}: ${f.description}`).join('\n') || undefined
            })
          )
          .catch(scanErr => notifyError(scanErr, translateNow('skills.hub.scanFailed')))
    }
  })
}
