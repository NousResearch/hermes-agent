import { atom, map } from 'nanostores'

import { getActionStatus, installSkillFromHub, uninstallSkillFromHub, updateSkillsFromHub } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { upsertDesktopActionTask } from '@/store/activity'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

const POLL_MS = 1200

// Shared with hub.tsx's sources useQuery so a finished action refreshes the
// installed map.
export const HUB_SOURCES_KEY = ['skill-hub-sources'] as const
// The Capabilities Skills-list query key (see app/skills/index.tsx) — kept in
// sync here so a hub (un)install updates the Skills tab, not just the hub.
const SKILLS_LIST_KEY = ['skills-list'] as const
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
async function runHubAction(key: string, kind: HubActionKind, spawn: () => Promise<{ name: string }>): Promise<void> {
  const epoch = _hubEpoch
  const switched = () => _hubEpoch !== epoch

  $hubActions.setKey(key, { kind, running: true, lines: [] })
  $hubActiveLog.set(key)

  try {
    const started = await spawn()
    let exitCode: number | null = null

    for (;;) {
      const status = await getActionStatus(started.name, 200)

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
    void queryClient.invalidateQueries({ queryKey: SKILLS_LIST_KEY })
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

export function installHubSkill(identifier: string): Promise<void> {
  return runHubAction(identifier, 'install', () => installSkillFromHub(identifier))
}

export function uninstallHubSkill(identifier: string, name: string): Promise<void> {
  return runHubAction(identifier, 'uninstall', () => uninstallSkillFromHub(name))
}

export function updateHubSkills(): Promise<void> {
  return runHubAction(UPDATE_ALL_KEY, 'update', () => updateSkillsFromHub())
}

export function closeHubLog(): void {
  $hubActiveLog.set(null)
}
