import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { getApiRequestConnection, getApiRequestProfile, type ProfileScope, profileScopeKey } from '@/api/client'
import {
  type DelegationMode,
  type DelegationModelMap,
  type DelegationProfileRoutes,
  type DelegationRouting,
  getDelegationRouting,
  saveDelegationModelRoutes,
  saveDelegationProfileRoutes,
  selectDelegationBackend
} from '@/api/delegation'
import { queryClient } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import { notifyError } from '@/store/notifications'
import { $activeProfile } from '@/store/profile'

/** The official plugin that provides the two route backends (`delegation.backend` values below). */
export const ADVANCED_DELEGATION_PLUGIN = 'advanced-delegation'
export const PLUGIN_MODES = ['model_map', 'profile_routes'] as const satisfies readonly DelegationMode[]
export const MODES = ['default', ...PLUGIN_MODES] as const satisfies readonly DelegationMode[]

// An unpinned scope follows the app's active profile and connection, so both are part of the key.
export const routingQueryKey = (scope: ProfileScope) =>
  [
    'delegation-routing',
    profileScopeKey(scope ?? getApiRequestProfile()),
    getApiRequestConnection() ?? 'local'
  ] as const

// A backend older than the routing endpoint answers 404: a missing capability, not a failure to retry.
const isMissingEndpoint = (error: unknown) => /\b404\b/.test(error instanceof Error ? error.message : String(error))

/** The owner's delegation routing (backend, route blocks, registered backends), shared by every row of the page.
 *  `null` data = the backend has no routing endpoint, so the page keeps its pre-routing shape. */
export function useDelegationRouting(scope: ProfileScope) {
  useStore($activeProfile)
  useStore($activeConnectionId)

  return useQuery({
    queryKey: routingQueryKey(scope),
    queryFn: async (): Promise<DelegationRouting | null> => {
      try {
        return await getDelegationRouting(scope)
      } catch (error) {
        if (isMissingEndpoint(error)) {
          return null
        }

        throw error
      }
    },
    retry: false,
    // Plugin installs and toggles elsewhere (Plugins page, CLI) change which backends exist: re-read on mount.
    staleTime: 0
  })
}

export const isPluginMode = (name: string): name is (typeof PLUGIN_MODES)[number] =>
  (PLUGIN_MODES as readonly string[]).includes(name)

/** Every mode the picker lists: the three it edits, each other registered backend (a third-party plugin's), then
 *  a saved backend nothing registers, so the saved choice is always on screen even when it cannot run. */
export function modeChoices(routing: DelegationRouting): string[] {
  const known = new Set<string>(MODES)
  const names = [...MODES, ...routing.backends.filter(b => !b.builtin && !known.has(b.name)).map(b => b.name)]

  return names.includes(routing.configured) ? names : [...names, routing.configured]
}

/** Whether a mode can be chosen for this owner: `default` always, a plugin mode once its backend can run. */
export function modeAvailable(routing: DelegationRouting, mode: string): boolean {
  return routing.backends.some(backend => backend.name === mode && backend.available)
}

/** Any mode but Default decides each subagent's model itself (a task with no route runs on the main agent's
 *  model), so `delegation.model` / `delegation.provider` never apply then, routes or not. */
export function routesOwnSubagentModel(routing: DelegationRouting): boolean {
  return routing.configured !== 'default'
}

// Writes for one owner run one at a time, in the order the user made them: each body is computed from the page
// as it stood after the previous edit, so the server must apply them in that order too.
const saveChains = new Map<string, Promise<unknown>>()
const latestWrite = new Map<string, number>()

/** Paint `patch` now, persist with `write` after this owner's earlier writes, and adopt the server's answer only
 *  when no newer edit is queued behind it (an older answer would undo that edit on screen). A failed write
 *  re-reads the owner's routing instead of restoring a snapshot, which would also undo edits saved since. */
async function writeThrough(
  scope: ProfileScope,
  patch: (current: DelegationRouting) => DelegationRouting,
  write: () => Promise<Partial<DelegationRouting>>,
  failMessage: string
): Promise<boolean> {
  const key = routingQueryKey(scope)
  const id = JSON.stringify(key)

  if (!queryClient.getQueryData<DelegationRouting>(key)) {
    return false
  }

  queryClient.setQueryData<DelegationRouting>(key, current => current && patch(current))

  const generation = (latestWrite.get(id) ?? 0) + 1
  latestWrite.set(id, generation)
  const run = (saveChains.get(id) ?? Promise.resolve()).catch(() => undefined).then(write)
  saveChains.set(id, run)

  try {
    const saved = await run

    if (latestWrite.get(id) === generation) {
      queryClient.setQueryData<DelegationRouting>(key, current => (current ? { ...current, ...saved } : current))
    }

    return true
  } catch (err) {
    notifyError(err, failMessage)
    await saveChains.get(id)?.catch(() => undefined)
    await queryClient.invalidateQueries({ queryKey: key })

    return false
  }
}

/** Only a mode that can run is offered, and the server refuses any other: a saved choice is never blocked. */
export function chooseDelegationMode(scope: ProfileScope, mode: string, failMessage: string) {
  return writeThrough(
    scope,
    current => ({ ...current, configured: mode, blocked_reason: null }),
    async () => {
      const { backend } = await selectDelegationBackend(mode, scope)

      return { configured: backend, blocked_reason: null }
    },
    failMessage
  )
}

export function saveModelRoutes(scope: ProfileScope, next: DelegationModelMap, failMessage: string) {
  return writeThrough(
    scope,
    current => ({ ...current, model_map: next }),
    async () => ({ model_map: (await saveDelegationModelRoutes(next, scope)).model_map }),
    failMessage
  )
}

export function saveProfileRoutes(scope: ProfileScope, next: DelegationProfileRoutes, failMessage: string) {
  return writeThrough(
    scope,
    current => ({ ...current, profile_routes: next }),
    async () => ({ profile_routes: (await saveDelegationProfileRoutes(next, scope)).profile_routes }),
    failMessage
  )
}

export function refreshDelegationRouting(scope: ProfileScope) {
  return queryClient.invalidateQueries({ queryKey: routingQueryKey(scope) })
}
