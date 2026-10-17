import { Codecs, persistentAtom } from '@/lib/persisted'

interface StatusDrawerScope {
  connectionId: string | null
  profile: string
  targetProfile: string
  sessionId: string
}

const MAX_SAVED_DRAWERS = 512

function decodeScope(key: string): StatusDrawerScope | null {
  try {
    const value: unknown = JSON.parse(key)

    if (
      !Array.isArray(value) ||
      value.length !== 4 ||
      (value[0] !== null && typeof value[0] !== 'string') ||
      !value.slice(1).every(part => typeof part === 'string' && part.length > 0)
    ) {
      return null
    }

    const [connectionId, profile, targetProfile, sessionId] = value as [string | null, string, string, string]

    return { connectionId, profile, targetProfile, sessionId }
  } catch {
    return null
  }
}

export function statusDrawerKey(scope: StatusDrawerScope): string {
  return JSON.stringify([scope.connectionId, scope.profile, scope.targetProfile, scope.sessionId])
}

/** Only hidden drawers need an entry; new conversations keep the existing open default. */
export const $collapsedStatusDrawers = persistentAtom<string[]>(
  'hermes.desktop.collapsedStatusDrawers.v1',
  [],
  Codecs.json(value =>
    Array.isArray(value)
      ? [...new Set(value.filter((key): key is string => typeof key === 'string' && decodeScope(key) !== null))].slice(
          -MAX_SAVED_DRAWERS
        )
      : []
  )
)

export function setStatusDrawerCollapsed(key: string, collapsed: boolean): void {
  const next = $collapsedStatusDrawers.get().filter(saved => saved !== key)

  if (collapsed) {
    next.push(key)
  }

  $collapsedStatusDrawers.set(next.slice(-MAX_SAVED_DRAWERS))
}

export function migrateStatusDrawersForProfile(from: string, to: string): void {
  const renamed = $collapsedStatusDrawers.get().map(key => {
    const scope = decodeScope(key)

    if (!scope || (scope.connectionId !== null && scope.connectionId !== 'local')) {
      return key
    }

    return statusDrawerKey({
      ...scope,
      profile: scope.profile === from ? to : scope.profile,
      targetProfile: scope.targetProfile === from ? to : scope.targetProfile
    })
  })

  $collapsedStatusDrawers.set([...new Set(renamed)])
}

export function dropStatusDrawersForProfile(
  profile: string,
  route?: { connectionId?: string; profile?: string; targetProfile?: string }
): void {
  $collapsedStatusDrawers.set(
    $collapsedStatusDrawers.get().filter(key => {
      const scope = decodeScope(key)

      if (!scope) {
        return false
      }

      const matches = route
        ? scope.profile === route.profile?.trim() &&
          (!route.targetProfile || scope.targetProfile === route.targetProfile.trim()) &&
          (!route.connectionId || scope.connectionId === route.connectionId.trim())
        : (scope.connectionId === null || scope.connectionId === 'local') &&
          (scope.profile === profile || scope.targetProfile === profile)

      return !matches
    })
  )
}
