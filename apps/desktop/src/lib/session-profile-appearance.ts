import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { requestGatewayForAgent } from '@/store/gateway'
import { $gatewayState } from '@/store/session'
import { knownOwnerForSession } from '@/store/session-states'

export interface ExactSessionOwner {
  connectionId: string
  profile: string
  targetProfile: string
}

export interface SessionProfileAppearance {
  avatarDataUrl: null | string
  displayName: string
  role: null | string
}

type Request = (
  connectionId: string,
  profile: string,
  method: string,
  params: Record<string, unknown>
) => Promise<unknown>

interface ProfileRow {
  display_name?: unknown
  has_avatar?: unknown
  name?: unknown
  title?: unknown
  ui_meta?: unknown
}

const normalized = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

export function normalizeExactSessionOwner(owner: unknown): ExactSessionOwner | null {
  if (!owner || typeof owner !== 'object') {
    return null
  }

  const route = owner as { connectionId?: unknown; profile?: unknown; targetProfile?: unknown }
  const connectionId = normalized(route.connectionId)
  const profile = normalized(route.profile)
  const targetProfile = normalized(route.targetProfile) || profile

  return connectionId && profile && targetProfile ? { connectionId, profile, targetProfile } : null
}

export function exactSessionOwner(sessionId: null | string | undefined): ExactSessionOwner | null {
  return normalizeExactSessionOwner(knownOwnerForSession(sessionId))
}

export function sessionProfileAppearanceKey(owner: ExactSessionOwner): string {
  return JSON.stringify([owner.connectionId, owner.profile, owner.targetProfile, 'avatar'])
}

function roleFor(row: ProfileRow): null | string {
  const direct = normalized(row.title)

  if (direct) {
    return direct
  }

  const uiMeta = row.ui_meta
  const pluginMeta =
    uiMeta && typeof uiMeta === 'object'
      ? (uiMeta as Record<string, unknown>)['hermes-bots']
      : null

  return pluginMeta && typeof pluginMeta === 'object'
    ? normalized((pluginMeta as Record<string, unknown>).title) || null
    : null
}

export function createSessionProfileAppearanceResolver(request: Request) {
  const cache = new Map<string, SessionProfileAppearance>()
  const inflight = new Map<string, Promise<null | SessionProfileAppearance>>()
  const keyEpochs = new Map<string, number>()
  const listeners = new Set<() => void>()
  let globalEpoch = 0

  const notify = () => {
    for (const listener of listeners) {
      listener()
    }
  }

  const epochFor = (key: string): readonly [number, number] => [globalEpoch, keyEpochs.get(key) ?? 0]
  const isCurrent = (key: string, epoch: readonly [number, number]): boolean => {
    const current = epochFor(key)

    return current[0] === epoch[0] && current[1] === epoch[1]
  }
  const invalidateKey = (key: string): void => {
    keyEpochs.set(key, (keyEpochs.get(key) ?? 0) + 1)
    cache.delete(key)
    inflight.delete(key)
  }

  const resolve = async (
    owner: ExactSessionOwner,
    options: { revalidate?: boolean } = {}
  ): Promise<null | SessionProfileAppearance> => {
    const key = sessionProfileAppearanceKey(owner)
    const pending = inflight.get(key)

    if (pending) {
      return pending
    }

    if (!options.revalidate && cache.has(key)) {
      return cache.get(key) ?? null
    }

    const epoch = epochFor(key)
    let task!: Promise<null | SessionProfileAppearance>

    task = (async () => {
      try {
        const result = (await request(owner.connectionId, owner.profile, 'profiles.list', {})) as {
          profiles?: ProfileRow[]
        }

        if (!isCurrent(key, epoch)) {
          return null
        }

        const row = (Array.isArray(result?.profiles) ? result.profiles : []).find(
          candidate => normalized(candidate?.name) === owner.targetProfile
        )

        if (!row) {
          cache.delete(key)
          notify()
          return null
        }

        let avatarDataUrl: null | string = null

        if (row.has_avatar === true) {
          const asset = (await request(owner.connectionId, owner.profile, 'profiles.get_asset', {
            asset: 'avatar',
            name: owner.targetProfile
          })) as { data?: unknown; found?: unknown }

          if (!isCurrent(key, epoch)) {
            return null
          }

          if (asset?.found === true && normalized(asset.data).startsWith('data:image/')) {
            avatarDataUrl = normalized(asset.data)
          }
        }

        const appearance: SessionProfileAppearance = {
          avatarDataUrl,
          displayName: normalized(row.display_name) || owner.targetProfile,
          role: roleFor(row)
        }

        if (!isCurrent(key, epoch)) {
          return null
        }

        cache.set(key, appearance)
        notify()
        return appearance
      } catch {
        if (isCurrent(key, epoch)) {
          cache.delete(key)
          notify()
        }

        return null
      } finally {
        if (inflight.get(key) === task) {
          inflight.delete(key)
        }
      }
    })()

    inflight.set(key, task)
    return task
  }

  return {
    clear() {
      globalEpoch += 1
      cache.clear()
      inflight.clear()
      keyEpochs.clear()
      notify()
    },
    invalidateOwner(owner: ExactSessionOwner) {
      invalidateKey(sessionProfileAppearanceKey(owner))
      notify()
    },
    peek(owner: ExactSessionOwner) {
      return cache.get(sessionProfileAppearanceKey(owner)) ?? null
    },
    purgeConnection(connectionId: string) {
      const keys = new Set([...cache.keys(), ...inflight.keys()])

      for (const key of keys) {
        const parsed = JSON.parse(key) as string[]

        if (parsed[0] === connectionId) {
          invalidateKey(key)
        }
      }

      notify()
    },
    resolve,
    subscribe(listener: () => void) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    }
  }
}

export const sessionProfileAppearanceResolver = createSessionProfileAppearanceResolver(
  (connectionId, profile, method, params) => requestGatewayForAgent(connectionId, profile, method, params)
)

export function shouldPresentSessionAppearance(
  connectionState: string,
  validatedKey: string,
  ownerKey: string
): boolean {
  return connectionState !== 'open' || (ownerKey !== '' && validatedKey === ownerKey)
}

let lifecycleBound = false

function bindConnectionLifecycle(): void {
  if (lifecycleBound || typeof window === 'undefined') {
    return
  }

  const onChanged = window.hermesDesktop?.connections?.onChanged

  if (!onChanged) {
    return
  }

  lifecycleBound = true
  onChanged(change => {
    const connectionId = normalized(change?.connectionId)

    if (!connectionId) {
      return
    }

    sessionProfileAppearanceResolver.purgeConnection(connectionId)
  })
}

export function useSessionProfileAppearance(sessionId: null | string): {
  appearance: null | SessionProfileAppearance
  owner: ExactSessionOwner | null
} {
  const owner = useMemo(() => exactSessionOwner(sessionId), [sessionId])
  const key = owner ? sessionProfileAppearanceKey(owner) : ''
  const connectionState = useStore($gatewayState)
  const [validatedKey, setValidatedKey] = useState('')
  const [, setRevision] = useState(0)

  useEffect(() => {
    bindConnectionLifecycle()
    return sessionProfileAppearanceResolver.subscribe(() => setRevision(value => value + 1))
  }, [])

  useEffect(() => {
    let current = true

    if (!owner || connectionState !== 'open') {
      setValidatedKey('')

      return () => {
        current = false
      }
    }

    setValidatedKey('')
    void sessionProfileAppearanceResolver.resolve(owner, { revalidate: true }).then(() => {
      if (current) {
        setValidatedKey(key)
      }
    })

    return () => {
      current = false
    }
  }, [connectionState, key])

  const presentationReady = shouldPresentSessionAppearance(connectionState, validatedKey, key)

  return {
    appearance: owner && presentationReady ? sessionProfileAppearanceResolver.peek(owner) : null,
    owner
  }
}
