import { registry } from '@/contrib/registry'
import type { Contribution } from '@/contrib/types'

export const COMPOSER_BEHAVIOR_VERSION = 2
export const COMPOSER_BEHAVIOR_AREA = 'composer.behavior.v2'
export interface BehaviorContext {
  connectionId: string
  profile: string
  sessionId: string
  storedId: string
  busy: boolean
}
export interface BehaviorKey {
  key: string
  busy: boolean
  ctrlKey?: boolean
  metaKey?: boolean
  altKey?: boolean
  shiftKey?: boolean
}
export type BehaviorLifecycle =
  { kind: 'drop'; profile: string; connectionId: string } | { kind: 'rename'; from: string; to: string }
export interface BehaviorController {
  update: (state: { enabled: boolean; empty: boolean }) => void
  decide: (status: 'consumed' | 'dismissed') => null | string
  refresh: () => void
  dispose: () => void
}
export interface BehaviorEnvironment {
  context: BehaviorContext
  publish: (text: null | string) => void
  request: <T>(path: string, body: unknown) => Promise<T>
  onSettled: (listener: () => void) => () => void
}
export interface ComposerBehavior {
  protocol: 2
  create: (environment: BehaviorEnvironment) => BehaviorController
  keyAction?: (event: BehaviorKey) => 'queue' | 'steer' | undefined
  readOnlyKeys?: Record<string, readonly string[]>
  lifecycle?: (event: BehaviorLifecycle) => void
}

export function composerBehavior(contributions: readonly Contribution[] = registry.getArea(COMPOSER_BEHAVIOR_AREA)) {
  const providers = contributions.filter(item => {
    const data = item.data as Partial<ComposerBehavior> | undefined

    return (
      /^plugin:[a-z0-9][a-z0-9_-]*$/.test(item.source || '') &&
      data?.protocol === 2 &&
      typeof data.create === 'function'
    )
  })

  return providers.length === 1 ? (providers[0] as (typeof providers)[number] & { data: ComposerBehavior }) : undefined
}

export function behaviorRequestPath(source: string, path: string): string {
  if (!/^plugin:[a-z0-9][a-z0-9_-]*$/.test(source) || !/^\/[a-zA-Z][a-zA-Z0-9_-]*$/.test(path)) {
    throw new Error('Invalid composer extension namespace/path')
  }

  return `/api/plugins/${source.slice(7)}${path}`
}

export function behaviorKeyAction(event: BehaviorKey): 'queue' | 'steer' | undefined {
  try {
    const result = composerBehavior()?.data.keyAction?.(event)

    return result === 'queue' || result === 'steer' ? result : undefined
  } catch {
    return undefined
  }
}

export function behaviorKeys(id: string, fallback: readonly string[]): readonly string[] {
  const keys = composerBehavior()?.data.readOnlyKeys?.[id]

  return Array.isArray(keys) && keys.length > 0 && keys.every(key => typeof key === 'string') ? keys : fallback
}

export function notifyBehaviorLifecycle(event: BehaviorLifecycle): void {
  try {
    composerBehavior()?.data.lifecycle?.(event)
  } catch {
    /* A plugin cannot break native profile operations. */
  }
}
