import { useEffect, useMemo, useState } from 'react'

import { hermesApiAs, resolveOwnerNow } from '@/api/client'
import { useContributions } from '@/contrib/react/use-contributions'
import {
  type BehaviorController,
  behaviorRequestPath,
  COMPOSER_BEHAVIOR_AREA,
  composerBehavior
} from '@/store/composer-behavior'

interface Options {
  gateway: unknown
  profile?: null | string
  sessionId?: null | string
  storedId?: null | string
  busy: boolean
  enabled: boolean
  empty: boolean
  revision: string
}

const validText = (text: unknown): text is string =>
  typeof text === 'string' &&
  text.trim() === text &&
  text.length > 0 &&
  Array.from(text).length <= 240 &&
  !/[\p{Cc}\p{Cf}\u2028\u2029]/u.test(text)

export function useComposerBehavior({
  gateway,
  profile,
  sessionId,
  storedId,
  busy,
  enabled,
  empty,
  revision
}: Options) {
  const contributions = useContributions(COMPOSER_BEHAVIOR_AREA)
  const provider = useMemo(() => composerBehavior(contributions), [contributions])
  const { connectionId, profile: resolvedProfile } = resolveOwnerNow({ profile })

  const scope = useMemo(
    () => ({
      owner: { connectionId, profile: resolvedProfile },
      gateway,
      sessionId: sessionId || '',
      storedId: storedId || sessionId || '',
      busy,
      revision
    }),
    [gateway, connectionId, resolvedProfile, sessionId, storedId, busy, revision]
  )

  // This is an owned controller lifetime, not a mirror of reactive app state.
  const binding = useMemo(() => ({ scope, provider, controller: null as BehaviorController | null }), [scope, provider])
  const [value, setValue] = useState<{ binding: typeof binding; text: string } | null>(null)

  useEffect(() => {
    if (!provider || !scope.gateway) {
      return
    }

    let disposed = false
    const subscriptions = new Set<() => void>()

    const source = scope.gateway as {
      onEvent?: (
        fn: (event: { type: string; session_id?: string; payload?: { running?: boolean } }) => void
      ) => () => void
    }

    try {
      const controller = provider.data.create({
        context: {
          connectionId: scope.owner.connectionId || 'local',
          profile: scope.owner.profile || 'default',
          sessionId: scope.sessionId,
          storedId: scope.storedId,
          busy: scope.busy
        },
        publish: text => {
          if (!disposed) {
            setValue(validText(text) ? { binding, text } : null)
          }
        },
        request: <T>(path: string, body: unknown) => {
          if (disposed) {
            return Promise.reject(new Error('Composer scope disposed'))
          }

          return hermesApiAs<T>(scope.owner, {
            path: behaviorRequestPath(provider.source || '', path),
            method: 'POST',
            body,
            timeoutMs: 32_000,
            priority: undefined
          })
        },
        onSettled: listener => {
          const off =
            source.onEvent?.(event => {
              if (
                !disposed &&
                event.type === 'session.info' &&
                event.session_id === scope.sessionId &&
                event.payload?.running === false
              ) {
                listener()
              }
            }) || (() => {})

          subscriptions.add(off)

          return () => {
            subscriptions.delete(off)
            off()
          }
        }
      })

      if (
        !controller ||
        !['update', 'decide', 'refresh', 'dispose'].every(
          key => typeof controller[key as keyof BehaviorController] === 'function'
        )
      ) {
        throw new Error('Invalid composer controller')
      }

      binding.controller = controller
    } catch {
      disposed = true
      setValue(null)

      for (const off of subscriptions) {
        off()
      }

      subscriptions.clear()
    }

    return () => {
      disposed = true

      try {
        binding.controller?.dispose()
      } catch {
        /* Extension cleanup cannot break native input. */
      }

      binding.controller = null

      for (const off of subscriptions) {
        off()
      }
    }
  }, [binding, scope, provider])

  useEffect(() => {
    try {
      binding.controller?.update({ enabled, empty })
    } catch {
      setValue(null)
    }
  }, [binding, enabled, empty])

  const text = value?.binding === binding && enabled && empty && !busy ? value.text : null

  return {
    text,
    decide: (status: 'consumed' | 'dismissed'): null | string => {
      if (!text || !binding.controller) {
        return null
      }

      try {
        const result = binding.controller.decide(status)

        return result === text ? result : null
      } catch {
        return null
      }
    },
    refresh: () => {
      try {
        binding.controller?.refresh()
      } catch {
        /* Optional extension only. */
      }
    }
  }
}
