import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { computed } from 'nanostores'
import { useEffect, useRef, useState } from 'react'

import { capabilityScoped, type ProfileScope } from '@/api/client'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { useI18n } from '@/i18n'
import { confirm } from '@/store/confirm'
import { activeGatewayConnectionId, activeGatewayProfileKey } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import { $focusedRuntimeId, $sessionStates, knownOwnerForSession } from '@/store/session-states'

import { type PresetAction, presetRequest } from './preset-api'
import { refreshDelegationRouting, routingQueryKey } from './routing-data'
const $presetChatOwner = computed([$focusedRuntimeId, $sessionStates], runtime => knownOwnerForSession(runtime))

export function useModelPresets(scope: ProfileScope) {
  const { t } = useI18n()
  const p = t.settings.model.subagents.routing.presets
  const { requestGateway } = useGatewayRequest()
  const sessionId = useStore($focusedRuntimeId)
  const owner = useStore($presetChatOwner)
  const [selected, setSelected] = useState('')
  const [editing, setEditing] = useState<'save' | 'rename' | null>(null)
  const [name, setName] = useState('')
  const routeKey = JSON.stringify(routingQueryKey(scope))
  const [dialogRoute, setDialogRoute] = useState(routeKey)
  const currentRoute = useRef(routeKey)
  // eslint-disable-next-line no-restricted-syntax -- scope is a prop, not an atom; bind pending dialogs to their originating route.
  useEffect(() => {
    currentRoute.current = routeKey

    return () => {
      currentRoute.current = ''
    }
  }, [routeKey])
  const [busy, setBusy] = useState(false)
  const [output, setOutput] = useState<{ text: string; target: string } | null>(null)
  const outputTarget = JSON.stringify([routeKey, sessionId, owner])

  function isCurrentChat() {
    return (
      currentRoute.current === routeKey &&
      $focusedRuntimeId.get() === sessionId &&
      $presetChatOwner.get() === owner &&
      owner &&
      typeof owner === 'object' &&
      owner.connectionId === (activeGatewayConnectionId() || 'local') &&
      normalizeProfileKey(owner.profile) === normalizeProfileKey(activeGatewayProfileKey())
    )
  }

  const query = useQuery({
    queryKey: [...routingQueryKey(scope), 'presets'],
    queryFn: async () => {
      try {
        return await presetRequest(scope)
      } catch (error) {
        if (/\b404\b/.test(String(error))) {
          return null
        }

        throw error
      }
    },
    retry: false,
    staleTime: 0
  })

  const state = query.data
  const choice = state?.names.includes(selected) ? selected : state?.active || ''

  const canApply = Boolean(
    sessionId &&
    owner &&
    typeof owner === 'object' &&
    owner.connectionId === (activeGatewayConnectionId() || 'local') &&
    normalizeProfileKey(owner.profile) === normalizeProfileKey(activeGatewayProfileKey()) &&
    owner.connectionId === (capabilityScoped(scope).connectionId || activeGatewayConnectionId() || 'local') &&
    normalizeProfileKey(owner.profile) === normalizeProfileKey(capabilityScoped(scope).profile)
  )

  async function mutate(action: PresetAction) {
    if (currentRoute.current !== routeKey) {
      return
    }

    setBusy(true)
    setOutput(null)

    try {
      await presetRequest(scope, action)

      if (currentRoute.current !== routeKey) {
        return
      }

      await Promise.all([query.refetch(), refreshDelegationRouting(scope)])

      if (currentRoute.current !== routeKey) {
        return
      }

      setSelected(action.action === 'delete' ? '' : action.action === 'rename' ? action.new_name! : action.name)
      setEditing(null)
    } catch (error) {
      notifyError(error, p.failed)
    } finally {
      setBusy(false)
    }
  }

  async function apply() {
    if (!canApply || !choice) {
      return
    }

    const ok = await confirm({ title: p.apply, description: p.warning, confirmLabel: p.apply })

    // A focus/profile change while confirmation was open must not retarget the command.
    if (!ok || !isCurrentChat()) {
      return
    }

    setBusy(true)

    try {
      const result = await requestGateway<{ output?: string; error?: string }>('slash.exec', {
        session_id: sessionId,
        command: `delegation preset ${choice} --now`
      })

      if (!isCurrentChat()) {
        return
      }

      if (result.error) {
        throw new Error(result.error)
      }

      setOutput({ text: result.output || p.applied, target: outputTarget })
      await Promise.all([query.refetch(), refreshDelegationRouting(scope)])
    } catch (error) {
      if (isCurrentChat()) {
        notifyError(error, p.failed)
      }
    } finally {
      setBusy(false)
    }
  }

  return {
    t,
    p,
    query,
    state,
    choice,
    canApply,
    busy,
    output,
    outputTarget,
    setSelected,
    editing,
    setEditing,
    name,
    setName,
    routeKey,
    dialogRoute,
    setDialogRoute,
    mutate,
    apply
  }
}
