import { gatewayActivationEpoch, host, useGatewayActivationEpoch, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalRoom } from './canonical-groups'

interface PreviewResult {
  room_id: string
  authority_gateway_id: string
  authority_epoch: number
  revision: number
  members: { member_id: string; handle: string }[]
}

export function CanonicalGroupPreview({ binding, draft, room, visible }: {
  binding: CanonicalGroupBinding; draft: string; room?: CanonicalRoom; visible: boolean
}) {
  const labels = useCanonicalGroupLabels()
  const connection = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const epoch = useGatewayActivationEpoch()
  const roster = JSON.stringify(room?.members ?? [])

  const signature = JSON.stringify([binding.connectionId, binding.profile, binding.roomId, visible, connection, profile, gateway,
    epoch, draft, room?.authority_gateway_id, room?.authority_epoch, room?.revision, roster])

  // A → B → A is a new observation, not permission to reuse the old A response.
  const observation = useRef({ signature, revision: 0 })

  if (observation.current.signature !== signature) {
    observation.current = { signature, revision: observation.current.revision + 1 }
  }

  const key = String(observation.current.revision)
  const [result, setResult] = useState<{ key: string; names: string[] } | null>(null)

  useEffect(() => {
    let cancelled = false

    if (!visible || !draft.trim() || new TextEncoder().encode(draft).length > 65536 || !room ||
      !Array.isArray(room.members) || typeof room.revision !== 'number' ||
      typeof room.authority_gateway_id !== 'string' || typeof room.authority_epoch !== 'number' ||
      gateway !== 'open' || connection !== binding.connectionId || profile !== binding.profile) {return}

    const current = () => !cancelled && gatewayActivationEpoch() === epoch &&
      host.state.gateway.get() === 'open' && host.state.connectionId.get() === binding.connectionId &&
      host.state.profile.get() === binding.profile

    const timer = setTimeout(() => {
      void canonicalGroupRequest<{ methods?: unknown; driver?: unknown }>(binding, 'groups.capabilities').then(capabilities => {
        if (!current() || capabilities.driver !== true || !Array.isArray(capabilities.methods) || !capabilities.methods.includes('groups.preview')) {return null}

        return canonicalGroupRequest<PreviewResult>(binding, 'groups.preview', { room_id: binding.roomId, text: draft })
      }).then(value => {
        if (!current() || !value || value.room_id !== binding.roomId || value.authority_gateway_id !== room.authority_gateway_id ||
          value.authority_epoch !== room.authority_epoch || value.revision !== room.revision || !Array.isArray(value.members)) {return}

        const byId = new Map(room.members.map(member => [member.member_id, member]))
        const members = value.members.map(member => byId.get(member.member_id))

        if (members.some((member, i) => !member || member.handle !== value.members[i].handle)) {return}
        setResult({ key, names: members.map(member => member!.display_name || member!.handle) })
      }).catch(() => { /* advisory failure never affects the composer */ })
    }, 250)

    return () => { cancelled = true; clearTimeout(timer) }
  }, [binding, connection, draft, epoch, gateway, key, profile, room, roster, visible])

  if (!visible || !draft.trim() || !result || result.key !== key) {return null}

  return <p aria-live="polite" className="text-xs text-(--ui-text-tertiary)" role="status">
    {labels.firstRound} {result.names.join(', ')}. {labels.firstRoundCaveat}
  </p>
}
