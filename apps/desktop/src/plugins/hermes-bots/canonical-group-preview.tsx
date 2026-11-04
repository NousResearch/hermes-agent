import { gatewayActivationEpoch, host, useGatewayActivationEpoch, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalRoom, CanonicalRoomContext } from './canonical-groups'
import { fill } from './group-room-ui'

interface PreviewMember {
  member_id: string
  handle: string
  display_name?: string | null
}

interface PreviewResult {
  room_id: string
  authority_gateway_id: string
  authority_epoch: number
  revision: number
  latest_seq: number
  mode: 'members' | 'lead'
  members: PreviewMember[]
  lead?: PreviewMember | null
}

/** What the last authoritative observation told the composer. `key` fences it to that observation. */
interface Shown {
  key: string
  mode: 'members' | 'lead'
  names: string[]
  lead?: string
}

export function CanonicalGroupPreview({ binding, draft, room, visible, context, latestSeq, thread }: {
  binding: CanonicalGroupBinding; draft: string; room?: CanonicalRoom; visible: boolean
  context?: CanonicalRoomContext | null
  /** The room log tail this composer has observed; the server refuses a preview of any other snapshot. */
  latestSeq: number
  /** The thread a reply sends into. Absent is the new thread a fresh send mints. */
  thread?: string
}) {
  const labels = useCanonicalGroupLabels()
  const connection = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const epoch = useGatewayActivationEpoch()
  const roster = JSON.stringify(room?.members ?? [])
  const routing = context?.routing === 'lead' ? 'lead' : 'everyone'
  const leadMemberId = context?.lead_member_id ?? ''
  const threadId = thread ?? ''

  // Room revision does not move for messages or settings events, so the log tail and the routing
  // fields are part of the observation: a settings save or a new message is a new projection even
  // when the roster and revision are unchanged.
  const signature = JSON.stringify([binding.connectionId, binding.profile, binding.roomId, visible, connection, profile, gateway,
    epoch, draft, room?.authority_gateway_id, room?.authority_epoch, room?.revision, roster, routing, leadMemberId, latestSeq, threadId])

  // A → B → A is a new observation, not permission to reuse the old A response.
  const observation = useRef({ signature, revision: 0 })

  if (observation.current.signature !== signature) {
    observation.current = { signature, revision: observation.current.revision + 1 }
  }

  const key = String(observation.current.revision)
  const [result, setResult] = useState<Shown | null>(null)
  const resultRef = useRef<Shown | null>(null)
  resultRef.current = result
  // One in-flight request per observation. The generation discards anything that a newer observation
  // (or an unmount) has superseded, so a late failure never hides newer data and a late success never
  // overwrites it. Equivalent polling snapshots do not restart work that is already running.
  const pending = useRef<{ key: string; id: number; timer?: ReturnType<typeof setTimeout> } | null>(null)
  const generation = useRef(0)
  const mounted = useRef(true)

  useEffect(() => () => { mounted.current = false; generation.current += 1 }, [])

  const eligible = visible && !!draft.trim() && new TextEncoder().encode(draft).length <= 65536 && !!room &&
    Array.isArray(room.members) && typeof room.revision === 'number' &&
    typeof room.authority_gateway_id === 'string' && typeof room.authority_epoch === 'number' &&
    Number.isSafeInteger(latestSeq) && latestSeq >= 0 &&
    gateway === 'open' && connection === binding.connectionId && profile === binding.profile

  // Request generation, pending token and mount fences — not an atom mirror.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref writes (see eslint rule comment)
  useEffect(() => {
    if (!eligible) {
      generation.current += 1
      if (pending.current?.timer) {clearTimeout(pending.current.timer)}

      pending.current = null
      if (resultRef.current) {setResult(null)}

      return
    }

    // A fresh observation invalidates the shown answer immediately (the render guard hides it by key),
    // while an equivalent room snapshot revalidates the existing answer without clearing it first.
    if (pending.current?.key === key) {return}

    if (pending.current?.timer) {clearTimeout(pending.current.timer)}

    if (resultRef.current && resultRef.current.key !== key) {setResult(null)}

    const id = ++generation.current
    const text = draft
    const expectedSeq = latestSeq
    const expectedThread = threadId
    const snapshot = room
    const timer = setTimeout(() => {
      const current = () => mounted.current && id === generation.current && gatewayActivationEpoch() === epoch &&
        host.state.gateway.get() === 'open' && host.state.connectionId.get() === binding.connectionId &&
        host.state.profile.get() === binding.profile
      const fail = () => { if (current()) {setResult(null)} }
      const settle = () => { if (pending.current?.id === id) {pending.current = null} }

      void canonicalGroupRequest<{ methods?: unknown; driver?: unknown }>(binding, 'groups.capabilities').then(capabilities => {
        if (!current()) {return null}

        if (capabilities.driver !== true || !Array.isArray(capabilities.methods) || !capabilities.methods.includes('groups.preview')) {return null}

        return canonicalGroupRequest<PreviewResult>(binding, 'groups.preview', {
          room_id: binding.roomId, text, latest_seq: expectedSeq, ...(expectedThread ? { thread_id: expectedThread } : {})
        })
      }).then(value => {
        if (!current()) {return}

        if (!value || typeof value !== 'object' || value.room_id !== binding.roomId ||
          value.authority_gateway_id !== snapshot.authority_gateway_id ||
          value.authority_epoch !== snapshot.authority_epoch || value.revision !== snapshot.revision ||
          value.latest_seq !== expectedSeq || !Array.isArray(value.members)) {fail(); return}

        const byId = new Map(snapshot.members.map(member => [member.member_id, member]))
        const named = (member: PreviewMember) => byId.get(member.member_id)

        if (value.mode === 'lead') {
          const lead = value.lead ? named(value.lead) : undefined

          if (!lead || !value.lead || lead.handle !== value.lead.handle || value.members.length) {fail(); return}

          setResult({ key, mode: 'lead', names: [], lead: lead.display_name || lead.handle })
          return
        }

        if (value.mode !== 'members' || !value.members.length) {fail(); return}

        const members = value.members.map(named)

        if (members.some((member, i) => !member || member.handle !== value.members[i].handle)) {fail(); return}

        setResult({ key, mode: 'members', names: members.map(member => member!.display_name || member!.handle) })
      }).catch(fail).finally(settle)
    }, 250)

    pending.current = { key, id, timer }
  }, [binding, connection, draft, eligible, epoch, gateway, key, latestSeq, leadMemberId, profile, room, routing, threadId, visible])

  if (!eligible || !result || result.key !== key) {return null}

  return <p aria-live="polite" className="text-xs text-(--ui-text-tertiary)" role="status">
    {result.mode === 'lead'
      ? fill(labels.firstRoundLead, { lead: result.lead ?? '' })
      : `${labels.firstRound} ${result.names.join(', ')}.`} {labels.firstRoundCaveat}
  </p>
}
