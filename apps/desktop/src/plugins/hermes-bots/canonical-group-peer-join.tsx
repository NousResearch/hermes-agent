import { Button, ConfirmDialog, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { canonicalMemberName } from './canonical-group-activity'
import { type GroupMethods, offersGroupMethod } from './canonical-group-capabilities'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { $peerJoinErrors, canRejoinPeer, PeerJoinError, peerJoinKey, peerJoinText, peerMemberStates, peerRenewalSettled, rejoinPeer, renewPeers, startAgainWithout } from './canonical-group-peers'
import { endCanonicalGroup, registerCanonicalGroup } from './canonical-group-registry'
import type { CanonicalGroupBinding, CanonicalRoom, CanonicalRoomMember, GroupCreationIntent } from './canonical-groups'
import { cachedUnionRoster } from './data'

type RenewalState = { room: CanonicalRoom; driver_status?: { peer_routes?: unknown } } | null

function renewRoom(binding: CanonicalGroupBinding, state: RenewalState, methods: GroupMethods) {
  if (state?.driver_status && offersGroupMethod(methods, 'groups.peer.register')) {
    renewPeers(binding, state.room, state.driver_status.peer_routes, cachedUnionRoster()?.profiles ?? [])
  }
}

/** Keep this open room's members of other gateways seated (`renewPeers`) where its gateway registers them. */
export function usePeerRenewal(binding: CanonicalGroupBinding, state: RenewalState, methods: GroupMethods, visible: boolean) {
  useEffect(() => { if (visible) {renewRoom(binding, state, methods)} }, [binding, methods, state, visible])
}

/** Before a send leaves: start a renewal that is due, then wait for the room's renewals in flight. */
export function peerRenewalBeforeSend(binding: CanonicalGroupBinding, state: RenewalState, methods: GroupMethods) {
  renewRoom(binding, state, methods)

  return peerRenewalSettled(binding)
}

const STATUS_WORDS = { missing: 'peerNotJoined', needs_reauthorization: 'peerRejoin', unavailable: 'peerAway' } as const

// A socket reset unmounts the controls. Keep uncertain attempts until acknowledgement or refusal.
const intents: Record<string, GroupCreationIntent> = {}

interface JoinRowsProps {
  binding: CanonicalGroupBinding
  methods: GroupMethods
  onChanged: () => void
  onReplaced?: (key: string) => void
}

/** Missing and unavailable peers, with Retry where Desktop can invite them and a new group
 *  without the member. A stored unavailable route still belongs to a joined member. */
export function CanonicalPeerJoinRows({ state, ...props }: JoinRowsProps & {
  state: { room: CanonicalRoom; driver_status?: { peer_routes?: unknown } } | null
}) {
  if (!state?.driver_status || !offersGroupMethod(props.methods, 'groups.peer.register')) {return null}

  return <PeerJoinRows {...props} room={state.room} routes={state.driver_status.peer_routes} />
}

function PeerJoinRows({ binding, room, routes, methods, onChanged, onReplaced }: JoinRowsProps & { room: CanonicalRoom; routes: unknown }) {
  const labels = useCanonicalGroupLabels()
  const errors = useValue($peerJoinErrors)
  const [busy, setBusy] = useState(false)
  const [removing, setRemoving] = useState<CanonicalRoomMember | null>(null)
  const away = peerMemberStates(room.members, routes).filter(({ status }) => status !== 'ready')
  const roster = () => cachedUnionRoster()?.profiles ?? []

  const retry = async (member: CanonicalRoomMember) => {
    setBusy(true)

    try {
      await rejoinPeer(binding, room, member, roster())
    } catch {
      // The row says why ($peerJoinErrors).
    } finally {
      setBusy(false)
      onChanged()
    }
  }

  // ConfirmDialog keeps itself open and shows a thrown message.
  const remove = async () => {
    if (!removing) {return}
    const attemptKey = peerJoinKey(binding, removing.member_id)
    const intent = intents[attemptKey]?.payload ? intents[attemptKey] : (intents[attemptKey] = {})

    const next = await startAgainWithout(binding, room, removing.member_id, roster(), labels, intent)
      .finally(() => { if (intents[attemptKey] === intent && !intent.payload) {delete intents[attemptKey]} })

    const key = registerCanonicalGroup({ connectionId: binding.connectionId, profile: binding.profile }, next.room)

    if (next.ended) {endCanonicalGroup(binding)} else {
      host.notify({ kind: 'error', message: next.unjoined.length
        ? labels.peerStartWithoutKept.replace('{reason}', next.unjoined.map(({ member, error }) => peerJoinText(error, canonicalMemberName(member), labels)).join(' ') +
          (next.unjoined.some(({ error }) => error instanceof PeerJoinError && error.reason === 'registration_unconfirmed') ? '' : ' ' + labels.peerStartWithoutRetry))
        : next.disbandError instanceof Error ? next.disbandError.message
        : labels.peerStartWithoutKept.replace('{reason}', labels.disbandUnconfirmed) })
    }

    setRemoving(null)
    onReplaced?.(key)
  }

  if (!away.length) {return null}

  return <div className="px-4 py-2"><div className="mx-auto grid w-full max-w-3xl gap-2" data-peer-join="">
    {away.map(({ member, status }) => {
      const name = canonicalMemberName(member)
      const error = errors[peerJoinKey(binding, member.member_id)]
      const words = labels[STATUS_WORDS[status as keyof typeof STATUS_WORDS] ?? 'peerAway']
      const canRetry = status === 'missing' || canRejoinPeer(binding, member, roster())

      return <div className="grid gap-1 text-xs" data-member-id={member.member_id} key={member.member_id} role="status">
        <p>{words.replace('{name}', name)}</p>
        {error !== undefined && <p className="text-(--ui-text-secondary)">{status === 'missing' ? peerJoinText(error, name, labels) : labels.peerRetryFailed.replace('{name}', name)}</p>}
        <div className="flex flex-wrap gap-2">
          {canRetry && <Button disabled={busy} onClick={() => void retry(member)}>{labels.retry}</Button>}
          {room.members.length > 2 && offersGroupMethod(methods, 'groups.disband') &&
            <Button className="max-w-full whitespace-normal text-start" disabled={busy} onClick={() => setRemoving(member)}>{labels.peerStartWithout.replace('{name}', name)}</Button>}
        </div>
      </div>
    })}
    {removing && <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={labels.startGatewayGroup} description={labels.peerStartWithoutConfirm}
      destructive onClose={() => setRemoving(null)}
      onConfirm={remove} open title={labels.peerStartWithout.replace('{name}', canonicalMemberName(removing))} />}
  </div></div>
}
