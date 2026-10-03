import { host } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Dispatch, ReactNode, SetStateAction } from 'react'

import type { CanonicalGroupAttachment } from './canonical-group-attachments'
import { CarefulMoveWarning, MoveBackNudge, SplitBanner, useSplitEnding } from './canonical-group-automatic'
import type { CanonicalGroupEvent } from './canonical-group-history'
import { acceptedCanonicalGroupSend, clearUnsavedCanonicalGroupSends, holdCanonicalGroupSend, listUnsavedCanonicalGroupSends,
  reclaimRefusedCanonicalGroupSend, refuseReofferedCanonicalGroupSend, reofferedCanonicalGroupSend, sendOutcome, settleCanonicalGroupSend }
  from './canonical-group-send'
import type { AcceptedCanonicalGroupSend, PreparedCanonicalGroupSend, RecoverableCanonicalGroupSend } from './canonical-group-send'
import { readProtectedSeq, SUCCESSION_POLL_MS } from './canonical-group-succession'
import { type ContinuedOn, useCanonicalGroupSuccession } from './canonical-group-succession-state'
import { CanonicalGroupMovedAway, CanonicalGroupSuccessionBanner, computerName } from './canonical-group-succession-view'
import { canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoomMember } from './canonical-groups'
import { useBots } from './i18n'

interface DriverTask { task_id?: unknown; member_id?: unknown; state?: unknown; resource?: unknown; host_name?: unknown }

/** Journal entries being offered again or handed back, so two views of one group never handle the same message at once. */
const following = new Set<string>()

/** Your messages each room view read, by route, and the ones carried across a move to the route the group moved to. */
const ownSeen = new Map<string, CanonicalGroupEvent[]>()
const carried = new Map<string, { to: string; events: CanonicalGroupEvent[] }>()
const routeKey = (binding: CanonicalGroupBinding) => JSON.stringify([binding.connectionId, binding.profile, binding.roomId])

const own = (event: CanonicalGroupEvent) =>
  (event.actor?.kind === 'user' || event.kind === 'message.user') && event.actor?.id === 'desktop' && !!event.event_id

const carriedTo = (binding: CanonicalGroupBinding) => {
  const entry = carried.get(binding.roomId)

  return entry?.to === routeKey(binding) ? entry.events : []
}

/** When the group moves, the messages you sent that this view showed go along, to be checked against the new host's
 * log. Nothing is sent again: the old host may already have started work for them. */
export function carryOwnMessages(from: CanonicalGroupBinding, to: CanonicalGroupBinding) {
  const events = [...carriedTo(from), ...ownSeen.get(routeKey(from)) ?? []]

  if (events.length) {carried.set(from.roomId, { to: routeKey(to), events: [...new Map(events.map(event => [event.event_id, event])).values()] })}
}

/** After a move, a message the old host held alone goes to the new one with its original identity: the new host
 * recognizes it if its copy already has it. One the new host refuses for good waits to be edited. Messages the running
 * host now reports held by enough computers are done. */
async function followUpUnsaved(binding: CanonicalGroupBinding, records: RecoverableCanonicalGroupSend[]) {
  for (const record of records.filter(record => record.entry.unsaved?.reoffer && !following.has(record.storageKey))) {
    following.add(record.storageKey)

    try {
      const result = await canonicalGroupRequest<unknown>(binding, 'groups.send', record.entry.params)
      await reofferedCanonicalGroupSend(record, acceptedCanonicalGroupSend(result, record.entry, 'Unconfirmed group Send'))
    } catch (error) {
      if (sendOutcome(error) === 'refused') {await refuseReofferedCanonicalGroupSend(record, String((error as { data?: { reason?: unknown } }).data?.reason))}
      else {console.warn('A message could not be offered to the group’s new host yet', error)}
    } finally {following.delete(record.storageKey)}
  }

  if (!records.some(record => !record.entry.unsaved?.reoffer && !record.entry.unsaved?.refused)) {return}
  const covered = await readProtectedSeq(binding, binding.roomId).catch(() => null)

  if (covered !== null) {await clearUnsavedCanonicalGroupSends(binding, covered)}
}

/** What a room view shows because its host can go offline, kept out of the room view itself: the controller, a split
 * Desktop can see, and the lines derived from the latest status. `onMoved` must keep its identity. Messages waiting
 * for another computer are read once the room view `restored` its own journal entry. */
export function useRoomContinuity({ binding, visible, readError, events, roomName, onMoved, restored, composer }: {
  binding: CanonicalGroupBinding; visible: boolean; readError: string; events: CanonicalGroupEvent[]; roomName: string
  onMoved: (route: CanonicalGroupRoute) => void; restored: boolean
  /** Where a refused message goes back to be edited. */
  composer: { setDraft: Dispatch<SetStateAction<string>>; setAttachments: Dispatch<SetStateAction<CanonicalGroupAttachment[]>>; setHint: (hint: string) => void }
}) {
  const words = useBots().succession

  const onContinued = useCallback(({ status, previousHost }: ContinuedOn) => {
    host.notify({ kind: 'success', message: words.continuedSummary(status.host.name, status.work?.unknown ?? 0,
      status.work?.waiting_for_host ?? 0, previousHost) })
  }, [words])

  const [unsaved, setUnsaved] = useState<RecoverableCanonicalGroupSend[]>([])
  const [, setCarriedVersion] = useState(0)
  // After the host refused a message without storing it, it is offered again at most once per status reading.
  const [retryAfter, setRetryAfter] = useState(0)

  const reload = useCallback(() => listUnsavedCanonicalGroupSends(binding)
    .then(next => setUnsaved(current => JSON.stringify(current) === JSON.stringify(next) ? current : next))
    .catch(error => console.warn('Messages waiting for another computer could not be read', error)), [binding])

  const controller = useCanonicalGroupSuccession({ binding, visible, hostFailing: !!readError, events, onMoved, onContinued, watch: unsaved.length > 0 })
  const split = useSplitEnding({ binding, controller, visible, group: roomName })
  const status = controller.status
  const unsavedIds = useMemo(() => new Set<string | undefined>(unsaved.map(record => record.entry.unsaved?.event_id)), [unsaved])
  const logIds = useMemo(() => new Set(events.map(event => event.event_id)), [events])
  const refused = unsaved.find(record => record.entry.unsaved?.refused)

  useEffect(() => {if (restored) {void reload()}}, [restored, reload])

  // Your messages this view shows, for a move that may follow.
  useEffect(() => {ownSeen.set(routeKey(binding), events.filter(own))}, [binding, events])

  // On each reading while messages wait. Only the running host knows what is held where; a copy's view clears nothing.
  useEffect(() => {
    if (status?.state === 'ok' && status.this_install.role === 'host' && unsaved.length) {
      void followUpUnsaved(binding, unsaved).then(reload)
    }
  }, [status, unsaved, binding, reload])

  const previous = status?.previous_host
  const previousName = previous ? computerName(controller, previous) : null
  const hostName = status ? computerName(controller, status.host) : null
  const movedAway = controller.fromBinding && status?.state === 'moved_away'
  // Messages are held whenever the host can't take them, and while the group moves: even a host that still answers has
  // promised the group to another computer and stores nothing. Stop still reaches a host that answers.
  const holding = controller.paused || status?.state === 'moving'

  return {
    controller,
    split,
    /** The host can't take messages right now: Send keeps them durably for when the group resumes. */
    paused: holding,
    /** This computer only keeps a copy now; the group continues on its new host. */
    movedAway,
    /** A message can be written: the host takes it now, or it is kept for when the group resumes. */
    composable: (driverStatus: unknown) => Boolean(driverStatus) || holding,
    /** Stop needs a host that answers here. */
    stoppable: (active: boolean) => active && !controller.paused && !movedAway,
    /** Held messages go out only once the group can take them again. */
    resumable: () => !holding && !movedAway && Date.now() >= retryAfter,
    /** A status explains the room's state, so the generic "unavailable" line isn't needed. */
    explained: !!status,
    unavailable: Object.fromEntries((status?.unavailable_bots ?? []).map(bot => [bot.member_id, bot.on?.reachable
      ? words.unavailableUntilBack(computerName(controller, bot.on)) : words.unavailableWhileOffline(previousName)])),
    unknownTitle: previous ? words.unknownAfterMove(previousName) : undefined,
    pausedHint: holding ? words.pausedComposer : '',

    /** Work waiting for the computer that has its Bot or file. */
    waiting: (tasks: DriverTask[] | undefined) => (tasks ?? []).flatMap(task => task.state === 'waiting_for_host' && typeof task.task_id === 'string' &&
      typeof task.member_id === 'string' ? [{ task_id: task.task_id, member_id: task.member_id,
        text: words.waitingTask(typeof task.host_name === 'string' && task.host_name ? task.host_name : previousName, String(task.resource ?? '')) }] : []),

    computerName: (installId: string) => installId === status?.host.install_id ? computerName(controller, status.host) ?? undefined
      : controller.computerFor(installId)?.label,

    /** Messages the host accepted that no other computer holds yet, by their event id in the log. */
    unsaved: unsavedIds,
    /** Your messages the old host showed that the new host's log doesn't have. Only a tap sends one again. */
    missing: { events: events.length ? carriedTo(binding).filter(event => !logIds.has(event.event_id) && !unsavedIds.has(event.event_id)) : [],
      note: words.didntReach(hostName, previousName) },

    /** "Send again": from now on it is a new message, and the old one is no longer shown as missing. */
    resend: (event: CanonicalGroupEvent) => {
      const entry = carried.get(binding.roomId)

      if (entry) {entry.events = entry.events.filter(candidate => candidate.event_id !== event.event_id)}
      setCarriedVersion(version => version + 1)

      return { text: event.payload.text ?? event.payload.content ?? '', attachments: event.payload.attachments ?? [] }
    },

    /** A message the new host refused for good comes back to the composer as an editable draft, with the reason. Call it
     * when the composer is empty; anything typed meanwhile is kept, and the message goes after it. */
    reclaim: () => {
      const { setDraft, setAttachments, setHint } = composer

      if (!refused || following.has(refused.storageKey)) {return}
      following.add(refused.storageKey)
      void reclaimRefusedCanonicalGroupSend(refused).then(payload => {
        if (!payload) {return}
        const text = String(payload.text ?? '')
        setDraft(current => current.trim() ? `${current}\n\n${text}` : text)
        setAttachments(current => [...current, ...(payload.attachments as CanonicalGroupAttachment[] | undefined) ?? []])
        setHint(words.reofferRefused(hostName, refused.entry.unsaved?.refused ?? ''))
      }).catch(error => console.warn('A refused message could not be handed back yet', error))
        .finally(() => {following.delete(refused.storageKey); void reload()})
    },
    /** The host stored nothing (it paused, or a move waits): the message is held, and the status read again to say why. */
    hold: async (entry: PreparedCanonicalGroupSend) => {
      await holdCanonicalGroupSend(binding, entry)
      setRetryAfter(Date.now() + SUCCESSION_POLL_MS)
      controller.refresh()
    },

    /** Settles an accepted Send; one the host holds alone stays in the journal and is shown as not yet saved. */
    settle: async (entry: PreparedCanonicalGroupSend, accepted: AcceptedCanonicalGroupSend) => {
      await settleCanonicalGroupSend(binding, entry, accepted)

      if (entry.unsaved) {await reload()}
    }
  }
}

export type RoomContinuity = ReturnType<typeof useRoomContinuity>

/** Everything that appears above the transcript because of where the group runs. */
export function RoomContinuityBanners({ continuity, binding, members, group, events, visible }: {
  continuity: RoomContinuity; binding: CanonicalGroupBinding; members: CanonicalRoomMember[]; group: string; events: CanonicalGroupEvent[]
  visible: boolean
}) {
  if (!visible) {return null}

  return <>
    {continuity.split ? <SplitBanner controller={continuity.controller} group={group} split={continuity.split} />
      : <CanonicalGroupSuccessionBanner binding={binding} controller={continuity.controller} group={group} members={members} />}
    <CarefulMoveWarning controller={continuity.controller} events={events} group={group} roomId={binding.roomId} />
    <MoveBackNudge controller={continuity.controller} group={group} members={members} />
  </>
}

/** On the computer the group moved away from, the composer gives way to "Open on {target}". */
export function RoomComposerSlot({ continuity, children }: { continuity: RoomContinuity; children: ReactNode }) {
  return continuity.movedAway ? <CanonicalGroupMovedAway controller={continuity.controller} /> : <>{children}</>
}
