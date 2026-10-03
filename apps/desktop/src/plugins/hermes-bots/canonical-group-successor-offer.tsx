import { Button, cn, gatewayActivationEpoch, host, Switch } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { moveCanonicalGroup } from './canonical-group-registry'
import { desktopComputers, successionAdvertised } from './canonical-group-succession'
import { canonicalPeerGroupEligibility, readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'
import { durableGroupChatMembers } from './group-membership'
import { useBots } from './i18n'
import type { GroupMember, RosterRow } from './types'

type Surface = Awaited<ReturnType<typeof readGroupExecutionMode>>
type Words = ReturnType<typeof useBots>['succession']

/** What each computer a new group would span reports (`groups.capabilities`): who runs it, whether it stays on, and
 * whether it can join a group hosted elsewhere. Only the computers of the Bots you chose are asked. */
function useSurfaces(connectionIds: string[], enabled: boolean) {
  const [surfaces, setSurfaces] = useState<Record<string, Surface>>({})
  const key = JSON.stringify([...new Set(connectionIds)].sort())

  useEffect(() => {
    let current = true

    if (!enabled) {return}
    void Promise.all((JSON.parse(key) as string[]).map(connectionId => readGroupExecutionMode({ connectionId, profile: 'default' },
      gatewayActivationEpoch()).then(surface => [connectionId, surface] as const))).then(entries => {if (current) {setSurfaces(Object.fromEntries(entries))}})

    return () => {current = false}
  }, [key, enabled])

  return surfaces
}

/** Whose a computer is, from the operator names both sides report. Only two names that are set say anything: equal is
 * yours, different is someone else's, and a name missing on either side means it isn't known. */
export function ownership(mine: string | null | undefined, theirs: string | null | undefined): 'own' | 'guest' | 'unknown' {
  return mine && theirs ? mine === theirs ? 'own' : 'guest' : 'unknown'
}

/** What another computer keeping the group's whole history means, said before it's added: whose computer it is when both
 * names are known, or just which computer when they aren't. */
export function historyNotices(words: Words, computers: { label: string; owner: ReturnType<typeof ownership>; person?: string | null }[]) {
  return [...new Set(computers.flatMap(computer => computer.owner === 'guest' && computer.person ? [words.guestHistory(computer.person)]
    : computer.owner === 'unknown' ? [words.unknownHistory(computer.label)] : []))]
}

export function HistoryNotices({ notices }: { notices: string[] }) {
  if (!notices.length) {return null}

  return <div className="grid gap-1 text-sm text-(--ui-text-secondary)" data-slot="guest-history">
    {notices.map(notice => <p key={notice}>{notice}</p>)}
  </div>
}

/** `own`: run by the same person as the current connection. Only those are preselected; one whose owner isn't known is only
 * suggested. */
interface HostChoice { id: string; label: string; alwaysOn: boolean; owner: ReturnType<typeof ownership>; eligible: boolean; reason: string | null }

const placeOf = (member: GroupMember, current: string) => member.route?.connectionId ?? member.connectionId ?? current
const profileOf = (member: GroupMember) => member.route?.targetProfile ?? member.targetProfile ?? member.name

/** Each computer the group could run on. The current connection always can, as before; another one only when every Bot
 * elsewhere could join it: on the default profile, from a computer that can join groups hosted elsewhere. */
function hostChoices(words: Words, current: string, members: GroupMember[], surfaces: Record<string, Surface>, labels: Record<string, string>) {
  const computers = [...new Set([current, ...members.map(member => placeOf(member, current))])]
  const label = (id: string) => labels[id] ?? id

  return computers.map((id): HostChoice => {
    const others = computers.filter(other => other !== id && members.some(member => placeOf(member, current) === other))
    const named = others.find(other => members.some(member => placeOf(member, current) === other && profileOf(member) !== 'default'))
    const unready = others.find(other => !surfaces[other]?.peer)

    const reason = id === current || !surfaces[id] ? null : surfaces[id].mode !== 'canonical' ? words.hostUnavailable(label(id))
      : named ? words.hostProfileBlocked(label(named)) : unready ? words.hostPeerNotReady(label(unready))
      : canonicalPeerGroupEligibility({ connectionId: id, profile: 'default' }, members) ? null : words.hostUnavailable(label(id))

    return { id, label: label(id), alwaysOn: surfaces[id]?.alwaysOn === true,
      owner: id === current ? 'own' : ownership(surfaces[current]?.operatorName, surfaces[id]?.operatorName),
      eligible: id === current || !!surfaces[id] && !reason, reason }
  })
}

const onMac = () => typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || '')

/** Creating a group with Bots on your other computers: they keep a full copy, and with this on (the default) they
 * also allow and are designated to continue the group, in one step. Offered only when the host can designate them.
 * Where the group runs is a choice too: an always-on computer is preselected, so the group keeps running when this
 * computer sleeps. Without one, or with all Bots on one computer, the group runs on the current connection as before. */
export function useSuccessorOffer({ open, connectionId, profile, peerEligible, eligible, selected }: {
  open: boolean; connectionId: string | null; profile: string; peerEligible: boolean; eligible: boolean; selected: RosterRow[]
}) {
  const words = useBots().succession
  // Only a group with Bots on other computers has computers that could continue it, or someone else's computer.
  const peerSelection = peerEligible && !eligible
  const members = durableGroupChatMembers(selected)
  const current = connectionId ?? ''
  const computers = [...new Set([current, ...members.map(member => placeOf(member, current))])].filter(Boolean)
  const surfaces = useSurfaces(computers, open && peerSelection)
  const [enabled, setEnabled] = useState(true)
  const [chosen, setChosen] = useState<string | null>(null)
  const [support, setSupport] = useState<{ source: string; designates: boolean; labels: Record<string, string> } | null>(null)
  const source = JSON.stringify([connectionId, profile])

  useEffect(() => {if (open) {setEnabled(true); setChosen(null)}}, [open, source])

  useEffect(() => {
    let current = true

    if (!open || !connectionId || !peerSelection) {return}
    void Promise.all([readGroupExecutionMode({ connectionId, profile }, gatewayActivationEpoch()), desktopComputers()]).then(([surface, found]) => {
      if (current) {
        setSupport({ source, labels: Object.fromEntries(found.map(computer => [computer.connectionId, computer.label])),
          designates: !!surface.methods?.includes('groups.custody.designate') && successionAdvertised(surface.methods) })
      }
    })

    return () => {current = false}
  }, [open, connectionId, profile, source, peerSelection])

  const offered = peerSelection && support?.source === source && support.designates
  const choices = peerSelection && connectionId ? hostChoices(words, connectionId, members, surfaces, support?.labels ?? {}) : []
  const shown = choices.some(choice => choice.id !== connectionId && choice.eligible)
  // One of your own always-on computers is preselected. One whose owner isn't known is only suggested, and someone else's
  // computer is never chosen for you.
  const preferred = choices.find(choice => choice.eligible && choice.alwaysOn && choice.owner === 'own')?.id ?? connectionId
  const picked = shown && choices.some(choice => choice.id === chosen && choice.eligible) ? chosen : shown ? preferred : connectionId
  const pickedChoice = choices.find(choice => choice.id === picked)
  const standby = choices.find(choice => choice.eligible && choice.alwaysOn && choice.owner === 'own' && choice.id !== picked)
  const suggested = pickedChoice?.alwaysOn ? undefined : choices.find(choice => choice.eligible && choice.alwaysOn && choice.owner === 'unknown')

  const others = computers.filter(id => id !== connectionId).map(id => ({ label: support?.labels[id] ?? id,
    owner: ownership(connectionId ? surfaces[connectionId]?.operatorName : undefined, surfaces[id]?.operatorName), person: surfaces[id]?.operatorName }))

  // The route the room is created on: the chosen computer's default profile, or the current connection unchanged.
  const home = (route: CanonicalGroupRoute): CanonicalGroupRoute => shown && picked && picked !== route.connectionId
    ? { connectionId: picked, profile: 'default' } : route

  return {
    offered,
    enabled,
    setEnabled,
    host: connectionId ? support?.labels[connectionId] ?? null : null,
    hosts: { shown, choices, picked, pick: setChosen, line: !shown || !pickedChoice ? null : pickedChoice.alwaysOn
      ? words.hostedAlwaysOn(pickedChoice.label, onMac()) : standby ? words.hostSleeps(pickedChoice.label, standby.label) : null,
      suggestion: shown && suggested ? { id: suggested.id, text: words.hostTip(suggested.label, onMac()) } : null },
    /** What the other computers keeping the group's history means: whose they are, or which ones when that isn't known. */
    notices: peerSelection && connectionId && Object.keys(surfaces).length ? historyNotices(words, others) : [],
    /** What the create request asks for. */
    requested: offered && enabled,
    home,

    /** The group exists either way: a designation that didn't land is said once, never rolled back. A group created on
     * another computer stays listed where you created it, bound to the computer that hosts it. */
    report(created: { room: CanonicalRoom; successors?: 'designated' | 'failed' }, route: CanonicalGroupRoute) {
      const at = home(route)

      if (at !== route) {moveCanonicalGroup({ ...route, roomId: created.room.room_id }, { ...at, roomId: created.room.room_id })}

      if (created.successors === 'failed') {host.notify({ kind: 'info', message: words.createdWithoutSuccessors(created.room.name) })}
    }
  }
}

type Offer = ReturnType<typeof useSuccessorOffer>

/** Where the group runs, when the Bots span computers and another one could host it. */
function HostPicker({ hosts, disabled }: { hosts: Offer['hosts']; disabled: boolean }) {
  const words = useBots().succession

  if (!hosts.shown) {return null}

  return <fieldset className="grid gap-1.5 text-sm text-(--ui-text-secondary)" data-slot="group-host">
    <legend className="mb-1 font-medium text-(--ui-text-primary)">{words.hostLabel}</legend>
    {hosts.choices.map(choice => <label className={cn('flex items-start gap-2', !choice.eligible && 'opacity-60')} key={choice.id}>
      <input checked={hosts.picked === choice.id} className="mt-1" disabled={disabled || !choice.eligible} name="group-host"
        onChange={() => hosts.pick(choice.id)} type="radio" />
      <span className="grid min-w-0">
        <span className="text-(--ui-text-primary)">{choice.label}</span>
        {choice.reason && <span className="text-xs text-(--ui-text-tertiary)">{choice.reason}</span>}
      </span>
    </label>)}
    {hosts.line && <p className="text-xs" data-slot="group-host-line">{hosts.line}</p>}
    {hosts.suggestion && <div><Button className="h-auto whitespace-normal text-left" disabled={disabled} onClick={() => hosts.pick(hosts.suggestion!.id)}
      size="inline" variant="text">{hosts.suggestion.text}</Button></div>}
  </fieldset>
}

export function SuccessorOffer({ offer, disabled }: { offer: Offer; disabled: boolean }) {
  const words = useBots().succession

  return <>
    <HostPicker disabled={disabled} hosts={offer.hosts} />
    <HistoryNotices notices={offer.notices} />
    {offer.offered && <div className="grid gap-2 text-sm text-(--ui-text-secondary)" data-slot="group-successors">
      <p>{words.successorInfo(offer.host)}</p>
      <label className="flex items-center gap-2 text-(--ui-text-primary)">
        <Switch aria-label={words.successorToggle} checked={offer.enabled} disabled={disabled} onCheckedChange={offer.setEnabled} size="xs" />
        {words.successorToggle}
      </label>
    </div>}
  </>
}
