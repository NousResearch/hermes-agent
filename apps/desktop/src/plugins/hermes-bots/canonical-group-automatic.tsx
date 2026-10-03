import { Button, ConfirmDialog, Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, Switch, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useMemo, useRef, useState } from 'react'

import type { CanonicalGroupEvent } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { confirmComputer, desktopComputers, learnSuccession, offeredTargets, offers, readSuccessionStatus, successionAdvertised }
  from './canonical-group-succession'
import type { SuccessionComputer, SuccessionStatus } from './canonical-group-succession'
import type { SuccessionController } from './canonical-group-succession-state'
import { computerName, readinessItem } from './canonical-group-succession-view'
import { canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoomMember } from './canonical-groups'
import { useBots } from './i18n'
import { getPluginCtx } from './shared'

type Words = ReturnType<typeof useBots>['succession']

export const momentLabel = (seconds: number | null | undefined, locale: string | undefined) => {
  if (!seconds) {return null}
  const at = new Date(seconds * 1000)
  const today = at.toDateString() === new Date().toDateString()

  return new Intl.DateTimeFormat(locale, today ? { hour: 'numeric', minute: '2-digit' }
    : { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(at)
}

const list = (locale: string | undefined, names: string[]) => new Intl.ListFormat(locale, { type: 'conjunction' }).format(names)

/** The readiness line everyone sees, under "If a computer goes offline". A computer without a name is numbered. */
function readinessLine(words: Words, controller: SuccessionController, status: SuccessionStatus, host: string | null, locale: string | undefined) {
  const automatic = status.automatic
  const standby = computerName(controller, automatic?.standby)
  const offline = (automatic?.offline ?? []).map((computer, index) => computerName(controller, computer) ?? words.computerNumber(index + 1))

  switch (automatic?.state) {
    case 'ready': return automatic.mode === 'ask' ? words.automaticOff
      : automatic.mode === 'careful' ? words.readyCareful(host, standby) : words.readyMajority(host, standby)

    case 'not_ready': return words.notReady(list(locale, offline), offline.length, host)

    case 'unavailable': return words.needsComputers(host, Math.max(automatic.needed, 1))

    case 'off': return words.automaticOff

    default: return null
  }
}

/** The owner's planned move, confirmed: a handover with nothing at risk. One move at a time: a second click while the
 * first runs does nothing. */
function MoveConfirm({ controller, group, host, target, onClose }: {
  controller: SuccessionController; group: string; host: string | null; target: SuccessionComputer | null; onClose: () => void
}) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const inFlight = useRef(false)
  const name = computerName(controller, target)

  return <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={words.moveConfirm(name)} description={words.moveBody(host)}
    onClose={onClose} onConfirm={async () => {
      if (!target || inFlight.current) {return}
      inFlight.current = true

      try {await controller.move(target.install_id)} catch {throw new Error(words.continueFailed(name, words.errorGeneric))}
      finally {inFlight.current = false}
    }} open={!!target} title={words.moveTitle(group, name)} />
}

/** After a move, Bots left on a computer that answers again can take part once the group moves back there: offered to the
 * owner, as the same planned move. */
export function MoveBackNudge({ controller, group, members }: { controller: SuccessionController; group: string; members: CanonicalRoomMember[] }) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()
  const [target, setTarget] = useState<SuccessionComputer | null>(null)
  const status = controller.status
  const moves = offeredTargets(status, 'move')
  const back = new Map<string, { computer: SuccessionComputer; names: string[] }>()

  for (const bot of status?.unavailable_bots ?? []) {
    if (!bot.on?.reachable || !moves.includes(bot.on.install_id)) {continue}
    const entry = back.get(bot.on.install_id) ?? { computer: bot.on, names: [] }
    entry.names.push(bot.name ?? members.find(member => member.member_id === bot.member_id)?.display_name ?? labels.unknownBot)
    back.set(bot.on.install_id, entry)
  }

  if (!status || !back.size) {return null}

  return <>
    {[...back.values()].map(({ computer, names }) => <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)"
      data-slot="group-move-back" key={computer.install_id} role="status">
      <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center gap-2 px-4 py-2.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
        <p className="min-w-0 flex-1">{words.moveBackNudge(list(locale, names), names.length, computerName(controller, computer))}</p>
        <Button onClick={() => setTarget(computer)} size="sm" variant="secondary">{words.moveBack}</Button>
      </div>
    </div>)}
    <MoveConfirm controller={controller} group={group} host={computerName(controller, status.host)} onClose={() => setTarget(null)} target={target} />
  </>
}

/** Group info, "If a computer goes offline": readiness for everyone; the owner's switch and planned move from `actions`. */
export function AutomaticSection({ controller, group, onAddBackup }: { controller: SuccessionController; group: string; onAddBackup?: () => void }) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()
  const [moving, setMoving] = useState(false)
  const [target, setTarget] = useState<SuccessionComputer | null>(null)
  const [failed, setFailed] = useState(false)
  const status = controller.status

  if (!status?.automatic) {return null}
  const host = computerName(controller, status.host)
  const setting = status.actions.find(entry => entry.action === 'automatic')
  // A change waits for the other computers to store it: the switch shows what was asked, and says it's waiting.
  const committed = status.automatic.enabled ?? setting?.enabled !== false
  const requested = status.automatic.pending ?? committed
  const moves = offeredTargets(status, 'move')
  const needs = status.automatic.state === 'unavailable' && offers(status, 'add_backup') && onAddBackup
  const computer = (id: string) => status.backups.find(backup => backup.install_id === id) ?? { install_id: id, name: null }

  return <section aria-label={words.offlineHeading} className="grid gap-1.5" data-slot="group-automatic">
    <p className="font-medium text-(--ui-text-primary)">{words.offlineHeading}</p>
    <p>{readinessLine(words, controller, status, host, locale)}</p>
    {status.automatic.state === 'ready' && status.automatic.mode === 'careful' && <p className="text-(--ui-text-tertiary)">{words.carefulHelp}</p>}
    {setting && <label className="flex items-center gap-2 pt-1">
      <Switch aria-label={words.automaticSwitch} checked={requested} onCheckedChange={enabled => {
        setFailed(false)
        void controller.setAutomatic(enabled).catch(() => setFailed(true))
      }} size="xs" />
      {words.automaticSwitch}
    </label>}
    {setting && <p className="text-(--ui-text-tertiary)">{requested === committed ? words.automaticHelp : requested ? words.turningOn : words.turningOff}</p>}
    {failed && <p className="text-destructive" role="alert">{words.changeFailed}</p>}
    {needs && <div><Button onClick={onAddBackup} size="sm" variant="secondary">{words.addBackup}</Button></div>}
    {!!moves.length && <div><Button onClick={() => setMoving(true)} size="sm" variant="secondary">{words.moveToAnother}</Button></div>}
    <Dialog onOpenChange={open => {if (!open) {setMoving(false)}}} open={moving && !target}>
      <DialogContent className="max-w-sm">
        <DialogHeader><DialogTitle>{words.moveToAnother}</DialogTitle><DialogDescription>{words.moveBody(host)}</DialogDescription></DialogHeader>
        <ul className="grid gap-1" data-slot="move-targets">
          {moves.map(id => <li key={id}><Button className="w-full justify-start" onClick={() => setTarget(computer(id))} size="sm" variant="ghost">
            {readinessItem(words, status.backups.find(backup => backup.install_id === id), computerName(controller, computer(id)))}
          </Button></li>)}
        </ul>
      </DialogContent>
    </Dialog>
    <MoveConfirm controller={controller} group={group} host={host} onClose={() => {setTarget(null); setMoving(false)}} target={target} />
  </section>
}

/** Derived per computer row, only where automatic moves are reported. */
export function automaticSublabel(words: Words, status: SuccessionStatus | null, alwaysOn: boolean, successor: boolean) {
  return status?.automatic && successor ? alwaysOn ? words.takesOverAutomatically : words.continuesWhenYouChoose : null
}

const NOTICES_KEY = 'hermes.desktop.canonicalGroupNotices.v1'

function readNotices(): Record<string, { dismissed?: string[]; notified?: string[] }> {
  try {return JSON.parse(window.localStorage.getItem(NOTICES_KEY) || '{}') ?? {}} catch {return {}}
}

function markNotice(roomId: string, field: 'dismissed' | 'notified', id: string) {
  try {
    const notices = readNotices()
    const room = notices[roomId] ?? {}
    room[field] = [...(room[field] ?? []).slice(-19), id]
    window.localStorage.setItem(NOTICES_KEY, JSON.stringify({ ...notices, [roomId]: room }))
  } catch {/* The notice may show again; nothing else depends on this hint. */}
}

/** OS notification for something the owner must decide; Desktop shows it only while the user is away. */
function notifyOs(title: string, body: string) {
  try {void getPluginCtx()?.os?.notify?.({ title, body })} catch {/* Best effort. */}
}

/** After a careful move (`proof_kind: "evidence"`): the owner gets one warning per transition, with its choices, until
 * acknowledged. Who decides is read from `actions`; everyone else sees an info line while the host reports the move. */
export function CarefulMoveWarning({ controller, roomId, events, group }: {
  controller: SuccessionController; roomId: string; events: CanonicalGroupEvent[]; group: string
}) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()
  const [dismissed, setDismissed] = useState<string | null>(null)
  const [goingBack, setGoingBack] = useState(false)
  const [failed, setFailed] = useState(false)
  const status = controller.status

  const latest = useMemo(() => [...events].reverse().find(event => event.kind === 'authority.transition'), [events])
  const evidence = latest?.payload.proof_kind === 'evidence' && latest.event_id ? latest : null
  const seen = evidence ? readNotices()[roomId] : undefined
  const to = evidence ? evidence.payload.to_name ?? computerName(controller, status?.host) : null
  const from = evidence ? evidence.payload.from_name ?? computerName(controller, status?.previous_host) : null
  const fromInstall = status?.previous_host?.install_id
  const goBack = !!fromInstall && offeredTargets(status, 'keep').includes(fromInstall)
  const askFirst = offers(status, 'automatic')
  const owner = goBack || askFirst
  const shown = evidence && owner && dismissed !== evidence.event_id && !seen?.dismissed?.includes(evidence.event_id!)

  useEffect(() => {
    if (!shown || !evidence?.event_id || readNotices()[roomId]?.notified?.includes(evidence.event_id)) {return}
    markNotice(roomId, 'notified', evidence.event_id)
    notifyOs(words.carefulTitle(group, to), words.carefulBody(from, to))
  }, [shown, evidence, roomId, group, to, from, words])

  if (evidence && status && !owner && status.moved_in?.proof_kind === 'evidence') {
    return <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)" data-slot="group-careful-info" role="status">
      <div className="mx-auto grid w-full max-w-3xl gap-1 px-4 py-2.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
        <p className="text-sm font-medium text-(--ui-text-primary)">{words.carefulTitle(group, to)}</p>
        <p>{words.carefulBody(from, to)}</p>
      </div>
    </div>
  }

  if (!shown || !evidence?.event_id) {return null}
  const id = evidence.event_id
  const time = momentLabel(evidence.created_at, locale)

  const acknowledge = () => {markNotice(roomId, 'dismissed', id); setDismissed(id)}

  return <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)" data-slot="group-careful-warning" role="alert">
    <div className="mx-auto grid w-full max-w-3xl gap-1.5 px-4 py-2.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
      <p className="text-sm font-medium text-(--ui-text-primary)">{words.carefulTitle(group, to)}</p>
      <p>{words.carefulBody(from, to)}</p>
      <div className="flex flex-wrap items-center gap-2 pt-0.5">
        <Button onClick={acknowledge} size="sm">{words.keepGoing(to)}</Button>
        {goBack && <Button onClick={() => setGoingBack(true)} size="sm" variant="secondary">{words.goBack(from)}</Button>}
        {askFirst && <Button onClick={() => {
          setFailed(false)
          void controller.setAutomatic(false).then(acknowledge).catch(() => setFailed(true))
        }} size="sm" variant="text">{words.askFirst}</Button>}
      </div>
      {failed && <p className="text-destructive">{words.changeFailed}</p>}
    </div>
    <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={words.goBack(from)} description={words.goBackBody(to, from, time)}
      onClose={() => setGoingBack(false)} onConfirm={async () => {
        if (!fromInstall) {return}

        try {await controller.keep(fromInstall)} catch {throw new Error(words.changeFailed)}
        acknowledge()
      }} open={goingBack} title={words.goBackTitle(from)} />
  </div>
}

interface Split { hosts: SuccessionComputer[]; keep: string[]; other: CanonicalGroupRoute }

async function epochOf(route: CanonicalGroupRoute, roomId: string) {
  const state = await canonicalGroupRequest<{ room?: { authority_epoch?: unknown } }>(route, 'groups.state', { room_id: roomId })
  const epoch = state?.room?.authority_epoch

  return typeof epoch === 'number' && Number.isSafeInteger(epoch) ? epoch : null
}

/** The newer side's transitions after the older epoch, with the configurations they verify against, in log order. */
async function transitionChain(route: CanonicalGroupRoute, roomId: string, olderEpoch: number) {
  const chain: CanonicalGroupEvent[] = []
  let started = false

  for (let since = 0, page = 0; page < 50; page++) {
    const result = await canonicalGroupRequest<{ events?: CanonicalGroupEvent[]; has_more?: boolean }>(route, 'groups.log',
      { room_id: roomId, since_seq: since, limit: 100 })

    const events = result?.events ?? []

    for (const event of events) {
      const epoch = (event.payload as { to_epoch?: unknown }).to_epoch
      started ||= event.kind === 'authority.transition' && typeof epoch === 'number' && epoch > olderEpoch

      if (started && (event.kind === 'authority.transition' || event.kind === 'custody.configured')) {chain.push(event)}
    }

    const next = events.at(-1)?.seq

    if (!result?.has_more || !next || next <= since) {break}
    since = next
  }

  return chain
}

/** Desktop ends splits it can see: two of its connections both host the room. It hands the older host the newer
 * side's chain once per incident, quietly; only a split that remains is shown, with an OS notification. */
export function useSplitEnding({ binding, controller, visible, group }: {
  binding: CanonicalGroupBinding; controller: SuccessionController; visible: boolean; group: string
}) {
  const words = useBots().succession
  const [split, setSplit] = useState<Split | null>(null)
  const handled = useRef(new Set<string>())
  const status = controller.status
  // Only two hosts that both report serving are a split Desktop has to end; one the gateways already know about shows as a conflict.
  const ours = controller.fromBinding && controller.answeredByHost && status?.state === 'ok' ? status : null

  useEffect(() => {
    if (!visible || !ours) {return}
    let stopped = false

    void (async () => {
      const found = await desktopComputers()

      for (const backup of ours.backups) {
        const computer = found.find(entry => entry.installId === backup.install_id && entry.connectionId !== binding.connectionId)
        const confirmed = computer && await confirmComputer(computer).catch(() => null)

        if (stopped) {return}

        if (!confirmed || !successionAdvertised(confirmed.methods)) {continue}
        const theirs = await readSuccessionStatus(confirmed.route, binding.roomId).catch(() => null)

        if (stopped) {return}

        if (theirs?.this_install.role !== 'host' || theirs.state !== 'ok') {continue}
        const [mine, other] = await Promise.all([epochOf(binding, binding.roomId), epochOf(confirmed.route, binding.roomId)]).catch(() => [null, null])
        const key = JSON.stringify([binding.roomId, ...[mine, other].sort()])
        let remains = true

        // Hand the older host the newer chain once; whether it stepped down is read every time.
        if (mine !== null && other !== null && mine !== other) {
          const [newer, older, olderEpoch] = mine > other ? [binding, confirmed.route, other] : [confirmed.route, binding, mine]

          if (!handled.current.has(key)) {
            handled.current.add(key)
            const chain = await transitionChain(newer, binding.roomId, olderEpoch).catch(() => [])

            if (chain.length) {await learnSuccession(older, binding.roomId, chain).catch(() => undefined)}
          }

          remains = (await readSuccessionStatus(older, binding.roomId).catch(() => null))?.this_install.role === 'host'
        }

        if (stopped) {return}
        const hosts = [ours.this_install, theirs.this_install]

        if (remains && !handled.current.has(`shown:${key}`)) {
          handled.current.add(`shown:${key}`)
          notifyOs(words.runningTwiceTitle, words.runningTwiceBody(computerName(controller, hosts[0]) ?? words.computerNumber(1),
            computerName(controller, hosts[1]) ?? words.computerNumber(2), group))
        }

        setSplit(remains ? { hosts, other: confirmed.route,
          keep: [...new Set([...offeredTargets(ours, 'keep'), ...offeredTargets(theirs, 'keep')])] } : null)

        if (!remains) {controller.refresh()}

        return
      }

      if (!stopped) {setSplit(null)}
    })()

    return () => {stopped = true}
    // The controller's methods are rebuilt each render; the status object is what changes the answer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, ours, binding, group, words])

  return ours ? split : null
}

/** "This group is running in two places": loud, owner-only Keep, only when ending it quietly didn't work. */
export function SplitBanner({ controller, split, group }: { controller: SuccessionController; split: Split; group: string }) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const [keeping, setKeeping] = useState<SuccessionComputer | null>(null)
  const named = split.hosts.map((computer, index) => ({ computer, name: computerName(controller, computer) ?? words.computerNumber(index + 1) }))
  const keeper = named.find(entry => entry.computer.install_id === keeping?.install_id)

  return <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)" data-slot="group-split" role="alert">
    <div className="mx-auto grid w-full max-w-3xl gap-1.5 px-4 py-2.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
      <p className="text-sm font-medium text-destructive">{words.runningTwiceTitle}</p>
      <p>{words.runningTwiceBody(named[0].name, named[1].name, group)}</p>
      {!!split.keep.length && <div className="flex flex-wrap items-center gap-2 pt-0.5">
        {named.filter(entry => split.keep.includes(entry.computer.install_id)).map(entry =>
          <Button key={entry.computer.install_id} onClick={() => setKeeping(entry.computer)} size="sm" variant="secondary">{words.keep(entry.name)}</Button>)}
      </div>}
    </div>
    <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={keeper ? words.keep(keeper.name) : ''}
      description={named.find(entry => entry.computer.install_id !== keeping?.install_id)?.name
        ? words.keepBody(named.find(entry => entry.computer.install_id !== keeping?.install_id)!.name) : undefined}
      onClose={() => setKeeping(null)} onConfirm={async () => {if (keeping) {await controller.keep(keeping.install_id)}}} open={!!keeping}
      title={keeper ? words.keepTitle(keeper.name) : ''} />
  </div>
}
