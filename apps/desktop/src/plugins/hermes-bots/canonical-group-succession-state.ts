import { gatewayActivationEpoch } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { CanonicalGroupEvent } from './canonical-group-history'
import { allowSuccessor, confirmComputer, CONSENT_CONFIRM_MS, continueAnyway, designateBackup, desktopComputers, HOSTED_POLL_MS, keepSuccession,
  moveGroup, moveNow, MOVING_BACKOFF_MAX_MS, MOVING_POLL_MS, offers, prepareSuccession, promoteSuccession, readSuccessionStatus, recallBackups,
  rememberBackups, removeBackup, setAutomatic, SUCCESSION_POLL_MS, successionAdvertised, successionFailure } from './canonical-group-succession'
import type { DesktopComputer, SuccessionComputer, SuccessionPreview, SuccessionStatus } from './canonical-group-succession'
import { readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute } from './canonical-groups'

/** Log kinds after which the host's view of the group may have changed. */
const STATE_KINDS = new Set(['succession.state', 'authority.transition', 'custody.configured'])
/** While these show, the host can't take new messages: Send keeps them for later. */
const PAUSED_STATES = new Set(['host_unreachable', 'host_restarting', 'moving'])

export interface SuccessionMoveFailure { target: SuccessionComputer; reason: string; other: SuccessionComputer | null }
export interface PendingSwitch { on: boolean; since: number; error?: boolean }
export interface ContinuedOn { status: SuccessionStatus; preview: SuccessionPreview | null; previousHost: string | null }

interface Reading { status: SuccessionStatus; route: CanonicalGroupRoute; fromBinding: boolean }
/** A move in progress: where to watch it, and where the room goes once it lands. */
interface Moving { target: SuccessionComputer; watch: CanonicalGroupRoute; follow: CanonicalGroupRoute | null; preview: SuccessionPreview | null
  previousHost: string | null }

const defaultRoute = (computer: DesktopComputer) => ({ connectionId: computer.connectionId, profile: 'default' })

interface ReadContext {
  binding: CanonicalGroupBinding
  surface: { methods: string[]; installId?: string }
  stopped: () => boolean
  follow: (route: CanonicalGroupRoute) => void
  onContinued: (continued: ContinuedOn) => void
  setReading: (reading: Reading | null) => void
  setMoving: (moving: Moving | null) => void
  setFailure: (failure: SuccessionMoveFailure) => void
  setComputers: (computers: DesktopComputer[]) => void
}

/** A computer listed with a copy of the room answers as a backup: while its host is up, the room opens there. One that
 * answers as the room's host now (the group moved to it) is where the room goes. */
async function followHost(context: ReadContext, next: SuccessionStatus, hostInstall: string | undefined, found?: DesktopComputer[],
  answered?: CanonicalGroupRoute) {
  if (next.state !== 'ok' || next.host.install_id === hostInstall) {return}

  if (next.this_install.role === 'host') {
    if (answered && !context.stopped()) {context.follow(answered)}

    return
  }

  const host = (found ?? await desktopComputers()).find(entry => entry.installId === next.host.install_id)

  if (host && !context.stopped()) {context.follow(defaultRoute(host))}
}

/** False when the computer watching the move didn't answer. */
async function readMove(context: ReadContext, move: Moving) {
  const next = await readSuccessionStatus(move.watch, context.binding.roomId).catch(() => null)

  if (context.stopped() || !next) {return !!next}

  if (next.state === 'ok' && next.host.install_id === move.target.install_id) {
    context.onContinued({ status: next, preview: move.preview, previousHost: move.previousHost })

    if (move.follow) {context.follow(move.follow)}
    else {context.setMoving(null)}
  } else if (next.state !== 'moving') {
    context.setMoving(null)

    if (next.last_attempt) {context.setFailure({ target: next.last_attempt.to, reason: next.last_attempt.error, other: null })}
  }

  context.setReading({ status: next, route: move.watch, fromBinding: move.watch === context.binding })

  return true
}

/** `hosted` when the room's own connection answered as its host. */
async function readBinding(context: ReadContext) {
  const { binding, surface } = context

  if (!successionAdvertised(surface.methods)) {return}
  const next = await readSuccessionStatus(binding, binding.roomId).catch(() => null)

  if (context.stopped()) {return}

  if (next) {rememberBackups(binding.roomId, next)}
  context.setReading(next && { status: next, route: binding, fromBinding: true })

  if (next) {await followHost(context, next, surface.installId)}

  return next?.this_install.role === 'host' ? 'hosted' : undefined
}

/** The host can't be reached: ask only the computers that held copies of this room, never anything else. */
async function readBackups(context: ReadContext) {
  const { binding, surface } = context
  const known = recallBackups(binding.roomId)
  const found = known ? await desktopComputers() : []

  for (const backup of known?.backups ?? []) {
    const computer = found.find(entry => entry.installId === backup.install_id && entry.connectionId !== binding.connectionId)
    const confirmed = computer && backup.install_id !== known?.host && await confirmComputer(computer, true).catch(() => null)

    if (context.stopped()) {return}

    if (!confirmed || !successionAdvertised(confirmed.methods)) {continue}
    const next = await readSuccessionStatus(confirmed.route, binding.roomId).catch(() => null)

    if (context.stopped()) {return}

    if (!next) {continue}
    context.setComputers(found)
    context.setReading({ status: next, route: confirmed.route, fromBinding: false })
    // Another computer already hosts the room and Desktop reaches it: the room follows it there.
    await followHost(context, next, surface.installId ?? known?.host, found, confirmed.route)

    return
  }

  if (!context.stopped()) {context.setReading(null)}
}

/** One room's continuation state. Reads come from the room's host. When its connection fails they come only
 * from the room's last known backups that Desktop already has connections to, and while moving from where the
 * move is watched. `onMoved` and `onContinued` must keep their identity for the lifetime of the room view.
 * `watch` keeps reading while something here waits for the status, such as a message no other computer holds yet. */
export function useCanonicalGroupSuccession({ binding, visible, hostFailing, events, onMoved, onContinued, watch = false }: {
  binding: CanonicalGroupBinding; visible: boolean; hostFailing: boolean; events: CanonicalGroupEvent[]
  onMoved: (route: CanonicalGroupRoute) => void
  onContinued: (continued: ContinuedOn) => void
  watch?: boolean
}) {
  const [surface, setSurface] = useState<{ methods: string[]; installId?: string; operatorName?: string } | null>(null)
  const [reading, setReading] = useState<Reading | null>(null)
  const [computers, setComputers] = useState<DesktopComputer[]>([])
  const [moving, setMoving] = useState<Moving | null>(null)
  const [failure, setFailure] = useState<SuccessionMoveFailure | null>(null)
  const [switches, setSwitches] = useState<Record<string, PendingSwitch>>({})
  const [tick, setTick] = useState(0)
  const [, setClock] = useState(0)
  const moved = useRef(false)
  const marker = useMemo(() => events.reduce((seq, event) => STATE_KINDS.has(event.kind) ? Math.max(seq, event.seq) : seq, 0), [events])
  const status = reading?.status ?? null
  const settled = !status || status.state === 'ok'
  const hasReading = !!reading

  useEffect(() => {
    let current = true
    void readGroupExecutionMode(binding, gatewayActivationEpoch()).then(result => {
      if (current) {setSurface({ methods: result.methods ?? [], installId: result.installId, operatorName: result.operatorName })}
    })

    return () => {current = false}
  }, [binding])

  useEffect(() => {
    if (!visible || !hasReading && !hostFailing) {return}
    let current = true
    void desktopComputers().then(found => {if (current) {setComputers(found)}})

    return () => {current = false}
  }, [visible, hasReading, hostFailing, tick])

  // `moved` is a one-way latch that this room view already handed the room to another route, not a mirror.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (!visible || !surface) {return}
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let misses = 0
    const interval = moving ? MOVING_POLL_MS : hostFailing || !settled || watch ? SUCCESSION_POLL_MS : 0

    const context: ReadContext = { binding, surface, stopped: () => stopped, onContinued, setReading, setMoving, setFailure, setComputers,
      follow: route => {if (!moved.current) {moved.current = true; onMoved(route)}} }

    const cycle = async () => {
      const reached = await (moving ? readMove(context, moving) : hostFailing ? readBackups(context) : readBinding(context)).catch(() => false)
      misses = reached === false ? misses + 1 : 0
      // While its own host answers, the room is still read now and then: a host that pauses appends nothing to show it.
      const delay = moving ? Math.min(interval * 2 ** misses, MOVING_BACKOFF_MAX_MS) : interval || (reached === 'hosted' ? HOSTED_POLL_MS : 0)

      if (!stopped && delay) {timer = setTimeout(() => void cycle(), delay)}
    }

    void cycle()

    return () => {stopped = true; clearTimeout(timer)}
  }, [visible, surface, hostFailing, moving, settled, watch, marker, tick, binding, onMoved, onContinued])

  // Optimistic switches settle when status agrees. A failed write keeps its error until dismissed.
  useEffect(() => {
    const backups = reading?.status.backups ?? []
    setSwitches(current => {
      const next = Object.fromEntries(Object.entries(current).filter(([installId, pending]) =>
        pending.error || backups.find(backup => backup.install_id === installId)?.successor !== pending.on))

      return Object.keys(next).length === Object.keys(current).length ? current : next
    })
  }, [reading])

  useEffect(() => {
    const waiting = Object.values(switches).filter(pending => !pending.error)

    if (!waiting.length) {return}
    const due = Math.min(...waiting.map(pending => pending.since + CONSENT_CONFIRM_MS)) - Date.now()
    const timer = setTimeout(() => setClock(value => value + 1), Math.max(0, due) + 50)

    return () => clearTimeout(timer)
  }, [switches])

  const refresh = useCallback(() => setTick(value => value + 1), [])
  const computerFor = (installId: string | undefined) => installId ? computers.find(computer => computer.installId === installId) : undefined
  const answeredByHost = status?.this_install.role === 'host'
  const hostRoute = answeredByHost ? reading?.route : undefined

  const follow = (route: CanonicalGroupRoute) => {
    if (!moved.current) {moved.current = true; onMoved(route)}
  }

  return {
    status,
    computers,
    computerFor,
    hostRoute,
    hostInstall: surface?.installId,
    /** Who runs the computer this room is connected to, when its operator set a name. */
    operatorName: surface?.operatorName,
    /** The status came from the room's own route (its host, or a computer listed with a copy). */
    fromBinding: !!reading?.fromBinding,
    answeredByHost,
    /** The host can't take messages right now; Send holds them for when the group resumes. That includes a host that paused
     * itself to stay safe, and the side of a conflict that stopped serving. */
    paused: !!status && (PAUSED_STATES.has(status.state) && !answeredByHost || status.state === 'paused' || answeredByHost &&
      status.state === 'continued_on_two' && !!status.conflict_running_on && status.conflict_running_on.install_id !== status.this_install.install_id),
    moving: moving?.target ?? null,
    failure,
    switches,
    refresh,
    clearFailure: () => setFailure(null),
    ...successionActions({ binding, status, reading, hostRoute, computerFor, refresh, follow, onContinued, setFailure, setMoving,
      setReading, setSwitches })
  }
}

/** Everything the room view can ask of the gateways, each on the computer the contract names. */
function successionActions({ binding, status, reading, hostRoute, computerFor, refresh, follow, onContinued, setFailure, setMoving,
  setReading, setSwitches }: {
  binding: CanonicalGroupBinding; status: SuccessionStatus | null; reading: Reading | null; hostRoute?: CanonicalGroupRoute
  computerFor: (installId: string | undefined) => DesktopComputer | undefined; refresh: () => void
  follow: (route: CanonicalGroupRoute) => void; onContinued: (continued: ContinuedOn) => void
  setFailure: (failure: SuccessionMoveFailure | null) => void; setMoving: (moving: Moving | null) => void
  setReading: (reading: Reading) => void
  setSwitches: (update: (current: Record<string, PendingSwitch>) => Record<string, PendingSwitch>) => void
}) {
  const roomId = binding.roomId

  const watch = (next: SuccessionStatus | null, move: Moving) => {
    if (next?.state === 'ok' && next.host.install_id === move.target.install_id) {
      onContinued({ status: next, preview: move.preview, previousHost: move.previousHost })

      if (move.follow) {follow(move.follow)} else {refresh()}

      return
    }

    setMoving(move)

    if (next) {setReading({ status: next, route: move.watch, fromBinding: move.watch === binding })}
  }

  return {
    /** Every Continue on… item runs on the target computer's own connection. */
    async prepare(installId: string) {
      const target = computerFor(installId)
      const confirmed = target && await confirmComputer(target, true)

      if (!target || !confirmed) {throw Object.assign(new Error('target_not_local'), { code: 4001, data: { reason: 'target_not_local' } })}

      return { target, route: confirmed.route, preview: await prepareSuccession(confirmed.route, roomId, installId) }
    },

    async promote(target: DesktopComputer, route: CanonicalGroupRoute, preview: SuccessionPreview) {
      setFailure(null)
      watch(await promoteSuccession(route, roomId, target.installId, preview.preview_id), { watch: route, follow: route, preview,
        target: { install_id: target.installId, name: preview.target.name ?? target.label }, previousHost: status?.host.name ?? null })
    },

    /** A planned move while the host is up: a handover from the host, watched there until the target hosts. */
    async move(installId: string) {
      if (!hostRoute) {return}
      const known = status?.backups.find(backup => backup.install_id === installId)
      const desktop = computerFor(installId)
      watch(await moveGroup(hostRoute, roomId, installId), { watch: hostRoute, follow: desktop ? defaultRoute(desktop) : null, preview: null,
        target: { install_id: installId, name: known?.name ?? desktop?.label ?? null }, previousHost: status?.host.name ?? null })
    },

    recordFailure(target: SuccessionComputer, error: unknown) {
      const typed = successionFailure(error)
      setFailure({ target, reason: typed?.reason ?? 'unreachable', other: typed?.other ?? null })
    },

    async keep(installId: string) {
      if (!reading) {return}
      await keepSuccession(reading.route, roomId, installId)
      refresh()
    },

    /** A planned move waiting for running turns: the owner moves now, and those turns show as unknown there. */
    async moveNow() {
      if (!hostRoute) {return}
      await moveNow(hostRoute, roomId)
      refresh()
    },

    /** "Paused to stay safe": the owner continues on the host anyway. */
    async continueAnyway() {
      if (!hostRoute) {return}
      await continueAnyway(hostRoute, roomId)
      refresh()
    },

    async setAutomatic(enabled: boolean) {
      if (!hostRoute) {return}
      await setAutomatic(hostRoute, roomId, enabled)
      refresh()
    },

    openOn(installId: string) {
      const target = computerFor(installId)

      if (target) {follow(defaultRoute(target))}
    },

    /** On your own computer, switching on also records its operator's consent there. Someone else's computer
     * only gets the owner's designation, and its switch stays disabled until its operator allows it. */
    async setSuccessor(installId: string, on: boolean) {
      const backup = status?.backups.find(entry => entry.install_id === installId)

      if (!backup || !hostRoute) {return}
      setSwitches(current => ({ ...current, [installId]: { on, since: Date.now() } }))

      try {
        if (on && !backup.allowed) {
          const own = computerFor(installId)
          const confirmed = own && await confirmComputer(own, true)

          if (!confirmed?.methods.includes('groups.custody.allow')) {throw new Error('consent_unavailable')}
          await allowSuccessor(confirmed.route, roomId, true)
        }

        await designateBackup(hostRoute, roomId, installId, on)
        refresh()
      } catch (error) {
        setSwitches(current => ({ ...current, [installId]: { on: backup.successor, since: Date.now(), error: true } }))
        throw error
      }
    },

    async removeBackup(installId: string) {
      if (!hostRoute) {return}
      await removeBackup(hostRoute, roomId, installId)
      refresh()
    },

    /** Grants stay in Electron: the renderer only names the two computers. */
    async addBackup(computer: DesktopComputer) {
      const add = window.hermesDesktop?.roomSetup?.addBackup

      if (!hostRoute || !add || !offers(status, 'add_backup')) {throw new Error('add_backup_unavailable')}

      const result = await add({ home: { connectionId: hostRoute.connectionId, profile: hostRoute.profile }, roomId,
        backup: defaultRoute(computer), successor: true })

      if (!result.ok) {throw Object.assign(new Error(result.reason || 'setup_failed'), { roomSetupReason: result.reason })}
      refresh()
    },

    dismissSwitchError(installId: string) {
      setSwitches(current => Object.fromEntries(Object.entries(current).filter(([id]) => id !== installId)))
    }
  }
}

export type SuccessionController = ReturnType<typeof useCanonicalGroupSuccession>
