import { gatewayActivationEpoch } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { CanonicalGroupEvent } from './canonical-group-history'
import { allowSuccessor, confirmComputer, CONSENT_CONFIRM_MS, designateBackup, desktopComputers, keepSuccession,
  MOVING_POLL_MS, offers, prepareSuccession, promoteSuccession, readSuccessionStatus, recallBackups, rememberBackups,
  removeBackup, SUCCESSION_POLL_MS, successionAdvertised, successionFailure } from './canonical-group-succession'
import type { DesktopComputer, SuccessionComputer, SuccessionPreview, SuccessionStatus } from './canonical-group-succession'
import { readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute } from './canonical-groups'

/** Log kinds after which the host's view of the group may have changed. */
const STATE_KINDS = new Set(['succession.state', 'authority.transition', 'custody.configured'])
/** While these show, the host can't take new messages: Send keeps them for later. */
const PAUSED_STATES = new Set(['host_unreachable', 'host_restarting', 'moving'])

export interface SuccessionMoveFailure { target: SuccessionComputer; reason: string; other: SuccessionComputer | null }
export interface PendingSwitch { on: boolean; since: number; error?: boolean }
export interface ContinuedOn { status: SuccessionStatus; preview: SuccessionPreview; previousHost: string | null }

interface Reading { status: SuccessionStatus; route: CanonicalGroupRoute; fromBinding: boolean }
interface Moving { target: DesktopComputer; route: CanonicalGroupRoute; preview: SuccessionPreview; previousHost: string | null }

const defaultRoute = (computer: DesktopComputer) => ({ connectionId: computer.connectionId, profile: 'default' })

/** One room's continuation state. Reads come from the room's host. When its connection fails they come only
 * from the room's last known backups that Desktop already has connections to, and while moving from the target.
 * `onMoved` and `onContinued` must keep their identity for the lifetime of the room view. */
export function useCanonicalGroupSuccession({ binding, visible, hostFailing, events, onMoved, onContinued }: {
  binding: CanonicalGroupBinding; visible: boolean; hostFailing: boolean; events: CanonicalGroupEvent[]
  onMoved: (route: CanonicalGroupRoute) => void
  onContinued: (continued: ContinuedOn) => void
}) {
  const [surface, setSurface] = useState<{ methods: string[]; installId?: string } | null>(null)
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
      if (current) {setSurface({ methods: result.methods ?? [], installId: result.installId })}
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
    const interval = moving ? MOVING_POLL_MS : hostFailing || !settled ? SUCCESSION_POLL_MS : 0

    const follow = (route: CanonicalGroupRoute) => {
      if (!moved.current) {moved.current = true; onMoved(route)}
    }

    const fromTarget = async (move: Moving) => {
      const next = await readSuccessionStatus(move.route, binding.roomId).catch(() => null)

      if (stopped || !next) {return}

      if (next.state === 'ok' && next.host.install_id === move.target.installId) {
        onContinued({ status: next, preview: move.preview, previousHost: move.previousHost })
        follow(move.route)

        return
      }

      if (next.state !== 'moving') {
        setMoving(null)

        if (next.last_attempt) {setFailure({ target: next.last_attempt.to, reason: next.last_attempt.error, other: null })}
      }

      setReading({ status: next, route: move.route, fromBinding: false })
    }

    // A computer listed with a copy of the room answers as a backup: while its host is up, the room opens there.
    // A backup answering as the room's host now (the group moved to it) is where the room goes.
    const followHost = async (next: SuccessionStatus, hostInstall: string | undefined, found?: DesktopComputer[], answered?: CanonicalGroupRoute) => {
      if (next.state !== 'ok' || next.host.install_id === hostInstall) {return}

      if (next.this_install.role === 'host') {
        if (answered && !stopped) {follow(answered)}

        return
      }

      const host = (found ?? await desktopComputers()).find(entry => entry.installId === next.host.install_id)

      if (host && !stopped) {follow(defaultRoute(host))}
    }

    const fromBinding = async () => {
      if (!successionAdvertised(surface.methods)) {return}
      const next = await readSuccessionStatus(binding, binding.roomId).catch(() => null)

      if (stopped) {return}

      if (next) {rememberBackups(binding.roomId, next)}
      setReading(next && { status: next, route: binding, fromBinding: true })

      if (next) {await followHost(next, surface.installId)}
    }

    // The host can't be reached: ask only the computers that held copies of this room.
    const fromBackups = async () => {
      const known = recallBackups(binding.roomId)
      const found = known ? await desktopComputers() : []

      for (const backup of known?.backups ?? []) {
        const computer = found.find(entry => entry.installId === backup.install_id && entry.connectionId !== binding.connectionId)
        const confirmed = computer && backup.install_id !== known?.host && await confirmComputer(computer, true).catch(() => null)

        if (stopped) {return}

        if (!confirmed || !successionAdvertised(confirmed.methods)) {continue}
        const next = await readSuccessionStatus(confirmed.route, binding.roomId).catch(() => null)

        if (stopped) {return}

        if (!next) {continue}
        setComputers(found)
        setReading({ status: next, route: confirmed.route, fromBinding: false })
        // Another computer already hosts the room and Desktop reaches it: the room follows it there.
        await followHost(next, surface.installId ?? known?.host, found, confirmed.route)

        return
      }

      if (!stopped) {setReading(null)}
    }

    const cycle = async () => {
      await (moving ? fromTarget(moving) : hostFailing ? fromBackups() : fromBinding()).catch(() => undefined)

      if (!stopped && interval) {timer = setTimeout(() => void cycle(), interval)}
    }

    void cycle()

    return () => {stopped = true; clearTimeout(timer)}
  }, [visible, surface, hostFailing, moving, settled, marker, tick, binding, onMoved, onContinued])

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

  return {
    status,
    computers,
    computerFor,
    hostRoute,
    hostInstall: surface?.installId,
    /** The status came from the room's own route (its host, or a computer listed with a copy). */
    fromBinding: !!reading?.fromBinding,
    answeredByHost,
    /** The host can't take messages right now; Send holds them for when the group resumes. */
    paused: !!status && PAUSED_STATES.has(status.state) && !answeredByHost,
    moving: moving?.target ?? null,
    failure,
    switches,
    refresh,
    clearFailure: () => setFailure(null),

    /** Every Continue on… item runs on the target computer's own connection. */
    async prepare(installId: string) {
      const target = computerFor(installId)
      const confirmed = target && await confirmComputer(target, true)

      if (!target || !confirmed) {throw Object.assign(new Error('target_not_local'), { code: 4001, data: { reason: 'target_not_local' } })}

      return { target, route: confirmed.route, preview: await prepareSuccession(confirmed.route, binding.roomId, installId) }
    },

    async promote(target: DesktopComputer, route: CanonicalGroupRoute, preview: SuccessionPreview) {
      setFailure(null)
      const previousHost = status?.host.name ?? null
      const next = await promoteSuccession(route, binding.roomId, target.installId, preview.preview_id)

      if (next?.state === 'ok' && next.host.install_id === target.installId) {
        onContinued({ status: next, preview, previousHost })

        if (!moved.current) {moved.current = true; onMoved(route)}

        return
      }

      setMoving({ target, route, preview, previousHost })

      if (next) {setReading({ status: next, route, fromBinding: false })}
    },

    recordFailure(target: SuccessionComputer, error: unknown) {
      const typed = successionFailure(error)
      setFailure({ target, reason: typed?.reason ?? 'unreachable', other: typed?.other ?? null })
    },

    async keep(installId: string) {
      if (!reading) {return}
      await keepSuccession(reading.route, binding.roomId, installId)
      refresh()
    },

    openOn(installId: string) {
      const target = computerFor(installId)

      if (target && !moved.current) {moved.current = true; onMoved(defaultRoute(target))}
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
          await allowSuccessor(confirmed.route, binding.roomId, true)
        }

        await designateBackup(hostRoute, binding.roomId, installId, on)
        refresh()
      } catch (error) {
        setSwitches(current => ({ ...current, [installId]: { on: backup.successor, since: Date.now(), error: true } }))
        throw error
      }
    },

    async removeBackup(installId: string) {
      if (!hostRoute) {return}
      await removeBackup(hostRoute, binding.roomId, installId)
      refresh()
    },

    /** Grants stay in Electron: the renderer only names the two computers. */
    async addBackup(computer: DesktopComputer) {
      const add = window.hermesDesktop?.roomSetup?.addBackup

      if (!hostRoute || !add || !offers(status, 'add_backup')) {throw new Error('add_backup_unavailable')}

      const result = await add({ home: { connectionId: hostRoute.connectionId, profile: hostRoute.profile }, roomId: binding.roomId,
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
