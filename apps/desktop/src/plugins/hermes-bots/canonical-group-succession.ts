import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'

import { canonicalGroupRequest, readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupRoute } from './canonical-groups'

/** Continuing a group on another computer (`groups.succession.*`). The gateways own every decision:
 * Desktop reads their state, words it, and offers only the actions they advertise. */

export const SUCCESSION_POLL_MS = 15_000
/** While the room's own host answers: a host that pauses to stay safe appends nothing, so only a status read shows it. */
export const HOSTED_POLL_MS = 30_000
export const MOVING_POLL_MS = 2_000
/** A move watched on a computer that doesn't answer is read less and less often, up to this. */
export const MOVING_BACKOFF_MAX_MS = 30_000
export const CONSENT_CONFIRM_MS = 30_000

const STATES = ['ok', 'host_unreachable', 'host_restarting', 'moving', 'continued_on_two', 'moved_away', 'paused'] as const
const AUTOMATIC = ['ready', 'not_ready', 'unavailable', 'off'] as const
const MODES = ['majority', 'careful', 'ask'] as const
const READINESS = ['caught_up', 'behind', 'offline', 'unknown', 'unsupported', 'needs_reauthorization'] as const
const STEPS = ['waiting_for_turns', 'fencing', 'catching_up', 'reconciling', 'finishing'] as const
const UNAVAILABLE = ['not_owner', 'no_successor', 'successor_behind_offline', 'host_reachable', 'takeover_waiting'] as const

export type SuccessionState = typeof STATES[number]
export type BackupReadiness = typeof READINESS[number]
export type MoveStep = typeof STEPS[number]

export interface SuccessionComputer { install_id: string; name: string | null }
export interface SuccessionBackup extends SuccessionComputer {
  successor: boolean
  allowed: boolean
  designated: boolean
  kind: 'member' | 'backup'
  operator_name: string | null
  readiness: BackupReadiness
  behind_by: number
  last_seen: number | null
  /** Counts toward the majority that lets the group move by itself. */
  voter: boolean
  /** An always-on computer can take over automatically; a laptop continues when you choose. */
  always_on: boolean
}
/** Whether the group moves by itself if its host goes offline, and why not. */
export interface SuccessionAutomatic {
  state: typeof AUTOMATIC[number]
  mode: typeof MODES[number] | null
  standby: SuccessionComputer | null
  reason: string | null
  /** Voters that are offline; a name may be unknown. */
  offline: SuccessionComputer[]
  needed: number
  /** The owner's setting as the computers have stored it, and as requested while they haven't yet. */
  enabled: boolean | null
  pending: boolean | null
}
export interface SuccessionWork { completed: number; elsewhere: number; unknown: number; waiting_for_host: number }
/** A Bot that can't take part after a move; `on` is the computer it runs on, and whether that computer answers now. */
export interface SuccessionBot { member_id: string; name: string | null; on: (SuccessionComputer & { reachable: boolean }) | null }
export interface SuccessionStatus {
  state: SuccessionState
  host: SuccessionComputer & { reachable: boolean; since: number | null }
  this_install: SuccessionComputer & { role: 'host' | 'backup' | 'member' | 'none' }
  owner: { name: string | null }
  backups: SuccessionBackup[]
  at_risk: number
  /** `running`: turns a planned move waits for (`step: "waiting_for_turns"`). */
  moving: { to: SuccessionComputer; step: MoveStep | null; reason: string | null; running: number } | null
  conflict: SuccessionComputer[]
  /** When two computers both ran the group (Unix seconds), if known. */
  conflict_window: { start: number; end: number } | null
  /** The side of a conflict that keeps serving; the other stopped. */
  conflict_running_on: SuccessionComputer | null
  automatic: SuccessionAutomatic | null
  /** The host stopped itself to stay safe (`state: "paused"`): it can't reach a majority (`lost_majority`, `isolated`), or
   * its connection to the other computers isn't ready (`no_lease_layer`). */
  paused: { reason: string; waiting_for: SuccessionComputer[] } | null
  /** A move into this host, until the old host is a copy again; after a careful move (`evidence`) the owner may go back. */
  moved_in: { from: SuccessionComputer | null; proof_kind: string | null } | null
  moved: { to: SuccessionComputer; branch_id: string | null; separate_events: number } | null
  work: SuccessionWork | null
  /** `turns_off_automatic`: continuing anyway also turns automatic moves off for the group. */
  actions: { action: string; targets: string[]; enabled?: boolean; turns_off_automatic?: boolean }[]
  unavailable_reason: typeof UNAVAILABLE[number] | null
  previous_host: (SuccessionComputer & { offline_since: number | null }) | null
  unavailable_bots: SuccessionBot[]
  last_attempt: { to: SuccessionComputer; error: string } | null
}
export interface SuccessionCaution { code: string; names: string[]; count: number }
export interface SuccessionPreview {
  preview_id: string
  target: SuccessionComputer & { operator_name: string | null }
  owner: { name: string | null }
  /** Messages the target is still catching up from another computer. */
  behind_by: number
  /** Recent messages that only the old host has. */
  at_risk: number
  work: SuccessionWork | null
  unavailable_bots: SuccessionBot[]
  cautions: SuccessionCaution[]
}

type Json = Record<string, unknown>
const record = (value: unknown): Json | null => value && typeof value === 'object' && !Array.isArray(value) ? value as Json : null
const oneOf = <T extends string>(values: readonly T[], value: unknown): T | null => values.includes(value as T) ? value as T : null
const count = (value: unknown) => Number.isSafeInteger(value) && (value as number) > 0 ? value as number : 0
const position = (value: unknown) => Number.isSafeInteger(value) && (value as number) >= 0 ? value as number : null
const seconds = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null

/** Display labels only: bounded, never an identifier substitute. */
export const displayLabel = (value: unknown) => typeof value === 'string' && value.trim() ? value.trim().slice(0, 200) : null

function computer(value: unknown): SuccessionComputer | null {
  const item = record(value)
  const id = item?.install_id

  return typeof id === 'string' && id ? { install_id: id, name: displayLabel(item?.name) } : null
}

function work(value: unknown): SuccessionWork | null {
  const item = record(value)

  return item && { completed: count(item.completed), elsewhere: count(item.elsewhere), unknown: count(item.unknown),
    waiting_for_host: count(item.waiting_for_host) }
}

function bots(value: unknown): SuccessionBot[] {
  return Array.isArray(value) ? value.flatMap(item => {
    const bot = record(item)

    const on = computer(bot?.on)

    return typeof bot?.member_id === 'string' && bot.member_id ? [{ member_id: bot.member_id, name: displayLabel(bot.name),
      on: on && { ...on, reachable: record(bot?.on)?.reachable === true } }] : []
  }) : []
}

function backup(value: unknown): SuccessionBackup[] {
  const item = record(value), base = computer(value)

  if (!item || !base) {return []}
  const allowed = item.allowed === true, designated = item.designated === true

  return [{ ...base, allowed, designated, successor: item.successor === true && allowed && designated,
    kind: item.kind === 'backup' ? 'backup' : 'member', operator_name: displayLabel(item.operator_name),
    readiness: oneOf(READINESS, item.readiness) ?? 'unknown', behind_by: count(item.behind_by), last_seen: seconds(item.last_seen),
    voter: item.voter === true, always_on: item.always_on === true }]
}

function actions(value: unknown) {
  return Array.isArray(value) ? value.flatMap(item => {
    const entry = record(item)

    if (typeof entry?.action !== 'string') {return []}
    const targets = [...Array.isArray(entry.targets) ? entry.targets : [], entry.target].filter((id): id is string => typeof id === 'string' && !!id)

    return [{ action: entry.action, targets, ...typeof entry.enabled === 'boolean' ? { enabled: entry.enabled } : {},
      ...entry.turns_off_automatic === true ? { turns_off_automatic: true } : {} }]
  }) : []
}

/** Unknown states and malformed payloads show nothing rather than a guess. */
export function parseSuccessionStatus(value: unknown): SuccessionStatus | null {
  const item = record(value)
  const state = oneOf(STATES, item?.state)
  const hostRecord = record(item?.host), hostComputer = computer(item?.host)
  const self = record(item?.this_install), selfComputer = computer(item?.this_install)

  if (!item || !state || !hostRecord || !hostComputer) {return null}
  const moving = record(item.moving), moved = record(item.moved), conflict = record(item.conflict)
  const previous = record(item.previous_host), previousComputer = computer(item.previous_host)
  const attempt = record(item.last_attempt), attemptTarget = computer(attempt?.to)
  const movingTo = computer(moving?.to), movedTo = computer(moved?.to)
  const automatic = record(item.automatic), automaticState = oneOf(AUTOMATIC, automatic?.state), paused = record(item.paused)
  const movedIn = record(item.moved_in), flag = (value: unknown) => typeof value === 'boolean' ? value : null
  const start = seconds(conflict?.start), end = seconds(conflict?.end)

  return {
    state,
    host: { ...hostComputer, reachable: hostRecord.reachable === true, since: seconds(hostRecord.since) },
    this_install: { ...selfComputer ?? { install_id: '', name: null },
      role: oneOf(['host', 'backup', 'member', 'none'] as const, self?.role) ?? 'none' },
    owner: { name: displayLabel(record(item.owner)?.name) },
    backups: Array.isArray(item.backups) ? item.backups.flatMap(backup) : [],
    at_risk: count(record(item.at_risk)?.count),
    moving: movingTo ? { to: movingTo, step: oneOf(STEPS, moving?.step), reason: typeof moving?.reason === 'string' ? moving.reason : null,
      running: count(moving?.running) } : null,
    conflict: Array.isArray(conflict?.hosts) ? conflict.hosts.flatMap(entry => computer(entry) ?? []) : [],
    conflict_window: start && end ? { start, end } : null,
    conflict_running_on: computer(conflict?.running_on),
    automatic: automaticState ? { state: automaticState, mode: oneOf(MODES, automatic?.mode), standby: computer(automatic?.standby),
      reason: typeof automatic?.reason === 'string' ? automatic.reason : null,
      offline: Array.isArray(automatic?.offline) ? automatic.offline.flatMap(entry => typeof entry === 'string' ? [{ install_id: '', name: displayLabel(entry) }]
        : computer(entry) ?? []) : [],
      needed: count(automatic?.needed), enabled: flag(automatic?.enabled), pending: flag(automatic?.pending) } : null,
    paused: state === 'paused' ? { reason: typeof paused?.reason === 'string' ? paused.reason : 'lost_majority',
      waiting_for: Array.isArray(paused?.waiting_for) ? paused.waiting_for.flatMap(entry => computer(entry) ?? []) : [] } : null,
    moved: movedTo ? { to: movedTo, branch_id: typeof moved?.branch_id === 'string' && moved.branch_id ? moved.branch_id : null,
      separate_events: count(moved?.separate_events) } : null,
    work: work(item.work),
    actions: actions(item.actions),
    unavailable_reason: oneOf(UNAVAILABLE, item.unavailable_reason),
    previous_host: previousComputer ? { ...previousComputer, offline_since: seconds(previous?.offline_since) } : null,
    unavailable_bots: bots(item.unavailable_bots),
    last_attempt: attemptTarget && typeof attempt?.error === 'string' && attempt.error ? { to: attemptTarget, error: attempt.error } : null,
    moved_in: movedIn ? { from: computer(movedIn.from), proof_kind: typeof movedIn.proof_kind === 'string' ? movedIn.proof_kind : null } : null
  }
}

export function parseSuccessionPreview(value: unknown): SuccessionPreview | null {
  const item = record(value), target = computer(item?.target)

  if (!item || !target || typeof item.preview_id !== 'string' || !item.preview_id) {return null}

  return {
    preview_id: item.preview_id,
    target: { ...target, operator_name: displayLabel(record(item.target)?.operator_name) },
    owner: { name: displayLabel(record(item.owner)?.name) },
    behind_by: count(item.behind_by),
    at_risk: count(record(item.at_risk)?.count),
    work: work(item.work),
    unavailable_bots: bots(item.unavailable_bots),
    cautions: Array.isArray(item.cautions) ? item.cautions.flatMap(entry => {
      const caution = record(entry)

      if (typeof caution?.code !== 'string') {return []}
      const names = Array.isArray(caution.names) ? caution.names.map(displayLabel).filter((name): name is string => !!name) : []

      return [{ code: caution.code, names, count: Math.max(count(caution.count), names.length) }]
    }) : []
  }
}

/** Targets the gateway offers for one action. Owner-only actions are simply absent for everyone else. */
export function offeredTargets(status: SuccessionStatus | null, action: string): string[] {
  return status?.actions.filter(entry => entry.action === action).flatMap(entry => entry.targets) ?? []
}

export const offers = (status: SuccessionStatus | null, action: string) => !!status?.actions.some(entry => entry.action === action)

export const successionAdvertised = (methods: readonly string[] | undefined) => !!methods?.includes('groups.succession.status')

export interface SuccessionFailure { reason: string; other: SuccessionComputer | null; target: SuccessionComputer | null }

/** A typed gateway refusal, or null for transport failures and anything untyped. */
export function successionFailure(error: unknown): SuccessionFailure | null {
  const failure = error as { code?: unknown; data?: { reason?: unknown; other?: unknown; target?: unknown } } | null

  if (failure?.code !== 4001 || typeof failure.data?.reason !== 'string') {return null}

  return { reason: failure.data.reason, other: computer(failure.data.other), target: computer(failure.data.target) }
}

/** `room_not_found` and a computer with no copy or membership are both "no answer here". */
export async function readSuccessionStatus(route: CanonicalGroupRoute, roomId: string): Promise<SuccessionStatus | null> {
  try {
    const status = parseSuccessionStatus(await canonicalGroupRequest<unknown>(route, 'groups.succession.status', { room_id: roomId }))

    return status && status.this_install.role !== 'none' ? status : null
  } catch (error) {
    if (successionFailure(error)?.reason === 'room_not_found') {return null}
    throw error
  }
}

export async function prepareSuccession(route: CanonicalGroupRoute, roomId: string, targetInstallId: string) {
  const preview = parseSuccessionPreview(await canonicalGroupRequest<unknown>(route, 'groups.succession.prepare',
    { room_id: roomId, target_install_id: targetInstallId }))

  if (!preview || preview.target.install_id !== targetInstallId) {throw new Error('Invalid continuation preview')}

  return preview
}

export async function promoteSuccession(route: CanonicalGroupRoute, roomId: string, targetInstallId: string, previewId: string) {
  return parseSuccessionStatus(await canonicalGroupRequest<unknown>(route, 'groups.succession.promote',
    { room_id: roomId, target_install_id: targetInstallId, preview_id: previewId, confirm: true }))
}

export const keepSuccession = (route: CanonicalGroupRoute, roomId: string, installId: string) =>
  canonicalGroupRequest<unknown>(route, 'groups.succession.keep', { room_id: roomId, install_id: installId })

export const readSeparateEvents = (route: CanonicalGroupRoute, roomId: string, branchId: string, afterSeq: number) =>
  canonicalGroupRequest<{ events?: unknown; has_more?: unknown }>(route, 'groups.succession.branch_log',
    { room_id: roomId, branch_id: branchId, after_seq: afterSeq, limit: 100 })

/** Owner designation, on the host. */
export const designateBackup = (hostRoute: CanonicalGroupRoute, roomId: string, installId: string, successor: boolean) =>
  canonicalGroupRequest<unknown>(hostRoute, 'groups.custody.designate', { room_id: roomId, install_id: installId, successor })

/** The computer's own operator consent, on that computer. The host confirms it at its next exchange. */
export async function allowSuccessor(ownRoute: CanonicalGroupRoute, roomId: string, successor: boolean) {
  const result = record(await canonicalGroupRequest<unknown>(ownRoute, 'groups.custody.allow', { room_id: roomId, successor }))

  if (!result || result.allowed !== successor) {throw new Error('Consent was not recorded')}

  return { confirmed: result.confirmed === true }
}

/** The owner's setting, on the host: whether the group may move by itself. */
export async function setAutomatic(hostRoute: CanonicalGroupRoute, roomId: string, enabled: boolean) {
  const result = record(await canonicalGroupRequest<unknown>(hostRoute, 'groups.custody.automatic', { room_id: roomId, enabled }))

  if (result?.automatic !== enabled) {throw new Error('The setting was not recorded')}
}

/** The highest seq a majority of the group's computers holds, from its host (`groups.custody.status`). */
export async function readProtectedSeq(hostRoute: CanonicalGroupRoute, roomId: string) {
  return position(record(await canonicalGroupRequest<unknown>(hostRoute, 'groups.custody.status', { room_id: roomId }))?.protected_seq)
}

/** A planned move while the host is up (a handover): no caution is needed. */
export async function moveGroup(hostRoute: CanonicalGroupRoute, roomId: string, targetInstallId: string) {
  return parseSuccessionStatus(await canonicalGroupRequest<unknown>(hostRoute, 'groups.succession.move', { room_id: roomId, target_install_id: targetInstallId }))
}

/** The owner doesn't wait for running turns: the planned move goes ahead, on the host. */
export const moveNow = (hostRoute: CanonicalGroupRoute, roomId: string) =>
  canonicalGroupRequest<unknown>(hostRoute, 'groups.succession.move_now', { room_id: roomId })

/** The owner overrides "paused to stay safe" on the host. */
export const continueAnyway = (hostRoute: CanonicalGroupRoute, roomId: string) =>
  canonicalGroupRequest<unknown>(hostRoute, 'groups.succession.continue_anyway', { room_id: roomId })

/** Hand an older host the newer side's transitions (and the configurations they verify against), in log order. */
export const learnSuccession = (olderRoute: CanonicalGroupRoute, roomId: string, events: unknown[]) =>
  canonicalGroupRequest<unknown>(olderRoute, 'groups.succession.learn', { room_id: roomId, events })

export const removeBackup = (hostRoute: CanonicalGroupRoute, roomId: string, installId: string) =>
  canonicalGroupRequest<unknown>(hostRoute, 'groups.custody.remove', { room_id: roomId, install_id: installId })

/** A computer Desktop already has a connection to, matched by the installation id Electron holds for it. */
export interface DesktopComputer { connectionId: string; label: string; installId: string }

/** Desktop's own registry only: matching never probes unrelated endpoints or credential scopes. */
export async function desktopComputers(): Promise<DesktopComputer[]> {
  let rows: unknown

  try {rows = await host.connections?.()} catch {return []}

  return Array.isArray(rows) ? rows.flatMap(row => {
    const entry = record(row)
    const id = typeof entry?.installId === 'string' ? entry.installId.trim().toLowerCase() : ''

    return typeof entry?.id === 'string' && /^[0-9a-f]{32}$/.test(id)
      ? [{ connectionId: entry.id, label: displayLabel(entry.label) ?? entry.id, installId: `install:${id}` }] : []
  }) : []
}

/** Route to a matched computer only after it confirms the same installation. Calls there use its default profile. */
export async function confirmComputer(computer: DesktopComputer, refresh = false) {
  const route = { connectionId: computer.connectionId, profile: 'default' }
  const surface = await readGroupExecutionMode(route, gatewayActivationEpoch(), refresh)

  return surface.installId === computer.installId ? { route, methods: surface.methods ?? [] } : null
}

const BACKUPS_KEY = 'hermes.desktop.canonicalGroupBackups.v1'
const REMEMBERED_ROOMS = 100

export interface RememberedBackups { host: string; backups: SuccessionComputer[]; at: number }

function readBackups(): Record<string, RememberedBackups> {
  try {
    const value = record(JSON.parse(window.localStorage.getItem(BACKUPS_KEY) || '{}'))

    return (value ?? {}) as Record<string, RememberedBackups>
  } catch {return {}}
}

/** Per room (room ids are global). Only installation ids and display labels: enough to find the
 * room's other computers when its host can't be reached, never credentials or routes. */
export function rememberBackups(roomId: string, status: SuccessionStatus) {
  if (status.state !== 'ok' || status.this_install.role !== 'host') {return}

  try {
    const rooms = readBackups()
    const backups = status.backups.map(({ install_id, name }) => ({ install_id, name }))
    const previous = rooms[roomId]

    if (previous?.host === status.host.install_id && JSON.stringify(previous.backups) === JSON.stringify(backups)) {return}
    rooms[roomId] = { host: status.host.install_id, backups, at: Date.now() }
    const kept = Object.entries(rooms).sort(([, a], [, b]) => b.at - a.at).slice(0, REMEMBERED_ROOMS)
    window.localStorage.setItem(BACKUPS_KEY, JSON.stringify(Object.fromEntries(kept)))
  } catch {/* A missing hint only means failover can't look for this room's copies. */}
}

export function recallBackups(roomId: string): RememberedBackups | null {
  const entry = record(readBackups()[roomId])

  if (!entry || typeof entry.host !== 'string' || !Array.isArray(entry.backups)) {return null}

  return { host: entry.host, at: count(entry.at), backups: entry.backups.flatMap(item => computer(item) ?? []) }
}
