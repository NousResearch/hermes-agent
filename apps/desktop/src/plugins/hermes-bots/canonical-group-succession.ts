import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'

import { canonicalGroupRequest, readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupRoute } from './canonical-groups'

/** Continuing a group on another computer (`groups.succession.*`). The gateways own every decision:
 * Desktop reads their state, words it, and offers only the actions they advertise. */

export const SUCCESSION_POLL_MS = 15_000
export const MOVING_POLL_MS = 2_000
export const CONSENT_CONFIRM_MS = 30_000

const STATES = ['ok', 'host_unreachable', 'host_restarting', 'moving', 'continued_on_two', 'moved_away'] as const
const READINESS = ['caught_up', 'behind', 'offline', 'unknown', 'unsupported', 'needs_reauthorization'] as const
const STEPS = ['fencing', 'catching_up', 'reconciling', 'finishing'] as const
const UNAVAILABLE = ['not_owner', 'no_successor', 'successor_behind_offline', 'host_reachable'] as const

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
}
export interface SuccessionWork { completed: number; elsewhere: number; unknown: number; waiting_for_host: number }
export interface SuccessionBot { member_id: string; name: string | null }
export interface SuccessionStatus {
  state: SuccessionState
  host: SuccessionComputer & { reachable: boolean; since: number | null }
  this_install: SuccessionComputer & { role: 'host' | 'backup' | 'member' | 'none' }
  owner: { name: string | null }
  backups: SuccessionBackup[]
  at_risk: number
  moving: { to: SuccessionComputer; step: MoveStep | null } | null
  conflict: SuccessionComputer[]
  moved: { to: SuccessionComputer; branch_id: string | null; separate_events: number } | null
  work: SuccessionWork | null
  actions: { action: string; targets: string[] }[]
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
  behind_by: number
  work: SuccessionWork | null
  unavailable_bots: SuccessionBot[]
  cautions: SuccessionCaution[]
}

type Json = Record<string, unknown>
const record = (value: unknown): Json | null => value && typeof value === 'object' && !Array.isArray(value) ? value as Json : null
const oneOf = <T extends string>(values: readonly T[], value: unknown): T | null => values.includes(value as T) ? value as T : null
const count = (value: unknown) => Number.isSafeInteger(value) && (value as number) > 0 ? value as number : 0
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

    return typeof bot?.member_id === 'string' && bot.member_id ? [{ member_id: bot.member_id, name: displayLabel(bot.name) }] : []
  }) : []
}

function backup(value: unknown): SuccessionBackup[] {
  const item = record(value), base = computer(value)

  if (!item || !base) {return []}
  const allowed = item.allowed === true, designated = item.designated === true

  return [{ ...base, allowed, designated, successor: item.successor === true && allowed && designated,
    kind: item.kind === 'backup' ? 'backup' : 'member', operator_name: displayLabel(item.operator_name),
    readiness: oneOf(READINESS, item.readiness) ?? 'unknown', behind_by: count(item.behind_by), last_seen: seconds(item.last_seen) }]
}

function actions(value: unknown) {
  return Array.isArray(value) ? value.flatMap(item => {
    const entry = record(item)

    if (typeof entry?.action !== 'string') {return []}
    const targets = [...Array.isArray(entry.targets) ? entry.targets : [], entry.target].filter((id): id is string => typeof id === 'string' && !!id)

    return [{ action: entry.action, targets }]
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

  return {
    state,
    host: { ...hostComputer, reachable: hostRecord.reachable === true, since: seconds(hostRecord.since) },
    this_install: { ...selfComputer ?? { install_id: '', name: null },
      role: oneOf(['host', 'backup', 'member', 'none'] as const, self?.role) ?? 'none' },
    owner: { name: displayLabel(record(item.owner)?.name) },
    backups: Array.isArray(item.backups) ? item.backups.flatMap(backup) : [],
    at_risk: count(record(item.at_risk)?.count),
    moving: movingTo ? { to: movingTo, step: oneOf(STEPS, moving?.step) } : null,
    conflict: Array.isArray(conflict?.hosts) ? conflict.hosts.flatMap(entry => computer(entry) ?? []) : [],
    moved: movedTo ? { to: movedTo, branch_id: typeof moved?.branch_id === 'string' && moved.branch_id ? moved.branch_id : null,
      separate_events: count(moved?.separate_events) } : null,
    work: work(item.work),
    actions: actions(item.actions),
    unavailable_reason: oneOf(UNAVAILABLE, item.unavailable_reason),
    previous_host: previousComputer ? { ...previousComputer, offline_since: seconds(previous?.offline_since) } : null,
    unavailable_bots: bots(item.unavailable_bots),
    last_attempt: attemptTarget && typeof attempt?.error === 'string' && attempt.error ? { to: attemptTarget, error: attempt.error } : null
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
