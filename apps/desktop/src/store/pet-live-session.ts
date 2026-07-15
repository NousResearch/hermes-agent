import { atom } from 'nanostores'

import { clearPetUnread } from './pet'
import { normalizeProfileKey } from './profile-key'
import { profileSessionKey } from './session-identity'

export type PetLiveSessionActivityKind = 'reasoning' | 'tool'
export type PetLiveSessionOutcome = 'done' | 'failed'

export interface PetLiveSessionReply {
  streaming: boolean
  text: string
}

export interface PetLiveSessionSnapshot {
  profile: string
  runtimeSessionId: string
  storedSessionId: string | null
  busy: boolean
  needsInput: boolean
  awaitingResponse: boolean
  turnStartedAt: number | null
  activityKind: PetLiveSessionActivityKind | null
  activityName: string | null
  outcome: PetLiveSessionOutcome | null
  reply: PetLiveSessionReply | null
  replyArmed: boolean
  updatedAt: number
}

export interface PetLiveSessionStateInput {
  profile: string | null | undefined
  runtimeSessionId: string
  storedSessionId: string | null
  busy: boolean
  needsInput: boolean
  awaitingResponse: boolean
  turnStartedAt: number | null
}

export interface PetLiveSessionFocus {
  profile: string | null | undefined
  runtimeSessionId: string
}

export const $petLiveSessions = atom<PetLiveSessionSnapshot[]>([])

let focusedIdentity: { profile: string; runtimeSessionId: string } | null = null
const pendingReplyArms = new Set<string>()
const pendingProfileReplyArms = new Set<string>()

export function armPetLiveSessionReply(profile: string, runtimeSessionId: string): void {
  const sessionId = runtimeSessionId.trim()

  if (sessionId) {
    pendingReplyArms.add(profileSessionKey(profile, sessionId))
  }
}

export function armNextReplyForProfile(profile: string): void {
  pendingProfileReplyArms.add(normalizeProfileKey(profile))
}

export function consumePetReplyArm(profile: string, runtimeSessionId: string): boolean {
  const normalizedProfile = normalizeProfileKey(profile)
  const sessionKey = profileSessionKey(normalizedProfile, runtimeSessionId.trim())

  if (pendingReplyArms.delete(sessionKey)) {
    return true
  }

  return pendingProfileReplyArms.delete(normalizedProfile)
}

export function disarmPetReplyArm(profile: string, runtimeSessionId: string): void {
  const sessionId = runtimeSessionId.trim()

  if (sessionId) {
    pendingReplyArms.delete(profileSessionKey(profile, sessionId))
  }
}

export function disarmNextReplyForProfile(profile: string): void {
  pendingProfileReplyArms.delete(normalizeProfileKey(profile))
}

const boundedActivityName = (value: string | null | undefined): string | null => {
  const trimmed = value?.trim() ?? ''

  return trimmed ? trimmed.slice(0, 120) : null
}

const normalizedFocus = (focus: PetLiveSessionFocus | null | undefined) => {
  const runtimeSessionId = focus?.runtimeSessionId.trim() ?? ''

  return runtimeSessionId
    ? { profile: normalizeProfileKey(focus?.profile), runtimeSessionId }
    : null
}

const isFocused = (profile: string, runtimeSessionId: string): boolean =>
  focusedIdentity?.profile === profile && focusedIdentity.runtimeSessionId === runtimeSessionId

const snapshotKey = (snapshot: Pick<PetLiveSessionSnapshot, 'profile' | 'runtimeSessionId'>): string =>
  profileSessionKey(snapshot.profile, snapshot.runtimeSessionId)

const sameSnapshot = (left: PetLiveSessionSnapshot, right: Omit<PetLiveSessionSnapshot, 'updatedAt'>): boolean =>
  left.profile === right.profile &&
  left.runtimeSessionId === right.runtimeSessionId &&
  left.storedSessionId === right.storedSessionId &&
  left.busy === right.busy &&
  left.needsInput === right.needsInput &&
  left.awaitingResponse === right.awaitingResponse &&
  left.turnStartedAt === right.turnStartedAt &&
  left.activityKind === right.activityKind &&
  left.activityName === right.activityName &&
  left.outcome === right.outcome &&
  left.replyArmed === right.replyArmed &&
  left.reply?.streaming === right.reply?.streaming &&
  left.reply?.text === right.reply?.text

const shouldRetain = (snapshot: Omit<PetLiveSessionSnapshot, 'updatedAt'>): boolean =>
  snapshot.busy || snapshot.needsInput || snapshot.outcome !== null || isFocused(snapshot.profile, snapshot.runtimeSessionId)

function writeSnapshot(next: Omit<PetLiveSessionSnapshot, 'updatedAt'>): void {
  const current = $petLiveSessions.get()
  const key = snapshotKey(next)
  const index = current.findIndex(candidate => snapshotKey(candidate) === key)
  const previous = index >= 0 ? current[index] : undefined

  if (!shouldRetain(next)) {
    if (index < 0) {
      return
    }

    $petLiveSessions.set([...current.slice(0, index), ...current.slice(index + 1)])

    return
  }

  if (previous && sameSnapshot(previous, next)) {
    return
  }

  const stamped: PetLiveSessionSnapshot = { ...next, updatedAt: Date.now() }

  if (index < 0) {
    $petLiveSessions.set([...current, stamped])
  } else {
    const updated = [...current]
    updated[index] = stamped
    $petLiveSessions.set(updated)
  }
}

function existingSnapshot(profile: string | null | undefined, runtimeSessionId: string): PetLiveSessionSnapshot | null {
  const key = profileSessionKey(profile, runtimeSessionId)

  return $petLiveSessions.get().find(candidate => snapshotKey(candidate) === key) ?? null
}

function emptySnapshot(
  profile: string | null | undefined,
  runtimeSessionId: string
): Omit<PetLiveSessionSnapshot, 'updatedAt'> {
  return {
    profile: normalizeProfileKey(profile),
    runtimeSessionId: runtimeSessionId.trim(),
    storedSessionId: null,
    busy: false,
    needsInput: false,
    awaitingResponse: false,
    turnStartedAt: null,
    activityKind: null,
    activityName: null,
    outcome: null,
    reply: null,
    replyArmed: false
  }
}

function startedSnapshot(
  snapshot: Omit<PetLiveSessionSnapshot, 'updatedAt'>,
  storedSessionId?: string | null
): Omit<PetLiveSessionSnapshot, 'updatedAt'> {
  return {
    ...snapshot,
    storedSessionId: storedSessionId?.trim() || snapshot.storedSessionId,
    busy: true,
    awaitingResponse: true,
    needsInput: false,
    turnStartedAt: snapshot.turnStartedAt ?? Date.now(),
    activityKind: null,
    activityName: null,
    outcome: null,
    reply: null,
    replyArmed: false
  }
}

function unstamped(snapshot: PetLiveSessionSnapshot): Omit<PetLiveSessionSnapshot, 'updatedAt'> {
  const { updatedAt: _updatedAt, ...value } = snapshot

  return value
}

export function syncPetLiveSessionState(
  input: PetLiveSessionStateInput,
  focus: PetLiveSessionFocus | null | undefined = focusedIdentity
): void {
  const runtimeSessionId = input.runtimeSessionId.trim()

  if (!runtimeSessionId) {
    return
  }

  focusedIdentity = normalizedFocus(focus)
  const profile = normalizeProfileKey(input.profile)
  const previous = existingSnapshot(profile, runtimeSessionId)
  const busy = Boolean(input.busy)

  writeSnapshot({
    profile,
    runtimeSessionId,
    storedSessionId: input.storedSessionId?.trim() || null,
    busy,
    needsInput: Boolean(input.needsInput),
    awaitingResponse: Boolean(input.awaitingResponse),
    turnStartedAt: input.turnStartedAt,
    activityKind: busy ? (previous?.activityKind ?? null) : null,
    activityName: busy ? (previous?.activityName ?? null) : null,
    outcome: busy ? null : (previous?.outcome ?? null),
    reply: previous?.reply ?? null,
    replyArmed: previous?.replyArmed ?? false
  })
}

const MAX_PET_REPLY_LENGTH = 24_000

function boundedReplyText(value: string): string {
  return value.length <= MAX_PET_REPLY_LENGTH ? value : `${value.slice(0, MAX_PET_REPLY_LENGTH - 1)}…`
}

export function beginPetLiveSessionReply(
  profile: string,
  runtimeSessionId: string,
  storedSessionId?: string | null
): void {
  const previous = existingSnapshot(profile, runtimeSessionId)
  const next = previous ? unstamped(previous) : emptySnapshot(profile, runtimeSessionId)

  writeSnapshot({
    ...startedSnapshot(next, storedSessionId),
    reply: next.reply?.streaming ? next.reply : { streaming: true, text: '' },
    replyArmed: true
  })
}

export function appendPetLiveSessionReply(profile: string, runtimeSessionId: string, delta: string): void {
  const previous = existingSnapshot(profile, runtimeSessionId)

  if (!previous?.reply?.streaming || !delta) {
    return
  }

  writeSnapshot({
    ...previous,
    reply: { streaming: true, text: boundedReplyText(previous.reply.text + delta) }
  })
}

export function completePetLiveSessionReply(profile: string, runtimeSessionId: string, finalText: string): void {
  const previous = existingSnapshot(profile, runtimeSessionId)

  if (!previous?.reply?.streaming) {
    return
  }

  const text = finalText.trim() || previous.reply.text

  writeSnapshot({
    ...previous,
    reply: { streaming: false, text: boundedReplyText(text) },
    replyArmed: false
  })
}

export function failPetLiveSessionReply(profile: string, runtimeSessionId: string): void {
  const previous = existingSnapshot(profile, runtimeSessionId)

  if (!previous?.reply?.streaming) {
    return
  }

  writeSnapshot({
    ...previous,
    reply: { streaming: false, text: previous.reply.text },
    replyArmed: false
  })
}

export function beginPetLiveSession(
  profile: string,
  runtimeSessionId: string,
  storedSessionId?: string | null
): void {
  const previous = existingSnapshot(profile, runtimeSessionId)
  const next = previous ? unstamped(previous) : emptySnapshot(profile, runtimeSessionId)

  writeSnapshot(startedSnapshot(next, storedSessionId))
}

export function replacePetLiveSessionRuntime(
  profile: string,
  previousRuntimeSessionId: string,
  recoveredRuntimeSessionId: string,
  storedSessionId?: string | null
): void {
  const normalizedProfile = normalizeProfileKey(profile)
  const previousRuntime = previousRuntimeSessionId.trim()
  const recoveredRuntime = recoveredRuntimeSessionId.trim()

  if (!previousRuntime || !recoveredRuntime) {
    return
  }

  if (previousRuntime === recoveredRuntime) {
    beginPetLiveSession(normalizedProfile, recoveredRuntime, storedSessionId)

    return
  }

  const current = $petLiveSessions.get()
  const previousKey = profileSessionKey(normalizedProfile, previousRuntime)
  const recoveredKey = profileSessionKey(normalizedProfile, recoveredRuntime)

  const matchesRotatedRuntime = (snapshot: PetLiveSessionSnapshot): boolean => {
    const key = snapshotKey(snapshot)

    return key === previousKey || key === recoveredKey
  }

  const previous = current.find(snapshot => snapshotKey(snapshot) === previousKey)
  const recovered = current.find(snapshot => snapshotKey(snapshot) === recoveredKey)

  const base = recovered
    ? unstamped(recovered)
    : previous
      ? { ...unstamped(previous), runtimeSessionId: recoveredRuntime }
      : emptySnapshot(normalizedProfile, recoveredRuntime)

  const recoveredIsAuthoritative = Boolean(
    recovered && (recovered.busy || recovered.needsInput || recovered.outcome !== null)
  )

  const next = recoveredIsAuthoritative
    ? {
        ...base,
        storedSessionId: storedSessionId?.trim() || base.storedSessionId,
        profile: normalizedProfile,
        runtimeSessionId: recoveredRuntime
      }
    : startedSnapshot(
        { ...base, profile: normalizedProfile, runtimeSessionId: recoveredRuntime },
        storedSessionId
      )

  const firstRotatedIndex = current.findIndex(matchesRotatedRuntime)
  const retained = current.filter(snapshot => !matchesRotatedRuntime(snapshot))

  const insertionIndex =
    firstRotatedIndex < 0
      ? retained.length
      : current.slice(0, firstRotatedIndex).filter(snapshot => !matchesRotatedRuntime(snapshot)).length

  if (isFocused(normalizedProfile, previousRuntime)) {
    focusedIdentity = { profile: normalizedProfile, runtimeSessionId: recoveredRuntime }
  }

  retained.splice(insertionIndex, 0, { ...next, updatedAt: Date.now() })
  $petLiveSessions.set(retained)
}

export function setPetLiveSessionActivity(
  profile: string,
  runtimeSessionId: string,
  kind: PetLiveSessionActivityKind,
  name?: string | null
): void {
  const previous = existingSnapshot(profile, runtimeSessionId)
  const next = previous ?? emptySnapshot(profile, runtimeSessionId)

  writeSnapshot({
    ...next,
    busy: true,
    awaitingResponse: true,
    activityKind: kind,
    activityName: kind === 'tool' ? boundedActivityName(name) : null,
    outcome: null
  })
}

export function clearPetLiveSessionActivity(
  profile: string,
  runtimeSessionId: string,
  kind?: PetLiveSessionActivityKind,
  name?: string | null
): void {
  const previous = existingSnapshot(profile, runtimeSessionId)

  if (!previous || (kind && previous.activityKind !== kind)) {
    return
  }

  const expectedName = kind === 'tool' ? boundedActivityName(name) : null

  if (kind === 'tool' && previous.activityName !== expectedName) {
    return
  }

  writeSnapshot({ ...previous, activityKind: null, activityName: null })
}

export function completePetLiveSession(
  profile: string,
  runtimeSessionId: string,
  outcome: PetLiveSessionOutcome
): void {
  const previous = existingSnapshot(profile, runtimeSessionId)
  const next = previous ?? emptySnapshot(profile, runtimeSessionId)

  writeSnapshot({
    ...next,
    busy: false,
    needsInput: false,
    awaitingResponse: false,
    turnStartedAt: null,
    activityKind: null,
    activityName: null,
    outcome,
    replyArmed: false
  })
}

export function acknowledgePetLiveSession(profile: string, runtimeSessionId: string): boolean {
  const previous = existingSnapshot(profile, runtimeSessionId)

  if (!previous?.outcome) {
    return false
  }

  writeSnapshot({ ...previous, outcome: null })

  if (!$petLiveSessions.get().some(snapshot => snapshot.outcome !== null)) {
    clearPetUnread()
  }

  return true
}

export function reconcilePetLiveSessionFocus(
  focus: PetLiveSessionFocus | null,
  cachedState?: PetLiveSessionStateInput | null
): void {
  focusedIdentity = normalizedFocus(focus)
  const current = $petLiveSessions.get()

  const retained = current.filter(
    snapshot => snapshot.busy || snapshot.needsInput || snapshot.outcome !== null || isFocused(snapshot.profile, snapshot.runtimeSessionId)
  )

  if (retained.length !== current.length) {
    $petLiveSessions.set(retained)
  }

  if (focusedIdentity && cachedState) {
    syncPetLiveSessionState(cachedState, focusedIdentity)
  }
}

export function resetPetLiveSessions(): void {
  focusedIdentity = null
  pendingReplyArms.clear()
  pendingProfileReplyArms.clear()
  $petLiveSessions.set([])
}
