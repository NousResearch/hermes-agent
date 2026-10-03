import { atom } from 'nanostores'

import { $activeGatewayProfile, resolveNewChatOwnerRoute } from '@/store/profile'
import {
  $activeSessionId,
  $connection,
  $currentModel,
  $currentProvider,
  $selectedStoredSessionId,
  $sessions,
  resolveComposerSessionKey
} from '@/store/session'
import { $sessionStates } from '@/store/session-states'

// Daybreak is a choice for a conversation, not a credential. Explicit choices
// stay in memory; without one a conversation follows the profile default
// (`agent.daybreak`), which the gateway applies to eligible models itself.
function draftKey(): string {
  const owner = resolveNewChatOwnerRoute()
  const profile = owner?.profile ?? $activeGatewayProfile.get()
  const connection = owner?.connectionId ?? $connection.get()?.connectionId ?? 'local'

  return `__new_chat__:${connection}:${profile}`
}

// A choice made on a model row that isn't selected yet, per conversation and
// provider::model. Like a speed preset, it applies when that model is selected.
export const $daybreakModelChoices = atom<Record<string, Record<string, boolean>>>({})

/** These model ids require a Daybreak program even when no switch choice is sent. */
export function daybreakOnlyModel(model: string): boolean {
  const slug = model.trim().toLowerCase()

  return (
    slug.startsWith('gpt-daybreak-blue-') || slug.startsWith('gpt-daybreak-red-') || slug.startsWith('gpt-5.6-cyber')
  )
}

export const daybreakKeyFor = (storedSessionId: null | string, runtimeId?: null | string): string =>
  resolveComposerSessionKey(storedSessionId, $sessions.get()) || storedSessionId || runtimeId || draftKey()

function currentModelKey(storedSessionId: null | string, runtimeId?: null | string): string {
  const states = $sessionStates.get()

  const state =
    (runtimeId ? states[runtimeId] : undefined) ??
    Object.values(states).find(
      row =>
        storedSessionId !== null &&
        row.storedSessionId &&
        daybreakKeyFor(row.storedSessionId) === daybreakKeyFor(storedSessionId)
    )

  if (state) {
    return `${state.provider}::${state.model}`
  }

  const primaryKey = daybreakKeyFor($selectedStoredSessionId.get(), $activeSessionId.get())

  return storedSessionId === null || daybreakKeyFor(storedSessionId, runtimeId) === primaryKey
    ? `${$currentProvider.get()}::${$currentModel.get()}`
    : '::'
}

export function daybreakSelectionFor(storedSessionId: null | string, runtimeId?: null | string): boolean | undefined {
  return daybreakModelChoiceFor(storedSessionId, currentModelKey(storedSessionId, runtimeId), runtimeId)
}

export function setDaybreakSelection(
  storedSessionId: null | string,
  enabled: boolean,
  runtimeId?: null | string
): void {
  setDaybreakModelChoice(storedSessionId, currentModelKey(storedSessionId, runtimeId), enabled, runtimeId)
}

/** Forget an ineligible model's choice; other model choices remain independent. */
export function clearDaybreakSelection(
  storedSessionId: null | string,
  runtimeId?: null | string,
  modelKey = currentModelKey(storedSessionId, runtimeId)
): void {
  const choices = $daybreakModelChoices.get()
  const key = daybreakKeyFor(storedSessionId, runtimeId)
  const next = { ...choices }

  for (const id of new Set([key, runtimeId].filter((id): id is string => Boolean(id)))) {
    if (choices[id]?.[modelKey] !== undefined) {
      const { [modelKey]: _removed, ...rest } = choices[id]
      next[id] = rest
    }
  }

  $daybreakModelChoices.set(next)
}

export function daybreakModelChoiceFor(
  storedSessionId: null | string,
  modelKey: string,
  runtimeId?: null | string
): boolean | undefined {
  const choices = $daybreakModelChoices.get()

  return (
    choices[daybreakKeyFor(storedSessionId, runtimeId)]?.[modelKey] ??
    (runtimeId ? choices[runtimeId]?.[modelKey] : undefined)
  )
}

export function setDaybreakModelChoice(
  storedSessionId: null | string,
  modelKey: string,
  enabled: boolean,
  runtimeId?: null | string
): void {
  const choices = $daybreakModelChoices.get()
  const key = daybreakKeyFor(storedSessionId, runtimeId)
  $daybreakModelChoices.set({ ...choices, [key]: { ...choices[key], [modelKey]: enabled } })
}

export function adoptDraftDaybreakSelection(storedSessionId: string, sourceDraftKey: string): void {
  const choices = $daybreakModelChoices.get()

  if (sourceDraftKey in choices) {
    const { [sourceDraftKey]: draftChoices, ...restChoices } = choices
    $daybreakModelChoices.set({ ...restChoices, [storedSessionId]: { ...choices[storedSessionId], ...draftChoices } })
  }
}
