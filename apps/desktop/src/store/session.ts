import { atom } from 'nanostores'

import type { ContextSuggestion } from '@/app/types'
import type { HermesConnection } from '@/global'
import type { ChatMessage } from '@/lib/chat-messages'
import type { SessionInfo, UsageStats } from '@/types/hermes'

type Updater<T> = T | ((current: T) => T)

interface AppAtom<T> {
  get: () => T
  set: (value: T) => void
}

function updateAtom<T>(store: AppAtom<T>, next: Updater<T>) {
  store.set(typeof next === 'function' ? (next as (current: T) => T)(store.get()) : next)
}

export const $connection = atom<HermesConnection | null>(null)
export const $gatewayState = atom('idle')
export const $sessions = atom<SessionInfo[]>([])
export const $sessionsTotal = atom<number>(0)
export const $sessionsLoading = atom(true)
export const $workingSessionIds = atom<string[]>([])
export const $activeSessionId = atom<string | null>(null)
export const $selectedStoredSessionId = atom<string | null>(null)
export const $messages = atom<ChatMessage[]>([])
export const $freshDraftReady = atom(false)
export const $busy = atom(false)
export const $awaitingResponse = atom(false)
export const $currentModel = atom('')
export const $currentProvider = atom('')
export const $currentReasoningEffort = atom('')
export const $currentServiceTier = atom('')
export const $currentFastMode = atom(false)
export const $currentCwd = atom('')
export const $currentBranch = atom('')
export const $currentUsage = atom<UsageStats>({
  calls: 0,
  input: 0,
  output: 0,
  total: 0
})
export const $sessionStartedAt = atom<number | null>(null)
export const $turnStartedAt = atom<number | null>(null)
export const $introPersonality = atom('')
export const $currentPersonality = atom('')
export const $availablePersonalities = atom<string[]>([])
export const $introSeed = atom(0)
export const $contextSuggestions = atom<ContextSuggestion[]>([])
export const $modelPickerOpen = atom(false)

export const setConnection = (next: Updater<HermesConnection | null>) => updateAtom($connection, next)
export const setGatewayState = (next: Updater<string>) => updateAtom($gatewayState, next)
export const setSessions = (next: Updater<SessionInfo[]>) => updateAtom($sessions, next)
export const setSessionsTotal = (next: Updater<number>) => updateAtom($sessionsTotal, next)
export const setSessionsLoading = (next: Updater<boolean>) => updateAtom($sessionsLoading, next)
export const setWorkingSessionIds = (next: Updater<string[]>) => updateAtom($workingSessionIds, next)
export const setActiveSessionId = (next: Updater<string | null>) => updateAtom($activeSessionId, next)
export const setSelectedStoredSessionId = (next: Updater<string | null>) => updateAtom($selectedStoredSessionId, next)
export const setMessages = (next: Updater<ChatMessage[]>) => updateAtom($messages, next)
export const setFreshDraftReady = (next: Updater<boolean>) => updateAtom($freshDraftReady, next)
export const setBusy = (next: Updater<boolean>) => updateAtom($busy, next)
export const setAwaitingResponse = (next: Updater<boolean>) => updateAtom($awaitingResponse, next)
export const setCurrentModel = (next: Updater<string>) => updateAtom($currentModel, next)
export const setCurrentProvider = (next: Updater<string>) => updateAtom($currentProvider, next)
export const setCurrentReasoningEffort = (next: Updater<string>) => updateAtom($currentReasoningEffort, next)
export const setCurrentServiceTier = (next: Updater<string>) => updateAtom($currentServiceTier, next)
export const setCurrentFastMode = (next: Updater<boolean>) => updateAtom($currentFastMode, next)
export const setCurrentCwd = (next: Updater<string>) => updateAtom($currentCwd, next)
export const setCurrentBranch = (next: Updater<string>) => updateAtom($currentBranch, next)
export const setCurrentUsage = (next: Updater<UsageStats>) => updateAtom($currentUsage, next)
export const setSessionStartedAt = (next: Updater<number | null>) => updateAtom($sessionStartedAt, next)
export const setTurnStartedAt = (next: Updater<number | null>) => updateAtom($turnStartedAt, next)
export const setIntroPersonality = (next: Updater<string>) => updateAtom($introPersonality, next)
export const setCurrentPersonality = (next: Updater<string>) => updateAtom($currentPersonality, next)
export const setAvailablePersonalities = (next: Updater<string[]>) => updateAtom($availablePersonalities, next)
export const setIntroSeed = (next: Updater<number>) => updateAtom($introSeed, next)
export const setContextSuggestions = (next: Updater<ContextSuggestion[]>) => updateAtom($contextSuggestions, next)
export const setModelPickerOpen = (next: Updater<boolean>) => updateAtom($modelPickerOpen, next)

export function setSessionWorking(sessionId: string | null | undefined, working: boolean) {
  if (!sessionId) {
    return
  }

  setWorkingSessionIds(current => {
    const alreadyWorking = current.includes(sessionId)

    if (working) {
      return alreadyWorking ? current : [...current, sessionId]
    }

    return alreadyWorking ? current.filter(id => id !== sessionId) : current
  })
}
