import { atom, computed, type ReadableAtom, type WritableAtom } from 'nanostores'
import { createContext, type MutableRefObject, useContext } from 'react'

import type { ContextSuggestion } from '@/app/types'
import type { ChatMessage } from '@/lib/chat-messages'
import { $composerAttachments, type ComposerAttachment } from '@/store/composer'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $contextSuggestions,
  $currentBranch,
  $currentCwd,
  $currentFastMode,
  $currentModel,
  $currentPersonality,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  $currentUsage,
  $freshDraftReady,
  $lastVisibleMessageIsUser,
  $messages,
  $messagesEmpty,
  $resumeExhaustedSessionId,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $sessionStartedAt,
  $turnStartedAt,
  $yoloActive,
  getRememberedWorkspaceCwd,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setContextSuggestions,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setFreshDraftReady,
  setMessages,
  setResumeExhaustedSessionId,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessionStartedAt,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import { $activePaneId, $splitPaneSession, type PaneId, setSplitPaneFreshDraft, setSplitPaneSession } from '@/store/split'
import {
  createThreadScrollAtoms,
  mainThreadScroll,
  type ThreadScrollInstance
} from '@/store/thread-scroll'
import type { UsageStats } from '@/types/hermes'

import { lastVisibleMessageIsUser } from './thread-loading'

type Updater<T> = T | ((current: T) => T)

/**
 * One chat pane's view-state bundle — the atoms/setters/refs a mounted
 * ChatView + composer + thread subtree reads and writes, plus the pane's
 * routing surface. The MAIN pane's bundle IS the existing `store/session.ts`
 * singleton block (aliased, not copied), so every consumer migrated to
 * `usePaneView()` keeps reading the identical atom objects in single-pane
 * mode — bit-for-bit today's behavior. The split pane gets a fresh,
 * factory-created instance whose setters are plain atom writes (no
 * localStorage side effects), so a background stream can never clobber the
 * sticky composer-model keys.
 */
export interface PaneSessionView {
  paneId: PaneId

  // ── View atoms (see store/session.ts for meanings) ─────────────────────
  $activeSessionId: WritableAtom<string | null>
  $selectedStoredSessionId: WritableAtom<string | null>
  $messages: WritableAtom<ChatMessage[]>
  $messagesEmpty: ReadableAtom<boolean>
  $lastVisibleMessageIsUser: ReadableAtom<boolean>
  $busy: WritableAtom<boolean>
  $awaitingResponse: WritableAtom<boolean>
  $freshDraftReady: WritableAtom<boolean>
  $resumeFailedSessionId: WritableAtom<string | null>
  $resumeExhaustedSessionId: WritableAtom<string | null>
  $currentModel: WritableAtom<string>
  $currentProvider: WritableAtom<string>
  $currentReasoningEffort: WritableAtom<string>
  $currentServiceTier: WritableAtom<string>
  $currentFastMode: WritableAtom<boolean>
  $yoloActive: WritableAtom<boolean>
  $currentCwd: WritableAtom<string>
  $currentBranch: WritableAtom<string>
  $currentUsage: WritableAtom<UsageStats>
  $sessionStartedAt: WritableAtom<number | null>
  $turnStartedAt: WritableAtom<number | null>
  $currentPersonality: WritableAtom<string>
  $contextSuggestions: WritableAtom<ContextSuggestion[]>
  $composerAttachments: WritableAtom<ComposerAttachment[]>
  $threadScrolledUp: WritableAtom<boolean>
  $threadJumpButtonVisible: WritableAtom<boolean>

  // ── Setters (Updater form, mirroring store/session.ts signatures) ──────
  setActiveSessionId: (next: Updater<string | null>) => void
  setSelectedStoredSessionId: (next: Updater<string | null>) => void
  setMessages: (next: Updater<ChatMessage[]>) => void
  setBusy: (next: Updater<boolean>) => void
  setAwaitingResponse: (next: Updater<boolean>) => void
  setFreshDraftReady: (next: Updater<boolean>) => void
  setResumeFailedSessionId: (next: Updater<string | null>) => void
  setResumeExhaustedSessionId: (next: Updater<string | null>) => void
  setCurrentModel: (next: Updater<string>) => void
  setCurrentProvider: (next: Updater<string>) => void
  setCurrentReasoningEffort: (next: Updater<string>) => void
  setCurrentServiceTier: (next: Updater<string>) => void
  setCurrentFastMode: (next: Updater<boolean>) => void
  setYoloActive: (next: Updater<boolean>) => void
  setCurrentCwd: (next: Updater<string>) => void
  setCurrentBranch: (next: Updater<string>) => void
  setCurrentUsage: (next: Updater<UsageStats>) => void
  setSessionStartedAt: (next: Updater<number | null>) => void
  setTurnStartedAt: (next: Updater<number | null>) => void
  setCurrentPersonality: (next: Updater<string>) => void
  setContextSuggestions: (next: Updater<ContextSuggestion[]>) => void

  // ── Thread-scroll instance ops (per-pane; see store/thread-scroll.ts) ──
  setThreadAtBottom: (isAtBottom: boolean) => void
  resetThreadScroll: () => void
  requestScrollToBottom: () => void
  onScrollToBottomRequest: (handler: () => void) => () => void

  // ── Refs owned by the pane ──────────────────────────────────────────────
  activeSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  creatingSessionRef: MutableRefObject<boolean>
  /** Runtime id whose transcript currently occupies this pane's $messages. */
  viewSessionIdRef: MutableRefObject<string | null>

  // ── Pane routing surface ────────────────────────────────────────────────
  /** Stored id this pane should be showing (main: route; split: $splitPaneSession). */
  getTargetSessionId: () => string | null
  /** main: router navigate(sessionRoute(id)); split: $splitPaneSession swap. */
  navigateToSession: (storedId: string, replace?: boolean) => void
  /** main: router navigate(NEW_CHAT_ROUTE); split: pane-local fresh-draft flag. */
  navigateToNewChat: (replace?: boolean) => void
  isActive: () => boolean
}

const ref = <T>(value: T): MutableRefObject<T> => ({ current: value })

const paneThreadScrollMembers = (threadScroll: ThreadScrollInstance) => ({
  $threadScrolledUp: threadScroll.$threadScrolledUp,
  $threadJumpButtonVisible: threadScroll.$threadJumpButtonVisible,
  setThreadAtBottom: threadScroll.setThreadAtBottom,
  resetThreadScroll: threadScroll.resetThreadScroll,
  requestScrollToBottom: threadScroll.requestScrollToBottom,
  onScrollToBottomRequest: threadScroll.onScrollToBottomRequest
})

/**
 * The main pane's bundle: the existing global singletons, aliased. Nothing is
 * copied — `MAIN_PANE_VIEW.$messages === store/session.$messages` — which is
 * the whole compat shim: with the split closed, every migrated consumer is
 * reading exactly the atoms it read before.
 *
 * The routing surface + pane refs need the router / state-cache instances that
 * only exist once DesktopController mounts, so they start as inert defaults
 * and the controller injects the live bindings via {@link bindMainPaneView}.
 */
export const MAIN_PANE_VIEW: PaneSessionView = {
  paneId: 'main',

  $activeSessionId,
  $selectedStoredSessionId,
  $messages,
  $messagesEmpty,
  $lastVisibleMessageIsUser,
  $busy,
  $awaitingResponse,
  $freshDraftReady,
  $resumeFailedSessionId,
  $resumeExhaustedSessionId,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  $currentFastMode,
  $yoloActive,
  $currentCwd,
  $currentBranch,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $currentPersonality,
  $contextSuggestions,
  $composerAttachments,

  setActiveSessionId,
  setSelectedStoredSessionId,
  setMessages,
  setBusy,
  setAwaitingResponse,
  setFreshDraftReady,
  setResumeFailedSessionId,
  setResumeExhaustedSessionId,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentFastMode,
  setYoloActive,
  setCurrentCwd,
  setCurrentBranch,
  setCurrentUsage,
  setSessionStartedAt,
  setTurnStartedAt,
  setCurrentPersonality,
  setContextSuggestions,

  ...paneThreadScrollMembers(mainThreadScroll),

  activeSessionIdRef: ref<string | null>(null),
  selectedStoredSessionIdRef: ref<string | null>(null),
  busyRef: ref(false),
  creatingSessionRef: ref(false),
  viewSessionIdRef: ref<string | null>(null),

  getTargetSessionId: () => null,
  navigateToSession: () => undefined,
  navigateToNewChat: () => undefined,
  isActive: () => $activePaneId.get() === 'main'
}

export type MainPaneViewBindings = Partial<
  Pick<
    PaneSessionView,
    | 'activeSessionIdRef'
    | 'busyRef'
    | 'creatingSessionRef'
    | 'getTargetSessionId'
    | 'navigateToNewChat'
    | 'navigateToSession'
    | 'selectedStoredSessionIdRef'
    | 'viewSessionIdRef'
  >
>

/** Inject the router + state-cache bindings into the main bundle at controller
 *  mount. Everything else on MAIN_PANE_VIEW is a module-level alias and never
 *  changes; only the per-mount surfaces are late-bound. */
export function bindMainPaneView(bindings: MainPaneViewBindings): void {
  Object.assign(MAIN_PANE_VIEW, bindings)
}

// Split setters are plain atom writes (Updater unwrap only) — deliberately NOT
// the store/session setters, whose model/cwd variants persist to localStorage.
// Structural atom shape (get/set on the same T) mirrors store/session's
// updateAtom so inference lands on the mutable array type.
interface PaneAtom<T> {
  get: () => T
  set: (value: T) => void
}

const update = <T>(store: PaneAtom<T>, next: Updater<T>) =>
  store.set(typeof next === 'function' ? (next as (current: T) => T)(store.get()) : next)

/** A fresh, self-contained bundle for the split pane. Same defaults as the
 *  main block; the composer-sticky atoms seed from the live sticky values so
 *  a split new-chat opens on your last pick — but split setters are plain
 *  atom writes, so nothing here ever touches the persisted keys. */
export function createSplitPaneView(): PaneSessionView {
  const $paneActiveSessionId = atom<string | null>(null)
  const $paneSelectedStoredSessionId = atom<string | null>(null)
  const $paneMessages = atom<ChatMessage[]>([])
  const $paneBusy = atom(false)
  const $paneAwaitingResponse = atom(false)
  const $paneFreshDraftReady = atom(false)
  const $paneResumeFailedSessionId = atom<string | null>(null)
  const $paneResumeExhaustedSessionId = atom<string | null>(null)
  const $paneCurrentModel = atom($currentModel.get())
  const $paneCurrentProvider = atom($currentProvider.get())
  const $paneCurrentReasoningEffort = atom($currentReasoningEffort.get())
  const $paneCurrentServiceTier = atom('')
  const $paneCurrentFastMode = atom($currentFastMode.get())
  const $paneYoloActive = atom(false)
  const $paneCurrentCwd = atom(getRememberedWorkspaceCwd())
  const $paneCurrentBranch = atom('')

  const $paneCurrentUsage = atom<UsageStats>({
    calls: 0,
    input: 0,
    output: 0,
    total: 0
  })

  const $paneSessionStartedAt = atom<number | null>(null)
  const $paneTurnStartedAt = atom<number | null>(null)
  const $paneCurrentPersonality = atom('')
  const $paneContextSuggestions = atom<ContextSuggestion[]>([])
  const $paneComposerAttachments = atom<ComposerAttachment[]>([])

  return {
    paneId: 'split',

    $activeSessionId: $paneActiveSessionId,
    $selectedStoredSessionId: $paneSelectedStoredSessionId,
    $messages: $paneMessages,
    $messagesEmpty: computed($paneMessages, messages => messages.length === 0),
    $lastVisibleMessageIsUser: computed($paneMessages, lastVisibleMessageIsUser),
    $busy: $paneBusy,
    $awaitingResponse: $paneAwaitingResponse,
    $freshDraftReady: $paneFreshDraftReady,
    $resumeFailedSessionId: $paneResumeFailedSessionId,
    $resumeExhaustedSessionId: $paneResumeExhaustedSessionId,
    $currentModel: $paneCurrentModel,
    $currentProvider: $paneCurrentProvider,
    $currentReasoningEffort: $paneCurrentReasoningEffort,
    $currentServiceTier: $paneCurrentServiceTier,
    $currentFastMode: $paneCurrentFastMode,
    $yoloActive: $paneYoloActive,
    $currentCwd: $paneCurrentCwd,
    $currentBranch: $paneCurrentBranch,
    $currentUsage: $paneCurrentUsage,
    $sessionStartedAt: $paneSessionStartedAt,
    $turnStartedAt: $paneTurnStartedAt,
    $currentPersonality: $paneCurrentPersonality,
    $contextSuggestions: $paneContextSuggestions,
    $composerAttachments: $paneComposerAttachments,

    setActiveSessionId: next => update($paneActiveSessionId, next),
    setSelectedStoredSessionId: next => update($paneSelectedStoredSessionId, next),
    setMessages: next => update($paneMessages, next),
    setBusy: next => update($paneBusy, next),
    setAwaitingResponse: next => update($paneAwaitingResponse, next),
    setFreshDraftReady: next => update($paneFreshDraftReady, next),
    setResumeFailedSessionId: next => update($paneResumeFailedSessionId, next),
    setResumeExhaustedSessionId: next => update($paneResumeExhaustedSessionId, next),
    setCurrentModel: next => update($paneCurrentModel, next),
    setCurrentProvider: next => update($paneCurrentProvider, next),
    setCurrentReasoningEffort: next => update($paneCurrentReasoningEffort, next),
    setCurrentServiceTier: next => update($paneCurrentServiceTier, next),
    setCurrentFastMode: next => update($paneCurrentFastMode, next),
    setYoloActive: next => update($paneYoloActive, next),
    setCurrentCwd: next => update($paneCurrentCwd, next),
    setCurrentBranch: next => update($paneCurrentBranch, next),
    setCurrentUsage: next => update($paneCurrentUsage, next),
    setSessionStartedAt: next => update($paneSessionStartedAt, next),
    setTurnStartedAt: next => update($paneTurnStartedAt, next),
    setCurrentPersonality: next => update($paneCurrentPersonality, next),
    setContextSuggestions: next => update($paneContextSuggestions, next),

    ...paneThreadScrollMembers(createThreadScrollAtoms()),

    activeSessionIdRef: ref<string | null>(null),
    selectedStoredSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    viewSessionIdRef: ref<string | null>(null),

    getTargetSessionId: () => $splitPaneSession.get()?.storedId ?? null,
    navigateToSession: storedId => setSplitPaneSession(storedId),
    navigateToNewChat: () => setSplitPaneFreshDraft(),
    isActive: () => $activePaneId.get() === 'split'
  }
}

/** DEFAULT = the main bundle, so every consumer that never sees a provider
 *  keeps today's behavior exactly. Only SplitChatPane mounts a provider. */
export const PaneViewContext = createContext<PaneSessionView>(MAIN_PANE_VIEW)

export const usePaneView = (): PaneSessionView => useContext(PaneViewContext)
