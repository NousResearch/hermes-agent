import { useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { closeActiveTab } from '@/app/chat/close-tab'
import { hudTargetSessionId } from '@/app/hud/handoff'
import { setTerminalTakeover } from '@/app/right-sidebar/store'
import { toggleTerminalPane } from '@/app/right-sidebar/terminal/reveal-focus'
import { closeActiveTerminal, createTerminal, cycleTerminal } from '@/app/right-sidebar/terminal/terminals'
import { appViewForPath, isOverlayView } from '@/app/routes'
import {
  activateTreeTabSlot,
  cycleTreeTabInFocusedZone,
  isPaneVisible,
  toggleTargetZoneTabStrip
} from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { onReleaseTypingFocus } from '@/components/ui/keyboard-first'
import { translateNow } from '@/i18n/runtime'
import { findBarClaimsCombo } from '@/lib/find-in-page'
import {
  contributedKeybindHandler,
  keybindAction,
  PROFILE_SLOT_COUNT,
  SESSION_SLOT_COUNT,
  TAB_SLOT_COUNT
} from '@/lib/keybinds/actions'
import { handleApprovalKey, releaseApprovalKey } from '@/lib/keybinds/approval-keys'
import { actionAllowedInInput, comboFromEvent, IS_MAC, isEditableTarget, isFocusWithin } from '@/lib/keybinds/combo'
import { composerFocusKeysAllowed, isComposerFocusSoftCombo, typeToFocusChar } from '@/lib/keybinds/composer-focus-keys'
import { stepReasoningEffort, writeSessionReasoningEffort } from '@/lib/reasoning-step'
import { toggleAgentsPanel } from '@/store/agents-panel'
import { openWorktreeDialog } from '@/store/coding-status'
import { $commandPaletteOpen, openCommandPalettePage, toggleCommandPalette } from '@/store/command-palette'
import { recordAction, recordDislike } from '@/store/desktop-metrics'
import {
  $findInPage,
  findNext as findNextMatch,
  findPrevious as findPreviousMatch,
  openFindBar
} from '@/store/find-in-page'
import { toggleHud } from '@/store/hud'
import { toggleSimpleMode } from '@/store/interface-mode'
import { $capture, $comboIndex, captureStep, endCapture, setBinding } from '@/store/keybinds'
import {
  cycleSidebarGrouping,
  layoutHasRightSide,
  requestSessionSearchFocus,
  setFileBrowserOpen,
  togglePanesFlipped,
  toggleRightSide,
  toggleSidebarOpen
} from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { toggleBrowserTab } from '@/store/preview'
import {
  $newChatProfile,
  cycleProfile,
  requestProfileCreate,
  switchProfileToSlot,
  switchToDefaultProfile,
  toggleShowAllProfiles
} from '@/store/profile'
import { toggleProfileRailVisible } from '@/store/profile-rail-prefs'
import { openFolderAsProject } from '@/store/projects'
import { toggleReview } from '@/store/review'
import {
  $activeSessionId,
  $currentReasoningEffort,
  $defaultReasoningEffort,
  $selectedStoredSessionId,
  markComposerSelectionManual,
  setCurrentReasoningEffort,
  setModelPickerOpen
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { reopenLastClosedTile } from '@/store/session-states'
import {
  $switcherOpen,
  closeSwitcher,
  commitOnCtrlUp,
  onSwitcherTabDown,
  onSwitcherTabUp,
  openOrAdvanceSwitcher,
  slotSessionId,
  switcherActive,
  switcherJustClosed
} from '@/store/session-switcher'
import { toggleStatusbarVisible } from '@/store/statusbar-prefs'
import { requestThreadPageScroll } from '@/store/thread-scroll'
import { openNewWindow } from '@/store/windows'
import { useTheme } from '@/themes/context'

import {
  requestComposerDictation,
  requestComposerFocus,
  requestModelMenuToggle,
  requestVoiceToggle
} from '../chat/composer/focus'
import { handleComposerFocusChord } from '../chat/composer/focus-chord'
import { handleWindowPaste } from '../chat/composer/paste-to-focus'
import { openSession } from '../open-session'
import {
  $workspaceIsPage,

  ARTIFACTS_ROUTE,
  CAPABILITIES_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  navigateToWorkspacePage,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  sessionRoute,
  SETTINGS_ROUTE
} from '../routes'

export interface KeybindRuntimeDeps {
  /** Gateway RPC requester for session-scoped model controls (reasoning). */
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  /** Open/close the command center overlay (sessions / system / usage). */
  toggleCommandCenter: () => void
  /** Drop to a fresh new-session draft. */
  startFreshSession: () => void
  /** Open a fresh session as a tab in the main zone (⌘T), leaving the primary. */
  openNewSessionTab: () => void
  /** Pin/unpin the active session. */
  toggleSelectedPin: () => void
  /** Archive the active session. */
  archiveSelectedSession: () => void
}

/** A handler returns `false` to decline the chord (see `passthrough`); any other
 *  return value (void, a navigate() promise, …) means it ran. */
type HandlerMap = Record<string, () => unknown>

// Mount once near the top of the app. Owns the single global keydown listener
// for every rebindable hotkey: it runs the matched action, or — while capture
// mode is active (edit overlay / panel rebind) — records the pressed combo.
export function useKeybinds(deps: KeybindRuntimeDeps): void {
  const navigate = useNavigate()
  const location = useLocation()
  const { resolvedMode, setMode } = useTheme()

  // Keep the latest closures without re-subscribing the listener.
  const handlersRef = useRef<HandlerMap>({})
  const commitSwitcherRef = useRef<() => void>(() => {})

  const profileSwitchHandlers: HandlerMap = {}

  // A tab key that lands on the WORKSPACE tab while a full page (skills /
  // messaging / artifacts / a plugin route) covers it must also route back to
  // the chat: the workspace pane is already the zone's active tab behind the
  // page, so fronting it alone changes nothing on screen and the key reads
  // dead. Mirrors `openSession`'s full-page rule — only a route change puts
  // the chat back.
  const leavePageForWorkspaceChat = (paneId: null | string) => {
    if (paneId === 'workspace' && $workspaceIsPage.get()) {
      const selected = $selectedStoredSessionId.get()

      navigate(selected ? sessionRoute(selected) : NEW_CHAT_ROUTE)
    }
  }

  for (let slot = 1; slot <= PROFILE_SLOT_COUNT; slot += 1) {
    // Unconditional: the ⌘1…⌘9 tab dispatch is view.tabSlot.N, which sits
    // ahead of this action on the same chord and passes through when no tab
    // strip is eligible (#92569).
    profileSwitchHandlers[`profile.switch.${slot}`] = () => {
      switchProfileToSlot(slot)
    }
  }

  const goToSession = (sessionId: null | string) => {
    if (sessionId) {
      openSession(sessionId, navigate)
    }
  }

  // ^N jumps straight to the Nth recent session and dismisses the switcher.
  const sessionSlotHandlers: HandlerMap = {}

  for (let slot = 1; slot <= SESSION_SLOT_COUNT; slot += 1) {
    sessionSlotHandlers[`session.slot.${slot}`] = () => {
      closeSwitcher()
      goToSession(slotSessionId(slot))
    }
  }

  // view.tabSlot.N: activate the Nth visible tab of the hovered / focused /
  // workspace zone (`activateTreeTabSlot`'s ladder). Declines when no rung is
  // a real tab strip, so the chord falls through to profile.switch.N.
  const tabSlotHandlers: HandlerMap = {}

  for (let slot = 1; slot <= TAB_SLOT_COUNT; slot += 1) {
    tabSlotHandlers[`view.tabSlot.${slot}`] = () => {
      const pane = activateTreeTabSlot(slot)

      if (!pane) {
        return false
      }

      leavePageForWorkspaceChat(pane)
    }
  }

  commitSwitcherRef.current = () => goToSession(commitOnCtrlUp())

  const stepSession = (direction: 1 | -1) => {
    onSwitcherTabDown()
    goToSession(openOrAdvanceSwitcher(direction))
  }

  // ⌃Tab cycles the focused session/main tab strip; only a non-tabbed focus
  // falls through to the recent-session switcher. Landing on the workspace
  // under a full page routes back to the chat (same as view.tabSlot.1).
  const cycleTab = (direction: 1 | -1) => {
    const pane = cycleTreeTabInFocusedZone(direction)

    if (pane) {
      leavePageForWorkspaceChat(pane)
    } else {
      stepSession(direction)
    }
  }

  const showFiles = () => {
    setFileBrowserOpen(true)
    setTerminalTakeover(false)
  }

  // Reasoning level up/down (#71627): step the ACTIVE session one notch
  // through off → minimal → … → xhigh (clamped; max/ultra stay behind the
  // menu). Optimistic store write with rollback, and a monotonic sequence so
  // a slow earlier response can't revert a newer press.
  const reasoningRequestSeqRef = useRef(0)

  const stepSessionReasoning = (direction: 1 | -1) => {
    const sessionId = $activeSessionId.get()

    // No live session: the draft's pick still steps (it ships on the next
    // session.create); no `config.set` — without a session the RPC falls
    // back to the persistent profile config and would rewrite the default.
    const rollback = $currentReasoningEffort.get()
    const fallback = $defaultReasoningEffort.get() || undefined
    const next = stepReasoningEffort(rollback, direction, fallback)

    if (next === rollback.trim().toLowerCase()) {
      return
    }

    markComposerSelectionManual()
    setCurrentReasoningEffort(next)

    if (!sessionId) {
      return
    }

    const requestSeq = reasoningRequestSeqRef.current + 1

    reasoningRequestSeqRef.current = requestSeq

    void writeSessionReasoningEffort(deps.requestGateway, sessionId, next)
      .then(value => {
        if (reasoningRequestSeqRef.current === requestSeq) {
          setCurrentReasoningEffort(value)
        }
      })
      .catch(error => {
        if (reasoningRequestSeqRef.current === requestSeq) {
          setCurrentReasoningEffort(rollback)
          notifyError(error, translateNow('shell.modelOptions.updateFailed'))
        }
      })
  }

  handlersRef.current = {
    'keybinds.openPanel': () => navigate(`${SETTINGS_ROUTE}?tab=keybinds`),

    'composer.focus': () => requestComposerFocus('active'),
    // Toggle the composer pill's live model dropdown (pane under the pointer,
    // else active composer); no chat surface on screen → the full dialog.
    'composer.modelPicker': () => {
      if (!requestModelMenuToggle()) {
        setModelPickerOpen(true)
      }
    },
    'composer.voice': requestVoiceToggle,
    'composer.dictate': requestComposerDictation,
    'composer.reasoningUp': () => stepSessionReasoning(1),
    'composer.reasoningDown': () => stepSessionReasoning(-1),

    // On the Settings overlay, ⌘K scopes to settings search; the second press
    // (or Esc) still closes as usual via toggle.
    'nav.commandPalette': () => {
      if (!$commandPaletteOpen.get() && appViewForPath(location.pathname) === 'settings') {
        openCommandPalettePage('settings')

        return
      }

      toggleCommandPalette()
    },
    'nav.commandCenter': deps.toggleCommandCenter,
    'nav.settings': () => navigate(SETTINGS_ROUTE),
    'nav.profiles': () => navigate(PROFILES_ROUTE),
    'nav.capabilities': () => navigateToWorkspacePage(navigate, CAPABILITIES_ROUTE),
    'nav.messaging': () => navigateToWorkspacePage(navigate, MESSAGING_ROUTE),
    'nav.artifacts': () => navigateToWorkspacePage(navigate, ARTIFACTS_ROUTE),
    'nav.cron': () => navigate(CRON_ROUTE),
    'nav.agents': () => toggleAgentsPanel(),

    'session.new': () => {
      // Match the sidebar New Session button. A plain keyboard new chat should
      // target the current live profile, not a stale per-profile quick-create
      // selection from a prior action.
      setWorkspaceScope('sessions')
      $newChatProfile.set(null)
      deps.startFreshSession()
      window.dispatchEvent(new CustomEvent('hermes:new-session-shortcut'))
    },
    'session.newTab': () => deps.openNewSessionTab(),
    'session.newWindow': () => void openNewWindow(),
    'session.next': () => cycleTab(1),
    'session.prev': () => cycleTab(-1),
    ...sessionSlotHandlers,
    ...tabSlotHandlers,
    'session.focusSearch': requestSessionSearchFocus,
    'session.togglePin': deps.toggleSelectedPin,
    'session.archive': deps.archiveSelectedSession,
    'conversation.scrollPageUp': () => requestThreadPageScroll(-1, $focusedStoredSessionId.get()),
    'conversation.scrollPageDown': () => requestThreadPageScroll(1, $focusedStoredSessionId.get()),
    // openWorktreeDialog resolves the target. There is no test for a repo
    // here, so the key works from a detached session that sits inside a
    // project, and not only from a session with a repo. When no repo is in
    // reach, openWorktreeDialog does nothing.
    'workspace.newWorktree': () => void openWorktreeDialog(),
    // ⌘O: native folder picker → open the folder as a project (upsert) with a
    // fresh session anchored there.
    'workspace.openFolder': () => void openFolderAsProject(),

    // Narrow-viewport reveal is handled inside the store toggles now.
    'view.toggleSidebar': toggleSidebarOpen,
    'view.cycleSidebarGrouping': cycleSidebarGrouping,
    // ⌘J toggles the physical right side — whatever column lives there in the
    // live tree (the Browser preview column, the files column). Falls back to
    // the terminal when nothing lives on the right (terminal-on-bottom).
    'view.toggleRightSidebar': () => (layoutHasRightSide() ? toggleRightSide() : toggleTerminalPane()),
    'view.toggleReview': toggleReview,
    'view.toggleStatusbar': toggleStatusbarVisible,
    'view.toggleProfileRail': toggleProfileRailVisible,
    'view.toggleSimpleMode': toggleSimpleMode,
    'view.toggleTabStrip': () => void toggleTargetZoneTabStrip(),
    'view.showFiles': showFiles,
    'view.showBrowser': toggleBrowserTab,
    'view.toggleHud': () => toggleHud(hudTargetSessionId()),
    'view.showTerminal': () => toggleTerminalPane(),
    // Create first so the pane's open-effect ensure sees a non-empty set and
    // doesn't also spawn one — net effect is exactly one fresh terminal.
    'view.newTerminal': () => {
      createTerminal()
      setTerminalTakeover(true)
    },
    // Switch / close only act while the terminal is actually ON SCREEN — ask
    // the tree, not the toggle store (which stays true behind a stacked
    // sibling tab or a minimized zone).
    'view.nextTerminal': () => isPaneVisible('terminal') && cycleTerminal(1),
    'view.prevTerminal': () => isPaneVisible('terminal') && cycleTerminal(-1),
    'view.closeTerminal': () => isPaneVisible('terminal') && closeActiveTerminal(),
    'view.flipPanes': togglePanesFlipped,
    // ⌘W: close the focused tab (terminal / preview target / zone tree tab).
    // On the main tab with session tabs stacked, it shifts the next one in —
    // the loader navigates to that session's route (loads it into main). On
    // macOS the menu accelerator owns ⌘W and routes through the same
    // closeActiveTab via IPC (see use-desktop-integrations); this binding is
    // the Win/Linux path where ⌘W reaches the renderer directly.
    'view.closeTab': () => void closeActiveTab(id => navigate(sessionRoute(id))),
    'view.reopenTab': reopenLastClosedTile,
    'view.findInPage': () => {
      // Suppress on overlay routes so it doesn't collide with overlay-specific
      // search surfaces (e.g. Settings search bar).
      if (!isOverlayView(appViewForPath(location.pathname))) {
        openFindBar()
      }
    },
    // ⌘G / ⌘⇧G are handled by the find bar's own capture-phase listener while
    // it is open (so they don't collide with `view.toggleReview`). These
    // registry handlers cover a user-assigned dedicated chord: stepping is a
    // no-op unless the bar is open with a query, so a bound key can't search
    // invisibly.
    'view.findNext': findNextMatch,
    'view.findPrevious': findPreviousMatch,

    'appearance.toggleMode': () => setMode(resolvedMode === 'dark' ? 'light' : 'dark'),

    'profile.default': switchToDefaultProfile,
    ...profileSwitchHandlers,
    'profile.next': () => cycleProfile(1),
    'profile.prev': () => cycleProfile(-1),
    'profile.toggleAll': toggleShowAllProfiles,
    'profile.create': requestProfileCreate
  }

  // A keyboard-driven overlay closing hands typing back to the composer: Radix
  // restores focus to the trigger (a toolbar button for the model pill), so
  // without this the Enter that committed a model also eats the next keystroke.
  // Deferred one frame and skipped when something else editable has claimed
  // focus, because a palette action can legitimately open a dialog or navigate
  // — the release must never steal focus from the surface it just opened.
  useEffect(
    () =>
      onReleaseTypingFocus(() =>
        requestAnimationFrame(() => {
          if (!isEditableTarget(document.activeElement)) {
            requestComposerFocus('active')
          }
        })
      ),
    []
  )

  useEffect(() => {
    const updateF12Ownership = () => {
      const hasF12Binding = [...$comboIndex.get().keys()].some(combo => combo === 'f12' || combo.endsWith('+f12'))
      window.hermesDesktop?.setF12ShortcutActive?.(hasF12Binding || $capture.get() !== null)
    }

    const stopBindings = $comboIndex.subscribe(updateF12Ownership)
    const stopCapture = $capture.subscribe(updateF12Ownership)

    return () => {
      stopBindings()
      stopCapture()
      window.hermesDesktop?.setF12ShortcutActive?.(false)
    }
  }, [])

  useEffect(() => {
    const stopF12Shortcut = window.hermesDesktop?.onF12Shortcut?.(input => {
      const target = document.activeElement ?? document.body ?? document.documentElement
      target.dispatchEvent(
        new KeyboardEvent('keydown', {
          altKey: input.alt,
          bubbles: true,
          cancelable: true,
          code: input.code,
          ctrlKey: input.control,
          key: input.key,
          metaKey: input.meta,
          repeat: input.repeat,
          shiftKey: input.shift
        })
      )
    })

    return () => stopF12Shortcut?.()
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // An active IME composition owns the keyboard. Windows Chinese IMEs
      // (Microsoft Pinyin, Sogou) use Ctrl+, as their punctuation-mode toggle,
      // so without this guard that keystroke ALSO matched `nav.settings` and
      // navigated away mid-word — unmounting the composer and destroying the
      // unsent draft (#41079). The draft stash below makes navigation safe;
      // this makes the IME keystroke not navigate at all.
      if (event.isComposing) {
        return
      }

      // Capture mode: the next real key becomes the binding. Backspace/Delete
      // clears it (empty combos) so a shipped chord like the sidebar's mod+b
      // can be unbound. Escape cancels. Swallow everything so e.g. ⌘K rebinds
      // instead of opening the palette.
      const capturing = $capture.get()

      if (capturing) {
        event.preventDefault()
        event.stopPropagation()

        const step = captureStep(event.key, comboFromEvent(event))

        if (step.type === 'wait') {
          return
        }

        if (step.type === 'set') {
          setBinding(capturing, step.combos)
        } else {
          recordDislike('cancelled', 'keybind_capture')
        }

        endCapture()

        return
      }

      // While the session switcher is up, Esc abandons it (stay put) before any
      // combo dispatch — ⌃Tab keeps stepping through the existing handler.
      if (switcherActive() && event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        closeSwitcher()

        return
      }

      const combo = comboFromEvent(event)

      if (!combo) {
        return
      }

      // The open find bar owns ⌘G / ⌘⇧G / Escape. Its own capture-phase
      // listener runs those actions; bail here so the registry doesn't ALSO
      // fire the action bound to the same combo (⌘G = view.toggleReview,
      // Escape = composer.cancel, which would abort a live turn). Both
      // listeners are on `window`, so stopPropagation in the bar can't
      // suppress this one — the dispatcher has to yield explicitly.
      if ($findInPage.get().active && findBarClaimsCombo(combo)) {
        return
      }

      if (handleApprovalKey(event)) {
        return
      }

      const actionIds = $comboIndex.get().get(combo)

      // Unbound printable → type-to-focus. Bound chords (shift+n, …) win above.
      if (!actionIds) {
        const typeChar = typeToFocusChar(event)

        if (typeChar && composerFocusKeysAllowed(event, 'type')) {
          event.preventDefault()
          requestComposerFocus('active', { typeChar })
        }

        return
      }

      // Actions bound to the chord, in registration order. The first runs; a
      // `passthrough` action that declines hands the chord to the next.
      for (const actionId of actionIds) {
        if (isEditableTarget(event.target) && !actionAllowedInInput(actionId, combo)) {
          return
        }

        // Soft `/` / Enter: gated so dialogs/buttons/terminal keep those keys.
        // Rebound chords fall through to the normal handler.
        if (actionId === 'composer.focus' && isComposerFocusSoftCombo(combo)) {
          if (!composerFocusKeysAllowed(event, combo)) {
            return
          }

          event.preventDefault()
          requestComposerFocus('active', { typeChar: combo === '/' ? '/' : undefined })

          return
        }

        // The close-tab chord over a focused user terminal is the shell's word
        // erase (readline ^W). Main already claimed the default chord on
        // every platform (window-accelerator) and routes it through the IPC
        // rung above, so this guards the DEFAULT binding when it reaches the
        // dispatcher anyway, plus a chord rebound onto another key with the
        // same bare shape. Returning without preventDefault hands the key to
        // xterm, whose data handler writes the ^W byte to the PTY. Read-only
        // agent mirrors carry only [data-terminal], so they keep close.
        if (
          actionId === 'view.closeTab' &&
          event.key.toLowerCase() === 'w' &&
          (IS_MAC ? event.metaKey : event.ctrlKey) &&
          !event.altKey &&
          !event.shiftKey &&
          isFocusWithin('[data-interactive-terminal]')
        ) {
          return
        }

        // Built-in handlers first (they carry React context); contributed
        // actions bring their own `run` through the registry.
        const handler = handlersRef.current[actionId] ?? contributedKeybindHandler(actionId)

        if (!handler) {
          return
        }

        event.preventDefault()

        if (handler() === false && keybindAction(actionId)?.passthrough) {
          continue
        }

        recordAction(actionId, 'shortcut')

        return
      }
    }

    // Mac-app-switcher commit: lifting Ctrl with the overlay open lands on the
    // highlighted session. A window blur (Cmd+Tab away mid-switch) cancels so
    // the overlay never gets stranded waiting for a keyup that never comes.
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === 'Escape') {
        releaseApprovalKey()
      }

      if (event.key === 'Tab') {
        onSwitcherTabUp()
      }

      if (event.key === 'Control') {
        commitSwitcherRef.current()
      }
    }

    const onBlur = () => {
      releaseApprovalKey()

      if (switcherActive()) {
        closeSwitcher()
      }
    }

    // Swallow trailing contextmenu after Ctrl+click commit (Electron main menu).
    const onContextMenu = (event: MouseEvent) => {
      if ($switcherOpen.get() || switcherJustClosed()) {
        event.preventDefault()
        event.stopPropagation()
      }
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })
    window.addEventListener('keyup', onKeyUp, { capture: true })
    window.addEventListener('blur', onBlur)
    window.addEventListener('contextmenu', onContextMenu, { capture: true })
    // Paste twin of type-to-focus: ⌘V on non-editable chrome routes the
    // clipboard (text AND images) into the active composer. Bubble phase so
    // editables' own paste handlers run first and mark the event handled.
    window.addEventListener('paste', handleWindowPaste)
    // ⌘/Ctrl+L moves focus to the composer. Bubble phase so capture-phase
    // claimants run first; the priority ladder lives in focus-chord.ts.
    window.addEventListener('keydown', handleComposerFocusChord)

    return () => {
      window.removeEventListener('keydown', onKeyDown, { capture: true })
      window.removeEventListener('keyup', onKeyUp, { capture: true })
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('contextmenu', onContextMenu, { capture: true })
      window.removeEventListener('paste', handleWindowPaste)
      window.removeEventListener('keydown', handleComposerFocusChord)
    }
  }, [])
}
