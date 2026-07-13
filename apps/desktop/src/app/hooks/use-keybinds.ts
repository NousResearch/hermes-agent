import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { closeActiveTerminal, createTerminal, cycleTerminal } from '@/app/right-sidebar/terminal/terminals'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { matchesQuery } from '@/hooks/use-media-query'
import { PROFILE_SLOT_COUNT, SESSION_SLOT_COUNT } from '@/lib/keybinds/actions'
import { comboAllowedInInput, comboFromEvent, isEditableTarget } from '@/lib/keybinds/combo'
import { $repoStatus } from '@/store/coding-status'
import { toggleCommandPalette } from '@/store/command-palette'
import { $capture, $comboIndex, endCapture, setBinding, toggleKeybindPanel } from '@/store/keybinds'
import {
  CHAT_SIDEBAR_PANE_ID,
  FILE_BROWSER_PANE_ID,
  requestSessionSearchFocus,
  setFileBrowserOpen,
  toggleFileBrowserOpen,
  togglePanesFlipped,
  toggleSidebarOpen
} from '@/store/layout'
import {
  $newChatProfile,
  cycleProfile,
  ensureGatewayProfile,
  requestProfileCreate,
  switchProfileToSlot,
  switchToDefaultProfile,
  toggleShowAllProfiles
} from '@/store/profile'
import { requestNewWorktree } from '@/store/projects'
import { toggleReview } from '@/store/review'
import { setModelPickerOpen } from '@/store/session'
import type { SessionIdentity } from '@/store/session-identity'
import {
  $switcherOpen,
  closeSwitcher,
  commitOnCtrlUp,
  onSwitcherTabDown,
  onSwitcherTabUp,
  openOrAdvanceSwitcher,
  slotSession,
  switcherActive,
  switcherJustClosed
} from '@/store/session-switcher'
import { openNewSessionInNewWindow } from '@/store/windows'
import { useTheme } from '@/themes/context'

import { requestComposerFocus, requestVoiceToggle } from '../chat/composer/focus'
import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '../layout-constants'
import {
  AGENTS_ROUTE,
  ARTIFACTS_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  PROFILES_ROUTE,
  sessionRoute,
  SETTINGS_ROUTE,
  SKILLS_ROUTE
} from '../routes'

export interface KeybindRuntimeDeps {
  /** Open/close the command center overlay (sessions / system / usage). */
  toggleCommandCenter: () => void
  /** Drop to a fresh new-session draft. */
  startFreshSession: () => void
  /** Pin/unpin the active session. */
  toggleSelectedPin: () => void
}

type HandlerMap = Record<string, () => void>

// Mount once near the top of the app. Owns the single global keydown listener
// for every rebindable hotkey: it runs the matched action, or — while capture
// mode is active (edit overlay / panel rebind) — records the pressed combo.
export function useKeybinds(deps: KeybindRuntimeDeps): void {
  const navigate = useNavigate()
  const { resolvedMode, setMode } = useTheme()

  // Keep the latest closures without re-subscribing the listener.
  const handlersRef = useRef<HandlerMap>({})
  const commitSwitcherRef = useRef<() => void>(() => {})

  const profileSwitchHandlers: HandlerMap = {}

  for (let slot = 1; slot <= PROFILE_SLOT_COUNT; slot += 1) {
    profileSwitchHandlers[`profile.switch.${slot}`] = () => switchProfileToSlot(slot)
  }

  const goToSession = (session: SessionIdentity | null) => {
    if (session) {
      void ensureGatewayProfile(session.profile).then(() => navigate(sessionRoute(session.sessionId)))
    }
  }

  // ^N jumps straight to the Nth recent session and dismisses the switcher.
  const sessionSlotHandlers: HandlerMap = {}

  for (let slot = 1; slot <= SESSION_SLOT_COUNT; slot += 1) {
    sessionSlotHandlers[`session.slot.${slot}`] = () => {
      closeSwitcher()
      goToSession(slotSession(slot))
    }
  }

  commitSwitcherRef.current = () => goToSession(commitOnCtrlUp())

  const stepSession = (direction: 1 | -1) => {
    onSwitcherTabDown()
    goToSession(openOrAdvanceSwitcher(direction))
  }

  const showFiles = () => {
    setFileBrowserOpen(true)
    setTerminalTakeover(false)
  }

  handlersRef.current = {
    'keybinds.openPanel': toggleKeybindPanel,

    'composer.focus': () => requestComposerFocus('main'),
    'composer.modelPicker': () => setModelPickerOpen(true),
    'composer.voice': requestVoiceToggle,

    'nav.commandPalette': toggleCommandPalette,
    'nav.commandCenter': deps.toggleCommandCenter,
    'nav.settings': () => navigate(SETTINGS_ROUTE),
    'nav.profiles': () => navigate(PROFILES_ROUTE),
    'nav.skills': () => navigate(SKILLS_ROUTE),
    'nav.messaging': () => navigate(MESSAGING_ROUTE),
    'nav.artifacts': () => navigate(ARTIFACTS_ROUTE),
    'nav.cron': () => navigate(CRON_ROUTE),
    'nav.agents': () => navigate(AGENTS_ROUTE),

    'session.new': () => {
      // Match the sidebar New Session button. A plain keyboard new chat should
      // target the current live profile, not a stale per-profile quick-create
      // selection from a prior action.
      $newChatProfile.set(null)
      deps.startFreshSession()
      window.dispatchEvent(new CustomEvent('hermes:new-session-shortcut'))
    },
    'session.newWindow': () => void openNewSessionInNewWindow(),
    'session.next': () => stepSession(1),
    'session.prev': () => stepSession(-1),
    ...sessionSlotHandlers,
    'session.focusSearch': requestSessionSearchFocus,
    'session.togglePin': deps.toggleSelectedPin,
    // Only meaningful inside a git repo — a no-op otherwise (the key falls
    // through instead of silently doing nothing).
    'workspace.newWorktree': () => $repoStatus.get() && requestNewWorktree(),

    'view.toggleSidebar': () => {
      if (matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
        window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: CHAT_SIDEBAR_PANE_ID } }))
      } else {
        toggleSidebarOpen()
      }
    },
    'view.toggleRightSidebar': () => {
      if (matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
        window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: FILE_BROWSER_PANE_ID } }))
      } else {
        toggleFileBrowserOpen()
      }
    },
    'view.toggleReview': toggleReview,
    'view.showFiles': showFiles,
    'view.showTerminal': () => setTerminalTakeover(!$terminalTakeover.get()),
    // Create first so the pane's open-effect ensure sees a non-empty set and
    // doesn't also spawn one — net effect is exactly one fresh terminal.
    'view.newTerminal': () => {
      createTerminal()
      setTerminalTakeover(true)
    },
    // Switch / close only act while the pane is open (no focus-scoping here, so
    // this stands in for "terminal is showing").
    'view.nextTerminal': () => $terminalTakeover.get() && cycleTerminal(1),
    'view.prevTerminal': () => $terminalTakeover.get() && cycleTerminal(-1),
    'view.closeTerminal': () => $terminalTakeover.get() && closeActiveTerminal(),
    'view.flipPanes': togglePanesFlipped,

    'appearance.toggleMode': () => setMode(resolvedMode === 'dark' ? 'light' : 'dark'),

    'profile.default': switchToDefaultProfile,
    ...profileSwitchHandlers,
    'profile.next': () => cycleProfile(1),
    'profile.prev': () => cycleProfile(-1),
    'profile.toggleAll': toggleShowAllProfiles,
    'profile.create': requestProfileCreate
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Capture mode: the next real key becomes the binding. Swallow everything
      // so e.g. ⌘K rebinds instead of opening the palette.
      const capturing = $capture.get()

      if (capturing) {
        event.preventDefault()
        event.stopPropagation()

        if (event.key === 'Escape') {
          endCapture()

          return
        }

        const combo = comboFromEvent(event)

        if (!combo) {
          return
        }

        setBinding(capturing, [combo])
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

      const actionId = $comboIndex.get().get(combo)

      if (!actionId) {
        return
      }

      if (isEditableTarget(event.target) && !comboAllowedInInput(combo)) {
        return
      }

      const handler = handlersRef.current[actionId]

      if (!handler) {
        return
      }

      event.preventDefault()
      handler()
    }

    // Mac-app-switcher commit: lifting Ctrl with the overlay open lands on the
    // highlighted session. A window blur (Cmd+Tab away mid-switch) cancels so
    // the overlay never gets stranded waiting for a keyup that never comes.
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === 'Tab') {
        onSwitcherTabUp()
      }

      if (event.key === 'Control') {
        commitSwitcherRef.current()
      }
    }

    const onBlur = () => switcherActive() && closeSwitcher()

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

    return () => {
      window.removeEventListener('keydown', onKeyDown, { capture: true })
      window.removeEventListener('keyup', onKeyUp, { capture: true })
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('contextmenu', onContextMenu, { capture: true })
    }
  }, [])
}
