import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

import { setRightSidebarTab } from '@/app/right-sidebar/store'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { matchesQuery } from '@/hooks/use-media-query'
import { PROFILE_SLOT_COUNT } from '@/lib/keybinds/actions'
import { comboAllowedInInput, comboFromEvent, isEditableTarget } from '@/lib/keybinds/combo'
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
  cycleProfile,
  requestProfileCreate,
  switchProfileToSlot,
  switchToDefaultProfile,
  toggleShowAllProfiles
} from '@/store/profile'
import { $activeSessionId, $sessions, setModelPickerOpen } from '@/store/session'
import { useTheme } from '@/themes/context'

import { requestComposerFocus } from '../chat/composer/focus'
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

  const profileSwitchHandlers: HandlerMap = {}

  for (let slot = 1; slot <= PROFILE_SLOT_COUNT; slot += 1) {
    profileSwitchHandlers[`profile.switch.${slot}`] = () => switchProfileToSlot(slot)
  }

  // Move to the adjacent session in recency order, wrapping at the ends.
  const cycleSession = (direction: 1 | -1) => {
    const sessions = $sessions.get()

    if (sessions.length < 2) {
      return
    }

    const current = sessions.findIndex(session => session.id === $activeSessionId.get())
    const start = current === -1 ? (direction === 1 ? -1 : 0) : current
    const next = sessions[(start + direction + sessions.length) % sessions.length]

    if (next) {
      navigate(sessionRoute(next.id))
    }
  }

  const showRightSidebarTab = (tab: 'files' | 'terminal') => {
    setFileBrowserOpen(true)
    setRightSidebarTab(tab)
  }

  handlersRef.current = {
    'keybinds.openPanel': toggleKeybindPanel,

    'composer.focus': () => requestComposerFocus('main'),
    'composer.modelPicker': () => setModelPickerOpen(true),

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
      deps.startFreshSession()
      window.dispatchEvent(new CustomEvent('hermes:new-session-shortcut'))
    },
    'session.next': () => cycleSession(1),
    'session.prev': () => cycleSession(-1),
    'session.focusSearch': requestSessionSearchFocus,
    'session.togglePin': deps.toggleSelectedPin,

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
    'view.showFiles': () => showRightSidebarTab('files'),
    'view.showTerminal': () => showRightSidebarTab('terminal'),
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

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [])
}
