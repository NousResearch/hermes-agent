import { cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactElement } from 'react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useKeybinds } from './use-keybinds'

// The handler under test (`nav.backToChat`) only cares about the current
// pathname and `navigate` — the rest of the keybind runtime is irrelevant noise
// for these route-guard assertions. Mock the heavyweight modules so mounting
// `useKeybinds` (which wires a global keydown listener) is cheap and isolated.

const navigate = vi.fn()
const toggleCommandCenter = vi.fn()
const startFreshSession = vi.fn()
const toggleSelectedPin = vi.fn()

vi.mock('react-router-dom', async () => {
  // `typeof import('react-router-dom')` is the idiom vitest ships for typed
  // importActual, but the repo's `@typescript-eslint/consistent-type-imports`
  // rule flags inline `import()` type annotations — disable just here.
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')

  return {
    ...actual,
    useNavigate: () => navigate
  }
})

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() })
}))

vi.mock('@/store/keybinds', () => {
  const capture = atom<null | string>(null)
  const comboIndex = atom(new Map<string, string>([['escape', 'nav.backToChat']]))

  return {
    $capture: capture,
    $comboIndex: comboIndex,
    endCapture: vi.fn(),
    setBinding: vi.fn(),
    toggleKeybindPanel: vi.fn()
  }
})

vi.mock('@/store/command-palette', () => ({
  toggleCommandPalette: vi.fn()
}))

vi.mock('@/store/session-switcher', () => ({
  $switcherOpen: atom(false),
  closeSwitcher: vi.fn(),
  commitOnCtrlUp: vi.fn(() => null),
  onSwitcherTabDown: vi.fn(),
  onSwitcherTabUp: vi.fn(),
  openOrAdvanceSwitcher: vi.fn(() => null),
  slotSessionId: vi.fn(() => null),
  switcherActive: vi.fn(() => false),
  switcherJustClosed: vi.fn(() => false)
}))

vi.mock('@/store/session', () => ({
  setModelPickerOpen: vi.fn()
}))

vi.mock('@/store/layout', () => ({
  CHAT_SIDEBAR_PANE_ID: 'chat-sidebar',
  FILE_BROWSER_PANE_ID: 'file-browser',
  requestSessionSearchFocus: vi.fn(),
  setFileBrowserOpen: vi.fn(),
  toggleFileBrowserOpen: vi.fn(),
  togglePanesFlipped: vi.fn(),
  toggleSidebarOpen: vi.fn()
}))

vi.mock('@/store/profile', () => ({
  $newChatProfile: atom(null),
  cycleProfile: vi.fn(),
  requestProfileCreate: vi.fn(),
  switchProfileToSlot: vi.fn(),
  switchToDefaultProfile: vi.fn(),
  toggleShowAllProfiles: vi.fn()
}))

vi.mock('@/store/review', () => ({
  toggleReview: vi.fn()
}))

vi.mock('@/store/projects', () => ({
  requestNewWorktree: vi.fn()
}))

vi.mock('@/store/coding-status', () => ({
  $repoStatus: atom(null)
}))

vi.mock('@/store/windows', () => ({
  openNewSessionInNewWindow: vi.fn()
}))

vi.mock('@/app/right-sidebar/store', () => ({
  $terminalTakeover: atom(false),
  setTerminalTakeover: vi.fn()
}))

vi.mock('@/app/right-sidebar/terminal/terminals', () => ({
  closeActiveTerminal: vi.fn(),
  createTerminal: vi.fn(),
  cycleTerminal: vi.fn()
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestVoiceToggle: vi.fn()
}))

vi.mock('@/components/pane-shell', () => ({
  PANE_TOGGLE_REVEAL_EVENT: 'pane:toggle-reveal'
}))

vi.mock('@/hooks/use-media-query', () => ({
  matchesQuery: vi.fn(() => false),
  useMediaQuery: vi.fn(() => false)
}))

function HarnessProbe(): ReactElement {
  // Render the current location so the test can confirm `navigate` was called
  // by inspecting the post-Esc pathname rendered into the DOM.
  const location = useLocation()

  return <div data-testid="pathname">{location.pathname}</div>
}

function renderKeybindsAt(pathname: string) {
  return render(
    <MemoryRouter initialEntries={[pathname]}>
      <HarnessProbe />
    </MemoryRouter>
  )
}

function pressEscape(target: Element | Window = window): void {
  // useKeybinds subscribes via window.addEventListener('keydown', ..., {capture: true}).
  // fireEvent.keyDown(window, ...) bypasses the capture phase, so dispatch a
  // real KeyboardEvent on window instead — and set `code` so comboFromEvent
  // (which derives the binding token from `event.code`, not `event.key`) maps
  // it to the "escape" combo the actions.ts entry ships.
  const event = new KeyboardEvent('keydown', { key: 'Escape', code: 'Escape', bubbles: true, cancelable: true })

  ;(target as Window).dispatchEvent(event)
}

describe('useKeybinds nav.backToChat (Esc route guard)', () => {
  beforeEach(() => {
    navigate.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('navigates to "/" when Esc is pressed on /settings', async () => {
    // useKeybinds imported statically at top of file

    function Probe() {
      useKeybinds({
        startFreshSession,
        toggleCommandCenter,
        toggleSelectedPin
      })

      return <HarnessProbe />
    }

    render(
      <MemoryRouter initialEntries={['/settings']}>
        <Probe />
      </MemoryRouter>
    )

    pressEscape()

    expect(navigate).toHaveBeenCalledTimes(1)
    expect(navigate).toHaveBeenCalledWith('/')
  })

  it('navigates to "/" when Esc is pressed on /cron (overlay)', async () => {
    // useKeybinds imported statically at top of file

    function Probe() {
      useKeybinds({
        startFreshSession,
        toggleCommandCenter,
        toggleSelectedPin
      })

      return <HarnessProbe />
    }

    render(
      <MemoryRouter initialEntries={['/cron']}>
        <Probe />
      </MemoryRouter>
    )

    pressEscape()

    expect(navigate).toHaveBeenCalledTimes(1)
    expect(navigate).toHaveBeenCalledWith('/')
  })

  it('does NOT navigate when Esc is pressed on the chat view (/)', async () => {
    // useKeybinds imported statically at top of file

    function Probe() {
      useKeybinds({
        startFreshSession,
        toggleCommandCenter,
        toggleSelectedPin
      })

      return <HarnessProbe />
    }

    render(
      <MemoryRouter initialEntries={['/']}>
        <Probe />
      </MemoryRouter>
    )

    pressEscape()

    expect(navigate).not.toHaveBeenCalled()
  })

  it('does NOT navigate when Esc is pressed on a session route (/abc-123)', async () => {
    // useKeybinds imported statically at top of file

    function Probe() {
      useKeybinds({
        startFreshSession,
        toggleCommandCenter,
        toggleSelectedPin
      })

      return <HarnessProbe />
    }

    render(
      <MemoryRouter initialEntries={['/abc-123']}>
        <Probe />
      </MemoryRouter>
    )

    pressEscape()

    expect(navigate).not.toHaveBeenCalled()
  })

  // Quietly satisfy the helper so the import above does not flag unused locals.
  it('sanity: harness renders the current pathname', () => {
    const { getByTestId } = renderKeybindsAt('/settings')
    expect(getByTestId('pathname').textContent).toBe('/settings')
  })
})
