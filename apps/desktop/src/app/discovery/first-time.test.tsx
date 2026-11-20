import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $cronJobs } from '@/store/cron'
import { bindDesktopMetrics, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import { $firstTimeSplash, $firstTimeSplashHoldsSidebar } from '@/store/first-time-splash'
import { $guidanceLedger } from '@/store/guidance-ledger'
import { holdPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $newVersionReady } from '@/store/new-version'
import { $traySeen } from '@/store/notification-tray'
import { $portalAccent } from '@/store/portal-accent'
import { $renameSplash } from '@/store/rename-splash'
import { $sessions, $sessionsLoadError, $sessionsLoading } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'
import { $signInFailed, $visitorAccess } from '@/store/visitor-access'
import { ThemeProvider } from '@/themes/context'
import type { SessionInfo } from '@/types/hermes'

import { ComposerModelRow } from '../chat/composer/controls'
import type { ChatBarState } from '../chat/composer/types'
import { $artworkChoice, installArtworkAttributes } from '../redesign/artwork'

import { discoveryCopy } from './discovery-copy'
import { firstTimeCopy } from './first-time-copy'
import { HeldExchange } from './held-exchange'
import { NotificationBell } from './notification-tray'
import { SplashCoveredRoutes } from './splash-layer'
import { StarterIdeas } from './starter-ideas'

const copy = firstTimeCopy('en')
const tray = discoveryCopy('en').tray
const chat = { id: 'chat-1', title: 'Quarterly plan' } as SessionInfo
const SIGN_UP = { kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' } as const

/** The redesigned shell's pieces the splash deals with: the rail (with its bell) and the chat area. */
function renderShell() {
  return render(
    <ThemeProvider>
      <MemoryRouter>
        <nav data-slot="global-nav">
          <button data-nav-spot="cron" type="button">
            Routines
          </button>
          <NotificationBell />
        </nav>
        <main>
          <SplashCoveredRoutes chat={<Route element={<textarea aria-label="Message" />} index />} pages={null} />
        </main>
      </MemoryRouter>
    </ThemeProvider>
  )
}

/** The machine's first answer: nothing on it (a new install), or a chat (someone who used it). */
function settle(sessions: SessionInfo[]) {
  act(() => {
    $sessions.set(sessions)
    $sessionsLoading.set(false)
  })
}

const splash = () => screen.queryByRole('region', { name: copy.splash.title })
const firstTime = () => window.document.querySelector<HTMLElement>('[data-splash="first-time"]')

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $traySeen.set([])
  $renameSplash.set(null)
  $firstTimeSplash.set(null)
  $newVersionReady.set(false)
  $portalAccent.set('blue')
  $artworkChoice.set(null)
  $visitorAccess.set({ kind: 'open' })
  $signInFailed.set(false)
  takeHeldPrompt()
  $cronJobs.set([])
  $sessions.set([])
  $sessionsLoadError.set(false)
  $sessionsLoading.set(true)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  resetDesktopMetricsForTests()
})

describe('the first-time splash in the shell', () => {
  it('appears over an inert chat for a new install, and not for someone who used Hermes before', () => {
    renderShell()
    expect(splash()).toBeNull()

    settle([])
    expect(splash()).not.toBeNull()
    expect(screen.getByLabelText('Message').closest('[inert]')).not.toBeNull()
    cleanup()

    $guidanceLedger.set({})
    $firstTimeSplash.set(null)
    $sessionsLoading.set(true)
    renderShell()
    settle([chat])
    expect(splash()).toBeNull()
    expect(screen.getByRole('region', { name: discoveryCopy('en').renameSplash.title })).toBeTruthy()
  })

  it('says each place in one line, and its example, its lit place and its seen check once a row is reached', () => {
    renderShell()
    settle([])

    // Named by the rail's own label for the place, then its line; the example is its description.
    const routines = screen.getByRole('button', { name: name => name.endsWith(`: ${copy.splash.lines.routines.line}`) })
    const agents = screen.getByRole('button', { name: name => name.endsWith(`: ${copy.splash.lines.agents.line}`) })
    const routineRow = () => routines.closest('li')!
    expect(routineRow().dataset.active).toBeUndefined()
    expect(firstTime()?.dataset.place).toBeUndefined()
    expect(window.document.querySelectorAll('[data-slot="seen-check"]')).toHaveLength(0)
    expect($firstTimeSplashHoldsSidebar.get()).toBe(true)

    // The active row is the one whose example shows (discovery.css swaps it for the line).
    fireEvent.focus(routines)
    expect(routineRow().dataset.active).toBe('true')
    expect(firstTime()?.dataset.place).toBe('routines')
    expect(routines.getAttribute('aria-describedby')).toContain(' ')

    fireEvent.click(agents)
    expect(routineRow().dataset.active).toBeUndefined()
    expect(firstTime()?.dataset.place).toBe('agents')
    // Both rows reached keep their check for the rest of the showing.
    expect(routines.querySelector('[data-slot="seen-check"]')).not.toBeNull()
    expect(agents.querySelector('[data-slot="seen-check"]')).not.toBeNull()
    expect(window.document.querySelectorAll('[data-slot="seen-check"]')).toHaveLength(2)
  })

  it('hands the chat sidebar back while Chats is the place lit, and takes it again after', () => {
    renderShell()
    settle([])
    const chats = screen.getByRole('button', { name: name => name.endsWith(`: ${copy.splash.lines.chats.line}`) })
    const library = screen.getByRole('button', { name: name => name.endsWith(`: ${copy.splash.lines.library.line}`) })

    fireEvent.pointerEnter(chats, { pointerType: 'mouse' })
    expect($firstTimeSplashHoldsSidebar.get()).toBe(false)
    fireEvent.pointerEnter(library, { pointerType: 'mouse' })
    expect($firstTimeSplashHoldsSidebar.get()).toBe(true)

    fireEvent.keyDown(window, { key: 'Escape' })
    expect($firstTimeSplashHoldsSidebar.get()).toBe(false)
  })

  it('moves to Make it yours and back, with focus on each step', async () => {
    renderShell()
    settle([])

    fireEvent.click(screen.getByRole('button', { name: copy.splash.style.action }))
    const title = screen.getByRole('heading', { level: 1, name: copy.splash.theme.title })
    await vi.waitFor(() => expect(title.ownerDocument.activeElement).toBe(title))
    expect(screen.getByRole('region', { name: copy.splash.theme.title })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: copy.splash.theme.back }))
    const next = screen.getByRole('button', { name: copy.splash.style.action })
    await vi.waitFor(() => expect(next.ownerDocument.activeElement).toBe(next))
  })

  it('makes it yours: eight colours, light or dark and the artworks, as radio groups that change the app at once', () => {
    window.localStorage.setItem('hermes-boot-color-scheme', 'dark')
    const disposeArtwork = installArtworkAttributes(window.document.createElement('html'))
    renderShell()
    settle([])
    fireEvent.click(screen.getByRole('button', { name: copy.splash.style.action }))

    const colors = screen.getByRole('group', { name: copy.splash.theme.color })
    expect(
      within(colors)
        .getAllByRole('radio')
        .map(radio => radio.getAttribute('aria-label'))
    ).toEqual(Object.values(copy.splash.theme.colors))
    fireEvent.click(within(colors).getByRole('radio', { name: copy.splash.theme.colors.cyan }))
    expect($portalAccent.get()).toBe('cyan')

    const mode = screen.getByRole('group', { name: copy.splash.theme.mode })
    fireEvent.click(within(mode).getByRole('radio', { name: copy.splash.theme.dark }))
    expect(window.document.documentElement.classList.contains('dark')).toBe(true)

    const artworks = screen.getByRole('group', { name: copy.splash.theme.artwork.title })
    expect(within(artworks).getAllByRole('radio')).toHaveLength(5)
    fireEvent.click(within(artworks).getByRole('radio', { name: copy.splash.theme.artwork.names.karborn }))
    expect($artworkChoice.get()).toBe('karborn')
    expect(
      (within(artworks).getByRole('radio', { name: copy.splash.theme.artwork.names.karborn }) as HTMLInputElement)
        .checked
    ).toBe(true)
    expect(within(artworks).getByText(copy.splash.theme.artwork.selected)).toBeTruthy()

    // Light offers only what it draws.
    fireEvent.click(within(mode).getByRole('radio', { name: copy.splash.theme.light }))
    expect(
      within(screen.getByRole('group', { name: copy.splash.theme.artwork.title })).getAllByRole('radio')
    ).toHaveLength(2)
    disposeArtwork()
  })

  it('has no ×: Escape and any nav click close it, and it stays closed', () => {
    renderShell()
    settle([])
    expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(firstTime()).toBeNull()
    expect($guidanceLedger.get().firstTimeSplash).toBe('seen')

    act(() => $firstTimeSplash.set('reopened'))
    fireEvent.click(screen.getByRole('button', { name: 'Routines' }))
    expect(firstTime()).toBeNull()
    expect(screen.getByLabelText('Message').closest('[inert]')).toBeNull()
  })

  it('ends with one way in where nothing needs a sign up, and with Sign up or Look around where it does', () => {
    renderShell()
    settle([])
    fireEvent.click(screen.getByRole('button', { name: copy.splash.style.action }))
    expect(screen.getByRole('button', { name: copy.splash.start })).toBeTruthy()
    expect(screen.queryByRole('button', { name: copy.splash.signUp })).toBeNull()
    expect(screen.queryByRole('button', { name: copy.splash.lookAround })).toBeNull()
    cleanup()

    $guidanceLedger.set({})
    $firstTimeSplash.set(null)
    $sessionsLoading.set(true)
    $visitorAccess.set(SIGN_UP)
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })
    renderShell()
    settle([])
    fireEvent.click(screen.getByRole('button', { name: copy.splash.style.action }))
    expect(screen.queryByRole('button', { name: copy.splash.start })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: copy.splash.lookAround }))
    expect(firstTime()).toBeNull()
    expect(assign).not.toHaveBeenCalled()

    act(() => $firstTimeSplash.set('reopened'))
    fireEvent.click(screen.getByRole('button', { name: copy.splash.style.action }))
    fireEvent.click(screen.getByRole('button', { name: copy.splash.signUp }))
    expect(firstTime()).toBeNull()
    expect($guidanceLedger.get().firstTimeSplash).toBe('seen')
    expect(assign).toHaveBeenCalledTimes(1)
  })
})

describe('the tray for a first-timer', () => {
  it('carries sign up with the bell’s dot while it is required, then Meet Hermes once the splash was seen', () => {
    $visitorAccess.set(SIGN_UP)
    renderShell()
    const bell = screen.getByRole('button', { name: tray.title })
    expect(bell.getAttribute('aria-description')).toBe(tray.unseen)

    settle([])
    fireEvent.keyDown(window, { key: 'Escape' })
    fireEvent.click(bell)

    const panel = screen.getByRole('dialog', { name: tray.title })
    const cards = within(panel).getAllByRole('article')
    expect(within(cards[0]!).getByText(copy.tray.signUp.title)).toBeTruthy()
    expect(within(cards[0]!).getByText(copy.tray.signUp.description)).toBeTruthy()
    expect(within(cards[0]!).getByRole('button', { name: copy.tray.signUp.action })).toBeTruthy()
    expect(bell.getAttribute('aria-description')).toBeNull()

    const meet = cards.find(card => within(card).queryByText(copy.tray.meet.title))!
    expect(within(meet).getByText(copy.tray.meet.description)).toBeTruthy()
    fireEvent.click(within(meet).getByRole('button', { name: copy.tray.meet.action }))
    expect($firstTimeSplash.get()).toBe('reopened')
  })

  it('has no sign up for anyone who can already run things', () => {
    renderShell()
    fireEvent.click(screen.getByRole('button', { name: tray.title }))

    expect(screen.queryByText(copy.tray.signUp.title)).toBeNull()
  })

  it('says a failed sign in, with the way to try again, and lights the bell again', () => {
    $visitorAccess.set(SIGN_UP)
    renderShell()
    const bell = screen.getByRole('button', { name: tray.title })
    fireEvent.click(bell)
    fireEvent.click(bell)
    expect(bell.getAttribute('aria-description')).toBeNull()

    act(() => $signInFailed.set(true))
    expect(bell.getAttribute('aria-description')).toBe(tray.unseen)
    fireEvent.click(bell)
    expect(screen.getByText(copy.held.failedTitle)).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.held.tryAgain })).toBeTruthy()
  })
})

describe('the starter ideas', () => {
  const NEWCOMER = { renameSplash: 'not_applicable' } as const

  function renderIdeas() {
    const sent = vi.fn()
    const listener = (event: Event) => sent((event as CustomEvent<{ text: string }>).detail.text)
    window.addEventListener('hermes:composer-submit', listener)

    render(
      <div data-composer-surface-id="surface-1" data-composer-target="main">
        <StarterIdeas target="main" />
      </div>
    )

    return { sent, stop: () => window.removeEventListener('hermes:composer-submit', listener) }
  }

  it('are four fixed ideas, each sent through the composer as the person’s own message', () => {
    $guidanceLedger.set(NEWCOMER)
    const { sent, stop } = renderIdeas()

    const list = screen.getByRole('list', { name: copy.landing.ideasLabel })
    const ideas = Object.values(copy.landing.ideas)
    expect(
      within(list)
        .getAllByRole('button')
        .map(button => button.textContent)
    ).toEqual(ideas)
    expect(screen.getByRole('button', { name: copy.landing.hide })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: copy.landing.ideas.clean_inbox }))
    expect(sent).toHaveBeenCalledWith(copy.landing.ideas.clean_inbox)
    stop()
  })

  it('Hide ideas takes them away and leaves Show ideas focused; Show ideas brings them back, focused', async () => {
    $guidanceLedger.set(NEWCOMER)
    const { stop } = renderIdeas()

    fireEvent.click(screen.getByRole('button', { name: copy.landing.hide }))
    expect(screen.queryByRole('list', { name: copy.landing.ideasLabel })).toBeNull()
    const show = screen.getByRole('button', { name: copy.landing.showIdeas })
    await vi.waitFor(() => expect(show.ownerDocument.activeElement).toBe(show))

    fireEvent.click(show)
    const first = screen.getByRole('button', { name: copy.landing.ideas.morning_briefing })
    await vi.waitFor(() => expect(first.ownerDocument.activeElement).toBe(first))
    stop()
  })

  it('are not there for someone who used Hermes before', () => {
    $guidanceLedger.set({ renameSplash: 'seen' })
    const { stop } = renderIdeas()

    expect(screen.queryByRole('list', { name: copy.landing.ideasLabel })).toBeNull()
    expect(screen.queryByRole('button', { name: copy.landing.showIdeas })).toBeNull()
    stop()
  })
})

describe('the composer while sign up is required', () => {
  it('offers sign up where the model row would be', () => {
    $visitorAccess.set(SIGN_UP)
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })
    render(<ComposerModelRow disabled={false} model={{} as ChatBarState['model']} redesigned />)

    fireEvent.click(screen.getByRole('button', { name: copy.landing.signUp }))
    expect(assign).toHaveBeenCalledTimes(1)
  })
})

describe('a message held until sign up', () => {
  it('shows the message and Hermes’ local answer, with sign up, and says so when sign in failed', () => {
    $visitorAccess.set(SIGN_UP)
    holdPrompt('Compare the top 5 standing desks')
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })
    render(<HeldExchange primary routed={false} />)

    const yours = screen.getByRole('article', { name: copy.held.you })
    expect(within(yours).getByText('Compare the top 5 standing desks')).toBeTruthy()
    expect(within(yours).getByText(copy.held.justNow)).toBeTruthy()
    const reply = screen.getByRole('article', { name: copy.held.hermes })
    expect(within(reply).getByText(copy.held.title)).toBeTruthy()
    expect(within(reply).getByText(copy.held.body)).toBeTruthy()

    act(() => $signInFailed.set(true))
    expect(within(reply).getByText(copy.held.failedTitle)).toBeTruthy()
    expect(within(reply).getByText(copy.held.failedBody)).toBeTruthy()

    fireEvent.click(within(reply).getByRole('button', { name: copy.held.tryAgain }))
    expect(assign).toHaveBeenCalledTimes(1)
  })

  it('stands only in the window’s new chat, and only while sign up is required', () => {
    $visitorAccess.set(SIGN_UP)
    holdPrompt('Compare the top 5 standing desks')

    render(
      <>
        <HeldExchange primary routed />
        <HeldExchange primary={false} routed={false} />
      </>
    )
    expect(screen.queryByRole('article', { name: copy.held.you })).toBeNull()
    cleanup()

    $visitorAccess.set({ kind: 'open' })
    render(<HeldExchange primary routed={false} />)
    expect(screen.queryByRole('article', { name: copy.held.you })).toBeNull()
  })
})
