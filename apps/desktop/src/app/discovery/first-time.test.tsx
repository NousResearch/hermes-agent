import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $artifactRegistry, upsertArtifact } from '@/store/artifacts'
import { $cronJobs } from '@/store/cron'
import { bindDesktopMetrics, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import { $developerViewOfferOpen } from '@/store/developer-view-offer'
import { $firstTimeSplash, $firstTimeSplashPlace } from '@/store/first-time-splash'
import { $guidanceLedger } from '@/store/guidance-ledger'
import { holdPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $interfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { $newVersionReady } from '@/store/new-version'
import { $trayCleared, $traySeen } from '@/store/notification-tray'
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

import { DeveloperViewOffer } from './developer-view-offer'
import { discoveryCopy } from './discovery-copy'
import { firstTimeCopy } from './first-time-copy'
import { HeldExchange } from './held-exchange'
import { NotificationBell } from './notification-tray'
import { SplashCoveredRoutes } from './splash-layer'
import { LandingGreeting } from './starter-ideas'

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
          <button data-nav-spot="chat" type="button">
            Chats
          </button>
          <button data-nav-spot="profiles" type="button">
            Agents
          </button>
          <button data-nav-spot="capabilities" type="button">
            Connections
          </button>
          <button data-nav-spot="cron" type="button">
            Routines
          </button>
          <NotificationBell />
        </nav>
        {/* The chat sidebar's agent group: Chats must never light this (Figma 18924:28898). */}
        <div data-zone-pane="sessions">
          <div data-slot="agent-group">Quarterly plan</div>
        </div>
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
const spotlight = () => window.document.querySelector<HTMLElement>('[data-slot="nav-spotlight"]')

type Place = keyof typeof copy.splash.lines

/** The row whose accessible name carries the place's line and example: both always show, together. */
function placeRow(place: Place) {
  const { example, line } = copy.splash.lines[place]

  return screen.getByRole('button', { name: name => name.includes(line) && name.includes(example) })
}

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $traySeen.set([])
  $trayCleared.set([])
  $renameSplash.set(null)
  $firstTimeSplash.set(null)
  $firstTimeSplashPlace.set(null)
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
  vi.restoreAllMocks()
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

  it('shows all four places with their line and example together, at rest with nothing lit', () => {
    renderShell()
    settle([])

    const list = screen.getByRole('list', { name: copy.splash.listLabel })
    expect(within(list).getAllByRole('button')).toHaveLength(4)
    // The card opens on 18695:17147 with no row lit and nothing in the nav.
    expect($firstTimeSplashPlace.get()).toBeNull()
    expect(firstTime()?.dataset.place).toBeUndefined()
    expect(list.querySelectorAll('[data-active]')).toHaveLength(0)
    expect(spotlight()).toBeNull()
    expect(window.document.querySelectorAll('[data-splash-highlight]')).toHaveLength(0)
    expect(window.document.querySelectorAll('[data-slot="seen-check"]')).toHaveLength(0)
    expect(list.querySelector('.discovery-welcome-seen')).toBeNull()

    // Every row says its line and its example at once; nothing is swapped in under the pointer.
    for (const place of ['chats', 'agents', 'connections', 'routines'] as const) {
      const { example, line } = copy.splash.lines[place]
      const row = placeRow(place)

      expect(row.textContent).toContain(line)
      expect(row.textContent).toContain(example)
      expect(row.getAttribute('aria-label')).toContain(`${line} ${example}`)
    }

    fireEvent.focus(placeRow('routines'))
    expect(placeRow('routines').closest('li')?.dataset.active).toBe('true')
    expect(placeRow('agents').closest('li')?.dataset.active).toBeUndefined()
    expect(firstTime()?.dataset.place).toBe('routines')
    // Reaching a row is reading, not a check: no seen marks remain.
    expect(window.document.querySelectorAll('[data-slot="seen-check"]')).toHaveLength(0)
  })

  // A click leaves focus on the row. Only the keyboard's focus keeps the rest stepped back: a pointer
  // that has left would otherwise find "Make it yours" still greyed.
  it('lights one row under the pointer, dims the rest by a fifth, and returns to rest as the pointer leaves', async () => {
    renderShell()
    settle([])
    const list = screen.getByRole('list', { name: copy.splash.listLabel })

    fireEvent.pointerEnter(list)
    expect(spotlight()).toBeNull()

    fireEvent.pointerEnter(placeRow('chats'), { pointerType: 'mouse' })
    expect(placeRow('chats').closest('li')?.dataset.active).toBe('true')
    expect(firstTime()?.dataset.place).toBe('chats')
    expect(spotlight()?.dataset.dim).toBe('true')
    expect(spotlight()?.dataset.railHighlight).toBe('true')
    // No dashed elbow: the rail button's own glow carries the highlight.
    await vi.waitFor(() =>
      expect(spotlight()!.querySelector('svg path:not(.discovery-spotlight-veil)')!.getAttribute('d')).toBe('')
    )

    fireEvent.pointerEnter(placeRow('agents'), { pointerType: 'mouse' })
    expect(placeRow('chats').closest('li')?.dataset.active).toBeUndefined()
    expect(placeRow('agents').closest('li')?.dataset.active).toBe('true')
    expect(firstTime()?.dataset.place).toBe('agents')

    // A click leaves focus on its row; that alone must not hold the rest back after the pointer goes.
    fireEvent.click(placeRow('agents'))
    fireEvent.focus(placeRow('agents'))
    fireEvent.pointerLeave(list)
    expect($firstTimeSplashPlace.get()).toBeNull()
    expect(placeRow('agents').closest('li')?.dataset.active).toBeUndefined()
    expect(firstTime()?.dataset.place).toBeUndefined()
    expect(spotlight()).toBeNull()
  })

  it('keeps a keyboard-focused row lit when the pointer leaves, and clears it when focus leaves the list', () => {
    renderShell()
    settle([])
    const list = screen.getByRole('list', { name: copy.splash.listLabel })
    const connections = placeRow('connections')
    // jsdom never reports :focus-visible, so say this focus arrived by keyboard.
    const native = Element.prototype.matches
    const matches = vi
      .spyOn(Element.prototype, 'matches')
      .mockImplementation(function (this: Element, selector: string) {
        return selector === ':focus-visible' ? this === connections : native.call(this, selector)
      })

    fireEvent.focus(connections)
    expect(spotlight()?.dataset.dim).toBe('true')
    fireEvent.pointerEnter(list)
    fireEvent.pointerEnter(connections, { pointerType: 'mouse' })
    expect(connections.closest('li')?.dataset.active).toBe('true')

    fireEvent.pointerLeave(list)
    expect(connections.closest('li')?.dataset.active).toBe('true')
    // The keyboard still holds the row, so the rest stays stepped back around it.
    expect(spotlight()?.dataset.dim).toBe('true')

    fireEvent.blur(connections, { relatedTarget: window.document.body })
    expect($firstTimeSplashPlace.get()).toBeNull()
    expect(connections.closest('li')?.dataset.active).toBeUndefined()
    expect(spotlight()).toBeNull()
    matches.mockRestore()
  })

  it('points Chats at its rail button alone, never the sidebar’s agent group, and Connections at Capabilities', async () => {
    renderShell()
    settle([])
    const rail = (area: string) => window.document.querySelector<HTMLElement>(`[data-nav-spot="${area}"]`)!
    const chat = rail('chat')
    const capabilities = rail('capabilities')
    const agentGroup = window.document.querySelector<HTMLElement>('[data-slot="agent-group"]')!
    const box = (left: number) =>
      ({
        left,
        top: 100,
        right: left + 40,
        bottom: 132,
        width: 40,
        height: 32,
        x: left,
        y: 100,
        toJSON: () => ({})
      }) as DOMRect
    chat.getBoundingClientRect = () => box(10)
    capabilities.getBoundingClientRect = () => box(10)
    agentGroup.getBoundingClientRect = () => box(200)

    // jsdom has no layout: show the rail and the sidebar as visible, and let the hit test pick a place.
    const nativeComputed = window.getComputedStyle.bind(window)
    vi.stubGlobal('getComputedStyle', (element: Element) =>
      new Proxy(nativeComputed(element), {
        get(target, property) {
          if (property === 'overflowX' || property === 'overflowY') {
            return 'visible'
          }

          const value = Reflect.get(target, property)

          return typeof value === 'function' ? value.bind(target) : value
        }
      })
    )
    const nativeFromPoint = window.document.elementFromPoint
    const restoreFromPoint = () => {
      if (nativeFromPoint) {
        window.document.elementFromPoint = nativeFromPoint
      } else {
        Reflect.deleteProperty(window.document, 'elementFromPoint')
      }
    }
    window.document.elementFromPoint = (x: number) => (x >= 200 ? agentGroup : chat)

    try {
      fireEvent.pointerEnter(placeRow('chats'), { pointerType: 'mouse' })
      await vi.waitFor(() => expect(chat.getAttribute('data-splash-highlight')).toBe('true'))

      expect(agentGroup.hasAttribute('data-splash-highlight')).toBe(false)
      expect(window.document.querySelectorAll('[data-splash-highlight="true"]')).toHaveLength(1)
      expect(spotlight()?.dataset.railHighlight).toBe('true')
      expect(spotlight()!.querySelector('svg path:not(.discovery-spotlight-veil)')!.getAttribute('d')).toBe('')

      const connections = placeRow('connections')
      window.document.elementFromPoint = () => capabilities
      fireEvent.pointerEnter(connections, { pointerType: 'mouse' })
      await vi.waitFor(() => expect(capabilities.getAttribute('data-splash-highlight')).toBe('true'))

      expect(chat.hasAttribute('data-splash-highlight')).toBe(false)
      expect(agentGroup.hasAttribute('data-splash-highlight')).toBe(false)
      expect(window.document.querySelectorAll('[data-splash-highlight="true"]')).toHaveLength(1)
    } finally {
      restoreFromPoint()
    }
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
    expect(within(artworks).getAllByRole('radio')).toHaveLength(10)
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

  it('keeps a way back to Meet Hermes after its row is cleared', () => {
    renderShell()
    settle([])
    fireEvent.keyDown(window, { key: 'Escape' })
    fireEvent.click(screen.getByRole('button', { name: tray.title }))

    const panel = screen.getByRole('dialog', { name: tray.title })
    const meet = within(panel).getByText(copy.tray.meet.title).closest('article')!
    fireEvent.click(within(meet).getByRole('button', { name: tray.dismiss }))

    // The row is cleared to an empty tray, but the welcome screen is still reachable from here.
    expect(within(panel).queryByText(copy.tray.meet.title)).toBeNull()
    expect(within(panel).getByText(tray.empty)).toBeTruthy()
    fireEvent.click(within(panel).getByRole('button', { name: tray.openIntroduction }))
    expect($firstTimeSplash.get()).toBe('reopened')
  })

  it('has no sign up for anyone who can already run things', () => {
    renderShell()
    fireEvent.click(screen.getByRole('button', { name: tray.title }))

    expect(screen.queryByText(copy.tray.signUp.title)).toBeNull()
  })

  it('keeps the ordinary sign-up card when sign in failed: the recovery screen says that', () => {
    $visitorAccess.set(SIGN_UP)
    $signInFailed.set(true)
    renderShell()
    fireEvent.click(screen.getByRole('button', { name: tray.title }))

    expect(screen.getByText(copy.tray.signUp.title)).toBeTruthy()
  })
})

describe('a newcomer’s landing as they settle in (Onboarding 03)', () => {
  function renderLanding() {
    return render(
      <MemoryRouter>
        <LandingGreeting target="main" />
      </MemoryRouter>
    )
  }

  const heading = () => screen.getByRole('heading', { level: 1 }).textContent

  it('says Welcome back after a first task (C4, C5)', () => {
    $guidanceLedger.set({ renameSplash: 'not_applicable', finishedTaskChats: ['a'] })
    renderLanding()

    expect(heading()).toBe(copy.landing.back)
    expect(screen.getByText(copy.landing.subtitleRetired)).toBeTruthy()
  })

  it('after a month away says hello again, without the tour (C8)', () => {
    $guidanceLedger.set({ renameSplash: 'not_applicable', finishedTaskChats: ['a'] })
    const longAgo = Date.now() / 1000 - 40 * 24 * 60 * 60
    $sessions.set([{ ...chat, id: 'old', last_active: longAgo } as SessionInfo])
    renderLanding()

    expect(heading()).toBe(copy.landing.awayTitle)
    expect(screen.getByText(copy.landing.awayLine)).toBeTruthy()
    expect($firstTimeSplash.get()).toBeNull()
  })

  it('puts the check-in in the heading’s place at two finished chats (C7)', () => {
    $guidanceLedger.set({ renameSplash: 'not_applicable', finishedTaskChats: ['a', 'b'] })
    renderLanding()

    expect(screen.getByRole('region', { name: copy.checkIn.title })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: new RegExp(copy.checkIn.dismiss) }))
    expect(screen.queryByRole('region', { name: copy.checkIn.title })).toBeNull()
    expect(heading()).toBe(copy.landing.back)
    expect($guidanceLedger.get().checkIn).toBe('dismissed')
  })

  it('sends a suggestion through the composer like a typed message, and Open reopens the tour', () => {
    $guidanceLedger.set({ renameSplash: 'not_applicable', firstTimeSplash: 'seen', finishedTaskChats: ['a', 'b'] })
    const sent = vi.fn()
    const listener = (event: Event) => sent((event as CustomEvent<{ text: string }>).detail.text)
    window.addEventListener('hermes:composer-submit', listener)
    render(
      <MemoryRouter>
        <div data-composer-surface-id="surface-1" data-composer-target="main">
          <LandingGreeting target="main" />
        </div>
      </MemoryRouter>
    )

    const rows = screen.getByRole('list', { name: copy.checkIn.listLabel })
    fireEvent.click(within(rows).getByRole('button', { name: new RegExp(copy.checkIn.video) }))
    expect(sent).toHaveBeenCalledWith(copy.checkIn.video)
    fireEvent.click(within(rows).getByRole('button', { name: new RegExp(copy.checkIn.tour) }))
    expect($firstTimeSplash.get()).toBe('reopened')
    window.removeEventListener('hermes:composer-submit', listener)
  })

  it('comes back from the tray once dismissed', () => {
    $guidanceLedger.set({ renameSplash: 'not_applicable', finishedTaskChats: ['a', 'b'], checkIn: 'dismissed' })
    renderShell()
    fireEvent.click(screen.getByRole('button', { name: tray.title }))
    fireEvent.click(screen.getByRole('button', { name: copy.tray.checkIn.action }))
    cleanup()

    renderLanding()
    expect(screen.getByRole('region', { name: copy.checkIn.title })).toBeTruthy()
  })
})

describe('the developer-view offer in a chat about code (C6)', () => {
  afterEach(() => {
    $artifactRegistry.set({})
    setInterfaceMode('advanced')
  })

  it('is offered in Simple once an answer held a code artifact, and previews the view on hover', () => {
    setInterfaceMode('simple')
    const { rerender } = render(<DeveloperViewOffer sessionId="chat-1" />)
    expect(screen.queryByRole('region', { name: copy.developerView.title })).toBeNull()

    act(() => {
      upsertArtifact('chat-1', { kind: 'code', language: 'python', title: 'fix.py' }, 'print(1)')
    })
    rerender(<DeveloperViewOffer sessionId="chat-1" />)
    const preview = screen.getByRole('button', { name: new RegExp(copy.developerView.preview) })

    fireEvent.pointerEnter(preview)
    expect($interfaceMode.get()).toBe('advanced')
    expect(window.localStorage.getItem('hermes.desktop.interfaceMode.v1')).toBe('simple')
    fireEvent.pointerLeave(preview)
    expect($interfaceMode.get()).toBe('simple')

    fireEvent.click(screen.getByRole('button', { name: copy.developerView.notNow }))
    expect($developerViewOfferOpen.get()).toBe(false)
    expect(screen.queryByRole('region', { name: copy.developerView.title })).toBeNull()
  })

  it('either way the offer goes, the preview it started ends with it', async () => {
    setInterfaceMode('simple')
    act(() => {
      upsertArtifact('chat-1', { kind: 'code', language: 'python', title: 'fix.py' }, 'print(1)')
    })

    const { rerender, unmount } = render(<DeveloperViewOffer sessionId="chat-1" />)
    fireEvent.pointerEnter(screen.getByRole('button', { name: new RegExp(copy.developerView.preview) }))
    expect($interfaceMode.get()).toBe('advanced')

    // The chat changes out from under the card: it stops rendering without unmounting.
    rerender(<DeveloperViewOffer sessionId="chat-2" />)
    await waitFor(() => expect($interfaceMode.get()).toBe('simple'))

    rerender(<DeveloperViewOffer sessionId="chat-1" />)
    fireEvent.pointerEnter(screen.getByRole('button', { name: new RegExp(copy.developerView.preview) }))
    expect($interfaceMode.get()).toBe('advanced')

    unmount()
    await waitFor(() => expect($interfaceMode.get()).toBe('simple'))
  })

  it('turns the view on for good', () => {
    setInterfaceMode('simple')
    act(() => {
      upsertArtifact('chat-1', { kind: 'code', language: 'python', title: 'fix.py' }, 'print(1)')
    })
    render(<DeveloperViewOffer sessionId="chat-1" />)

    fireEvent.click(screen.getByRole('button', { name: copy.developerView.turnOn }))
    expect($interfaceMode.get()).toBe('advanced')
    expect($guidanceLedger.get().developerViewOffer).toBe('accepted')
  })
})

describe('the return from a sign up that did not sign the visitor in', () => {
  const recovery = () => screen.queryByRole('region', { name: copy.recovery.title })

  it('covers the chat with Let’s try signing in again, and every way on starts sign up again', () => {
    $visitorAccess.set(SIGN_UP)
    $signInFailed.set(true)
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })
    renderShell()

    expect(recovery()).not.toBeNull()
    expect(screen.getByLabelText('Message').closest('[inert]')).not.toBeNull()
    const steps = screen.getByRole('list', { name: copy.recovery.listLabel })
    expect(within(steps).getByText(copy.recovery.unverified)).toBeTruthy()
    expect(within(steps).getByText(copy.recovery.waiting)).toBeTruthy()
    // No message held: no request row, and the line does not promise one.
    expect(within(steps).queryByText(copy.recovery.saved)).toBeNull()
    expect(screen.getByText(copy.recovery.subtitle)).toBeTruthy()

    // Leaving for sign up clears the failure; each way on is tried from a fresh return.
    for (const name of [copy.recovery.tryAgain, copy.recovery.otherEmail, copy.recovery.requestNew]) {
      act(() => $signInFailed.set(true))
      fireEvent.click(screen.getByRole('button', { name }))
    }

    expect(assign).toHaveBeenCalledTimes(3)
  })

  it('names the held request as saved', () => {
    $visitorAccess.set(SIGN_UP)
    $signInFailed.set(true)
    holdPrompt('Compare the top 5 standing desks')
    renderShell()

    const steps = screen.getByRole('list', { name: copy.recovery.listLabel })
    expect(within(steps).getByText('Compare the top 5 standing desks')).toBeTruthy()
    expect(within(steps).getByText(copy.recovery.saved)).toBeTruthy()
    expect(screen.getByText(copy.recovery.subtitleHeld)).toBeTruthy()
  })

  it('goes away on Escape or a nav click, leaving the ordinary signed-out app', () => {
    $visitorAccess.set(SIGN_UP)
    $signInFailed.set(true)
    renderShell()

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(recovery()).toBeNull()
    expect($signInFailed.get()).toBe(false)

    act(() => $signInFailed.set(true))
    fireEvent.click(screen.getByRole('button', { name: 'Routines' }))
    expect(recovery()).toBeNull()
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
  it('shows the message and Hermes’ local answer, with sign up', () => {
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

    fireEvent.click(within(reply).getByRole('button', { name: copy.held.signUp }))
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
