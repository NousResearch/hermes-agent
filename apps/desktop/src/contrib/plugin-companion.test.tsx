import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CompanionBubble } from '@/components/companion/companion-bubble'
import { FloatingCompanion } from '@/components/companion/floating-companion'
import { $companionActivity, setCompanionInfo } from '@/store/companion'
import { $companionOverlayActive, popInCompanion, popOutCompanion } from '@/store/companion-overlay'
import { $companionPluginMessages, mirrorCompanionPluginMessages, resetCompanionPluginMessages } from '@/store/companion-plugin-messages'

import { createPluginContext } from './plugin'
import { publishPlugin } from './plugins-store'

// The in-window companion's gateway/theme plumbing is unrelated to the bubble; the
// sprite is a canvas jsdom can't paint, so stand it in with the same a11y
// label the real one carries.
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn(async () => null) })
}))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ resolvedMode: 'dark' }) }))
vi.mock('@/components/companion/companion-sprite', () => ({
  CompanionSprite: () => <canvas aria-label="Boba companion" />,
  roamWalkRow: () => ({ mirror: false, row: undefined })
}))

const COMPANION = {
  displayName: 'Boba',
  enabled: true,
  frameH: 208,
  frameW: 192,
  scale: 0.33,
  spritesheetBase64: 'stub'
}

// Each test gets its own plugin id, so per-plugin rate windows never leak
// between tests; afterEach resets the store so a live line cannot leak.
let n = 0

function plugin(name = 'Companion Wallet') {
  const id = `companion-wallet-${++n}`
  const disposers: Array<() => void> = []
  publishPlugin({ id, kind: 'disk', name, status: 'loaded' })
  const ctx = createPluginContext(id, dispose => disposers.push(dispose))

  return { ctx, id, unload: () => act(() => disposers.splice(0).forEach(dispose => dispose())) }
}

function mountCompanion() {
  return render(
    <MemoryRouter>
      <FloatingCompanion />
    </MemoryRouter>
  )
}

const bubble = () => document.querySelector('[data-slot="companion-plugin-bubble"]')

describe('ctx.companion — plugin lines in the core companion bubble', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $companionActivity.set({})
    $companionOverlayActive.set(false)
    setCompanionInfo(COMPANION)
  })

  afterEach(() => {
    cleanup()
    resetCompanionPluginMessages()
    vi.clearAllTimers()
    vi.useRealTimers()
    setCompanionInfo({ enabled: false })
  })

  it('shows the line above the in-window companion, labelled with the plugin name', () => {
    const { ctx, id } = plugin()
    mountCompanion()
    expect(bubble()).toBeNull()

    act(() => void ctx.companion.say('DeepSeek ¥12.40 left'))

    expect(screen.getByText('DeepSeek ¥12.40 left')).toBeTruthy()
    expect(bubble()?.getAttribute('data-companion-plugin')).toBe(id)
    expect(bubble()?.textContent).toContain('Companion Wallet')
    expect(ctx.companion.visible.get()).toBe(true)
  })

  it('clears the line when its TTL runs out (default 6 s, clamped)', () => {
    const { ctx } = plugin()
    mountCompanion()

    act(() => void ctx.companion.say('short', { ttlMs: 2_000 }))
    expect(screen.queryByText('short')).toBeTruthy()
    act(() => void vi.advanceTimersByTime(1_999))
    expect(screen.queryByText('short')).toBeTruthy()
    act(() => void vi.advanceTimersByTime(1))
    expect(screen.queryByText('short')).toBeNull()

    act(() => void ctx.companion.say('default'))
    act(() => void vi.advanceTimersByTime(5_999))
    expect(screen.queryByText('default')).toBeTruthy()
    act(() => void vi.advanceTimersByTime(1))
    expect(screen.queryByText('default')).toBeNull()

    // A huge TTL is capped at 30 s — a plugin can't pin the bubble forever.
    act(() => void ctx.companion.say('forever', { ttlMs: 10 * 60_000 }))
    act(() => void vi.advanceTimersByTime(30_000))
    expect(screen.queryByText('forever')).toBeNull()
  })

  it('clears the line through the returned disposer and through clear(id)', () => {
    const { ctx } = plugin()
    mountCompanion()

    let dispose = () => {}
    act(() => void (dispose = ctx.companion.say('bye soon')))
    act(() => dispose())
    expect(screen.queryByText('bye soon')).toBeNull()

    act(() => void ctx.companion.say('keyed', { id: 'balance' }))
    act(() => ctx.companion.clear('balance'))
    expect(bubble()).toBeNull()
  })

  it('replaces a line in place when the same id is said again', () => {
    const { ctx } = plugin()
    mountCompanion()

    let first = () => {}
    act(() => void (first = ctx.companion.say('balance ¥12', { id: 'balance' })))
    act(() => void ctx.companion.say('balance ¥11', { id: 'balance' }))
    expect(screen.queryByText('balance ¥12')).toBeNull()
    expect(screen.getByText('balance ¥11')).toBeTruthy()

    // The first line's disposer must not retire its replacement.
    act(() => first())
    expect(screen.getByText('balance ¥11')).toBeTruthy()
  })

  it('clears every line a plugin has up when the plugin unloads or is disabled', () => {
    const a = plugin('Companion Wallet')
    const b = plugin('Other')
    mountCompanion()

    act(() => void b.ctx.companion.say('from other', { ttlMs: 30_000 }))
    act(() => void a.ctx.companion.say('from wallet', { ttlMs: 30_000 }))
    expect(screen.getByText('from wallet')).toBeTruthy()

    a.unload()

    expect(screen.queryByText('from wallet')).toBeNull()
    // Only the unloaded plugin's lines go; another plugin's line resurfaces.
    expect(screen.getByText('from other')).toBeTruthy()
  })

  it('renders plain text only: markup stays literal, control chars go, length is capped', () => {
    const { ctx } = plugin('<b>Evil</b> Wallet')
    mountCompanion()

    act(() => void ctx.companion.say('<img src=x onerror="alert(1)"> hi\u202e\n\tthere'))
    expect(bubble()?.querySelector('img, b')).toBeNull()
    expect(screen.getByText('<img src=x onerror="alert(1)"> hi there')).toBeTruthy()
    expect(bubble()?.textContent).toContain('<b>Evil</b> Wallet')

    act(() => void ctx.companion.say('x'.repeat(500), { id: 'long' }))
    const shown = screen.getByText(/^x+…$/).textContent ?? ''
    expect(shown.length).toBe(120)

    // Nothing to say → nothing shown, and a no-op disposer.
    act(() => void ctx.companion.say('   \n  ', { id: 'blank' }))
    expect(screen.queryByText('   ')).toBeNull()
  })

  it('shows nothing while the companion is hidden or turned off', () => {
    const { ctx } = plugin()
    setCompanionInfo({ enabled: false })
    mountCompanion()

    act(() => void ctx.companion.say('anyone there?', { ttlMs: 30_000 }))
    expect(screen.queryByText('anyone there?')).toBeNull()
    expect(ctx.companion.visible.get()).toBe(false)

    // Turning the companion on while the line is live shows it — the host gates on
    // the user's companion setting, not the plugin.
    act(() => setCompanionInfo(COMPANION))
    expect(screen.getByText('anyone there?')).toBeTruthy()
    act(() => setCompanionInfo({ enabled: false }))
    expect(screen.queryByText('anyone there?')).toBeNull()
  })

  it('rate-limits a chatty plugin', () => {
    const { ctx } = plugin()
    mountCompanion()
    vi.spyOn(console, 'warn').mockImplementation(() => {})

    act(() => {
      for (let i = 0; i < 10; i++) {
        ctx.companion.say(`line ${i}`, { id: `l${i % 3}` })
      }
    })
    expect(screen.getByText('line 9')).toBeTruthy()

    act(() => void ctx.companion.say('one too many'))
    expect(screen.queryByText('one too many')).toBeNull()

    act(() => void vi.advanceTimersByTime(10_000))
    act(() => void ctx.companion.say('window reopened'))
    expect(screen.getByText('window reopened')).toBeTruthy()
  })
})

describe('CompanionBubble — core status keeps priority', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $companionActivity.set({})
  })

  afterEach(() => {
    cleanup()
    resetCompanionPluginMessages()
    vi.clearAllTimers()
    vi.useRealTimers()
    $companionActivity.set({})
  })

  it('lets error / waiting-on-you states win over a plugin line, then yields back', () => {
    const { ctx } = plugin()
    render(<CompanionBubble />)

    act(() => void ctx.companion.say('balance ¥12', { ttlMs: 30_000 }))
    expect(screen.getByText('balance ¥12')).toBeTruthy()

    act(() => $companionActivity.set({ error: true }))
    expect(screen.queryByText('balance ¥12')).toBeNull()

    act(() => $companionActivity.set({ awaitingInput: true }))
    expect(screen.queryByText('balance ¥12')).toBeNull()

    act(() => $companionActivity.set({}))
    expect(screen.getByText('balance ¥12')).toBeTruthy()
  })

  it('shows a plugin tone glyph and leaves the core status line to the overlay', () => {
    const { ctx } = plugin()
    render(<CompanionBubble showStatus={false} />)

    // In-window (showStatus=false) a busy turn says nothing on its own…
    act(() => $companionActivity.set({ busy: true }))
    expect(document.body.textContent).toBe('')

    // …but a plugin line still shows, with its tone glyph.
    act(() => void ctx.companion.say('quota low', { tone: 'wait' }))
    expect(screen.getByText('quota low')).toBeTruthy()
    expect(bubble()?.querySelector('svg')).toBeTruthy()
  })
})

describe('pop-out overlay', () => {
  afterEach(() => {
    popInCompanion()
    cleanup()
    resetCompanionPluginMessages()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
    setCompanionInfo({ enabled: false })
  })

  it('pushes live plugin lines to the overlay window, which renders them in its bubble', async () => {
    // The overlay is a separate gateway-less window: the main renderer pushes
    // its companion state over IPC. Capture that push and replay it the way the
    // overlay's onState handler does.
    const pushState = vi.fn()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      companionOverlay: { close: vi.fn(async () => ({ ok: true })), open: vi.fn(async () => ({ ok: true })), pushState }
    }
    setCompanionInfo(COMPANION)
    popOutCompanion({ height: 70, width: 64, x: 10, y: 10 })

    const { ctx, id } = plugin()
    act(() => void ctx.companion.say('popped-out hello', { ttlMs: 30_000 }))

    const last = pushState.mock.calls.at(-1)?.[0] as { pluginMessages?: unknown[] }
    expect(last.pluginMessages).toEqual([expect.objectContaining({ pluginId: id, text: 'popped-out hello' })])

    // Overlay side: adopt the pushed list verbatim and render the core bubble.
    const pushed = structuredClone(last.pluginMessages)
    resetCompanionPluginMessages()
    act(() => mirrorCompanionPluginMessages(pushed))
    render(<CompanionBubble />)
    expect(screen.getByText('popped-out hello')).toBeTruthy()

    // A malformed push clears rather than throws.
    act(() => mirrorCompanionPluginMessages(undefined))
    expect($companionPluginMessages.get()).toEqual([])
  })
})
