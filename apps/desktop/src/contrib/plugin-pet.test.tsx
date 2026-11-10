import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { FloatingPet } from '@/components/pet/floating-pet'
import { PetBubble } from '@/components/pet/pet-bubble'
import { $petActivity, setPetInfo } from '@/store/pet'
import { $petOverlayActive, popInPet, popOutPet } from '@/store/pet-overlay'
import { $petPluginMessages, mirrorPetPluginMessages, resetPetPluginMessages } from '@/store/pet-plugin-messages'

import { createPluginContext } from './plugin'
import { publishPlugin } from './plugins-store'

// The in-window pet's gateway/theme plumbing is unrelated to the bubble; the
// sprite is a canvas jsdom can't paint, so stand it in with the same a11y
// label the real one carries.
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn(async () => null) })
}))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ resolvedMode: 'dark' }) }))
vi.mock('@/components/pet/pet-sprite', () => ({
  PetSprite: () => <canvas aria-label="Boba pet" />,
  roamWalkRow: () => ({ mirror: false, row: undefined })
}))

const PET = {
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

function plugin(name = 'Pet Wallet') {
  const id = `pet-wallet-${++n}`
  const disposers: Array<() => void> = []
  publishPlugin({ id, kind: 'disk', name, status: 'loaded' })
  const ctx = createPluginContext(id, dispose => disposers.push(dispose))

  return { ctx, id, unload: () => act(() => disposers.splice(0).forEach(dispose => dispose())) }
}

function mountPet() {
  return render(
    <MemoryRouter>
      <FloatingPet />
    </MemoryRouter>
  )
}

const bubble = () => document.querySelector('[data-slot="pet-plugin-bubble"]')

describe('ctx.pet — plugin lines in the core pet bubble', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $petActivity.set({})
    $petOverlayActive.set(false)
    setPetInfo(PET)
  })

  afterEach(() => {
    cleanup()
    resetPetPluginMessages()
    vi.clearAllTimers()
    vi.useRealTimers()
    setPetInfo({ enabled: false })
  })

  it('shows the line above the in-window pet, labelled with the plugin name', () => {
    const { ctx, id } = plugin()
    mountPet()
    expect(bubble()).toBeNull()

    act(() => void ctx.pet.say('DeepSeek ¥12.40 left'))

    expect(screen.getByText('DeepSeek ¥12.40 left')).toBeTruthy()
    expect(bubble()?.getAttribute('data-pet-plugin')).toBe(id)
    expect(bubble()?.textContent).toContain('Pet Wallet')
    expect(ctx.pet.visible.get()).toBe(true)
  })

  it('clears the line when its TTL runs out (default 6 s, clamped)', () => {
    const { ctx } = plugin()
    mountPet()

    act(() => void ctx.pet.say('short', { ttlMs: 2_000 }))
    expect(screen.queryByText('short')).toBeTruthy()
    act(() => void vi.advanceTimersByTime(1_999))
    expect(screen.queryByText('short')).toBeTruthy()
    act(() => void vi.advanceTimersByTime(1))
    expect(screen.queryByText('short')).toBeNull()

    act(() => void ctx.pet.say('default'))
    act(() => void vi.advanceTimersByTime(5_999))
    expect(screen.queryByText('default')).toBeTruthy()
    act(() => void vi.advanceTimersByTime(1))
    expect(screen.queryByText('default')).toBeNull()

    // A huge TTL is capped at 30 s — a plugin can't pin the bubble forever.
    act(() => void ctx.pet.say('forever', { ttlMs: 10 * 60_000 }))
    act(() => void vi.advanceTimersByTime(30_000))
    expect(screen.queryByText('forever')).toBeNull()
  })

  it('clears the line through the returned disposer and through clear(id)', () => {
    const { ctx } = plugin()
    mountPet()

    let dispose = () => {}
    act(() => void (dispose = ctx.pet.say('bye soon')))
    act(() => dispose())
    expect(screen.queryByText('bye soon')).toBeNull()

    act(() => void ctx.pet.say('keyed', { id: 'balance' }))
    act(() => ctx.pet.clear('balance'))
    expect(bubble()).toBeNull()
  })

  it('replaces a line in place when the same id is said again', () => {
    const { ctx } = plugin()
    mountPet()

    let first = () => {}
    act(() => void (first = ctx.pet.say('balance ¥12', { id: 'balance' })))
    act(() => void ctx.pet.say('balance ¥11', { id: 'balance' }))
    expect(screen.queryByText('balance ¥12')).toBeNull()
    expect(screen.getByText('balance ¥11')).toBeTruthy()

    // The first line's disposer must not retire its replacement.
    act(() => first())
    expect(screen.getByText('balance ¥11')).toBeTruthy()
  })

  it('clears every line a plugin has up when the plugin unloads or is disabled', () => {
    const a = plugin('Pet Wallet')
    const b = plugin('Other')
    mountPet()

    act(() => void b.ctx.pet.say('from other', { ttlMs: 30_000 }))
    act(() => void a.ctx.pet.say('from wallet', { ttlMs: 30_000 }))
    expect(screen.getByText('from wallet')).toBeTruthy()

    a.unload()

    expect(screen.queryByText('from wallet')).toBeNull()
    // Only the unloaded plugin's lines go; another plugin's line resurfaces.
    expect(screen.getByText('from other')).toBeTruthy()
  })

  it('renders plain text only: markup stays literal, control chars go, length is capped', () => {
    const { ctx } = plugin('<b>Evil</b> Wallet')
    mountPet()

    act(() => void ctx.pet.say('<img src=x onerror="alert(1)"> hi\u202e\n\tthere'))
    expect(bubble()?.querySelector('img, b')).toBeNull()
    expect(screen.getByText('<img src=x onerror="alert(1)"> hi there')).toBeTruthy()
    expect(bubble()?.textContent).toContain('<b>Evil</b> Wallet')

    act(() => void ctx.pet.say('x'.repeat(500), { id: 'long' }))
    const shown = screen.getByText(/^x+…$/).textContent ?? ''
    expect(shown.length).toBe(120)

    // Nothing to say → nothing shown, and a no-op disposer.
    act(() => void ctx.pet.say('   \n  ', { id: 'blank' }))
    expect(screen.queryByText('   ')).toBeNull()
  })

  it('shows nothing while the pet is hidden or turned off', () => {
    const { ctx } = plugin()
    setPetInfo({ enabled: false })
    mountPet()

    act(() => void ctx.pet.say('anyone there?', { ttlMs: 30_000 }))
    expect(screen.queryByText('anyone there?')).toBeNull()
    expect(ctx.pet.visible.get()).toBe(false)

    // Turning the pet on while the line is live shows it — the host gates on
    // the user's pet setting, not the plugin.
    act(() => setPetInfo(PET))
    expect(screen.getByText('anyone there?')).toBeTruthy()
    act(() => setPetInfo({ enabled: false }))
    expect(screen.queryByText('anyone there?')).toBeNull()
  })

  it('rate-limits a chatty plugin', () => {
    const { ctx } = plugin()
    mountPet()
    vi.spyOn(console, 'warn').mockImplementation(() => {})

    act(() => {
      for (let i = 0; i < 10; i++) {
        ctx.pet.say(`line ${i}`, { id: `l${i % 3}` })
      }
    })
    expect(screen.getByText('line 9')).toBeTruthy()

    act(() => void ctx.pet.say('one too many'))
    expect(screen.queryByText('one too many')).toBeNull()

    act(() => void vi.advanceTimersByTime(10_000))
    act(() => void ctx.pet.say('window reopened'))
    expect(screen.getByText('window reopened')).toBeTruthy()
  })
})

describe('PetBubble — core status keeps priority', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $petActivity.set({})
  })

  afterEach(() => {
    cleanup()
    resetPetPluginMessages()
    vi.clearAllTimers()
    vi.useRealTimers()
    $petActivity.set({})
  })

  it('lets error / waiting-on-you states win over a plugin line, then yields back', () => {
    const { ctx } = plugin()
    render(<PetBubble />)

    act(() => void ctx.pet.say('balance ¥12', { ttlMs: 30_000 }))
    expect(screen.getByText('balance ¥12')).toBeTruthy()

    act(() => $petActivity.set({ error: true }))
    expect(screen.queryByText('balance ¥12')).toBeNull()

    act(() => $petActivity.set({ awaitingInput: true }))
    expect(screen.queryByText('balance ¥12')).toBeNull()

    act(() => $petActivity.set({}))
    expect(screen.getByText('balance ¥12')).toBeTruthy()
  })

  it('shows a plugin tone glyph and leaves the core status line to the overlay', () => {
    const { ctx } = plugin()
    render(<PetBubble showStatus={false} />)

    // In-window (showStatus=false) a busy turn says nothing on its own…
    act(() => $petActivity.set({ busy: true }))
    expect(document.body.textContent).toBe('')

    // …but a plugin line still shows, with its tone glyph.
    act(() => void ctx.pet.say('quota low', { tone: 'wait' }))
    expect(screen.getByText('quota low')).toBeTruthy()
    expect(bubble()?.querySelector('svg')).toBeTruthy()
  })
})

describe('pop-out overlay', () => {
  afterEach(() => {
    popInPet()
    cleanup()
    resetPetPluginMessages()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
    setPetInfo({ enabled: false })
  })

  it('pushes live plugin lines to the overlay window, which renders them in its bubble', async () => {
    // The overlay is a separate gateway-less window: the main renderer pushes
    // its pet state over IPC. Capture that push and replay it the way the
    // overlay's onState handler does.
    const pushState = vi.fn()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      petOverlay: { close: vi.fn(async () => ({ ok: true })), open: vi.fn(async () => ({ ok: true })), pushState }
    }
    setPetInfo(PET)
    popOutPet({ height: 70, width: 64, x: 10, y: 10 })

    const { ctx, id } = plugin()
    act(() => void ctx.pet.say('popped-out hello', { ttlMs: 30_000 }))

    const last = pushState.mock.calls.at(-1)?.[0] as { pluginMessages?: unknown[] }
    expect(last.pluginMessages).toEqual([expect.objectContaining({ pluginId: id, text: 'popped-out hello' })])

    // Overlay side: adopt the pushed list verbatim and render the core bubble.
    const pushed = structuredClone(last.pluginMessages)
    resetPetPluginMessages()
    act(() => mirrorPetPluginMessages(pushed))
    render(<PetBubble />)
    expect(screen.getByText('popped-out hello')).toBeTruthy()

    // A malformed push clears rather than throws.
    act(() => mirrorPetPluginMessages(undefined))
    expect($petPluginMessages.get()).toEqual([])
  })
})
