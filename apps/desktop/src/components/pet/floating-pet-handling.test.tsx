import { act, cleanup, fireEvent, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startPetNap } from '@/app/contrib/hooks/use-pet-bridge'
import {
  $petActivity,
  $petHandling,
  $petRoam,
  $petRoamDir,
  $petSpriteTarget,
  clearPetReaction,
  onPetInteract,
  type PetBehaviors,
  type PetInteraction,
  reactPet,
  setPetActivity,
  setPetInfo
} from '@/store/pet'
import { $petOverlayActive } from '@/store/pet-overlay'
import { mountPet, PET, press } from '@/test/floating-pet'
import { stubMenuDomApis } from '@/test/jsdom'

import { groundTop, snapshotLedges } from './roam-geometry'

// Gateway/theme plumbing is unrelated to handling; the sprite is a canvas jsdom
// can't paint.
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn(async () => null) })
}))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ resolvedMode: 'dark' }) }))
vi.mock('./pet-sprite', () => ({ PetSprite: () => <canvas aria-label="Boba pet" /> }))

const PET_H = 208 * 0.5

const MAPPED = { ...PET, behaviors: { rows: { fall: ['tumble'], held: ['dangle'], land: ['thud'], poke: ['giggle'] } } }

let frames: FrameRequestCallback[] = []
let heard: PetInteraction[] = []
let asleep: boolean[] = []

let off = () => {}

function runFrames() {
  let t = performance.now()

  while (frames.length) {
    const batch = frames.splice(0)
    t += 16
    act(() => batch.forEach(cb => cb(t)))
  }
}

describe('FloatingPet handling (pet.json poke / held / fall / land)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    stubMenuDomApis() // pointer capture
    $petActivity.set({})
    $petOverlayActive.set(false)
    frames = []
    heard = []
    asleep = []
    off = onPetInteract((kind, info) => {
      heard.push(kind)
      asleep.push(info.asleep)
    })
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => frames.push(cb))
    vi.stubGlobal('cancelAnimationFrame', () => {})
  })

  afterEach(() => {
    off()
    cleanup()
    vi.unstubAllGlobals()
    vi.clearAllTimers()
    vi.useRealTimers()
    setPetInfo({ enabled: false })
  })

  it('a press that moves less than the threshold is a poke: one loop of its row, no drag', () => {
    const el = mountPet(MAPPED)
    const top = el.style.top

    press(el, [10, 10], [13, 12])

    expect(heard).toEqual(['poke'])
    expect($petHandling.get()).toBe('poke')
    expect(el.style.top).toBe(top)
    act(() => void vi.advanceTimersByTime(1000))
    expect($petHandling.get()).toBeNull()
  })

  it('a carry plays held, and a pet without `fall` stays where it was dropped', () => {
    const el = mountPet({ ...PET, behaviors: { rows: { held: ['dangle'] } } })

    press(el, [0, 0], [40, 50], () => expect($petSpriteTarget.get().state).toBe('held'))
    runFrames()

    expect(heard).toEqual(['pickup', 'drop'])
    expect($petHandling.get()).toBeNull()
    expect(el.style.top).toBe('50px')
  })

  it('with `fall` mapped, a pet dropped in mid-air falls to the ground and lands: pickup → drop → land', () => {
    const el = mountPet(MAPPED)

    press(el, [0, 0], [40, 50])
    expect(heard).toEqual(['pickup', 'drop'])
    expect($petHandling.get()).toBe('fall')

    runFrames()

    expect(heard).toEqual(['pickup', 'drop', 'land'])
    expect(el.style.top).toBe(`${groundTop(snapshotLedges(192 * 0.5, PET_H)[0]!, PET_H)}px`)
    expect($petHandling.get()).toBe('land')
  })

  it('with reduced motion the drop snaps to the ground and the pet is free again', () => {
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: query.includes('reduced-motion') }))
    const el = mountPet(MAPPED)

    press(el, [0, 0], [40, 50])
    runFrames()
    expect(heard).toEqual(['pickup', 'drop', 'land'])
    expect($petHandling.get()).toBe('land')

    // Nothing is left "falling" (which would also freeze the wander): a new
    // press doesn't treat the landed pet as caught mid-air.
    act(() => void fireEvent.pointerDown(el, { clientX: 60, clientY: 60, pointerId: 1 }))
    expect($petHandling.get()).toBe('land')
  })

  it('a cancelled press is neither a poke nor stuck held', () => {
    const el = mountPet(MAPPED)

    fireEvent.pointerDown(el, { clientX: 0, clientY: 0, pointerId: 1 })
    fireEvent.pointerCancel(el, { pointerId: 1 })
    expect(heard).toEqual([])

    fireEvent.pointerDown(el, { clientX: 0, clientY: 0, pointerId: 1 })
    fireEvent.pointerMove(el, { clientX: 40, clientY: 50, pointerId: 1 })
    fireEvent.pointerCancel(el, { pointerId: 1 })
    expect(heard).toEqual(['pickup', 'drop'])
    expect($petHandling.get()).not.toBe('held')
  })

  describe('roaming', () => {
    afterEach(() => {
      clearPetReaction('test')
      $petRoam.set(false)
      $petRoamDir.set(0)
      vi.restoreAllMocks()
    })

    // Advance the clock frame by frame (the wander mixes timers and RAF).
    const tick = (ms: number) => {
      for (let t = 0; t < ms; t += 16) {
        act(() => {
          vi.advanceTimersByTime(16)
          frames.splice(0).forEach(cb => cb(performance.now()))
        })
      }
    }

    it('faces the way it travels whatever row shows: a directional row keeps its own facing, a pose faces travel', () => {
      mountPet({ ...PET, stateRows: ['idle', 'running-right', 'running-left'] })
      const facing = () => screen.getByLabelText('Boba pet').parentElement!.style.transform

      act(() => $petRoamDir.set(1))
      expect(facing()).toBe('none') // running-right already faces right

      act(() => void reactPet('test', 'wave', 5_000))
      expect(facing()).toBe('scaleX(-1)') // the wave art faces left: mirror it rightward

      act(() => clearPetReaction('test'))
      act(() => $petRoamDir.set(-1))
      expect(facing()).toBe('none')
    })

    it('a plugin reaction pauses the wander: the pet holds still while it plays, then roams on', () => {
      vi.spyOn(Math, 'random').mockReturnValue(0.9) // every beat moves
      $petRoam.set(true)
      const el = mountPet(PET)
      const at = () => [el.style.left, el.style.top]
      const start = at()

      act(() => void reactPet('test', 'wave', 5_000))
      tick(4_800)
      expect(at()).toEqual(start)

      tick(3_000)
      expect(at()).not.toEqual(start)
    })
  })

  describe('a sleeping pet (behaviors.sleep)', () => {
    let stopNap = () => {}

    // The real wiring: any interaction wakes the pet (startPetNap), so `asleep` must be read before it.
    beforeEach(() => {
      stopNap = startPetNap()
    })

    afterEach(() => stopNap())

    // Mount a pet with these handling rows (plus a `yawn` row) and let it doze off.
    const sleepyPet = (rows: PetBehaviors['rows'] = {}) => {
      const el = mountPet({ ...PET, stateRows: [...PET.stateRows, 'yawn'], behaviors: { rows, sleepAfterSeconds: 60 } })
      act(() => setPetActivity({ sleepy: true }))

      return el
    }

    // [case, rows, drag to, handling played]; plugins hear `asleep` before the wake.
    it.each<[string, PetBehaviors['rows'], [number, number] | undefined, string]>([
      ['a poke plays `wake` instead of poke', { poke: ['giggle'], wake: ['yawn'] }, undefined, 'wake'],
      ['without a `wake` row a poke falls back to poke', { poke: ['giggle'] }, undefined, 'poke'],
      ['picking it up plays held, not wake', { held: ['dangle'], wake: ['yawn'] }, [40, 50], 'held']
    ])('%s and wakes it', (_case, rows, to, state) => {
      let held: null | string = null
      press(sleepyPet(rows), [0, 0], to, () => (held = $petHandling.get()))

      expect(to ? held : $petHandling.get()).toBe(state)
      expect($petActivity.get().sleepy).toBe(false)
      expect(asleep).toEqual(to ? [true, false] : [true])
    })
  })
})
