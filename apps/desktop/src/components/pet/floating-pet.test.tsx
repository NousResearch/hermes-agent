import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// All mock state lives in vi.hoisted so the hoisted vi.mock factories can see
// it before module-level initialization (plain top-level refs hit the TDZ).
const mockStores = vi.hoisted(() => {
  const createStore = <T,>(initial: T) => {
    let value = initial
    const listeners = new Set<(next: T) => void>()

    return {
      get: () => value,
      set: (next: T) => {
        value = next
        listeners.forEach(listener => listener(value))
      },
      subscribe: (listener: (next: T) => void) => {
        listeners.add(listener)
        listener(value)

        return () => {
          listeners.delete(listener)
        }
      },
      listen: (listener: (next: T) => void) => {
        listeners.add(listener)

        return () => {
          listeners.delete(listener)
        }
      }
    }
  }

  return {
    gatewayState: createStore<'open' | 'closed'>('open'),
    overlayActive: createStore(false),
    activeGatewayProfile: createStore<string | null>(null),
    busy: createStore(false),
    requestGateway: vi.fn(async (method: string) => {
      if (method === 'pet.info') {
        return {
          enabled: true,
          slug: 'boba',
          displayName: 'Boba',
          spritesheetBase64: 'data:image/png;base64,AAAA',
          spritesheetRevision: 'sig:1',
          frameW: 192,
          frameH: 208,
          scale: 0.33
        }
      }

      return null
    })
  }
})

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: mockStores.requestGateway })
}))
vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' })
}))
vi.mock('@/app/hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: () => undefined
}))
vi.mock('@/app/hooks/use-route-overlay-active', () => ({
  useRouteOverlayActive: () => false
}))
vi.mock('@/components/chat/vibe-hearts', () => ({
  PetHeartField: () => null
}))
vi.mock('@/store/session', () => ({
  $gatewayState: mockStores.gatewayState,
  $busy: mockStores.busy
}))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: mockStores.activeGatewayProfile,
  normalizeProfileKey: (name: string | null | undefined) => name ?? 'default'
}))
vi.mock('@/store/pet-gallery', () => ({
  resetPetGallery: vi.fn(),
  setPetEnabled: vi.fn(),
  setPetScale: vi.fn()
}))
vi.mock('@/store/pet-overlay', () => ({
  $petOverlayActive: mockStores.overlayActive,
  initPetOverlayBridge: () => () => undefined,
  popOutPet: vi.fn(),
  restorePetOverlay: vi.fn()
}))
vi.mock('@/store/windows', () => ({
  isSecondaryWindow: () => false
}))
vi.mock('./use-pet-roam', () => ({
  usePetRoam: () => undefined
}))
vi.mock('./use-pet-zoom-gesture', () => ({
  usePetZoomGesture: () => undefined
}))
vi.mock('./pet-sprite', () => ({
  PetSprite: () => null,
  roamWalkRow: () => ({ mirror: false })
}))

import { notifyPetChanged, setChangeEventsAvailable } from '@/store/live-sync'
import {
  $petDismissed,
  $petInfo,
  $petMotion,
  setPetActivity,
  setPetBubble,
  setPetControls,
  setPetInfo
} from '@/store/pet'

import { FloatingPet } from './floating-pet'

let root: Root | null = null
let container: HTMLDivElement | null = null

function render(ui: ReactNode) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)

  act(() => {
    root!.render(ui)
  })
}

function cleanup() {
  if (root) {
    act(() => {
      root!.unmount()
    })
  }

  container?.remove()
  root = null
  container = null
}

beforeEach(() => {
  setPetInfo({ enabled: false })
  $petDismissed.set(false)
  setPetBubble(true)
  setPetControls(true)
  setPetActivity({})
  $petMotion.set(null)
  setChangeEventsAvailable(false)
  mockStores.gatewayState.set('open')
  mockStores.overlayActive.set(false)
  mockStores.requestGateway.mockClear()
})

afterEach(() => {
  cleanup()
})

describe('event-driven disable/re-enable reconcile', () => {
  it('clears $petDismissed on pet.changed(enabled=false) so a later enable is not discarded', async () => {
    setChangeEventsAvailable(true)
    // Hide was clicked: the gateway RPC is in flight and the dismiss flag is set.
    $petDismissed.set(true)

    render(<FloatingPet />)

    // The mount poll returns the pet as enabled, but the dismiss guard must
    // swallow it — otherwise Hide would be undone by a stale response.
    await act(async () => {})
    expect($petDismissed.get()).toBe(true)
    expect($petInfo.get().enabled).toBe(false)

    // Event-capable backend broadcasts the authoritative disable. The fast path
    // must reconcile the flag (this is the regression: previously it stayed set
    // forever on event-capable backends because the poll path never ran).
    await act(async () => {
      notifyPetChanged({ enabled: false })
    })

    expect($petDismissed.get()).toBe(false)
    expect($petInfo.get().enabled).toBe(false)

    // A later authoritative enable is now allowed through the guard.
    await act(async () => {
      notifyPetChanged({ enabled: true, slug: 'boba', displayName: 'Boba', spritesheetRevision: 'sig:1', scale: 0.33 })
    })

    expect($petInfo.get().enabled).toBe(true)
  })
})

describe('bubble edge positioning', () => {
  it('renders the bubble above the sprite so it cannot extend the constrained box past the viewport', () => {
    setPetBubble(true)
    setPetControls(false)
    setPetInfo({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      spritesheetBase64: 'data:image/png;base64,AAAA',
      spritesheetRevision: 'sig:1',
      frameW: 192,
      frameH: 208,
      scale: 0.33
    })

    render(<FloatingPet />)

    const host = container!.firstElementChild as HTMLElement | null
    expect(host).not.toBeNull()

    // The container box is sprite-only: the bubble must not inflate it, or the
    // clamp/roam geometry (which use sprite-only petH) would let the pet slip
    // below the viewport at the lower edge.
    expect(host!.style.height).toBe('')

    const bubbleHosts = Array.from(host!.querySelectorAll('div')).filter(
      el => el.style.position === 'absolute' && el.style.bottom === '100%'
    )
    expect(bubbleHosts).toHaveLength(1)

    // bottom: 100% pins the bubble above the sprite's top edge — at the lower
    // viewport bound the bubble extends *up*, never below the clamp boundary.
    const bubbleHost = bubbleHosts[0]
    expect(bubbleHost.style.bottom).toBe('100%')
    expect(bubbleHost.style.marginBottom).toBe('4px')
    expect(host!.contains(bubbleHost)).toBe(true)
  })
})
