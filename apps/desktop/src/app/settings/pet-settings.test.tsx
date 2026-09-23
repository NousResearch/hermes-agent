// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, test, vi } from 'vitest'

import { en } from '@/i18n/en'

const setPetEnabled = vi.hoisted(() => vi.fn(async () => true))
const setPetRoam = vi.hoisted(() => vi.fn())

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({ useGatewayRequest: () => ({ requestGateway: vi.fn() }) }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))
vi.mock('@/store/session', () => ({ $gatewayState: atom('closed') }))
vi.mock('@/store/pet', () => ({ $petInfo: atom({ enabled: false }), $petRoam: atom(false), setPetRoam }))
vi.mock('@/store/pet-gallery', () => ({
  $petBusy: atom(null),
  $petGallery: atom({ enabled: true, active: '', pets: [] }),
  $petGalleryError: atom(null),
  $petGalleryStatus: atom('ready'),
  adoptPet: vi.fn(),
  exportPet: vi.fn(),
  loadPetGallery: vi.fn(),
  loadPetThumb: vi.fn(),
  PET_SCALE_DEFAULT: 1,
  PET_SCALE_MAX: 2,
  PET_SCALE_MIN: 0.5,
  rankedGalleryPets: (gallery: { pets: [] }) => gallery?.pets ?? [],
  removePet: vi.fn(),
  renamePet: vi.fn(),
  setPetEnabled,
  setPetScale: vi.fn()
}))

import { PetSettings } from './pet-settings'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

test('uses named switches for pet enablement and roaming while preserving each state action', () => {
  render(<PetSettings />)

  const enable = screen.getByRole('switch', { name: en.settings.appearance.pet.chooseTitle })
  const roam = screen.getByRole('switch', { name: en.settings.appearance.pet.roamTitle })
  expect(enable).toHaveProperty('ariaChecked', 'true')
  expect(roam).toHaveProperty('ariaChecked', 'false')

  fireEvent.click(enable)
  fireEvent.click(roam)

  expect(setPetEnabled).toHaveBeenCalledWith(expect.any(Function), false, expect.any(Object))
  expect(setPetRoam).toHaveBeenCalledWith(true)
})
