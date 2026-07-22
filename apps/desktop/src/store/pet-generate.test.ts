import { beforeEach, describe, expect, it } from 'vitest'

import {
  $petGenConcurrency,
  $petGenDraftCount,
  $petGenerateAboveRouteOverlay,
  $petGenerateOpen,
  $petGenModel,
  $petGenPoseAttempts,
  $petGenProvider,
  $petGenProviders,
  $petGenSeed,
  $petGenStyle,
  cleanPetName,
  closePetGenerate,
  openPetGenerate,
  resetPetGen,
  setPetGenProvider,
  yieldPetGenerateToRouteOverlay
} from './pet-generate'

describe('pet generation options', () => {
  beforeEach(() => {
    $petGenProviders.set([])
    $petGenProvider.set('')
    resetPetGen()
  })

  it('keeps advanced generation settings run-scoped', () => {
    $petGenModel.set('custom-model')
    $petGenStyle.set('pixel watercolor')
    $petGenSeed.set('42')
    $petGenDraftCount.set(7)
    $petGenConcurrency.set(2)
    $petGenPoseAttempts.set(3)

    resetPetGen()

    expect($petGenModel.get()).toBe('')
    expect($petGenStyle.get()).toBe('auto')
    expect($petGenSeed.get()).toBe('')
    expect($petGenDraftCount.get()).toBe(4)
    expect($petGenConcurrency.get()).toBe(4)
    expect($petGenPoseAttempts.get()).toBe(2)
  })

  it('derives a short pet name without prompt filler', () => {
    expect(cleanPetName('a cute cyber fox in watercolor style')).toBe('Cyber Fox Watercolor')
  })

  it('moves provider and compatible default model together', () => {
    $petGenProviders.set([
      {
        name: 'openai',
        label: 'OpenAI',
        default: true,
        defaultModel: 'gpt-image-2',
        models: [{ id: 'gpt-image-2' }]
      },
      {
        name: 'fal',
        label: 'FAL.ai',
        default: false,
        defaultModel: 'fal-edit',
        models: [{ id: 'fal-edit', supportsSeed: true }]
      }
    ])

    setPetGenProvider('fal')
    expect($petGenProvider.get()).toBe('fal')
    expect($petGenModel.get()).toBe('fal-edit')

    setPetGenProvider('')
    expect($petGenProvider.get()).toBe('')
    expect($petGenModel.get()).toBe('gpt-image-2')
  })

  it('can open above Appearance without making that the default', () => {
    openPetGenerate({ aboveRouteOverlay: true })

    expect($petGenerateOpen.get()).toBe(true)
    expect($petGenerateAboveRouteOverlay.get()).toBe(true)

    yieldPetGenerateToRouteOverlay()
    expect($petGenerateOpen.get()).toBe(true)
    expect($petGenerateAboveRouteOverlay.get()).toBe(false)

    closePetGenerate()
    openPetGenerate()

    expect($petGenerateOpen.get()).toBe(true)
    expect($petGenerateAboveRouteOverlay.get()).toBe(false)
  })
})
