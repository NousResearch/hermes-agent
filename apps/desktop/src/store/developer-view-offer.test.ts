import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $developerViewOfferOpen,
  acceptDeveloperView,
  declineDeveloperView,
  previewDeveloperView
} from './developer-view-offer'
import { $guidanceLedger } from './guidance-ledger'
import { $interfaceMode, setInterfaceMode } from './interface-mode'
import { $shellChoice } from './shell-variant'

beforeEach(() => {
  window.localStorage.clear()
  $shellChoice.set(true)
  $guidanceLedger.set({})
  setInterfaceMode('simple')
})

afterEach(() => {
  previewDeveloperView(false)
  setInterfaceMode('advanced')
})

const MODE_KEY = 'hermes.desktop.interfaceMode.v1'

describe('the developer-view offer (C6)', () => {
  it('is for someone in Simple, in the redesign only', () => {
    expect($developerViewOfferOpen.get()).toBe(true)

    setInterfaceMode('advanced')
    expect($developerViewOfferOpen.get()).toBe(false)

    setInterfaceMode('simple')
    $shellChoice.set(false)
    expect($developerViewOfferOpen.get()).toBe(false)
  })

  it('shows the developer view only while previewed, never storing it, and keeps the offer up meanwhile', () => {
    previewDeveloperView(true)
    expect($interfaceMode.get()).toBe('advanced')
    expect(window.localStorage.getItem(MODE_KEY)).toBe('simple')
    expect($developerViewOfferOpen.get()).toBe(true)

    previewDeveloperView(false)
    expect($interfaceMode.get()).toBe('simple')
    expect($guidanceLedger.get().developerViewOffer).toBeUndefined()
  })

  it('leaves the stored choice Simple when the window goes away mid-preview', async () => {
    previewDeveloperView(true)

    // A later page load (the window closed, or another window of the machine) reads the store.
    vi.resetModules()
    const fresh = await import('./interface-mode')

    expect(fresh.$interfaceMode.get()).toBe('simple')
  })

  it('keeps it on for good when turned on mid-preview, and is not offered again', () => {
    previewDeveloperView(true)
    acceptDeveloperView()
    previewDeveloperView(false)

    expect($interfaceMode.get()).toBe('advanced')
    // Advanced is stored as the absence of a choice.
    expect(window.localStorage.getItem(MODE_KEY)).toBeNull()
    expect($guidanceLedger.get().developerViewOffer).toBe('accepted')
    setInterfaceMode('simple')
    expect($developerViewOfferOpen.get()).toBe(false)
  })

  it('retires on Not now, leaving Simple as it was', () => {
    previewDeveloperView(true)
    declineDeveloperView()

    expect($interfaceMode.get()).toBe('simple')
    expect($guidanceLedger.get().developerViewOffer).toBe('declined')
    expect($developerViewOfferOpen.get()).toBe(false)
  })
})
