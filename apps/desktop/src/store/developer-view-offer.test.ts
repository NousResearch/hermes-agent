import { afterEach, beforeEach, describe, expect, it } from 'vitest'

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

describe('the developer-view offer (C6)', () => {
  it('is for someone in Simple, in the redesign only', () => {
    expect($developerViewOfferOpen.get()).toBe(true)

    setInterfaceMode('advanced')
    expect($developerViewOfferOpen.get()).toBe(false)

    setInterfaceMode('simple')
    $shellChoice.set(false)
    expect($developerViewOfferOpen.get()).toBe(false)
  })

  it('shows the developer view only while previewed, and keeps the offer up meanwhile', () => {
    previewDeveloperView(true)
    expect($interfaceMode.get()).toBe('advanced')
    expect($developerViewOfferOpen.get()).toBe(true)

    previewDeveloperView(false)
    expect($interfaceMode.get()).toBe('simple')
    expect($guidanceLedger.get().developerViewOffer).toBeUndefined()
  })

  it('keeps it on for good when turned on mid-preview, and is not offered again', () => {
    previewDeveloperView(true)
    acceptDeveloperView()
    previewDeveloperView(false)

    expect($interfaceMode.get()).toBe('advanced')
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
