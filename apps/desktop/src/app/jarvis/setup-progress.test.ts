import { expect, it } from 'vitest'

import { buildSetupChecklist } from './setup-progress'

const fresh = { briefingTried: false, hasHistory: false, modelReady: false }

it('starts with the model and moves on only as things become true', () => {
  const start = buildSetupChecklist(fresh)

  expect(start.next).toBe('model')
  expect(start.doneCount).toBe(0)

  const later = buildSetupChecklist({ ...fresh, googleConnected: true, modelReady: true })

  expect(later.next).toBe('memory')
  expect(later.doneCount).toBe(2)
})

it('never counts an unknown answer as done, and completes only when everything is', () => {
  expect(buildSetupChecklist({ ...fresh, googleConnected: undefined }).steps.find(s => s.id === 'google')?.done).toBe(false)

  const all = buildSetupChecklist({
    briefingTried: true,
    browserSignedIn: true,
    googleConnected: true,
    hasHistory: true,
    modelReady: true,
    vaultExists: true
  })

  expect(all.complete).toBe(true)
  expect(all.next).toBeNull()
})
