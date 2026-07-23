import { describe, expect, it, vi } from 'vitest'

import { MOA_STUDIO_ROUTE, selectMoaPresetInChat } from './moa-studio-actions'

describe('MoA Studio shell actions', () => {
  it('uses the existing persistent model-selection callback for the active chat', async () => {
    const selectModel = vi.fn(async () => true)

    await expect(selectMoaPresetInChat(selectModel, 'BeastMode')).resolves.toBe(true)
    expect(selectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa' })
  })

  it('targets the dedicated settings deep link', () => {
    expect(MOA_STUDIO_ROUTE).toBe('/settings?tab=moa')
  })
})
