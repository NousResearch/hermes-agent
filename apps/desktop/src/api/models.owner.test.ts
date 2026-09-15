import { afterEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import type { MoaConfigResponse } from '@/types/hermes'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import * as models from './models'

const moa: MoaConfigResponse = {
  active_preset: 'fixture',
  default_preset: 'fixture',
  presets: {},
  aggregator: { provider: 'fixture-provider', model: 'fixture-model' },
  aggregator_temperature: 0,
  degraded_reference_policy: 'loud',
  enabled: false,
  reference_models: [],
  reference_temperature: 0,
  reference_timeout: null
}

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

it.each(['local', 'fixture-remote'])(
  'routes model reads and assignments to the explicit %s owner',
  async connectionId => {
    const owner = { connectionId, profile: 'shared-name' }

    const api = vi.fn(async (request: HermesApiRequest) => {
      expect.soft(request).toMatchObject(owner)

      return {}
    })

    vi.stubGlobal('hermesDesktop', { api })
    setApiRequestConnection('fixture-ambient')
    setApiRequestProfile('ambient-profile')

    await models.getGlobalModelInfo(owner)
    await models.getGlobalModelOptions(undefined, owner)
    await models.getRecommendedDefaultModel('fixture-provider', owner)
    await models.setGlobalModel('fixture-provider', 'fixture-model', owner)
    await models.getAuxiliaryModels(owner)
    await models.getMoaModels(owner)
    await models.saveMoaModels(moa, owner)
    await models.setModelAssignment({ scope: 'main', provider: 'fixture-provider', model: 'fixture-model' }, owner)
    await models.getUsageAnalytics(30, owner)
  }
)
