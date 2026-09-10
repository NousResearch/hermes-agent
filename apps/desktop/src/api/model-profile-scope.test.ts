import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import { getGlobalModelInfo, getGlobalModelOptions, setModelAssignment } from './models'
import { getProfiles } from './profiles'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
})

it('pins bulk model reads and writes to their target despite ambient profile changes', async () => {
  const api = vi.fn(async () => ({ ok: true }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  setApiRequestConnection('other-host')
  setApiRequestProfile('other-profile')
  const scope = { connectionId: 'local', profile: 'default' }
  await getProfiles(scope)
  await getGlobalModelOptions(undefined, scope)
  await getGlobalModelInfo(scope)
  await setModelAssignment({ scope: 'main', provider: 'example', model: 'example-model' }, scope)
  expect(api.mock.calls).toHaveLength(4)

  for (const [request] of api.mock.calls as unknown as [{ connectionId: string; profile: string }][]) {
    expect(request).toMatchObject(scope)
  }

  await getGlobalModelInfo('coder')
  expect(api).toHaveBeenLastCalledWith(expect.objectContaining({ connectionId: 'other-host', profile: 'coder' }))
})
