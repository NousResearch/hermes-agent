import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import { getComputerUseStatus, grantComputerUsePermissions } from './toolsets'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

it.each(['local', 'homelab'])(
  'routes computer readiness and permission actions to the explicit %s owner',
  async connectionId => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    vi.stubGlobal('hermesDesktop', { api })
    setApiRequestConnection('other-gateway')
    setApiRequestProfile('other-agent')
    const owner = { connectionId, profile: 'research' }

    await getComputerUseStatus(owner)
    await grantComputerUsePermissions(owner)

    expect(api).toHaveBeenNthCalledWith(1, {
      ...owner,
      priority: 'foreground',
      path: '/api/tools/computer-use/status'
    })
    expect(api).toHaveBeenNthCalledWith(2, {
      ...owner,
      priority: 'foreground',
      path: '/api/tools/computer-use/permissions/grant',
      method: 'POST'
    })
  }
)
