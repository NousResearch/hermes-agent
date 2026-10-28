import { afterAll, expect, it, vi } from 'vitest'

import { getApiRequestProfile } from '@/hermes'
import { syncApprovalModeForProfile } from '@/store/approval-mode'
import { activeGatewayProfileKey, requestGatewayForProfile, setPrimaryGateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

// A Webapp tab's one socket answers for the profile its gateway runs as, and the gateway refuses a
// request naming a profile it does not serve (`profile_mismatch`). Boot requests that name the tab's
// profile (approval mode, model options, the pet, shared metrics, setup checks) can go out before boot
// adopts the primary, so the renderer must start on the gateway's own profile, not on `default`. The
// page the gateway serves sets these globals before any module evaluates (web_server_dashboard.py).
type PageGlobals = { __HERMES_DASHBOARD_PROFILE__?: string; __HERMES_UI_SURFACE__?: string }

vi.hoisted(() => {
  Object.assign(window, { __HERMES_DASHBOARD_PROFILE__: 'solo', __HERMES_UI_SURFACE__: 'webapp' })
})

afterAll(() => {
  const page = window as Window & PageGlobals
  delete page.__HERMES_DASHBOARD_PROFILE__
  delete page.__HERMES_UI_SURFACE__
})

it('a Webapp tab starts on the profile its gateway runs as, so a boot request names that profile', async () => {
  expect($activeGatewayProfile.get()).toBe('solo')
  expect(activeGatewayProfileKey()).toBe('solo')
  expect(getApiRequestProfile()).toBe('solo')

  // The status bar's approval-mode read, as it runs on the tab's socket once it opens.
  const request = vi.fn(async () => ({ value: 'manual' }))
  setPrimaryGateway({ connectionState: 'open', request } as never, $activeGatewayProfile.get())
  await syncApprovalModeForProfile(
    (method, params) => requestGatewayForProfile($activeGatewayProfile.get(), method, params),
    $activeGatewayProfile.get()
  )

  expect(request).toHaveBeenCalledWith('config.get', { key: 'approvals.mode', profile: 'solo' })
})
