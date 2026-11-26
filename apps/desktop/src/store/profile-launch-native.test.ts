import { afterAll, expect, it, vi } from 'vitest'

import { activeGatewayProfileKey } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

// The twin of profile-launch-webapp.test.ts: only a browser host starts on its server's profile.
// Electron's preload installs the bridge before any renderer module evaluates, so native Desktop
// starts on `default` even if a page global names another profile.
vi.hoisted(() => {
  Object.assign(window, { __HERMES_DASHBOARD_PROFILE__: 'solo', hermesDesktop: {} })
})

afterAll(() => {
  Reflect.deleteProperty(window, '__HERMES_DASHBOARD_PROFILE__')
  Reflect.deleteProperty(window, 'hermesDesktop')
})

it('native Desktop starts on default', () => {
  expect($activeGatewayProfile.get()).toBe('default')
  expect(activeGatewayProfileKey()).toBe('default')
})
