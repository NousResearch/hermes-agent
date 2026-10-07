// @vitest-environment jsdom
import { expect, it, vi } from 'vitest'

import {
  $gatewayGroupAliases,
  $gatewayGroupCollapsed,
  $gatewayGroupOrder,
  renameGatewayGroup,
  reorderGatewayGroups,
  toggleGatewayGroup
} from './gateway-group-preferences'

it('persists identity-scoped edits and reorders newly discovered groups without forgetting hidden groups', async () => {
  const local = JSON.stringify(['local', 'default'])
  const remote = JSON.stringify(['remote-1', 'default'])
  const remote1 = JSON.stringify(['remote-2', 'default'])
  $gatewayGroupOrder.set([])
  reorderGatewayGroups([remote, local])
  expect($gatewayGroupOrder.get()).toEqual([remote, local])
  renameGatewayGroup(remote, ' Research lab ')
  toggleGatewayGroup(remote)
  reorderGatewayGroups([remote1, local])
  expect($gatewayGroupOrder.get()).toEqual([remote, remote1, local])
  expect($gatewayGroupAliases.get()).toEqual({ [remote]: 'Research lab' })
  expect($gatewayGroupCollapsed.get()).toEqual([remote])
  vi.resetModules()
  const restored = await import('./gateway-group-preferences')
  expect(restored.$gatewayGroupOrder.get()).toEqual([remote, remote1, local])
  expect(restored.$gatewayGroupAliases.get()).toEqual({ [remote]: 'Research lab' })
  expect(restored.$gatewayGroupCollapsed.get()).toEqual([remote])
  restored.renameGatewayGroup(remote, '  ')
  expect(restored.$gatewayGroupAliases.get()).toEqual({})
})
