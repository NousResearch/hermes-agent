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
  const cloud = JSON.stringify(['cloud-1', 'default'])
  $gatewayGroupOrder.set([])
  reorderGatewayGroups([remote, local])
  expect($gatewayGroupOrder.get()).toEqual([remote, local])
  renameGatewayGroup(remote, ' Research lab ')
  toggleGatewayGroup(remote)
  reorderGatewayGroups([cloud, local])
  expect($gatewayGroupOrder.get()).toEqual([remote, cloud, local])
  expect($gatewayGroupAliases.get()).toEqual({ [remote]: 'Research lab' })
  expect($gatewayGroupCollapsed.get()).toEqual([remote])
  vi.resetModules()
  const restored = await import('./gateway-group-preferences')
  expect(restored.$gatewayGroupOrder.get()).toEqual([remote, cloud, local])
  expect(restored.$gatewayGroupAliases.get()).toEqual({ [remote]: 'Research lab' })
  expect(restored.$gatewayGroupCollapsed.get()).toEqual([remote])
  restored.renameGatewayGroup(remote, '  ')
  expect(restored.$gatewayGroupAliases.get()).toEqual({})
})
