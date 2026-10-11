// @vitest-environment jsdom
import { expect, it, vi } from 'vitest'

import {
  $gatewayGroupAliases,
  $gatewayGroupCollapsed,
  $gatewayGroupHidden,
  $gatewayGroupOrder,
  renameGatewayGroup,
  reorderGatewayGroups,
  setGatewayGroupHidden,
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

// Regression for #96532: a visibility preference is per-connection renderer
// state (the rail's own store), scoped to one connection id and surviving a
// reload. It is NOT the registry entry, so un-hiding must restore exactly the
// gateway that was hidden and leave every other gateway untouched.
it('remembers which gateways the user hid, per connection, across a reload', async () => {
  $gatewayGroupHidden.set([])
  setGatewayGroupHidden('local', true)
  setGatewayGroupHidden('pandora', true)
  setGatewayGroupHidden('local', false)
  expect($gatewayGroupHidden.get()).toEqual(['pandora'])

  vi.resetModules()
  const restored = await import('./gateway-group-preferences')
  expect(restored.$gatewayGroupHidden.get()).toEqual(['pandora'])
  restored.setGatewayGroupHidden('pandora', false)
  expect(restored.$gatewayGroupHidden.get()).toEqual([])
})
