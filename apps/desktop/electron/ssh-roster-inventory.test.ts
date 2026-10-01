import { expect, test } from 'vitest'

import { readSshRosterInventory } from './ssh-roster-inventory'

const endpoint = { profile_id: '/synthetic/selected-home', instance_id: 'selected-owner', authority_epoch: 1,
  runtime_protocol: 1, api_origin: 'http://127.0.0.1:4321', capabilities: ['session-authority-v1'], supervisor: 'external' }

test('undialed inventory does not probe a home; classic selection stays explicit and retires behind native attachment', async () => {
  const states = new Map<string, any>()
  const request = async () => {throw new Error('No canonical request expected')}
  expect((await readSshRosterInventory({ connectionId: 'peer', states, request })).kind).toBe('undialed')
  states.set('classic', { registryConnectionId: 'peer', canonical: false })
  const classic = await readSshRosterInventory({ connectionId: 'peer', states, request })
  expect(classic.kind).toBe('classic')
  expect(classic.kind === 'classic' && classic.isCurrent()).toBe(true)
  states.set('native', { registryConnectionId: 'peer', canonical: true, baseUrl: 'http://127.0.0.1:8765', gatewayEndpoint: endpoint })
  expect(classic.kind === 'classic' && classic.isCurrent()).toBe(false)
  await expect(readSshRosterInventory({ connectionId: 'peer', states, request })).rejects.toThrow('No canonical request expected')
})

test('a replaced canonical descriptor cannot publish its old inventory or fall back to another home', async () => {
  const current = { registryConnectionId: 'peer', canonical: true, baseUrl: 'http://127.0.0.1:8765', gatewayEndpoint: endpoint }
  const states = new Map<string, any>([['active', current]])
  await expect(readSshRosterInventory({ connectionId: 'peer', states, request: async (descriptor, requestPath) => {
    expect(descriptor.gatewayEndpoint).toBe(endpoint)
    expect(descriptor.baseUrl).toBe(current.baseUrl)
    states.set('active', { ...current, gatewayEndpoint: { ...endpoint, instance_id: 'replacement' } })
    return requestPath === '/api/profiles' ? { profiles: [{ name: 'selected-only' }] } : { install_id: 'selected' }
  } })).rejects.toThrow('source changed')
})
