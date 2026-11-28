import { afterEach, expect, it } from 'vitest'

import { $connectionsRegistry } from '@/store/connections'
import { $connection } from '@/store/session'

import { defaultTerminalGateway, shellGatewayFor } from './terminal-gateway'
import { $terminals, closeAllTerminals, createTerminal } from './terminals'

const row = (id: string, kind: string) => ({ authMode: 'oauth', headerNames: [], id, kind, label: id, url: `https://${id}.example` })

afterEach(() => {
  closeAllTerminals()
  $connectionsRegistry.set(null)
  $connection.set(null)
  Reflect.deleteProperty(window, '__HERMES_HOST_AGENT__')
})

// The hosted dashboard's list holds the account's cloud agents and no This device; the page relays
// each agent's own shell, so a tab names its agent, which that list registers.
it('keeps a terminal on a host with no agent on its agent through a later list publication', () => {
  Object.assign(window, { __HERMES_HOST_AGENT__: false })
  const registry = { connections: [row('alpha', 'cloud'), row('beta', 'cloud')], primary: 'alpha', version: 2 } as never
  $connectionsRegistry.set(registry)
  $connection.set({ connectionId: 'alpha', profile: 'default' } as never)
  const id = createTerminal('/w', defaultTerminalGateway())

  $connectionsRegistry.set({ ...(registry as object), lastUsed: 'alpha' } as never)

  expect($terminals.get().find(term => term.id === id)).toMatchObject({ connectionId: 'alpha', profile: 'default' })
  expect($terminals.get().find(term => term.id === id)?.gatewayRemoved).toBeFalsy()
})

it('still runs a cloud gateway shell on This device where the host runs its own agent', () => {
  const registry = { connections: [row('local', 'local'), row('alpha', 'cloud')], version: 2 } as never

  expect(shellGatewayFor('alpha', 'default', registry)).toEqual({ connectionId: 'local' })
})
