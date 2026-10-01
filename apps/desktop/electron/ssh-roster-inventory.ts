import type { GatewayEndpoint } from './local-gateway'
import { rosterProfileMetadata } from './roster-profile-metadata'
import { fetchRosterSourceData } from './roster-source-fetch'

interface SshInventoryState {
  registryConnectionId?: string
  canonical?: boolean
  baseUrl?: string
  gatewayEndpoint?: GatewayEndpoint
}

/** Inspect only an already-attached source. Inventory must not spawn an owner
 * or substitute the SSH login shell's home for the configured Hermes launcher. */
export async function readSshRosterInventory(options: {
  connectionId: string
  states: Map<string, SshInventoryState>
  request: (descriptor: { baseUrl: string; gatewayEndpoint: GatewayEndpoint; authMode: 'native'; token: string }, path: string) => Promise<any>
}) {
  const owner = () => {
    const states = [...options.states.values()].filter(state => state.registryConnectionId === options.connectionId)

    return states.find(state => state.canonical) || states[0]
  }

  const state = owner()
  const isCurrent = () => owner() === state

  if (!state) {return { kind: 'undialed' as const, isCurrent }}

  if (!state.canonical) {return { kind: 'classic' as const, isCurrent }}

  if (!state.baseUrl || !state.gatewayEndpoint) {throw new Error('Canonical SSH inventory descriptor is unavailable')}
  const descriptor = { baseUrl: state.baseUrl, gatewayEndpoint: state.gatewayEndpoint, authMode: 'native' as const, token: '' }

  const { body, installId } = await fetchRosterSourceData(
    () => options.request(descriptor, '/api/profiles'),
    async () => {
      try {
        const status = await options.request(descriptor, '/api/status')

        return typeof status?.install_id === 'string' ? status.install_id.trim() || undefined : undefined
      } catch {return undefined} // Identity metadata may be unavailable; never borrow an ambient ID.
    }
  )

  if (!isCurrent()) {throw new Error('SSH inventory source changed during enumeration')}

  if (!Array.isArray(body?.profiles)) {throw new Error('Canonical SSH profile inventory is unavailable')}
  const profiles: string[] = body.profiles.map(profile => String(profile?.name || '').trim()).filter(Boolean)

  return { kind: 'canonical' as const, profiles, profileMetadata: rosterProfileMetadata(body.profiles), installId, isCurrent }
}
