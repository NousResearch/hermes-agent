import { describe, expect, it } from 'vitest'

import { deriveCards } from './derive'
import { isConnectorConfiguredActive } from './use-connector-summary'

describe('connector capability counts', () => {
  it('counts configured/enabled connectors as active even when runtime health is degraded', () => {
    const hosted = deriveCards({
      hosted: [
        {
          slug: 'drive',
          connected: true,
          enabled: true,
          connectionStatus: 'expired'
        }
      ],
      local: []
    })[0]

    const local = deriveCards({
      hosted: [],
      local: [
        {
          name: 'local-mcp',
          enabled: true,
          status: 'error',
          target: 'stdio'
        }
      ]
    })[0]

    expect(hosted.state).toBe('expired')
    expect(local.state).toBe('broken')
    expect(isConnectorConfiguredActive(hosted)).toBe(true)
    expect(isConnectorConfiguredActive(local)).toBe(true)
  })

  it('does not count explicitly disabled connector ways as active', () => {
    const hosted = deriveCards({
      hosted: [
        {
          slug: 'drive',
          connected: true,
          enabled: false
        }
      ],
      local: []
    })[0]

    const local = deriveCards({
      hosted: [],
      local: [
        {
          name: 'local-mcp',
          enabled: false,
          status: 'ok',
          target: 'stdio'
        }
      ]
    })[0]

    expect(isConnectorConfiguredActive(hosted)).toBe(false)
    expect(isConnectorConfiguredActive(local)).toBe(false)
  })
})
