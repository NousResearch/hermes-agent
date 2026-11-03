import { describe, expect, it } from 'vitest'

import { EMPTY_POLICY, joinHostedConnectors, joinLocalServers } from './data/join'
import { cardKey, deriveCards } from './derive'

const LINEAR = { category: 'work', description: 'Issues', name: 'Linear', slug: 'linear' }

function cardsFor(localStatus: 'needs-auth' | 'ok') {
  const hosted = joinHostedConnectors({ accounts: [], catalog: [LINEAR], list: [], policy: EMPTY_POLICY })

  const local = joinLocalServers({
    servers: { linear: { auth: 'oauth', url: 'https://mcp.linear.app/mcp' } },
    status: { linear: localStatus }
  })

  return deriveCards({ hosted, local, titles: { linear: 'Linear' } })
}

describe('a configured server that shares a name with a hosted connector', () => {
  it('keeps its own card beside the hosted card, each with its own action', () => {
    const cards = cardsFor('needs-auth')
    const hosted = cards.find(card => card.residency === 'hosted')
    const local = cards.find(card => card.residency === 'local')

    expect(cards.map(cardKey)).toEqual(['hosted:linear', 'local:linear'])

    expect(hosted?.ways.local).toBeNull()
    expect(hosted?.verb).toBe('connect')

    expect(local?.ways.hosted).toBeNull()
    expect(local?.ways.local).toMatchObject({
      serverEnabled: true,
      serverName: 'linear',
      target: 'https://mcp.linear.app/mcp'
    })
    expect(local?.verb).toBe('authenticate')
  })

  it('reports the server as on without borrowing the hosted card state', () => {
    const local = cardsFor('ok').find(card => card.residency === 'local')

    expect(local).toMatchObject({ state: 'connected', stateWord: 'serverOn' })
  })
})
