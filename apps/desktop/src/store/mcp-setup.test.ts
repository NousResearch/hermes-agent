import { describe, expect, it } from 'vitest'

import { buildSetupOutcome, type McpConnectorOutcome } from './mcp-setup'

const results = (...entries: McpConnectorOutcome[]) => Object.fromEntries(entries.map(entry => [entry.server, entry]))

describe('buildSetupOutcome', () => {
  it('reports partial when some connectors landed and some failed', () => {
    const outcome = buildSetupOutcome({
      names: ['linear', 'n8n', 'notion'],
      results: results(
        { server: 'linear', status: 'connected', tools: ['issues'] },
        { server: 'n8n', status: 'connected' },
        { detail: 'authorization denied', server: 'notion', status: 'error' }
      ),
      server: 'linear'
    })

    expect(outcome.status).toBe('partial')
    // The successes survive the failure — that's the whole point.
    expect(outcome.connectors.filter(row => row.status === 'connected').map(row => row.server)).toEqual([
      'linear',
      'n8n'
    ])
    expect(outcome.connectors.find(row => row.server === 'notion')?.detail).toBe('authorization denied')
  })

  it('is connected only when nothing failed', () => {
    const outcome = buildSetupOutcome({
      names: ['linear', 'n8n'],
      results: results({ server: 'linear', status: 'connected' }, { server: 'n8n', status: 'connected' }),
      server: 'linear'
    })

    expect(outcome.status).toBe('connected')
  })

  it('is error when every attempt failed', () => {
    const outcome = buildSetupOutcome({
      names: ['notion'],
      results: results({ detail: 'unreachable', server: 'notion', status: 'error' }),
      server: 'notion'
    })

    expect(outcome.status).toBe('error')
  })

  it('declines the cards that were dismissed without connecting', () => {
    const outcome = buildSetupOutcome({ names: ['linear', 'n8n'], results: {}, server: 'linear' })

    expect(outcome.status).toBe('declined')
    expect(outcome.connectors).toEqual([
      { server: 'linear', status: 'declined' },
      { server: 'n8n', status: 'declined' }
    ])
  })

  it('still reports connected when a sibling card was dismissed', () => {
    // Dismissing Notion's card is not a failure of the whole set — Linear
    // connected and the agent should get on with using it.
    const outcome = buildSetupOutcome({
      names: ['linear', 'notion'],
      results: results({ server: 'linear', status: 'connected' }),
      server: 'linear'
    })

    expect(outcome.status).toBe('connected')
    expect(outcome.connectors.find(row => row.server === 'notion')?.status).toBe('declined')
  })

  it('keeps every offered connector in the answer, in the order asked', () => {
    const outcome = buildSetupOutcome({
      names: ['notion', 'linear', 'n8n'],
      results: results({ server: 'linear', status: 'connected' }),
      server: 'notion'
    })

    expect(outcome.connectors.map(row => row.server)).toEqual(['notion', 'linear', 'n8n'])
  })
})
