// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import { connectionsSearchTerms } from './index'

const CAPABILITIES = 'Capabilities'

/** The palette's own search contract: every typed word appears in the label or keywords. */
const finds = (label: string, keywords: string[], query: string): boolean =>
  query.split(/\s+/).every(term => label.toLowerCase().includes(term) || keywords.some(key => key.includes(term)))

describe('the ⌘K Connections row', () => {
  it('is Capabilities: Connections in the redesign, and typing "Connections" finds it, not Machines', () => {
    const terms = connectionsSearchTerms({
      capLabel: CAPABILITIES,
      classicTitle: 'Connectors',
      portal: true,
      redesignTitle: 'Connections'
    })

    expect(terms.connectors.label).toBe('Capabilities: Connections')
    expect(terms.connectors.keywords).toEqual(expect.arrayContaining(['connections', 'connectors']))
    expect(finds(terms.connectors.label, terms.connectors.keywords, 'connections')).toBe(true)
    expect(finds('Gateways', terms.gateways, 'connections')).toBe(false)
    expect(terms.gateways).toEqual(expect.arrayContaining(['machines']))
  })

  it('keeps classic’s Connectors row and its Connections alias on the merged Machines page', () => {
    const terms = connectionsSearchTerms({
      capLabel: CAPABILITIES,
      classicTitle: 'Connectors',
      portal: false,
      redesignTitle: 'Connections'
    })

    expect(terms.connectors.label).toBe('Capabilities: Connectors')
    expect(terms.connectors.keywords).toContain('connectors')
    expect(terms.connectors.keywords).not.toContain('connections')
    expect(finds('Gateways', terms.gateways, 'connections')).toBe(true)
  })
})
