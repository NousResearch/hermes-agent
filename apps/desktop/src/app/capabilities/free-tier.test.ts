import { freeConnectorAllowed, freeListAllowed, FREE_CAPABILITY_LIMIT } from './free-tier'

describe('free capabilities', () => {
  it('allows only the first three names until the brand is unlocked', () => {
    const ids = ['delta', 'alpha', 'echo', 'bravo']

    expect(freeListAllowed(ids, 'alpha', false)).toBe(true)
    expect(freeListAllowed(ids, 'bravo', false)).toBe(true)
    expect(freeListAllowed(ids, 'delta', false)).toBe(true)
    expect(freeListAllowed(ids, 'echo', false)).toBe(false)
    expect(freeListAllowed(ids, 'echo', true)).toBe(true)
  })

  it('keeps CRM, Inbox, and Mail on the free connector list', () => {
    expect(freeConnectorAllowed('twenty', false)).toBe(true)
    expect(freeConnectorAllowed('chatwoot', false)).toBe(true)
    expect(freeConnectorAllowed('notifuse', false)).toBe(true)
    expect(freeConnectorAllowed('firecrawl', false)).toBe(false)
    expect(freeConnectorAllowed('n8n', true)).toBe(true)
    expect(FREE_CAPABILITY_LIMIT).toBe(3)
  })
})
