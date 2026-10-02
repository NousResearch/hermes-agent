import { connectorTitle } from './connector-tools'
import { hideVendorNames, productToolLabel } from './product-names'

describe('product tool labels', () => {
  it('shows the product name for a brand connector tool', () => {
    expect(productToolLabel('mcp__ivx_foundrly_chatwoot__chatwoot_list_conversations')).toBe(
      'Inbox Studio · List Conversations'
    )
    expect(productToolLabel('mcp__ivx_foundrly_firecrawl__firecrawl_search')).toBe('Firecrawl · Search')
    expect(productToolLabel('ivx-foundrly-n8n')).toBe('n8n')
    expect(productToolLabel('ivx-foundrly-automation-studio')).toBe('Automation Studio · Studio')
    expect(productToolLabel('mcp__ivx_foundrly_twenty__search')).toBe('CRM · Search')
    expect(productToolLabel('mcp__ivx_foundrly_notifuse__notifuse_lists_list')).toBe('Mail Studio · Lists List')
    expect(productToolLabel('Foundrly Twenty · Execute Tool')).toBe('CRM · Execute Tool')
  })

  it('names a connector card by the product', () => {
    expect(connectorTitle('ivx-foundrly-notifuse')).toBe('Mail Studio')
    expect(connectorTitle('ivx-foundrly-chatwoot')).toBe('Inbox Studio')
    expect(connectorTitle('ivx-foundrly-twenty')).toBe('CRM')
    expect(connectorTitle('gmail')).toBe('Gmail')
  })

  it('leaves ordinary tools alone', () => {
    expect(productToolLabel('terminal')).toBeNull()
    expect(productToolLabel('read_file')).toBeNull()
  })

  it('hides vendor words in a reply without rewriting a count', () => {
    const shown = hideVendorNames(
      'Checked chatwoot and the server ivx-foundrly-firecrawl. Twenty people were added yesterday.'
    )

    expect(shown).not.toMatch(/chatwoot/i)
    expect(shown).toContain('Inbox Studio')
    expect(shown).toContain('Firecrawl')
    expect(shown).toContain('Twenty people were added yesterday.')
  })
})
