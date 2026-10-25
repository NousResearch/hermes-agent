import { expect, it } from 'vitest'

import { reloadAfterAccountUpdate } from './use-mcp-servers'

it('keeps a newly created unauthenticated account out of backend MCP reloads', () => {
  expect(reloadAfterAccountUpdate({ name: 'notion-work', state: 'pending' })).toBe(false)
  expect(reloadAfterAccountUpdate({ name: 'notion-work', state: 'abandoned' })).toBe(false)
  expect(reloadAfterAccountUpdate({ name: 'notion-work', state: 'authorized', tools: [] })).toBe(true)
  expect(reloadAfterAccountUpdate()).toBe(true)
})
