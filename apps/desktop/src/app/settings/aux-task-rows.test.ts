import { describe, expect, it } from 'vitest'

import { auxTaskRows } from './aux-task-rows'

describe('auxTaskRows', () => {
  it('does not offer the retired skills_hub slot (#40703)', () => {
    const keys = auxTaskRows([{ base_url: '', model: '', provider: 'auto', task: 'vision' }]).map(row => row.key)

    expect(keys).toContain('vision')
    expect(keys).not.toContain('skills_hub')
  })
})
