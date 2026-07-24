import { describe, expect, it } from 'vitest'

import { summarizePetApproval } from './pet-bubble'

describe('pet approval summary', () => {
  it('prefers the command and keeps only its first line', () => {
    expect(summarizePetApproval('npm run build\nrm -rf dist', 'dangerous command')).toBe('npm run build')
  })

  it('falls back to the description and truncates long text', () => {
    expect(summarizePetApproval('', 'x'.repeat(50))).toBe(`${'x'.repeat(39)}…`)
  })

  it('uses a localized fallback when no details are available', () => {
    expect(summarizePetApproval(' ', ' ')).toBe('待审批操作')
  })
})
