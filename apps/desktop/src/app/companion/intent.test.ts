import { describe, expect, it } from 'vitest'

import { classifyCompanionIntent } from './intent'

describe('companion intent classification', () => {
  it('detects app launch requests', () => {
    expect(classifyCompanionIntent('打开记事本')).toMatchObject({
      kind: 'open-app',
      target: '记事本'
    })
  })

  it('detects known folder open requests', () => {
    expect(classifyCompanionIntent('打开桌面')).toMatchObject({
      kind: 'open-folder',
      target: '桌面'
    })
  })

  it('detects file search requests', () => {
    expect(classifyCompanionIntent('帮我找合同.docx')).toMatchObject({
      kind: 'find-file',
      target: '合同.docx'
    })
  })

  it('falls back to normal chat when no local action is requested', () => {
    expect(classifyCompanionIntent('今天心情有点差')).toMatchObject({
      kind: 'chat'
    })
  })
})
