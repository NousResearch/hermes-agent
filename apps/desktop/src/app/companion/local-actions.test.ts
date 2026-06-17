import { describe, expect, it } from 'vitest'

import { buildFileSearchActions, filePathToExternalUrl } from './local-actions'

describe('companion local actions', () => {
  it('builds open actions from search matches', () => {
    expect(buildFileSearchActions(['C:\\Work\\合同.docx', 'C:\\Work\\设计稿'])).toEqual([
      {
        id: 'open-0',
        kind: 'open-path',
        label: '打开 合同.docx',
        target: 'C:\\Work\\合同.docx'
      },
      {
        id: 'open-1',
        kind: 'open-path',
        label: '打开 设计稿',
        target: 'C:\\Work\\设计稿'
      }
    ])
  })

  it('converts a Windows path to a file url', () => {
    expect(filePathToExternalUrl('C:\\Work\\合同.docx')).toBe('file:///C:/Work/%E5%90%88%E5%90%8C.docx')
  })
})
