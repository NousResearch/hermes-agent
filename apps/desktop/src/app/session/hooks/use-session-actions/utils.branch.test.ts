import { expect, it } from 'vitest'

import { type ChatMessage, textPart } from '@/lib/chat-messages'

import { selectBranchMessages } from './utils'

const message = (id: string, role: ChatMessage['role'], text: string, rowId: number): ChatMessage => ({
  id,
  parts: [textPart(text)],
  role,
  rowId
})

it('maps a clicked reply through a folded authoritative source row', () => {
  const local = [
    message('tail-user', 'user', 'latest question', 13),
    message('tail-assistant', 'assistant', 'latest answer', 14)
  ]

  const authoritative = [
    message('old-user', 'user', 'first question', 11),
    message('old-assistant', 'assistant', 'first answer', 12),
    message('tail-user', 'user', 'latest question', 13),
    {
      id: 'folded-assistant',
      role: 'assistant',
      rowId: 10,
      parts: [
        { ...textPart('tool preface'), sourceRowId: 10 },
        { ...textPart('latest answer'), sourceRowId: 14 }
      ]
    }
  ] as ChatMessage[]

  expect(selectBranchMessages(local, authoritative, 'tail-assistant').map(branch => branch.content)).toEqual([
    'first question',
    'first answer',
    'latest question',
    'tool prefacelatest answer'
  ])
})
