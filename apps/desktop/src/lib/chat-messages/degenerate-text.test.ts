import { describe, expect, it } from 'vitest'

import { isDegenerateAssistantText } from './degenerate-text'

describe('isDegenerateAssistantText', () => {
  it.each(['8368e4f9-9f48-4ba6-a5b5-ffb40b2ce509', '?wq', ':wq', 'भू긴', '666\u200bចercher'])(
    'rejects %j',
    text => {
      expect(isDegenerateAssistantText(text)).toBe(true)
    }
  )

  it.each(['OK', 'Done.', 'שלום', 'All four links are in, post verified everywhere.', ''])(
    'keeps %j',
    text => {
      expect(isDegenerateAssistantText(text)).toBe(false)
    }
  )
})
