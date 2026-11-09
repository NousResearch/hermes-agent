import { afterEach, describe, expect, it, vi } from 'vitest'

import { countMessagesBelow } from './use-messages-below'

const rect = (top: number, bottom: number): DOMRect => ({ top, bottom, height: bottom - top, width: 800 }) as DOMRect

afterEach(() => {
  window.document.body.innerHTML = ''
  vi.restoreAllMocks()
})

describe('messages-below scroll cost', () => {
  it('does not measure every mounted group to count messages below the fold', () => {
    const count = 512
    const viewport = window.document.createElement('div')
    const content = window.document.createElement('div')
    viewport.append(content)
    window.document.body.append(viewport)
    const scrollTop = 300 * 100 + 50
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))
    let groupReads = 0

    for (let index = 0; index < count; index++) {
      const group = window.document.createElement('div')
      group.dataset.slot = 'aui_message-group'
      content.append(group)
      vi.spyOn(group, 'getBoundingClientRect').mockImplementation(() => {
        groupReads++

        return rect(index * 100 - scrollTop, index * 100 + 100 - scrollTop)
      })

      for (const slot of ['aui_user-message-root', 'aui_assistant-message-root']) {
        const message = window.document.createElement('div')
        message.dataset.slot = slot
        group.append(message)
        vi.spyOn(message, 'getBoundingClientRect').mockReturnValue(
          slot === 'aui_user-message-root'
            ? rect(index * 100 - scrollTop, index * 100 + 40 - scrollTop)
            : rect(index * 100 + 40 - scrollTop, index * 100 + 100 - scrollTop)
        )
      }
    }

    expect(countMessagesBelow(viewport, content)).toEqual({
      count: 1 + (count - 307) * 2,
      settled: true
    })
    expect(groupReads).toBeLessThanOrEqual(Math.ceil(Math.log2(count)) + 2)
  })
})
