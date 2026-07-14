import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useSlashCompletions } from './use-slash-completions'

describe('useSlashCompletions', () => {
  it('requests the empty catalog and preserves categorized skills', async () => {
    vi.useFakeTimers()
    const request = vi.fn().mockResolvedValue({
      pairs: [],
      categories: [{ name: 'Tools & Skills', pairs: [['/test-skill', 'A test skill']] }]
    })
    const { result } = renderHook(() => useSlashCompletions({ gateway: { request } as never }))

    await act(async () => {
      result.current.adapter.search('')
      await vi.advanceTimersByTimeAsync(60)
    })

    expect(request).toHaveBeenCalledWith('commands.catalog')
    expect(result.current.adapter.search('')).toEqual([
      expect.objectContaining({
        id: '/test-skill|0',
        label: 'test-skill',
        metadata: expect.objectContaining({ group: 'Tools & Skills' })
      })
    ])
    vi.useRealTimers()
  })
})
