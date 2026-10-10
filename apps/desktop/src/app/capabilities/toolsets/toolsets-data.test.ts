import { describe, expect, it } from 'vitest'

import type { ToolsetInfo } from '@/types/hermes'

import { filteredToolsets, toolsetSearchTerms, visibleToolsetCount } from './toolsets-data'

describe('Desktop toolset curation', () => {
  it('excludes platform-only toolsets from rows, counts and search suggestions without changing their configuration', () => {
    const row = (name: string): ToolsetInfo => ({
      name,
      label: name,
      description: name,
      tools: [name],
      enabled: true,
      configured: true
    })

    const web = row('web')

    const extension = row('matrix_custom')

    const rows = [
      web,
      row('matrix_read'),
      row('matrix_admin'),
      row('matrix_unread'),
      row('matrix_image_packs'),
      row('matrix_reaction'),
      row('matrix_followup'),
      extension
    ]

    const before = structuredClone(rows)

    const visible = filteredToolsets(rows, '', { web: 10 }, true)

    expect({
      visible,
      count: visibleToolsetCount(rows),
      suggestions: toolsetSearchTerms(rows),
      searched: filteredToolsets(rows, 'matrix', {}, true),
      configuration: rows
    }).toEqual({
      visible: [web, extension],
      count: 2,
      suggestions: ['web', 'matrix_custom'],
      searched: [extension],
      configuration: before
    })
  })
})
