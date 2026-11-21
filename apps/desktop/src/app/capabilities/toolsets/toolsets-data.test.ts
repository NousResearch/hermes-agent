import { describe, expect, it } from 'vitest'

import type { EnvVarInfo, ToolsetInfo } from '@/types/hermes'

import { enabledToolsetSummary, toolsetKeyGroup } from './toolsets-data'

const toolset = (name: string, label: string, tools: string[], enabled = true) =>
  ({ name, label, description: '', enabled, available: true, configured: true, tools }) as unknown as ToolsetInfo

const key = (tools: string[]) => ({ tools }) as unknown as EnvVarInfo

const toolsets = [
  toolset('web', 'Web Search & Scraping', ['web_search', 'web_extract']),
  toolset('search', 'Search', ['web_search'], false),
  toolset('browser', 'Browser Automation', ['browser_exec', 'browser_cdp'])
]

describe('toolsetKeyGroup', () => {
  it('files a key under the toolset that shares most of its tools, and the rest under Other, last', () => {
    const group = toolsetKeyGroup(toolsets, 'Other')

    expect(group(key(['web_search', 'web_extract']))).toEqual({ label: 'Web Search & Scraping', order: 0 })
    expect(group(key(['browser_exec']))).toEqual({ label: 'Browser Automation', order: 2 })
    expect(group(key([]))).toEqual({ label: 'Other', order: 3 })
    expect(group(key(['honcho_context']))).toEqual({ label: 'Other', order: 3 })
  })
})

describe('enabledToolsetSummary', () => {
  it('counts the agent’s enabled toolsets and the distinct tools they bring', () => {
    expect(enabledToolsetSummary(toolsets)).toEqual({ toolsets: 2, tools: 4 })
  })
})
