import { expect, it } from 'vitest'

import type { CronJob } from '@/hermes'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { agentCapabilityCounts } from './settings-counts'

// One definition per count, read by both the agent Overview rows and the settings menu.
it('counts toolsets that are on, skills that are on, and every routine the agent owns', () => {
  const tools = [
    { name: 'web', enabled: true, tools: ['web_search', 'web_extract'] },
    { name: 'file', enabled: true, tools: ['read_file', 'write_file', 'patch'] },
    { name: 'tts', enabled: false, tools: ['text_to_speech'] }
  ] as unknown as ToolsetInfo[]

  const skills = [{ name: 'a', enabled: true }, { name: 'b', enabled: false }] as unknown as SkillInfo[]
  // A paused routine is still one of the agent's routines: the sidebar chip and the Routines list show it.
  const jobs = [{ id: '1', enabled: true }, { id: '2', enabled: false }] as unknown as CronJob[]

  const counts = agentCapabilityCounts({ tools, skills, jobs, connectors: 3, plugins: 1 })

  expect(counts.tools).toEqual({ toolsets: 2, tools: 5 })
  expect(counts.nav).toMatchObject({ tools: 2, skills: 1, schedules: 2, connectors: 3, plugins: 1 })
  expect(counts.skills).toBe(1)
})
