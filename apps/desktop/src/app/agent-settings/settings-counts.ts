import type { CronJob, MessagingPlatformsResponse } from '@/hermes'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { enabledToolsetSummary } from '../capabilities/toolsets/toolsets-data'

import type { AgentSettingsSection } from './settings-routes'

/** What the agent page has already loaded about this agent, for the sub-nav counts (895:13886
 *  "Tools 13 · Skills 8 · Plugins 8 · Messaging 2 · Routines 2") and the capability page summaries. A section
 *  whose data has not arrived has no count; nothing here fetches. */
export interface AgentCapabilityCounts {
  nav: Partial<Record<AgentSettingsSection, number>>
  tools?: { toolsets: number; tools: number }
  skills?: number
  /** Not a settings page (Connectors are set up on Capabilities): the Overview's row counts them. */
  connectors?: number
}

interface AgentCapabilitySources {
  tools?: ToolsetInfo[]
  skills?: SkillInfo[]
  jobs?: CronJob[]
  messaging?: MessagingPlatformsResponse
  connectors?: number
  plugins?: number
}

export function agentCapabilityCounts({
  tools,
  skills,
  jobs,
  messaging,
  connectors,
  plugins
}: AgentCapabilitySources): AgentCapabilityCounts {
  const toolSummary = tools && enabledToolsetSummary(tools)
  const skillsOn = skills?.filter(skill => skill.enabled).length

  return {
    nav: {
      tools: toolSummary?.toolsets,
      skills: skillsOn,
      plugins,
      messaging: messaging?.platforms.filter(platform => platform.configured).length,
      // Every routine the agent owns, paused included: the Routines list and the sidebar chip count the same.
      schedules: jobs?.length
    },
    tools: toolSummary,
    skills: skillsOn,
    connectors
  }
}
