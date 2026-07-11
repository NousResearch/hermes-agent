export type LearnLinkKind = 'community' | 'official'

export type LearnLink = {
  description: string
  host: string
  id: string
  kind: LearnLinkKind
  title: string
  url: string
}

/** Checked-in allowlist only. No remote feed. Community entries are easy to drop. */
export const MENU_BAR_SHOW_COMMUNITY_LEARN_LINKS = true

export const OFFICIAL_LEARN_LINKS: readonly LearnLink[] = [
  {
    id: 'official-docs',
    kind: 'official',
    title: 'Official Docs',
    description: 'Hermes Agent documentation home',
    url: 'https://hermes-agent.nousresearch.com/docs/',
    host: 'hermes-agent.nousresearch.com'
  },
  {
    id: 'skills-system',
    kind: 'official',
    title: 'Skills System',
    description: 'How skills load, author, and improve',
    url: 'https://hermes-agent.nousresearch.com/docs/user-guide/features/skills',
    host: 'hermes-agent.nousresearch.com'
  },
  {
    id: 'skills-hub',
    kind: 'official',
    title: 'Skills Hub',
    description: 'Browse registries and installable skills',
    url: 'https://hermes-agent.nousresearch.com/docs/skills',
    host: 'hermes-agent.nousresearch.com'
  },
  {
    id: 'bundled-skills',
    kind: 'official',
    title: 'Bundled Skills Catalog',
    description: 'Skills shipped with Hermes',
    url: 'https://hermes-agent.nousresearch.com/docs/reference/skills-catalog',
    host: 'hermes-agent.nousresearch.com'
  },
  {
    id: 'agent-skills-standard',
    kind: 'official',
    title: 'Agent Skills Standard',
    description: 'agentskills.io open format',
    url: 'https://agentskills.io',
    host: 'agentskills.io'
  },
  {
    id: 'desktop-guide',
    kind: 'official',
    title: 'Desktop App Guide',
    description: 'Native desktop app docs',
    url: 'https://hermes-agent.nousresearch.com/docs/user-guide/desktop',
    host: 'hermes-agent.nousresearch.com'
  },
  {
    id: 'learning-path',
    kind: 'official',
    title: 'Learning Path',
    description: 'Find the right docs for your level',
    url: 'https://hermes-agent.nousresearch.com/docs/getting-started/learning-path',
    host: 'hermes-agent.nousresearch.com'
  },
  {
    id: 'github',
    kind: 'official',
    title: 'GitHub Repository',
    description: 'NousResearch/hermes-agent source',
    url: 'https://github.com/NousResearch/hermes-agent',
    host: 'github.com'
  }
] as const

export const COMMUNITY_LEARN_LINKS: readonly LearnLink[] = [
  {
    id: 'hermes-bible',
    kind: 'community',
    title: 'Hermes Bible',
    description: 'Unofficial searchable docs, flows, videos, and repos',
    url: 'https://www.hermesbible.com/',
    host: 'www.hermesbible.com'
  },
  {
    id: 'bible-flows',
    kind: 'community',
    title: 'Bible Flows',
    description: 'Community operator workflows',
    url: 'https://www.hermesbible.com/flows',
    host: 'www.hermesbible.com'
  },
  {
    id: 'hermes-atlas',
    kind: 'community',
    title: 'Hermes Atlas',
    description: 'Community ecosystem map of tools and integrations',
    url: 'https://hermesatlas.com/',
    host: 'hermesatlas.com'
  },
  {
    id: 'atlas-handbook',
    kind: 'community',
    title: 'Atlas Handbook',
    description: 'Beginner guide to Hermes Agent',
    url: 'https://hermesatlas.com/guide/',
    host: 'hermesatlas.com'
  }
] as const

export function visibleLearnLinks(): LearnLink[] {
  return [...OFFICIAL_LEARN_LINKS, ...(MENU_BAR_SHOW_COMMUNITY_LEARN_LINKS ? COMMUNITY_LEARN_LINKS : [])]
}

export function isAllowlistedLearnUrl(url: string): boolean {
  try {
    const parsed = new URL(url)

    if (parsed.protocol !== 'https:') {
      return false
    }

    return visibleLearnLinks().some(link => link.url === url && link.host === parsed.hostname)
  } catch {
    return false
  }
}
