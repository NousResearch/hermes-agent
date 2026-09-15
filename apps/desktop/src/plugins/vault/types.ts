export interface VaultNoteSummary {
  title: string
  rel_path: string
  tags: string[]
  mtime: number
  size: number
  links_count: number
  backlinks_count: number
}

export interface VaultWikilink {
  target: string
  section: string | null
  alias: string | null
  raw: string
}

export interface VaultNote {
  path: string
  rel_path: string
  title: string
  content: string
  frontmatter: Record<string, unknown>
  tags: string[]
  links: VaultWikilink[]
  backlinks: string[]
  forward_links: string[]
  mtime: number
  size: number
}

export interface VaultGraphNode {
  id: string
  label: string
  path: string
  tags: string[]
  weight: number
  group: string
}

export interface VaultGraphEdge {
  source: string
  target: string
}

export interface VaultGraph {
  nodes: VaultGraphNode[]
  edges: VaultGraphEdge[]
}

export interface VaultSuggestion {
  title: string
  alias?: string
  path: string
}
