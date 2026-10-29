export interface CatalogTranslations {
  add: string
  added: string
  discover: string
  featured: string
  explorePlugins: string
  exploreSkills: string
  mostStarred: string
  newest: string
  recentlyUpdated: string
  alphabetical: string
  sortBy: string
  seeAll: string
  related: string
  tags: string
  screenshots: string
  listView: string
  cardView: string
  installTitle: (name: string) => string
  installDescription: string
  installTo: string
  thisComputer: string
  installing: string
  installComplete: (name: string) => string
  destinationChanged: string
  install: string
  installed: string
  searchSkills: string
  searchPlugins: string
  allSources: string
  allCategories: string
  about: string
  author: string
  source: string
  category: string
  version: string
  platforms: string
  requires: string
  tools: string
  hooks: string
  middleware: string
  commands: string
  license: string
  addedDate: string
  updatedDate: string
  repository: string
  documentation: string
  noResults: string
  tryAnother: string
  clearFilters: string
  filters: string
  loadFailed: string
  retry: string
  more: string
  pinned: string
  snapshotHint: string
  installHint: string
  results: (count: number) => string
  back: string
  tier: string
  previousSlide: string
  nextSlide: string
  slideOf: (index: number, count: number) => string
  tierLabels: Record<'official' | 'verified' | 'community', string>
  tiers: {
    official: { title: string; blurb: string }
    verified: { title: string; blurb: string }
  }
  moreToExplore: string
  moreToExploreBlurb: string
  skillsFor: (category: string) => string
  discoverKind: (kind: string) => string
  developer: {
    heading: string
    title: string
    pitch: string
    pitchFollowUp: string
    submit: string
    docs: string
  }
  toolsets: {
    mostUsed: string
    mostUsedBlurb: string
    essentials: string
    essentialsBlurb: string
    toolCount: (count: number) => string
    callCount: (count: string, calls: number) => string
    themes: Record<'work' | 'create' | 'recall' | 'connect' | 'more', { label: string; blurb: string }>
  }
}
