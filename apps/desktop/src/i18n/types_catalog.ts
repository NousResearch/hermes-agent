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
}

export interface SkillDeepLinkTranslations {
  installTitle: (name: string) => string
  installDescription: string
  installTo: string
  thisComputer: string
  installing: string
  installComplete: (name: string) => string
  destinationChanged: string
  installed: string
  source: string
}
