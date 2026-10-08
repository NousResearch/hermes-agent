export interface ConnectorCardTranslations {
  title: string
  connect: string
  skip: string
  cancel: string
  retry: string
  grant: string
  connected: string
  checking: string
  notConnected: string
  skipped: string
  disabled: string
  failed: string
  needsAuth: string
  opening: string
  waiting: string
  timeout: string
  refresh: string
  connectError: string
  connectErrorFor: (app: string) => string
  unavailable: string
  ownerMissing: string
  search: string
  empty: string
  disclaimer: string
  execution: string
  setup: (server: string) => string
  openInBrowser: string
  setupCancel: string
  authorizedToolsUnavailable: string
  required: string
  namedAccount: (app: string, alias: string) => string
  renameAccount: string
}

export interface ConnectorAccountsTranslations {
  heading: string
  count: (count: number) => string
  addAnother: string
  aliasLabel: string
  aliasHint: string
  aliasInvalid: string
  aliasTaken: string
  add: string
  rename: string
  renameLabel: (name: string) => string
  remove: string
  removeTitle: (app: string, name: string) => string
  removeBody: string
  retiredCount: (count: number) => string
  status: {
    expired: string
    failed: string
    inactive: string
    pending: string
    retired: string
    revoked: string
  }
}
