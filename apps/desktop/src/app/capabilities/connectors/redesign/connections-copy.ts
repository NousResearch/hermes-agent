import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

import { ar, ja, ru, zh, zhHant } from './connections-copy-intl'
import { de, es, fr } from './connections-copy-latin'
import type { ToolFacet } from './filter-logic'
import type { ConnectionCategoryId, ConnectionSubgroupId } from './taxonomy'

export type ConnectionBadge = ToolFacet | 'off'

/**
 * The redesigned Connections surface's own words (Figma `1511:10722`).
 *
 * Nothing here reuses the classic connector vocabulary: the page speaks about
 * apps, tools, connections and accounts, never about the old surface's terms.
 */
export interface ConnectionsCopy {
  badges: Record<ConnectionBadge, string>
  bulk: {
    discard: string
    save: string
    someOn: string
    turnAllOff: string
    turnAllOn: string
  }
  card: {
    add: string
    connect: string
    connected: string
    disconnect: string
    manage: string
    menu: string
    notConnected: string
    reconnect: string
    refreshTools: string
    remove: string
    rename: string
    toolsOff: (count: number) => string
    toolsOn: (count: number) => string
    toolsSomeOn: (on: number, total: number) => string
    waiting: string
  }
  categories: Record<ConnectionCategoryId, { name: string; subtitle: string }>
  disconnect: {
    body: (name: string) => string
    cancel: string
    confirm: string
    title: (name: string) => string
  }
  errors: {
    connectBody: string
    connectTitle: (name: string) => string
    dismiss: string
    loadBody: string
    loadTitle: string
    tryAgain: string
  }
  facetTooltips: Record<ToolFacet, string>
  filters: {
    activeFilters: (count: number) => string
    addMcpServer: string
    all: string
    allCategories: string
    apply: string
    backToConnections: string
    button: string
    categories: string
    categoryChipLabel: string
    chooseOne: string
    clear: string
    close: string
    findCategory: string
    noResultsBody: string
    noResultsQuery: (query: string) => string
    noResultsTitle: string
    resultCount: (count: number) => string
    searchCategory: (name: string) => string
    selectedCount: (count: number) => string
    showConnections: (count: number) => string
    status: string
    statusAny: string
    statusChipLabel: string
    statusConnected: string
    statusNotConnected: string
    title: string
  }
  page: { subtitle: string; title: string }
  removeAccount: {
    body: (name: string) => string
    cancel: string
    confirm: string
    title: string
  }
  search: { clear: string; placeholder: string }
  sections: {
    connected: string
    other: string
    seeAll: (count: number) => string
    suggested: string
  }
  sheet: {
    account: string
    accounts: string
    addAnAccount: string
    addAnotherAccount: string
    availableSubtitle: string
    availableToAgents: string
    cancelConnection: string
    changeAfterConnect: string
    close: string
    connected: string
    connectedOn: (date: string) => string
    disconnectFailed: (app: string) => string
    facetDescriptions: Record<ToolFacet, string>
    finishSignIn: string
    finishSignInBody: (app: string) => string
    notConnected: string
    permissionsTooltip: (app: string) => string
    previewAll: (count: number) => string
    removeAccountBody: (app: string) => string
    removeAccountTitle: (alias: string) => string
    removeFailed: (alias: string) => string
    removedSuccess: (alias: string) => string
    renamePlaceholder: string
    reopenSignIn: string
    reviewAll: (count: number) => string
    reviewTools: string
    signInBody: (app: string) => string
    signInWith: (app: string) => string
    startsOff: string
    startsOn: string
    toolPermissions: string
    tools: string
    toolsCount: (count: number) => string
    waiting: string
    waitingFor: (app: string) => string
  }
  subgroups: Record<ConnectionSubgroupId, string>
  toast: {
    connected: (name: string) => string
    disconnected: (name: string) => string
    removed: (name: string) => string
    renamed: (name: string) => string
  }
  toolFilters: {
    all: string
    apply: string
    areas: string
    behavior: string
    clear: string
    close: string
    status: string
    title: string
    uncategorised: string
  }
  tools: {
    allAreas: string
    areaChipLabel: string
    back: string
    emptyBody: string
    emptyTitle: string
    errorBody: string
    errorTitle: string
    filter: string
    findArea: string
    loading: string
    lockedByOrg: string
    noResultsBody: string
    noResultsTitle: string
    off: string
    on: string
    pageSummary: (on: number, total: number, app: string) => string
    pageTitle: (app: string) => string
    permissionChipLabel: string
    searchAppTools: (app: string) => string
    searchPlaceholder: string
    showTools: (count: number) => string
    subtitle: (name: string) => string
    summary: (on: number, total: number) => string
    title: string
    toolNamePrefix: string
    unsavedChanges: (count: number) => string
    whatItDoes: string
    whatItDoesHint: string
  }
}

export const en: ConnectionsCopy = {
  badges: { destructive: 'Destructive', off: 'Off', read: 'Read', write: 'Write' },
  bulk: {
    discard: 'Discard',
    save: 'Save',
    someOn: 'Some on',
    turnAllOff: 'Turn all off',
    turnAllOn: 'Turn all on'
  },
  card: {
    add: 'Add',
    connect: 'Connect',
    connected: 'Connected',
    disconnect: 'Disconnect',
    manage: 'Manage',
    menu: 'More options',
    notConnected: 'Not connected',
    reconnect: 'Reconnect',
    refreshTools: 'Refresh tool list',
    remove: 'Remove',
    rename: 'Rename',
    toolsOff: count => `${count} ${count === 1 ? 'tool' : 'tools'} off`,
    toolsOn: count => `${count} ${count === 1 ? 'tool' : 'tools'} on`,
    toolsSomeOn: (on, total) => `${on} of ${total} tools on`,
    waiting: 'Waiting for sign-in'
  },
  categories: {
    'customers-growth': {
      name: 'Customers & growth',
      subtitle: 'Marketing, sales and support.'
    },
    'data-analytics': {
      name: 'Data & analytics',
      subtitle: 'Understand your product and your data.'
    },
    'design-content': {
      name: 'Design & content',
      subtitle: 'Create, design and publish.'
    },
    engineering: {
      name: 'Engineering',
      subtitle: 'Build, host and monitor.'
    },
    money: {
      name: 'Money',
      subtitle: 'Payments, banking and investments.'
    },
    reference: {
      name: 'Reference',
      subtitle: 'Look things up without leaving the chat.'
    },
    'travel-fitness': {
      name: 'Travel & fitness',
      subtitle: 'Plan trips and track your training.'
    },
    'work-productivity': {
      name: 'Work & productivity',
      subtitle: 'Docs, projects, meetings and the apps your team lives in.'
    }
  },
  disconnect: {
    body: name => `Every agent loses access to ${name}. You can connect again anytime.`,
    cancel: 'Cancel',
    confirm: 'Disconnect',
    title: name => `Disconnect ${name}?`
  },
  errors: {
    connectBody: 'Check that you finished signing in, then try again.',
    connectTitle: name => `Could not connect ${name}`,
    dismiss: 'Dismiss',
    loadBody: 'Your apps could not be checked. Anything loaded before is still here.',
    loadTitle: 'Connections could not load',
    tryAgain: 'Try again'
  },
  facetTooltips: {
    destructive: 'Destructive: can delete data or undo work.',
    read: 'Read: can look at your data but cannot change it.',
    write: 'Write: can create and change data.'
  },
  filters: {
    activeFilters: count => `${count} ${count === 1 ? 'filter' : 'filters'}`,
    addMcpServer: 'Add MCP server',
    all: 'All',
    allCategories: 'All categories',
    apply: 'Show results',
    backToConnections: 'Connections',
    button: 'Filters',
    categories: 'Categories',
    categoryChipLabel: 'Category',
    chooseOne: 'Choose one',
    clear: 'Clear all',
    close: 'Close',
    findCategory: 'Find a category',
    noResultsBody: 'Nothing matches these filters. Clear them to see every app.',
    noResultsQuery: query => `Nothing matches “${query}”. Try another search or clear your filters.`,
    noResultsTitle: 'No connections match',
    resultCount: count => `${count} ${count === 1 ? 'connection' : 'connections'}`,
    searchCategory: name => `Search ${name}`,
    selectedCount: count => `${count} selected`,
    showConnections: count => `Show ${count} connections`,
    status: 'Status',
    statusAny: 'All',
    statusChipLabel: 'Status',
    statusConnected: 'Connected',
    statusNotConnected: 'Not connected',
    title: 'Filters'
  },
  page: {
    subtitle: 'Connect the apps you already use. Connections follow your account; choose which agents use each one in Agent settings.',
    title: 'Connections'
  },
  removeAccount: {
    body: name => `${name} will be removed from this connection. The app stays connected with your other accounts.`,
    cancel: 'Cancel',
    confirm: 'Remove',
    title: 'Remove this account?'
  },
  search: { clear: 'Clear search', placeholder: 'Search for apps and keywords' },
  sections: {
    connected: 'Connected',
    other: 'Other apps',
    seeAll: count => `See all ${count}`,
    suggested: 'Suggested for you'
  },
  sheet: {
    account: 'Account',
    accounts: 'Accounts',
    addAnAccount: '+ Add an account',
    addAnotherAccount: '+ Add another account',
    availableSubtitle: 'Pause for all agents; stay signed in.',
    availableToAgents: 'Available to agents',
    cancelConnection: 'Cancel connection',
    changeAfterConnect: 'You can change these after you connect.',
    close: 'Close',
    connected: 'Connected',
    connectedOn: date => `Connected ${date}`,
    disconnectFailed: app => `Couldn’t disconnect ${app}. Check your connection and `,
    facetDescriptions: {
      destructive: 'Permanent deletion',
      read: 'View only',
      write: 'Create and edit'
    },
    finishSignIn: 'Finish signing in',
    finishSignInBody: app => `Approve access on the ${app} page in your browser, then come back here.`,
    notConnected: 'Not connected',
    permissionsTooltip: app =>
      `${app} decides whether each tool is Read, Write, or Destructive. You choose which ones are on.`,
    previewAll: count => `Preview all ${count}`,
    removeAccountBody: app => `Agents can no longer use this ${app} account. Other accounts stay connected.`,
    removeAccountTitle: alias => `Remove “${alias}”?`,
    removeFailed: alias => `Couldn’t remove “${alias}”. Check your connection and `,
    removedSuccess: alias => `Removed “${alias}” successfully`,
    renamePlaceholder: 'Account name',
    reopenSignIn: 'Reopen sign-in page',
    reviewAll: count => `Review all ${count}`,
    reviewTools: 'Review all tools',
    signInBody: app => `Opens ${app} in your browser. Hermes never sees your password.`,
    signInWith: app => `Sign in with ${app}`,
    startsOff: 'Starts off',
    startsOn: 'Starts on',
    toolPermissions: 'Tool permissions',
    tools: 'Tools',
    toolsCount: count => `${count} tools`,
    waiting: 'Waiting for sign-in',
    waitingFor: app => `Waiting for ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'AI, media & presentations',
    'automation-ai-apis': 'Automation, AI & APIs',
    'bi-search-warehouses': 'BI, search & warehouses',
    'code-review-ci': 'Code review & CI',
    communication: 'Communication',
    'customer-support': 'Customer support',
    'design-whiteboards': 'Design & whiteboards',
    'docs-files': 'Docs & files',
    'docs-knowledge': 'Docs & knowledge',
    'hiring-people': 'Hiring & people',
    'hosting-databases': 'Hosting & databases',
    'marketing-seo': 'Marketing & SEO',
    'meetings-scheduling': 'Meetings & scheduling',
    'monitoring-incidents': 'Monitoring & incidents',
    'outdoors-fitness': 'Outdoors & fitness',
    'payments-commerce': 'Payments & commerce',
    'product-analytics': 'Product analytics',
    'projects-tasks': 'Projects & tasks',
    'sales-crm': 'Sales & CRM',
    'spend-investing-crypto': 'Spending, investing & crypto',
    travel: 'Travel',
    'websites-assets': 'Websites & assets'
  },
  toast: {
    connected: name => `${name} is connected.`,
    disconnected: name => `${name} is disconnected.`,
    removed: name => `${name} was removed.`,
    renamed: name => `Renamed to ${name}.`
  },
  toolFilters: {
    all: 'All',
    apply: 'Show results',
    areas: 'Areas',
    behavior: 'Behavior',
    clear: 'Clear all',
    close: 'Close',
    status: 'Status',
    title: 'Filter tools',
    uncategorised: 'Uncategorised'
  },
  tools: {
    allAreas: 'All areas',
    areaChipLabel: 'Area',
    back: 'Back to connections',
    emptyBody: 'This app has not shared any tools.',
    emptyTitle: 'No tools yet',
    errorBody: 'The tool list could not be read. Try again.',
    errorTitle: 'Tools could not load',
    filter: 'Filters',
    findArea: 'Find an area',
    loading: 'Loading tools',
    lockedByOrg: 'Turned off by your organisation',
    noResultsBody: 'Nothing matches this search or these filters.',
    noResultsTitle: 'No tools match',
    off: 'Off',
    on: 'On',
    pageSummary: (on, total, app) =>
      `${on} of ${total} tools on. Changes save together and apply to every agent that uses ${app}.`,
    pageTitle: app => `${app} Tools`,
    permissionChipLabel: 'Permission',
    searchAppTools: app => `Search ${app} tools`,
    searchPlaceholder: 'Search tools',
    showTools: count => `Show ${count} tools`,
    subtitle: name => `Choose what ${name} can do.`,
    summary: (on, total) => `${on} of ${total} on`,
    title: 'All tools',
    toolNamePrefix: 'Tool name:',
    unsavedChanges: count => `${count} unsaved changes`,
    whatItDoes: 'What it does',
    whatItDoesHint: 'Choose one or more. All clears other choices.'
  }
}

const COPY: Record<BundledLocale, ConnectionsCopy> = {
  ar,
  de,
  en,
  es,
  fr,
  ja,
  ru,
  zh,
  'zh-hant': zhHant
}

/** Figma names the locale `zhHant`; the i18n catalog spells it `zh-hant`. */
function bundledLocaleOf(locale: string): BundledLocale {
  const normalized = locale.toLowerCase()

  if (isBundledLocale(normalized)) {
    return normalized
  }

  return normalized === 'zhhant' || normalized === 'zh_hant' ? 'zh-hant' : 'en'
}

export function connectionsCopy(locale: string): ConnectionsCopy {
  return COPY[bundledLocaleOf(locale)]
}

export function useConnectionsCopy(): ConnectionsCopy {
  return connectionsCopy(useI18n().locale)
}
