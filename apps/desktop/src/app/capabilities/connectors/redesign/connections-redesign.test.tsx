// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { previewInterfaceMode } from '@/store/interface-mode'
import { stubMenuDomApis } from '@/test/jsdom'

import type { McpServersController } from '../../mcp/use-mcp-servers'
import { type WriteOutcome } from '../data/mutations'
import type { HostedConnectorsView } from '../data/queries'
import { asConnectorError } from '../data/rpc'
import type { AccountRow, ConnectorCardModel } from '../types'

import { ConnectionSheet, type ConnectionSheetProps } from './connection-sheet'
import { ConnectionToolsPage, type ConnectionToolsPageProps } from './connection-tools-page'
import { ConnectionsView, type ConnectionsViewProps } from './connections-view'

const { TOOLS } = vi.hoisted(() => ({
  TOOLS: [
    {
      categories: ['Issues'],
      deprecated: false,
      description: 'List every issue.',
      facet: 'read',
      hints: [],
      name: 'list_issues',
      slug: 'list_issues'
    },
    {
      categories: ['Issues'],
      deprecated: false,
      description: 'Create an issue.',
      facet: 'write',
      hints: [],
      name: 'create_issue',
      slug: 'create_issue'
    },
    {
      categories: ['Issues'],
      deprecated: false,
      description: 'Delete an issue.',
      facet: 'destructive',
      hints: [],
      name: 'delete_issue',
      slug: 'delete_issue'
    },
    {
      categories: ['Admin'],
      deprecated: false,
      description: 'Old admin tool.',
      facet: 'read',
      hints: [],
      name: 'old_tool',
      slug: 'old_tool'
    }
  ]
}))

vi.mock('../data/queries', () => ({
  useConnectorTools: () => ({
    refresh: () => undefined,
    retry: () => undefined,
    signedOut: false,
    status: null,
    tools: TOOLS
  })
}))

const ok = (): Promise<WriteOutcome> => Promise.resolve({ ok: true })
const fail = (): Promise<WriteOutcome> => Promise.resolve({ ok: false, error: asConnectorError(new Error('nope')) })

const CONTROLLER = {
  authenticate: vi.fn(async () => undefined),
  probes: {},
  runProbe: vi.fn(async () => undefined),
  servers: {},
  setServerEnabled: vi.fn(async () => undefined),
  setServerTools: vi.fn(async () => true)
} as unknown as McpServersController

const WORK: AccountRow = {
  active: true,
  alias: 'Work',
  connection_id: 'ca_work',
  connector: 'gmail',
  created_at: '2026-10-05T00:00:00Z',
  label: 'Gmail account',
  status: 'active',
  updated_at: '2026-10-05T00:00:00Z'
}

function hostedView(overrides: Partial<HostedConnectorsView> = {}): HostedConnectorsView {
  return {
    accounts: [],
    accountsLoaded: true,
    listSlugs: new Set(),
    phase: 'ready',
    policy: {
      member: {
        allowed: null,
        denied: new Set(),
        revision: 'rev-1',
        tags: { disable: [], enable: [] },
        tools: {}
      },
      others: []
    },
    refetch: vi.fn(),
    retryRules: vi.fn(),
    rows: [],
    rulesFailed: false,
    rulesSignedOut: false,
    titles: {},
    ...overrides
  }
}

function hostedCard(
  name: string,
  slug: string,
  { accounts = [], connected = false }: { accounts?: AccountRow[]; connected?: boolean } = {}
): ConnectorCardModel {
  return {
    inCatalog: true,
    name,
    residency: 'hosted',
    slug,
    state: connected ? 'connected' : 'available',
    stateWord: connected ? 'connected' : 'available',
    verb: connected ? undefined : 'connect',
    ways: {
      hosted: {
        accounts,
        connected,
        retiredAccounts: [],
        state: connected ? 'connected' : 'available',
        verb: connected ? undefined : 'connect'
      },
      local: null
    }
  }
}

const SLACK = hostedCard('Slack', 'slack', { connected: true, accounts: [WORK] })
const NOTION = hostedCard('Notion', 'notion')
const LINEAR = hostedCard('Linear', 'linear')
const CANVA = hostedCard('Canva', 'canva')
const HIGGSFIELD = hostedCard('Higgsfield', 'higgsfield')
const FIGMA = hostedCard('Figma', 'figma')
const CLOUDFLARE = hostedCard('Cloudflare', 'cloudflare')
const CONTEXT7 = hostedCard('Context7', 'context7')

const DIRECTORY_CARDS = [SLACK, NOTION, LINEAR, CANVA, HIGGSFIELD, FIGMA, CLOUDFLARE, CONTEXT7]

function baseViewProps(overrides: Partial<ConnectionsViewProps> = {}): ConnectionsViewProps {
  return {
    agentControls: true,
    cards: DIRECTORY_CARDS,
    controller: CONTROLLER,
    hosted: hostedView(),
    onCloseCard: vi.fn(),
    onConnect: vi.fn(ok),
    onDisconnectApp: vi.fn(ok),
    onGiveUp: vi.fn(),
    onInstall: vi.fn(),
    onOpenCard: vi.fn(),
    onReconnect: vi.fn(),
    onRemoveAccount: vi.fn(ok),
    onRemoveLocalServer: vi.fn(),
    onRenameAccount: vi.fn(ok),
    onVerb: vi.fn(),
    openCard: null,
    profile: 'default',
    ...overrides
  }
}

function DirectoryHarness({
  cards = DIRECTORY_CARDS,
  onSubviewChange,
  hosted = hostedView()
}: {
  cards?: readonly ConnectorCardModel[]
  hosted?: HostedConnectorsView
  onSubviewChange?: (inSubview: boolean) => void
}) {
  const [open, setOpen] = useState<ConnectorCardModel | null>(null)

  return (
    <ConnectionsView
      {...baseViewProps({ cards, hosted, onSubviewChange })}
      onCloseCard={() => setOpen(null)}
      onOpenCard={setOpen}
      openCard={open}
    />
  )
}

function renderDirectory(props: Parameters<typeof DirectoryHarness>[0] = {}) {
  return render(
    <I18nProvider>
      <DirectoryHarness {...props} />
    </I18nProvider>
  )
}

function sectionFor(title: string): HTMLElement {
  const heading = screen.getByRole('heading', { level: 2, name: title })

  return heading.closest('section') as HTMLElement
}

beforeEach(() => {
  stubMenuDomApis()
  previewInterfaceMode('simple')
})

afterEach(() => {
  cleanup()
  previewInterfaceMode(null)
  vi.clearAllMocks()
})

describe('ConnectionsView directory', () => {
  it('leads with Connected then Suggested, and previews categories in Simple order', () => {
    renderDirectory()

    expect(screen.getByRole('heading', { level: 2, name: 'Connected' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Suggested for you' })).toBeTruthy()

    const order = screen.getAllByRole('heading', { level: 2 }).map(heading => heading.textContent ?? '')
    const work = order.indexOf('Work & productivity')
    const design = order.indexOf('Design & content')
    const engineering = order.indexOf('Engineering')

    expect(work).toBeGreaterThan(-1)
    expect(work).toBeLessThan(design)
    expect(design).toBeLessThan(engineering)
  })

  it('puts Engineering first in Advanced', () => {
    previewInterfaceMode('advanced')
    renderDirectory()

    const order = screen.getAllByRole('heading', { level: 2 }).map(heading => heading.textContent ?? '')

    expect(order.indexOf('Engineering')).toBeLessThan(order.indexOf('Work & productivity'))
    expect(order.indexOf('Work & productivity')).toBeLessThan(order.indexOf('Design & content'))
  })

  it('offers Manage, Connect and Add on the rows they belong to', () => {
    renderDirectory()

    expect(within(sectionFor('Connected')).getByRole('button', { name: 'Manage' })).toBeTruthy()
    expect(within(sectionFor('Work & productivity')).getByRole('button', { name: 'Connect' })).toBeTruthy()
    expect(within(sectionFor('Reference')).getByRole('button', { name: 'Add' })).toBeTruthy()
  })

  it('opens a category See-All page and returns to the directory', async () => {
    const onSubviewChange = vi.fn()
    renderDirectory({ onSubviewChange })

    fireEvent.click(within(sectionFor('Work & productivity')).getByRole('button', { name: 'See all 3' }))

    expect(screen.getByRole('heading', { level: 1, name: 'Work & productivity' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Docs & files' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Communication' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Projects & tasks' })).toBeTruthy()
    await waitFor(() => expect(onSubviewChange).toHaveBeenLastCalledWith(true))

    fireEvent.click(screen.getByRole('button', { name: 'Connections' }))

    expect(screen.queryByRole('heading', { level: 1, name: 'Work & productivity' })).toBeNull()
    expect(screen.getByRole('heading', { level: 2, name: 'Connected' })).toBeTruthy()
    await waitFor(() => expect(onSubviewChange).toHaveBeenLastCalledWith(false))
  })

  it('opens the app drawer on Manage', () => {
    renderDirectory()

    fireEvent.click(within(sectionFor('Connected')).getByRole('button', { name: 'Manage' }))

    expect(screen.getByRole('dialog', { name: 'Slack' })).toBeTruthy()
  })

  it('stays on the tools page when its Disconnect confirmation is canceled', async () => {
    renderDirectory()

    fireEvent.click(within(sectionFor('Connected')).getByRole('button', { name: 'Manage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Review all 4' }))
    expect(screen.getByRole('heading', { level: 1, name: 'Slack Tools' })).toBeTruthy()

    fireEvent.pointerDown(screen.getByRole('button', { name: 'More options' }), {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })
    fireEvent.click(await screen.findByRole('menuitem', { name: /^Disconnect/ }))

    const confirm = await screen.findByRole('alertdialog', { name: 'Disconnect Slack?' })
    fireEvent.click(within(confirm).getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    expect(screen.getByRole('heading', { level: 1, name: 'Slack Tools' })).toBeTruthy()
    expect(screen.queryByRole('dialog', { name: 'Slack' })).toBeNull()
  })

  it('filters by status and category, shows the matching count, and clears chips', async () => {
    renderDirectory()

    fireEvent.click(screen.getByRole('button', { name: /^Filters/ }))
    fireEvent.click(await screen.findByRole('radio', { name: /Connected/ }))
    fireEvent.click(screen.getByRole('button', { name: /All categories/ }))

    const list = screen.getByRole('listbox', { name: 'Categories' })
    fireEvent.click(within(list).getByRole('option', { name: /Work & productivity/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Show 1 connections' }))

    expect(screen.getByText('Slack')).toBeTruthy()
    expect(screen.queryAllByText('Figma')).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Clear all: Connected' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Clear all: Work & productivity' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Clear all: Work & productivity' }))
    expect(screen.queryByRole('button', { name: 'Clear all: Work & productivity' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }))
    expect(screen.getAllByText('Figma').length).toBeGreaterThan(0)
    expect(screen.queryByRole('button', { name: 'Clear all: Connected' })).toBeNull()
  })

  it('narrows the rows with search', async () => {
    renderDirectory()
    const search = screen.getByRole('textbox', { name: 'Search for apps and keywords' })

    fireEvent.change(search, { target: { value: 'figma' } })

    expect(screen.getAllByText('Figma').length).toBeGreaterThan(0)
    expect(screen.queryByText('Slack')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Clear search' }))
    expect(screen.getByText('Slack')).toBeTruthy()
  })
})

function baseSheetProps(overrides: Partial<ConnectionSheetProps> = {}): ConnectionSheetProps {
  return {
    agentControls: true,
    card: hostedCard('Gmail', 'gmail', { accounts: [WORK], connected: true }),
    controller: CONTROLLER,
    hosted: hostedView({ accounts: [WORK], listSlugs: new Set(['gmail']) }),
    installing: false,
    onClose: vi.fn(),
    onConnect: vi.fn(ok),
    onDisconnectApp: vi.fn(ok),
    onGiveUp: vi.fn(),
    onInstall: vi.fn(),
    onOpenToolsPage: vi.fn(),
    onReconnect: vi.fn(),
    onRemoveAccount: vi.fn(ok),
    onRemoveLocalServer: vi.fn(),
    onRenameAccount: vi.fn(ok),
    onToggleForMe: vi.fn(),
    profile: 'default',
    togglePending: false,
    ...overrides
  }
}

function renderSheet(overrides: Partial<ConnectionSheetProps> = {}) {
  return render(
    <I18nProvider>
      <ConnectionSheet {...baseSheetProps(overrides)} />
    </I18nProvider>
  )
}

async function openAccountMenu(): Promise<void> {
  const heading = screen.getByRole('heading', { level: 3, name: 'Account' })
  const accountCard = heading.closest('section') as HTMLElement
  const trigger = within(accountCard).getByRole('button', { name: 'More options' })
  fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Remove' }))
}

describe('ConnectionSheet', () => {
  it('shows the connected state, its availability switch and the account row', () => {
    const onToggleForMe = vi.fn()
    renderSheet({ onToggleForMe })

    expect(screen.getByRole('heading', { level: 2, name: 'Gmail' })).toBeTruthy()
    expect(screen.getByText('Connected')).toBeTruthy()
    expect(screen.getByRole('heading', { level: 3, name: 'Available to agents' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 3, name: 'Account' })).toBeTruthy()
    expect(screen.getByText('Work')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Review all 4' })).toBeTruthy()

    fireEvent.click(screen.getByRole('switch', { name: 'Available to agents' }))
    expect(onToggleForMe).toHaveBeenCalledWith(false)
  })

  it('removes an account inline and reports success in a banner', async () => {
    const onRemoveAccount = vi.fn(ok)
    renderSheet({ onRemoveAccount })

    await openAccountMenu()
    const confirm = screen.getByRole('alertdialog', { name: 'Remove “Work”?' })
    fireEvent.click(within(confirm).getByRole('button', { name: 'Remove' }))

    expect(onRemoveAccount).toHaveBeenCalledWith(WORK)
    expect((await screen.findByRole('status')).textContent).toContain('Removed “Work” successfully')
  })

  it('reports a failed account removal with a retry action', async () => {
    renderSheet({ onRemoveAccount: vi.fn(fail) })

    await openAccountMenu()
    fireEvent.click(
      within(screen.getByRole('alertdialog', { name: 'Remove “Work”?' })).getByRole('button', {
        name: 'Remove'
      })
    )

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('Couldn’t remove “Work”. Check your connection and')
    expect(within(alert).getByRole('button', { name: 'Try again' })).toBeTruthy()
  })

  it('opens the disconnect confirmation from the header menu', async () => {
    const onDisconnectApp = vi.fn(ok)
    renderSheet({ onDisconnectApp })

    const dialog = screen.getByRole('dialog', { name: 'Gmail' })
    fireEvent.pointerDown(within(dialog).getAllByRole('button', { name: 'More options' })[0], {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })
    fireEvent.click(await screen.findByRole('menuitem', { name: /^Disconnect/ }))

    const confirm = screen.getByRole('alertdialog', { name: 'Disconnect Gmail?' })
    expect(confirm).toBeTruthy()
    fireEvent.click(within(confirm).getByRole('button', { name: 'Disconnect' }))
    await waitFor(() => expect(onDisconnectApp).toHaveBeenCalledTimes(1))
  })

  it('asks the parent to clear its disconnect state when the confirmation is canceled', () => {
    const onDisconnectDismiss = vi.fn()
    renderSheet({ initialDisconnect: true, onDisconnectDismiss })

    const confirm = screen.getByRole('alertdialog', { name: 'Disconnect Gmail?' })
    fireEvent.click(within(confirm).getByRole('button', { name: 'Cancel' }))

    expect(onDisconnectDismiss).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('alertdialog', { name: 'Disconnect Gmail?' })).toBeNull()
  })

  it('shows the not-connected offer, its preview count, and the starting permissions', () => {
    renderSheet({
      card: hostedCard('Gmail', 'gmail'),
      hosted: hostedView({ listSlugs: new Set(['gmail']) })
    })

    expect(screen.getByRole('heading', { level: 3, name: 'Sign in with Gmail' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Preview all 4' })).toBeTruthy()
    expect(screen.getByText('Starts off')).toBeTruthy()
    expect(screen.getAllByText('Starts on').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Connect' })).toBeTruthy()
  })
})

function baseToolsProps(overrides: Partial<ConnectionToolsPageProps> = {}): ConnectionToolsPageProps {
  const member = hostedView().policy.member

  return {
    agentControls: true,
    card: hostedCard('Gmail', 'gmail', { accounts: [WORK], connected: true }),
    controller: CONTROLLER,
    hosted: {
      ...hostedView({ accounts: [WORK], listSlugs: new Set(['gmail']) }),
      policy: {
        member: member === null ? null : { ...member, tools: { gmail: ['old_tool'] } },
        others: []
      }
    },
    onBack: vi.fn(),
    onDisconnect: vi.fn(),
    onReconnect: vi.fn(),
    profile: 'default',
    ...overrides
  }
}

function renderTools(overrides: Partial<ConnectionToolsPageProps> = {}) {
  return render(
    <I18nProvider>
      <ConnectionToolsPage {...baseToolsProps(overrides)} />
    </I18nProvider>
  )
}

describe('ConnectionToolsPage', () => {
  it('lists areas, tool names, facets and the Off state, and expands a tool', () => {
    renderTools()

    expect(screen.getByRole('heading', { level: 1, name: 'Gmail Tools' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Issues (3)' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Admin (1)' })).toBeTruthy()
    expect(screen.getAllByText(/Read/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Write/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Destructive/).length).toBeGreaterThan(0)
    expect(screen.getByText('Off')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Show what List issues does' }))

    expect(screen.getByText('Tool name: list_issues')).toBeTruthy()
    expect(screen.getByText('List every issue.')).toBeTruthy()
  })

  it('filters tools from the Tool Filter popover and shows the active chips', async () => {
    renderTools()

    fireEvent.click(screen.getByRole('button', { name: 'Filters' }))
    expect(await screen.findByText('Filters', { selector: '.conn-filter-title' })).toBeTruthy()
    expect(screen.getByText('What it does')).toBeTruthy()

    fireEvent.click(screen.getByRole('checkbox', { name: /^Read/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Show 2 tools' }))

    expect(screen.getByRole('button', { name: 'Clear search: Read' })).toBeTruthy()
    expect(screen.getByText('List issues')).toBeTruthy()
    expect(screen.queryByText('Create issue')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Clear search: Read' }))
    expect(screen.getByText('Create issue')).toBeTruthy()
  })

  it('raises the dirty Save/Discard bar and discards cleanly', () => {
    renderTools()

    fireEvent.click(screen.getByRole('switch', { name: 'Turn list_issues off' }))

    const bar = screen.getByRole('region', { name: '1 unsaved changes' })
    expect(within(bar).getByRole('button', { name: 'Save' })).toBeTruthy()

    fireEvent.click(within(bar).getByRole('button', { name: 'Discard' }))

    expect(screen.queryByRole('region', { name: '1 unsaved changes' })).toBeNull()
  })

  it('keeps a paused connected app editable on the tools page', () => {
    const connected = hostedCard('Gmail', 'gmail', { accounts: [WORK], connected: true })

    renderTools({
      card: {
        ...connected,
        state: 'off',
        stateWord: 'offForYou',
        ways: {
          hosted: {
            accounts: [WORK],
            connected: true,
            offBy: 'me',
            retiredAccounts: [],
            state: 'off'
          },
          local: null
        }
      }
    })

    expect(screen.getByText('Connected')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'More options' })).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Turn list_issues off' }).hasAttribute('disabled')).toBe(false)
  })

  it('goes back to the app from the tools page', () => {
    const onBack = vi.fn()
    renderTools({ onBack })

    fireEvent.click(screen.getByRole('button', { name: 'Gmail' }))

    expect(onBack).toHaveBeenCalledTimes(1)
  })
})
