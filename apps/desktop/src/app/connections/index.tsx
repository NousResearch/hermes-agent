/**
 * Połączenia: what Jarvis can be connected to, how to get the key for each
 * model or tool, and how other apps can talk to Jarvis through its own API.
 *
 * The page connects nothing by itself. Every button either starts a guided
 * setup in a fresh conversation (the agent drives the skill; the person reads
 * the request first) or opens the real settings page where the credential is
 * entered. What exists and where it goes is `connections-catalog.ts`.
 */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import { CONNECTION_STATUS_KEY, type ConnectionState, getConnectionStatus } from '@/api/connections'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { ExternalLink, KeyRound, ShieldLock, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { requestComposerPrefill } from '@/store/composer'

import { CONNECTION_ICONS } from '../jarvis/connection-icons'
import {
  JARVIS_API_KEYS,
  JARVIS_API_SERVER,
  JARVIS_CONNECTIONS,
  JARVIS_KEY_ROUTES,
  jarvisApiCurlExample,
  jarvisApiPythonExample,
  type JarvisConnection
} from '../jarvis/connections-catalog'
import { readJarvisOnboardingState } from '../jarvis/onboarding-state'
import { PageSearchShell } from '../page-search-shell'
import { navigateToWorkspacePage, NEW_CHAT_ROUTE } from '../routes'

import { DriveMemoryPanel } from './drive-memory-panel'
import { GoogleConnectDialog } from './google-connect-dialog'

const TABS = ['connections', 'keys', 'api'] as const

type ConnectionsTab = (typeof TABS)[number]

const CARD = 'jarvis-glass rounded-3xl p-4'
const INTEGRATION_CONNECTIONS = new Set(['google', 'notion', 'github', 'smartHome', 'mcp', 'messaging'])

function ExternalAnchor({ href, label }: { href: string; label: string }) {
  return (
    <a
      className="inline-flex min-h-9 items-center gap-1.5 rounded-lg px-2 text-sm text-(--ui-accent) hover:underline"
      href={href}
      rel="noreferrer"
      target="_blank"
    >
      {label}
      <ExternalLink className="size-3.5" />
    </a>
  )
}

const WIZARD_COPY = {
  en: { connect: 'Connect Google', connected: 'Connected', manage: 'Manage' },
  pl: { connect: 'Połącz Google', connected: 'Połączono', manage: 'Zarządzaj' }
} as const

function ConnectionCard({ connection, state }: { connection: JarvisConnection; state?: ConnectionState }) {
  const { locale, t } = useI18n()
  const queryClient = useQueryClient()
  const [wizardOpen, setWizardOpen] = useState(false)
  const wizard = locale === 'pl' ? WIZARD_COPY.pl : WIZARD_COPY.en
  const navigate = useNavigate()
  const copy = t.jarvisConnections
  const entry = copy.entries[connection.id]
  const { icon: Icon, tile } = CONNECTION_ICONS[connection.id]
  const setup = connection.setup

  const startWithJarvis = () => {
    // A fresh conversation, with the request waiting in the composer: the
    // person reads it (and can edit it) before Jarvis gets it.
    requestComposerPrefill(entry.prompt)
    navigate(NEW_CHAT_ROUTE)
  }

  return (
    <article className={cn(CARD, 'flex flex-col gap-3')} data-connection-card={connection.id}>
      <header className="flex items-start gap-3">
        <span className={cn('grid size-10 shrink-0 place-items-center rounded-xl', tile)}>
          <Icon className="size-5" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-base font-semibold text-(--ui-text-primary)">{entry.name}</h3>
          <p className="text-sm text-(--ui-text-secondary)">{entry.description}</p>
        </div>
        {state === 'connected' ? (
          <span className="shrink-0 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
            {wizard.connected}
          </span>
        ) : (
          <span className="shrink-0 rounded-full bg-(--ui-bg-quaternary) px-2 py-0.5 text-xs text-(--ui-text-tertiary)">
            {copy.auth[connection.auth]}
          </span>
        )}
      </header>
      <p className="text-sm italic text-(--ui-text-tertiary)">{entry.examples}</p>
      {setup.kind === 'wizard' && state === 'connected' ? <DriveMemoryPanel /> : null}
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-[0.12em] text-(--ui-text-tertiary)">
          {copy.stepsLabel}
        </p>
        <ol className="grid gap-1.5">
          {entry.steps.map((step, index) => (
            <li className="flex gap-2 text-sm leading-5 text-(--ui-text-secondary)" key={step}>
              <span className="grid size-5 shrink-0 place-items-center rounded-full bg-(--ui-accent)/15 text-xs font-semibold text-(--ui-accent)">
                {index + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </div>
      <div className="mt-auto flex flex-wrap items-center gap-2 pt-1">
        {setup.kind === 'wizard' ? (
          <>
            <Button className="min-h-10 rounded-full" onClick={() => setWizardOpen(true)} type="button">
              <Sparkles />
              {state === 'connected' ? wizard.manage : wizard.connect}
            </Button>
            <GoogleConnectDialog
              onChanged={() => void queryClient.invalidateQueries({ queryKey: CONNECTION_STATUS_KEY })}
              onClose={() => setWizardOpen(false)}
              open={wizardOpen}
            />
          </>
        ) : setup.kind === 'agent' ? (
          <Button className="min-h-10 rounded-full" onClick={startWithJarvis} type="button">
            <Sparkles />
            {copy.setupWithJarvis}
          </Button>
        ) : (
          <Button
            className="min-h-10 rounded-full"
            onClick={() => navigateToWorkspacePage(navigate, setup.route)}
            type="button"
          >
            {copy.openSettings}
          </Button>
        )}
        {connection.credentialUrl ? (
          <ExternalAnchor href={connection.credentialUrl} label={copy.getCredential} />
        ) : null}
      </div>
    </article>
  )
}

function ConnectionsGrid({ query }: { query: string }) {
  const { t } = useI18n()
  const copy = t.jarvisConnections
  const chosen = readJarvisOnboardingState()?.selections?.connections ?? []
  const status = useQuery({ queryFn: () => getConnectionStatus(), queryKey: CONNECTION_STATUS_KEY, staleTime: 15_000 })
  const needle = query.trim().toLocaleLowerCase()

  const visible = JARVIS_CONNECTIONS.filter(connection => {
    if (!INTEGRATION_CONNECTIONS.has(connection.id)) {
      return false
    }

    const entry = copy.entries[connection.id]

    return !needle || `${entry.name} ${entry.description}`.toLocaleLowerCase().includes(needle)
  })

  const first = visible.filter(connection => chosen.includes(connection.id))
  const rest = visible.filter(connection => !chosen.includes(connection.id))

  const group = (label: string, items: readonly JarvisConnection[]) =>
    items.length > 0 ? (
      <section className="grid gap-3" key={label}>
        {label ? (
          <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-(--ui-text-tertiary)">{label}</h2>
        ) : null}
        <div className="grid gap-3 lg:grid-cols-2">
          {items.map(connection => (
            <ConnectionCard connection={connection} key={connection.id} state={status.data?.[connection.id]} />
          ))}
        </div>
      </section>
    ) : null

  return (
    <div className="grid gap-6">
      {group(copy.chosenLabel, first)}
      {group(first.length > 0 ? copy.allLabel : '', rest)}
    </div>
  )
}

function ApiKeysSection() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const copy = t.jarvisConnections.keys

  return (
    <div className="grid gap-4">
      <div className={cn(CARD, 'flex gap-3')}>
        <KeyRound className="mt-0.5 size-5 shrink-0 text-(--ui-accent)" />
        <p className="text-sm leading-6 text-(--ui-text-secondary)">{copy.body}</p>
      </div>
      {(['model', 'tool'] as const).map(kind => (
        <section className="grid gap-2" key={kind}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-(--ui-text-tertiary)">
              {kind === 'model' ? copy.model : copy.tool}
            </h2>
            <Button
              className="min-h-9 rounded-full"
              onClick={() => navigate(JARVIS_KEY_ROUTES[kind])}
              size="sm"
              type="button"
              variant="secondary"
            >
              {kind === 'model' ? copy.openModelKeys : copy.openToolKeys}
            </Button>
          </div>
          <ul className="grid gap-2 lg:grid-cols-2">
            {JARVIS_API_KEYS.filter(key => key.kind === kind).map(key => (
              <li className={cn(CARD, 'flex flex-col gap-1 p-3')} data-api-key={key.id} key={key.id}>
                <code className="text-xs font-semibold text-(--ui-text-primary)">{key.env}</code>
                <span className="text-sm text-(--ui-text-secondary)">{copy.purposes[key.id]}</span>
                <ExternalAnchor href={key.url} label={copy.getKey} />
              </li>
            ))}
          </ul>
        </section>
      ))}
      <p className="flex gap-2 text-sm leading-6 text-(--ui-text-tertiary)">
        <ShieldLock className="mt-1 size-4 shrink-0" />
        {copy.safety}
      </p>
    </div>
  )
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <figure className="grid gap-1.5">
      <figcaption className="flex items-center justify-between text-xs font-semibold uppercase tracking-[0.12em] text-(--ui-text-tertiary)">
        {label}
        <CopyButton appearance="icon" label={label} text={code} />
      </figcaption>
      <pre className="overflow-x-auto rounded-xl border border-(--ui-stroke-tertiary) bg-black/40 p-3 text-xs leading-5 text-[#dbe7ff]">
        <code>{code}</code>
      </pre>
    </figure>
  )
}

function ApiSection() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const copy = t.jarvisConnections.api

  return (
    <div className="grid gap-4">
      <div className={cn(CARD, 'grid gap-2')}>
        <p className="text-sm leading-6 text-(--ui-text-secondary)">{copy.body}</p>
        <p className="text-sm italic text-(--ui-text-tertiary)">{copy.uses}</p>
      </div>
      <ol className="grid gap-2">
        {copy.steps.map((step, index) => (
          <li className="flex gap-2 text-sm leading-6 text-(--ui-text-secondary)" key={step}>
            <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-(--ui-accent)/15 text-xs font-semibold text-(--ui-accent)">
              {index + 1}
            </span>
            {step}
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap items-center gap-3">
        <Button className="min-h-10 rounded-full" onClick={() => navigate(JARVIS_API_SERVER.route)} type="button">
          {copy.open}
        </Button>
        <code className="rounded-lg border border-(--ui-stroke-tertiary) px-2 py-1 text-xs text-(--ui-text-primary)">
          {JARVIS_API_SERVER.baseUrl}
        </code>
        <code className="rounded-lg border border-(--ui-stroke-tertiary) px-2 py-1 text-xs text-(--ui-text-primary)">
          model: {JARVIS_API_SERVER.model}
        </code>
      </div>
      <CodeBlock code={jarvisApiCurlExample()} label={copy.curlLabel} />
      <CodeBlock code={jarvisApiPythonExample()} label={copy.pythonLabel} />
      <p className="flex gap-2 text-sm leading-6 text-(--ui-text-tertiary)">
        <ShieldLock className="mt-1 size-4 shrink-0" />
        {copy.security}
      </p>
    </div>
  )
}

export function ConnectionsView() {
  const { t } = useI18n()
  const copy = t.jarvisConnections
  const [tab, setTab] = useState<ConnectionsTab>('connections')
  const [query, setQuery] = useState('')

  const tabs = useMemo(
    () => [
      { id: 'connections', label: copy.title },
      { id: 'keys', label: copy.keys.title },
      { id: 'api', label: copy.api.tab }
    ],
    [copy]
  )

  return (
    <PageSearchShell
      activeTab={tab}
      data-testid="connections-page"
      onSearchChange={setQuery}
      onTabChange={id => setTab(id as ConnectionsTab)}
      searchHidden={tab !== 'connections'}
      searchPlaceholder={copy.title}
      searchValue={query}
      tabs={tabs}
    >
      <div className="h-full overflow-y-auto">
        <div className="mx-auto grid max-w-5xl gap-6 px-4 py-6 md:px-6">
          <header className="grid gap-1">
            <h1 className="text-2xl font-semibold text-(--ui-text-primary)">
              {tab === 'connections' ? copy.title : tab === 'keys' ? copy.keys.title : copy.api.title}
            </h1>
            {tab === 'connections' ? (
              <p className="max-w-3xl text-sm text-(--ui-text-secondary)">{copy.subtitle}</p>
            ) : null}
          </header>
          {tab === 'connections' ? <ConnectionsGrid query={query} /> : null}
          {tab === 'keys' ? <ApiKeysSection /> : null}
          {tab === 'api' ? <ApiSection /> : null}
        </div>
      </div>
    </PageSearchShell>
  )
}
