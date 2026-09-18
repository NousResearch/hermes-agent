import { useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { AvatarChip } from '@/components/ui/avatar-chip'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorBanner } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import {
  getActionStatus,
  type HermesGateway,
  installMcpCatalogEntry,
  type McpCatalogEntry,
  type ProfileScope
} from '@/hermes'
import { useI18n } from '@/i18n'
import { brandFor } from '@/lib/mcp-brands'
import { type McpImportEntry, parseMcpImport } from '@/lib/mcp-import'
import { isToolEnabled } from '@/lib/mcp-tool-filter'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { ICON_BUTTON, MASTER_DETAIL_WIDE_COLS } from '../../master-detail'
import { PanelAddButton, PanelEmpty } from '../../overlays/panel'
import { prettyName } from '../../settings/helpers'
import { useDeepLinkHighlight } from '../../settings/use-deep-link-highlight'

import { parseServersDoc, serverEnabled } from './mcp-doc'
import { McpEditorPane } from './mcp-editor'
import {
  capabilitySummary,
  type Probe,
  type ServerCost,
  type ServerStatus,
  STATUS_DOT,
  statusLine,
  statusOf
} from './mcp-status'
import { useMcpServers } from './use-mcp-servers'

export function McpTab({ gateway, profile }: { gateway: HermesGateway | null; profile?: ProfileScope }) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const mcp = useMcpServers({ gateway, profile })

  const {
    availableCatalog,
    blocks,
    config,
    configError,
    configFailed,
    configLoading,
    draft,
    names,
    selected,
    servers
  } = mcp

  useDeepLinkHighlight({
    block: 'nearest',
    elementId: serverName => `mcp-server-${serverName}`,
    onResolve: mcp.focusServer,
    param: 'server',
    ready: serverName => blocks.some(block => block.name === serverName)
  })

  // Cached data paints instantly; a spinner only ever shows on the first-ever
  // load, and a failed load gets a real retry — never a silent blank pane.
  if (configFailed && !config) {
    return (
      <div className="flex h-full min-h-0 flex-1 items-center justify-center p-6">
        <ErrorBanner className="max-w-sm">
          <span className="flex flex-col gap-2">
            {configError instanceof Error ? configError.message : m.failedLoad}
            <Button className="self-start" onClick={mcp.refetchConfig} size="xs" variant="text">
              {m.reload}
            </Button>
          </span>
        </ErrorBanner>
      </div>
    )
  }

  if (!config) {
    return <PageLoader className="min-h-24" label={configLoading ? m.loading : t.skills.loading} />
  }

  // Selection may reference an unsaved block (freshly pasted) — fall back to
  // the draft's parsed entry so the config pane can still describe it.
  const savedEntry = selected ? servers[selected] : undefined

  const draftEntry = (() => {
    if (!selected || savedEntry) {
      return undefined
    }

    try {
      return parseServersDoc(draft)[selected]
    } catch {
      return undefined
    }
  })()

  const activeEntry = savedEntry ?? draftEntry

  return (
    <div className={cn('grid h-full min-h-0 grid-cols-1', MASTER_DETAIL_WIDE_COLS)}>
      {/* LEFT: the focused block's server config, or the unified fleet+catalog list. */}
      <aside className="flex min-h-0 flex-col overflow-hidden border-r border-(--ui-stroke-quaternary)">
        {selected && activeEntry ? (
          <ServerConfig
            authing={mcp.authing === selected}
            cost={mcp.costFor(selected, activeEntry)}
            description={mcp.descriptionFor(selected, activeEntry)}
            entry={activeEntry}
            name={selected}
            onAuthenticate={() => void mcp.authenticate(selected)}
            onBack={() => mcp.setCursor(0)}
            onProbe={() => void mcp.runProbe(selected)}
            onRemove={() => void mcp.removeServer(selected)}
            onToggle={checked => void mcp.setServerEnabled(selected, checked)}
            onToggleTool={toolName => void mcp.toggleTool(selected, toolName)}
            probe={mcp.probes[selected]}
            saved={savedEntry !== undefined}
            saving={mcp.saving}
          />
        ) : (
          <div className="flex min-h-0 flex-1 flex-col p-2">
            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]">
              {/* ONE coherent column: the configured fleet on top, the
                  Nous-approved catalog below it. Installed entries live in the
                  fleet list (with live status), so the catalog section only
                  offers what's NOT installed yet — no duplicate rows, no tab
                  flipping to find the install button. */}
              {/* Geometry mirrors ListStrip (mb-1 h-6 pl-2) so this header
                  lands on the exact line the sort link occupies in the
                  Skills/Tools views. */}
              <div className="mb-1 flex h-6 shrink-0 items-center pl-2 pr-1">
                <span className="flex-1 text-[0.72rem] font-medium text-(--ui-text-tertiary)">{m.tabServers}</span>
                <McpImportButton disabled={mcp.profilePending} onImport={mcp.importServers} />
              </div>
              {names.length === 0 ? (
                <PanelEmpty
                  action={
                    <Button onClick={mcp.addServer} size="sm">
                      {m.newServer}
                    </Button>
                  }
                  description={m.emptyDesc}
                  icon="plug"
                  title={m.emptyTitle}
                />
              ) : (
                <>
                  {names.map(serverName => {
                    const server = servers[serverName]
                    const status = statusOf(server, mcp.probes[serverName])
                    const cost = mcp.costFor(serverName, server)

                    return (
                      <McpRow
                        active={false}
                        busy={mcp.saving}
                        enabled={serverEnabled(server)}
                        key={serverName}
                        name={serverName}
                        onProbe={() => void mcp.runProbe(serverName)}
                        onRemove={() => void mcp.removeServer(serverName)}
                        onSelect={() => mcp.focusServer(serverName)}
                        onToggle={checked => void mcp.setServerEnabled(serverName, checked)}
                        status={status}
                        statusText={statusLine(m, status, mcp.probes[serverName], server, cost)}
                        unused={
                          serverEnabled(server) &&
                          status === 'ok' &&
                          cost.tokens !== null &&
                          cost.tokens > 0 &&
                          cost.uses === 0
                        }
                      />
                    )
                  })}
                  <PanelAddButton label={m.newServer} onClick={mcp.addServer} />
                </>
              )}
              {(mcp.catalogLoading || availableCatalog.length > 0) && (
                <>
                  <div className="mb-1 mt-3 flex h-6 shrink-0 items-center border-t border-(--ui-stroke-quaternary) pl-2 pr-1 pt-2">
                    <span className="text-[0.72rem] font-medium text-(--ui-text-tertiary)">{m.tabCatalog}</span>
                  </div>
                  <McpCatalog
                    entries={availableCatalog}
                    loading={mcp.catalogLoading}
                    onInstalled={mcp.onCatalogInstalled}
                    profile={profile}
                  />
                </>
              )}
            </div>
          </div>
        )}
      </aside>

      {/* RIGHT: the mcp.json editor, logs hard-pinned below. */}
      <McpEditorPane controller={mcp} />
    </div>
  )
}


function ServerConfig({
  authing,
  cost,
  description,
  entry,
  name,
  onAuthenticate,
  onBack,
  onProbe,
  onRemove,
  onToggle,
  onToggleTool,
  probe,
  saved,
  saving
}: {
  authing: boolean
  cost?: ServerCost
  description: null | string
  entry: Record<string, unknown>
  name: string
  onAuthenticate: () => void
  onBack: () => void
  onProbe: () => void
  onRemove: () => void
  onToggle: (checked: boolean) => void
  onToggleTool: (toolName: string) => void
  probe: Probe | undefined
  saved: boolean
  saving: boolean
}) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const status = statusOf(entry, probe)

  // OAuth is only offered to servers that are actually OAuth-shaped. A server
  // with `headers` uses API-key/bearer auth — a 401 there means a bad key, NOT
  // "log in with OAuth"; routing it through the browser flow would wrongly
  // rewrite its config to `auth: oauth`. So: explicit `auth: oauth` can re-auth
  // on failure; an auth-less HTTP server may try OAuth on a 401; header servers
  // never do.
  const hasHeaderAuth = !!entry.headers && typeof entry.headers === 'object'

  const canAuth =
    typeof entry.url === 'string' &&
    !hasHeaderAuth &&
    (entry.auth === 'oauth' ? status === 'needs-auth' || status === 'error' : !entry.auth && status === 'needs-auth')

  const summary = probe && probe !== 'probing' && probe.ok ? capabilitySummary(m, probe, entry, cost) : null

  return (
    // p-2 matches the list view's container so flipping list ⇄ config keeps
    // content anchored at the same origin.
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-2 [scrollbar-gutter:stable]">
      {/* Geometry cloned from McpRow so nothing jumps when flipping list ⇄
          config: items-start with per-element top margins that reproduce the
          row's h-11 centering exactly (h-5 controls → mt-3, size-6 avatar →
          mt-2.5, h-4 switch → mt-3.5) no matter how tall the text column gets. */}
      <div className="flex items-start gap-2 pr-1.5">
        <Tip label={m.allServers}>
          <Button
            aria-label={m.allServers}
            className={cn('mt-3', ICON_BUTTON)}
            onClick={onBack}
            size="icon"
            variant="ghost"
          >
            <Codicon name="chevron-left" size="0.8125rem" />
          </Button>
        </Tip>
        <McpAvatar className="mt-2.5" name={name} status={status} />
        <div className="min-w-0 flex-1 pt-1">
          <h3 className="min-w-0 truncate text-[0.9375rem] font-semibold tracking-tight">{prettyName(name)}</h3>
          <p className="mt-0.5 truncate text-[0.68rem] text-(--ui-text-tertiary)">
            {typeof entry.url === 'string' ? entry.url : [entry.command, ...((entry.args as string[]) ?? [])].join(' ')}
          </p>
          {summary && <p className="mt-0.5 text-[0.68rem] text-(--ui-text-tertiary)">{summary}</p>}
        </div>
        {saved && (
          // Direct row children (no wrapper): the icons↔switch gap must be the
          // row's own gap-2, byte-identical to McpRow.
          <>
            <ServerIconActions
              className="mt-3"
              onProbe={onProbe}
              onRemove={onRemove}
              probing={probe === 'probing'}
              saving={saving}
            />
            <ServerSwitch
              className="mt-3.5"
              disabled={saving}
              enabled={serverEnabled(entry)}
              name={name}
              onToggle={onToggle}
            />
          </>
        )}
      </div>

      {description && (
        <p className="mt-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </p>
      )}

      {canAuth && saved && (
        <div className="mt-3 flex justify-end">
          <Button disabled={authing} onClick={onAuthenticate} size="xs">
            {authing ? m.waitingForBrowser : m.authenticate}
          </Button>
        </div>
      )}
      {!saved && <p className="mt-3 text-[0.68rem] text-muted-foreground/60">{m.unsavedConnect}</p>}

      {status === 'probing' && <PageLoader className="min-h-24" label={t.skills.loading} />}

      {/* No inline error dump — the status dot/line says "Error"/"Needs
          authentication", and the actual failure lands in the logs pane below
          (and the console). A big red block here just shouts the same thing. */}

      {probe && probe !== 'probing' && probe.ok && probe.tools.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {/* Chip = a discovered tool; click to include/exclude it (struck
              through when excluded, so it won't register). The probe always
              lists every tool regardless of the filter. */}
          {probe.tools.map(tool => {
            const on = isToolEnabled(entry, tool.name)

            return (
              <button
                aria-pressed={on}
                className={cn(
                  'rounded-md px-1.5 py-0.5 font-mono text-[0.65rem] text-(--ui-text-tertiary) hover:text-foreground',
                  saved ? 'cursor-pointer' : 'cursor-default',
                  on ? 'bg-(--ui-bg-quinary)' : 'line-through opacity-70'
                )}
                disabled={!saved}
                key={tool.name}
                onClick={() => onToggleTool(tool.name)}
                title={on ? m.disableTool(tool.name) : m.enableTool(tool.name)}
                type="button"
              >
                {tool.name}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// The enable toggle, shared by the row and the config header. It reflects the
// configured `enabled` flag ONLY — full-strength when on, dimmed when off — so
// "is this on?" reads instantly from config, never gated on a probe that can
// take seconds (stdio servers spawn `npx`). Whether it's actually *connected*
// is the status dot's job, not the switch's.
function ServerSwitch({
  className,
  disabled,
  enabled,
  name,
  onToggle
}: {
  className?: string
  disabled: boolean
  enabled: boolean
  name: string
  onToggle: (checked: boolean) => void
}) {
  return (
    <Switch
      aria-label={name}
      checked={enabled}
      className={cn('shrink-0 cursor-pointer', !enabled && 'opacity-60', className)}
      disabled={disabled}
      onCheckedChange={onToggle}
      size="xs"
      title={name}
    />
  )
}

// Refresh + delete, identical beside every toggle (rows and config header).
function ServerIconActions({
  className,
  onProbe,
  onRemove,
  probing,
  saving
}: {
  className?: string
  onProbe: () => void
  onRemove: () => void
  probing: boolean
  saving: boolean
}) {
  const { t } = useI18n()
  const m = t.settings.mcp

  return (
    <span className={cn('flex items-center gap-0.5', className)}>
      <Tip label={m.reload}>
        <Button
          aria-label={m.reload}
          className={ICON_BUTTON}
          disabled={probing}
          onClick={onProbe}
          size="icon"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.8125rem" spinning={probing} />
        </Button>
      </Tip>
      <Tip label={m.remove}>
        <Button
          aria-label={m.remove}
          className={cn(ICON_BUTTON, 'hover:text-destructive')}
          disabled={saving}
          onClick={onRemove}
          size="icon"
          variant="ghost"
        >
          <Codicon name="trash" size="0.8125rem" />
        </Button>
      </Tip>
    </span>
  )
}

// Paste-anything import: a compact popover on the Servers header. Paste any
// README shape — mcp.json snippet, npx/docker command line, `claude mcp add`,
// a bare URL, or a Cursor deeplink — see the inferred name + config, then
// merge it into the editor draft (unsaved, like the "+" starter entry).
function McpImportButton({ disabled, onImport }: { disabled: boolean; onImport: (entries: McpImportEntry[]) => void }) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')

  const entries = useMemo(() => parseMcpImport(text), [text])

  const reset = () => {
    setText('')
  }

  const confirm = () => {
    if (!entries) {
      return
    }

    onImport(entries)
    setOpen(false)
    reset()
  }

  return (
    <Popover
      onOpenChange={next => {
        setOpen(next)

        if (!next) {
          reset()
        }
      }}
      open={open}
    >
      <PopoverTrigger asChild>
        <Button className="h-5 px-1 text-[0.68rem]" disabled={disabled} size="xs" variant="text">
          <Codicon name="clippy" size="0.75rem" />
          {m.importButton}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80">
        <div className="flex flex-col gap-2">
          <Textarea
            aria-label={m.importButton}
            autoFocus
            className="max-h-40 min-h-20 font-mono text-[0.68rem]"
            onChange={event => setText(event.currentTarget.value)}
            placeholder={m.importPlaceholder}
            value={text}
          />
          {entries ? (
            <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
              {entries.map((entry, index) => (
                <div className="rounded-md bg-(--ui-bg-tertiary) px-2 py-1.5" key={`${entry.name}-${index}`}>
                  <span className="block truncate text-[0.72rem] font-medium text-foreground/85">{entry.name}</span>
                  <span className="block truncate font-mono text-[0.62rem] text-muted-foreground/60">
                    {typeof entry.config.url === 'string'
                      ? entry.config.url
                      : [entry.config.command, ...((entry.config.args as string[]) ?? [])].join(' ')}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            text.trim() && <p className="px-0.5 text-[0.62rem] text-muted-foreground/60">{m.importNoMatch}</p>
          )}
          <div className="flex justify-end">
            <Button disabled={!entries} onClick={confirm} size="xs">
              {entries && entries.length > 1 ? m.importConfirmMany(entries.length) : m.importConfirm}
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

// Small gray attribute chip (transport / auth / needs-build), matching the
// catalog's flat row treatment.
function CatalogTag({ children }: { children: string }) {
  return (
    <span className="rounded bg-(--ui-bg-tertiary) px-1.5 py-0.5 text-[0.6rem] text-(--ui-text-secondary)">
      {children}
    </span>
  )
}

// The Nous-approved MCP catalog: one-click installs of curated servers, with an
// inline prompt for any required credentials (never shows stored values). On
// install the parent refetches config + catalog and reloads live sessions.
function McpCatalog({
  entries,
  loading,
  onInstalled,
  profile
}: {
  entries: McpCatalogEntry[]
  loading: boolean
  onInstalled: () => void
  profile?: ProfileScope
}) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const [installing, setInstalling] = useState<null | string>(null)
  const [envDrafts, setEnvDrafts] = useState<Record<string, Record<string, string>>>({})
  const [envOpenFor, setEnvOpenFor] = useState<null | string>(null)

  const install = async (entry: McpCatalogEntry) => {
    const required = entry.required_env.filter(env => env.required)
    const draft = envDrafts[entry.name] ?? {}

    // Reveal the credential prompt first; only error once it's shown and unfilled.
    if (required.some(env => !draft[env.name]?.trim())) {
      if (envOpenFor !== entry.name) {
        setEnvOpenFor(entry.name)

        return
      }

      notify({ kind: 'error', title: m.catalogEnvPrompt(entry.name), message: m.catalogEnvRequired })

      return
    }

    setInstalling(entry.name)

    try {
      const res = await installMcpCatalogEntry(entry.name, draft, profile ?? undefined)

      // Git-backed entries clone in the background — keep the row busy and poll
      // the action to completion before refetching / re-enabling, so a re-click
      // can't spawn a second install over the first's tracked process. A non-zero
      // exit is a real failure — surface it instead of a false success.
      if (res.background && res.action) {
        for (;;) {
          const status = await getActionStatus(res.action, 1, profile ?? undefined)

          if (!status.running) {
            if (status.exit_code !== 0) {
              throw new Error(m.catalogInstallFailed(entry.name))
            }

            break
          }

          await new Promise(resolve => setTimeout(resolve, CATALOG_INSTALL_POLL_MS))
        }
      }

      notify({ kind: 'success', title: m.catalogInstallStarted(entry.name), message: '' })
      setEnvOpenFor(null)
      onInstalled()
    } catch (err) {
      notifyError(err, m.catalogInstallFailed(entry.name))
    } finally {
      setInstalling(null)
    }
  }

  if (loading) {
    return <PageLoader className="min-h-24" label={m.catalogLoading} />
  }

  if (entries.length === 0) {
    return <PanelEmpty description={m.catalogEmpty} icon="plug" title={m.tabCatalog} />
  }

  return (
    <div className="flex flex-col">
      {entries.map(entry => {
        const draft = envDrafts[entry.name] ?? {}

        return (
          <div className="rounded-md px-2 py-2" key={entry.name}>
            <div className="flex items-start gap-2">
              {/* 2px nudge so the start-aligned avatar sits where McpRow's
                  center-aligned one does — no jump when flipping Servers⇄Catalog. */}
              <McpAvatar
                className="mt-0.5"
                name={entry.name}
                status={entry.installed ? (entry.enabled ? 'ok' : 'off') : 'unknown'}
              />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="truncate text-[0.78rem] font-medium text-foreground/85">
                    {prettyName(entry.name)}
                  </span>
                  <CatalogTag>{entry.transport}</CatalogTag>
                  {entry.auth_type === 'oauth' && <CatalogTag>OAuth</CatalogTag>}
                  {entry.auth_type === 'api_key' && <CatalogTag>API key</CatalogTag>}
                  {entry.needs_install && !entry.installed && <CatalogTag>{m.catalogNeedsInstall}</CatalogTag>}
                  {entry.installed && (
                    <span className="text-[0.6rem] text-emerald-400">
                      {entry.enabled ? m.catalogEnabled : m.catalogInstalled}
                    </span>
                  )}
                </div>
                <p className="mt-0.5 line-clamp-2 text-[0.68rem] text-muted-foreground/70">{entry.description}</p>
                {envOpenFor === entry.name && entry.required_env.length > 0 && (
                  <div className="mt-2 grid gap-2">
                    {entry.required_env.map(env => (
                      <label className="grid gap-1" key={env.name}>
                        <span className="text-[0.62rem] text-muted-foreground">
                          {env.prompt || env.name}
                          {env.required ? ' *' : ''}
                        </span>
                        <Input
                          className="h-7 text-xs"
                          onChange={event =>
                            setEnvDrafts(prev => ({
                              ...prev,
                              [entry.name]: { ...prev[entry.name], [env.name]: event.currentTarget.value }
                            }))
                          }
                          type="password"
                          value={draft[env.name] ?? ''}
                        />
                      </label>
                    ))}
                  </div>
                )}
              </div>
              <Button
                className="mt-0.5 shrink-0"
                disabled={entry.installed || installing !== null}
                onClick={() => void install(entry)}
                size="xs"
                variant="text"
              >
                {installing === entry.name
                  ? m.catalogInstalling
                  : entry.installed
                    ? m.catalogInstalled
                    : m.catalogInstall}
              </Button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// Cadence for polling a background (git-bootstrap) catalog install to completion.
const CATALOG_INSTALL_POLL_MS = 1500

// ---------------------------------------------------------------------------
// Avatars + list rows
// ---------------------------------------------------------------------------

// The shared identity chip (`ui/avatar-chip`) plus a status dot. Identity
// ladder: curated brand glyph (lib/mcp-brands, shared with the composer
// suggestion pills and the inline setup card) → letter monogram. Nothing here
// reaches the network for a mark: a configured MCP URL can be a private host,
// and the connector card's favicon rung only ever reads a public site's own
// markup, never a third-party icon service.
function McpAvatar({ className, name, status }: { className?: string; name: string; status: ServerStatus }) {
  return (
    <AvatarChip
      brand={brandFor(name)}
      className={className}
      name={name}
      overlay={
        <span
          aria-hidden
          className={cn(
            'absolute -bottom-0.5 -right-0.5 size-2 rounded-full ring-2 ring-(--ui-chat-surface-background)',
            STATUS_DOT[status]
          )}
        />
      }
    />
  )
}

function McpRow({
  active,
  busy,
  enabled,
  name,
  onProbe,
  onRemove,
  onSelect,
  onToggle,
  status,
  statusText,
  unused
}: {
  active: boolean
  busy: boolean
  enabled: boolean
  name: string
  onProbe: () => void
  onRemove: () => void
  onSelect: () => void
  onToggle: (checked: boolean) => void
  status: ServerStatus
  statusText: string
  unused?: boolean
}) {
  const { t } = useI18n()
  const m = t.settings.mcp

  return (
    <div
      className={cn(
        'group/row row-hover flex h-11 w-full shrink-0 items-center gap-2 rounded-md pl-2 pr-1.5 hover:text-foreground',
        active ? 'bg-(--ui-row-active-background) text-foreground' : 'text-(--ui-text-secondary)'
      )}
      id={`mcp-server-${name}`}
    >
      <button
        className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
        onClick={onSelect}
        type="button"
      >
        <McpAvatar name={name} status={status} />
        <span className="min-w-0 flex-1">
          <span className="flex min-w-0 items-center gap-1.5">
            <span
              className={cn(
                'min-w-0 truncate text-[0.78rem]',
                enabled ? 'font-medium text-foreground/85' : 'font-normal text-muted-foreground/60'
              )}
            >
              {prettyName(name)}
            </span>
            {/* Subtle "paying for schemas, not using them" hint — a muted pill,
                never a dialog. Shown only when the overlay KNOWS both halves:
                nonzero schema cost and zero 30-day uses. */}
            {unused && (
              <span className="shrink-0 rounded bg-(--ui-bg-tertiary) px-1 py-px text-[0.58rem] font-normal text-muted-foreground/60">
                {m.unusedPill}
              </span>
            )}
          </span>
          <span className="block truncate text-[0.62rem] text-muted-foreground/50">{statusText}</span>
        </span>
      </button>
      <ServerIconActions
        className="opacity-0 transition-opacity focus-within:opacity-100 group-hover/row:opacity-100"
        onProbe={onProbe}
        onRemove={onRemove}
        probing={status === 'probing'}
        saving={busy}
      />
      <ServerSwitch disabled={busy} enabled={enabled} name={name} onToggle={onToggle} />
    </div>
  )
}
