'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { ConnectorCard, ConnectorRow, type ConnectorRowMark, ConnectorSummary } from '@/components/ui/connector-card'
import { getActionStatus, getMcpCatalog, installMcpCatalogEntry, type McpCatalogEntry, setMcpServerEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { connectorText, type McpTarget, mcpTargets } from '@/lib/connector-tools'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import {
  type ConnectionRequest,
  type ConnectionTarget,
  type ConnectionTargetOutcome,
  continueConnectionRequest,
  respondToConnectionRequest,
  sessionConnectionRequest
} from '@/store/connection-request'
import { $gateway } from '@/store/gateway'
import { reconnectAction } from '@/store/gateway-reconnect'
import { notifyError } from '@/store/notifications'
import { invalidateMcpSuggestionIndex } from '@/store/suggestion-providers/mcp'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

type SetupAction = McpTarget['action']
type SetupCopy = ReturnType<typeof useI18n>['t']['assistant']['mcpSetup']

const CATALOG_INSTALL_POLL_MS = 1500

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

const TITLE = {
  authorize: (copy: SetupCopy) => copy.authorizeTitle,
  enable: (copy: SetupCopy) => copy.enableTitle,
  install: (copy: SetupCopy) => copy.installTitle
} satisfies Record<SetupAction, (copy: SetupCopy) => string>

const VERB = {
  authorize: (copy: SetupCopy) => copy.authorizeAction,
  enable: (copy: SetupCopy) => copy.enableAction,
  install: (copy: SetupCopy) => copy.installAction
} satisfies Record<SetupAction, (copy: SetupCopy) => string>

const DONE = {
  authorize: (copy: SetupCopy, server: string) => copy.authorized(server),
  enable: (copy: SetupCopy, server: string) => copy.enabled(server),
  install: (copy: SetupCopy, server: string) => copy.installed(server)
} satisfies Record<SetupAction, (copy: SetupCopy, server: string) => string>

// Mirrors `RESOLVED_STATES` in tools/connectors/contract.py.
const resolved = (target: ConnectionTarget): boolean =>
  target.state === 'connected' || target.state === 'skipped' || target.state === 'unavailable'

function readSetupAction(args: unknown): SetupAction {
  const [target] = mcpTargets('manage_connections', parseMaybeObject(args))

  return target?.action ?? 'install'
}

interface SettledTarget {
  name: string
  state: string
  tools: number
}

function readSetupResult(result: unknown): SettledTarget[] {
  const row = parseMaybeObject(result)
  const targets = Array.isArray(row.targets) ? row.targets.map(parseMaybeObject) : []

  return targets.flatMap(target => {
    const name = connectorText(target.name)

    return name
      ? [{ name, state: connectorText(target.state) ?? '', tools: Array.isArray(target.tools) ? target.tools.length : 0 }]
      : []
  })
}

export const McpSetupTool = (props: ToolCallMessagePartProps) => {
  if (props.result !== undefined) {
    return <McpSetupSettled {...props} />
  }

  return <McpSetupLive {...props} />
}

const McpSetupLive = (props: ToolCallMessagePartProps) => {
  const messageRunning = useAuiState(selectMessageRunning)

  if (!messageRunning) {
    return <ToolFallback {...props} />
  }

  return <McpSetupPending {...props} />
}

function McpSetupSettled({ args, result }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  const action = useMemo(() => readSetupAction(args), [args])
  const targets = useMemo(() => readSetupResult(result), [result])

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-1">
      {targets.map(target => {
        const title = prettyName(target.name)
        const connected = target.state === 'connected'

        const line = connected
          ? DONE[action](copy, title)
          : target.state === 'skipped'
            ? t.connectors.skipped
            : t.connectors.notConnected

        return (
          <ConnectorSummary
            connector={{ name: target.name, title }}
            key={target.name}
            meta={connected && target.tools > 0 ? `${line} · ${copy.toolCount(target.tools)}` : line}
            tone={connected ? 'ok' : undefined}
          />
        )
      })}
    </div>
  )
}

export function McpSetupPending({ args }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  // Use the rendering transcript's session, not the globally active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionConnectionRequest(sessionId), [sessionId])
  const request = useStore($request)
  const action = useMemo(() => readSetupAction(args), [args])
  const title = TITLE[action](copy)

  // `tool.start` arrives before `connection.request`.
  if (!request) {
    return (
      <div className={cn(SHELL_CLASS, 'my-1.5 flex items-center gap-2')} data-slot="connector-card">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
        <span className="text-(--ui-text-tertiary)">{title}</span>
      </div>
    )
  }

  const open = request.targets.filter(target => !resolved(target))

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-1" data-connector-offer>
      <ConnectorCard title={title}>
        {request.targets.map(target => (
          <McpSetupRow
            action={action}
            copy={copy}
            key={target.name}
            request={request}
            single={open.length === 1 && open[0] === target}
            target={target}
          />
        ))}
      </ConnectorCard>
      {open.length > 0 ? (
        <div className="px-3.5">
          <Button onClick={() => void continueConnectionRequest(request)} size="xs" variant="textStrong">
            {t.common.continue}
          </Button>
        </div>
      ) : null}
    </div>
  )
}

interface McpSetupRowProps {
  action: SetupAction
  copy: SetupCopy
  request: ConnectionRequest
  /** The only open row owns ⌘⏎; with several rows the buttons are the path. */
  single: boolean
  target: ConnectionTarget
}

function McpSetupRow({ action, copy, request, single, target }: McpSetupRowProps) {
  const { t } = useI18n()
  const gateway = useStore($gateway)
  const [working, setWorking] = useState(false)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [entry, setEntry] = useState<McpCatalogEntry | null | undefined>(undefined)
  const [envOpen, setEnvOpen] = useState(false)
  const server = target.name
  const displayName = prettyName(server)
  const done = resolved(target)

  const respond = async (outcome: ConnectionTargetOutcome) => {
    if (!gateway) {
      notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed, { action: reconnectAction() })

      return
    }

    if (outcome.status === 'connected') {
      invalidateMcpSuggestionIndex()
    }

    try {
      await respondToConnectionRequest(request, { targets: [outcome] })
    } catch (error) {
      notifyError(error, copy.sendFailed)
    }
  }

  const approve = async () => {
    const oauthScope = capabilityScoped()
    setWorking(true)

    try {
      if (action === 'enable') {
        await setMcpServerEnabled(server, true)
        triggerHaptic('submit')
        await respond({ name: server, status: 'connected' })

        return
      }

      if (action === 'authorize') {
        const flow = await completeMcpDesktopOAuth({ serverName: server, profile: oauthScope })

        triggerHaptic('submit')
        await respond({ name: server, status: 'connected', tools: (flow.tools ?? []).map(tool => tool.name) })

        return
      }

      let catalogEntry = entry

      if (catalogEntry === undefined) {
        const catalog = await getMcpCatalog()
        catalogEntry = catalog.entries.find(candidate => candidate.name === server) ?? null
        setEntry(catalogEntry)
      }

      if (!catalogEntry) {
        await respond({ detail: copy.notInCatalog(server), name: server, status: 'failed' })

        return
      }

      const required = catalogEntry.required_env.filter(env => env.required)

      if (required.some(env => !envDraft[env.name]?.trim())) {
        setEnvOpen(true)

        return
      }

      const res = await installMcpCatalogEntry(server, envDraft)

      // Poll background installs so non-zero exits cannot report false success.
      if (res.background && res.action) {
        for (;;) {
          const status = await getActionStatus(res.action, 1)

          if (!status.running) {
            if (status.exit_code !== 0) {
              throw new Error(copy.failed(server))
            }

            break
          }

          await new Promise(resolve => setTimeout(resolve, CATALOG_INSTALL_POLL_MS))
        }
      }

      triggerHaptic('submit')
      await respond({ name: server, status: 'connected' })
    } catch (error) {
      // The user closed the sign-in window; the row simply offers again.
      if (error instanceof McpOAuthCancelled) {
        return
      }

      notifyError(error, copy.failed(displayName))
      await respond({
        detail: error instanceof Error ? error.message : String(error),
        name: server,
        status: 'failed'
      })
    } finally {
      setWorking(false)
    }
  }

  // Do not capture the shortcut while a focusable control owns typed input.
  useEffect(() => {
    if (!single || done) {
      return
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented || !isSubmitEnter(event) || !(event.metaKey || event.ctrlKey)) {
        return
      }

      const active = document.activeElement as HTMLElement | null

      if (
        active &&
        (active.isContentEditable || active.matches('a[href], button, input, select, textarea, [role="button"]'))
      ) {
        return
      }

      if (!working) {
        event.preventDefault()
        void approve()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  })

  const waiting = working && action === 'authorize'
  const mark: ConnectorRowMark = done ? 'connected' : waiting ? 'waiting' : 'idle'

  return (
    <ConnectorRow
      action={done ? undefined : { busy: working, label: VERB[action](copy), onClick: () => void approve() }}
      connector={{ name: server, title: displayName }}
      cue={waiting ? t.connectors.waiting : undefined}
      envDraft={envDraft}
      envFields={entry?.required_env}
      envOpen={envOpen && !!entry && entry.required_env.length > 0}
      envRequired={copy.envRequired}
      mark={mark}
      markLabel={mark === 'connected' ? t.connectors.connected : waiting ? t.connectors.waiting : t.connectors.notConnected}
      onEnvChange={(key, value) => setEnvDraft(prev => ({ ...prev, [key]: value }))}
    />
  )
}
