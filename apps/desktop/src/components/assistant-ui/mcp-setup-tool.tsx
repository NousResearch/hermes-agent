'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import {
  addMcpServer,
  authMcpServer,
  cancelMcpOAuthFlow,
  getActionStatus,
  getMcpCatalog,
  getMcpOAuthFlow,
  installMcpCatalogEntry,
  type McpCatalogEntry,
  removeMcpServer,
  setMcpServerEnabled
} from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AlertCircle, CheckCircle2, Loader2 } from '@/lib/icons'
import { brandFor, brandGlyphStyle } from '@/lib/mcp-brands'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { directoryEntry } from '@/lib/mcp-directory'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { clearMcpSetupRequest, type McpSetupOutcome, sessionMcpSetupRequest } from '@/store/mcp-setup'
import { invalidateMcpSuggestionIndex } from '@/store/mcp-suggestions'
import { notifyError } from '@/store/notifications'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

type SetupAction = 'authorize' | 'enable' | 'install'

interface SetupArgs {
  server: string
  action: SetupAction
  reason: string
}

const CATALOG_INSTALL_POLL_MS = 1500

// Thrown by the in-flight flow when the user cancels — the declined respond
// has already been sent, so the catch path must swallow this, not report it.
const CANCELLED = Symbol('mcp-setup-cancelled')

function readSetupArgs(args: unknown): SetupArgs {
  const row = parseMaybeObject(args)
  const rawAction = typeof row.action === 'string' ? row.action : 'install'

  return {
    action: rawAction === 'enable' || rawAction === 'authorize' ? rawAction : 'install',
    reason: typeof row.reason === 'string' ? row.reason : '',
    server: typeof row.server === 'string' ? row.server : ''
  }
}

/** The tool's settled JSON — the card's outcome plus the tool-only
 *  `unanswered` status (timeout, no user action). */
type SettledResult = Omit<Partial<McpSetupOutcome>, 'status'> & {
  status?: McpSetupOutcome['status'] | 'unanswered'
  note?: string
}

function readSetupResult(result: unknown): SettledResult {
  return parseMaybeObject(result) as SettledResult
}

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

// Same platform sniff the approval bar uses for its accelerator hint.
const isMac = typeof navigator !== 'undefined' && /Mac|iP(hone|ad|od)/.test(navigator.platform)

const ICON_CLASS = 'mt-px size-4 shrink-0 text-(--ui-text-tertiary)'

function SetupLine({ children, trailing }: { children: ReactNode; trailing?: ReactNode }) {
  return (
    <div className="flex items-start gap-2">
      <div className="min-w-0 flex-1">{children}</div>
      {trailing}
    </div>
  )
}

export const McpSetupTool = (props: ToolCallMessagePartProps) => {
  // Settled → static outcome line (the flow already ran or was declined).
  if (props.result !== undefined) {
    return <McpSetupSettled {...props} />
  }

  return <McpSetupLive {...props} />
}

const McpSetupLive = (props: ToolCallMessagePartProps) => {
  const messageRunning = useAuiState(selectMessageRunning)

  // Stopped mid-prompt with no result — don't leave a dead interactive panel.
  if (!messageRunning) {
    return <ToolFallback {...props} />
  }

  return <McpSetupPending {...props} />
}

function McpSetupSettled({ args, result }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  const fromArgs = useMemo(() => readSetupArgs(args), [args])
  const fromResult = useMemo(() => readSetupResult(result), [result])

  const server = fromResult.server || fromArgs.server
  const status = fromResult.status ?? 'error'
  const displayName = prettyName(server)

  const line =
    status === 'installed'
      ? copy.installed(displayName)
      : status === 'enabled'
        ? copy.enabled(displayName)
        : status === 'authorized'
          ? copy.authorized(displayName)
          : status === 'declined'
            ? copy.declined
            : status === 'unanswered'
              ? copy.unanswered
              : copy.failed(displayName)

  const ok = status === 'installed' || status === 'enabled' || status === 'authorized'
  const neutral = status === 'declined' || status === 'unanswered'
  const toolCount = Array.isArray(fromResult.tools) ? fromResult.tools.length : 0
  const brand = brandFor(server)

  return (
    <div className={cn(SHELL_CLASS, 'my-1.5 grid gap-1.5')} data-slot="mcp-setup-inline">
      <SetupLine
        trailing={
          ok ? (
            <CheckCircle2 aria-hidden className={cn(ICON_CLASS, 'text-emerald-400')} />
          ) : neutral && brand ? (
            <brand.Icon aria-hidden className="mt-px size-4 shrink-0 opacity-60" style={brandGlyphStyle(brand)} />
          ) : neutral ? (
            <Codicon className={ICON_CLASS} name="plug" size="1rem" />
          ) : (
            <AlertCircle aria-hidden className={cn(ICON_CLASS, 'text-destructive')} />
          )
        }
      >
        <span className={cn('font-medium', neutral && 'italic text-(--ui-text-tertiary)')}>{line}</span>
        {ok && toolCount > 0 && (
          <span className="ml-2 text-(--ui-text-tertiary)">{copy.toolCount(toolCount)}</span>
        )}
        {!ok && !neutral && fromResult.detail ? (
          <p className="mt-0.5 text-(--ui-text-secondary)">{fromResult.detail}</p>
        ) : null}
      </SetupLine>
    </div>
  )
}

function McpSetupPending({ args }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.mcpSetup
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's request (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionMcpSetupRequest(sessionId), [sessionId])
  const request = useStore($request)
  const gateway = useStore($gateway)
  const fromArgs = useMemo(() => readSetupArgs(args), [args])

  const server = fromArgs.server || request?.server || ''
  const action: SetupAction = fromArgs.action ?? request?.action ?? 'install'
  const reason = fromArgs.reason || request?.reason || ''

  const [working, setWorking] = useState(false)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [entry, setEntry] = useState<McpCatalogEntry | null | undefined>(undefined)
  const [envOpen, setEnvOpen] = useState(false)
  // Set when the user cancels mid-flight (a stuck OAuth tab, a hung install).
  // The in-flight flow checks it at every poll boundary and aborts via the
  // CANCELLED sentinel; the declined respond has already been sent by then.
  const cancelRef = useRef(false)

  // Race: tool.start fires a tick before mcp.setup.request — hold the buttons
  // until the gateway request is wired (same spinner rule as clarify).
  const ready = Boolean(request?.requestId)

  const respond = useCallback(
    async (outcome: McpSetupOutcome) => {
      // Another path (cancel racing completion) may have already resolved this
      // request; the store is the single source of truth, so bail if this
      // session's entry is gone — same guard as the approval bar.
      if (!request || sessionMcpSetupRequest(request.sessionId).get()?.requestId !== request.requestId) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed)

        return
      }

      // Clear first: the answer is decided, and an in-flight RPC must not
      // leave a live card that can be answered a second time.
      clearMcpSetupRequest(request.requestId, request.sessionId)

      // A successful outcome changed mcp_servers — reload the live session
      // BEFORE unblocking the tool, or the agent resumes being told the
      // server is ready while its tool snapshot still lacks it (the same
      // write-through mcp-tab's silentReload does; consent was the card
      // click, so no confirm prompt). Reload failure isn't outcome failure:
      // the config landed, tools arrive next session — report it and move on.
      if (outcome.status === 'installed' || outcome.status === 'enabled' || outcome.status === 'authorized') {
        try {
          await gateway.request('reload.mcp', { confirm: true, session_id: request.sessionId ?? undefined })
        } catch (error) {
          notifyError(error, copy.reloadFailed)
        }

        // The just-set-up server must stop being suggested immediately.
        invalidateMcpSuggestionIndex()
      }

      try {
        await gateway.request<{ status?: string }>('mcp.setup.respond', {
          request_id: request.requestId,
          result: JSON.stringify(outcome)
        })
        // tool.complete lands next → McpSetupSettled.
      } catch (error) {
        notifyError(error, copy.sendFailed)
      }
    },
    [copy.gatewayDisconnected, copy.reloadFailed, copy.sendFailed, gateway, request]
  )

  const decline = useCallback(() => {
    // While a flow is in flight this is a CANCEL: answer declined right away
    // and let the abandoned work notice via cancelRef at its next poll.
    cancelRef.current = true
    triggerHaptic('cancel')
    void respond({ server, status: 'declined' })
  }, [respond, server])

  const approve = useCallback(async () => {
    cancelRef.current = false
    setWorking(true)

    // Poll-boundary abort for the background-install loop; the OAuth flows
    // carry their own cancel via completeMcpDesktopOAuth's `cancelled`.
    const throwIfCancelled = <T,>(value: T): T => {
      if (cancelRef.current) {
        throw CANCELLED
      }

      return value
    }

    try {
      if (action === 'enable') {
        await setMcpServerEnabled(server, true)
        triggerHaptic('submit')
        await respond({ server, status: 'enabled' })

        return
      }

      if (action === 'authorize') {
        const flow = await completeMcpDesktopOAuth({
          serverName: server,
          start: authMcpServer,
          status: getMcpOAuthFlow,
          cancelled: () => cancelRef.current,
          cancel: cancelMcpOAuthFlow,
          openExternal: url => window.hermesDesktop.openExternal(url)
        })

        triggerHaptic('submit')
        await respond({ server, status: 'authorized', tools: (flow.tools ?? []).map(tool => tool.name) })

        return
      }

      // Install: prefer the reviewed catalog entry when one exists; otherwise
      // fall back to the desktop suggestion directory (official URL-only
      // remotes), written through the same validated POST the dashboard's add
      // form uses. Required catalog credentials get an inline prompt first
      // (never pre-filled, never echoed back).
      let resolved = entry

      if (resolved === undefined) {
        const catalog = await getMcpCatalog()
        resolved = catalog.entries.find(candidate => candidate.name === server) ?? null
        setEntry(resolved)
      }

      if (!resolved) {
        const known = directoryEntry(server)

        if (!known) {
          await respond({ detail: copy.notInCatalog(server), server, status: 'error' })

          return
        }

        // URL-only remote: add to config, then run the OAuth/probe flow so
        // "Install" lands the user on a working server, not a 401. If the
        // flow dies after the config write (cancel, closed OAuth tab), roll
        // the write back — decline means "no server", not an unauthorized
        // entry squatting in mcp_servers (authoritative-write rule).
        await addMcpServer({ name: known.name, url: known.url })

        let flow

        try {
          flow = await completeMcpDesktopOAuth({
            serverName: known.name,
            start: authMcpServer,
            status: getMcpOAuthFlow,
            cancelled: () => cancelRef.current,
            cancel: cancelMcpOAuthFlow,
            openExternal: url => window.hermesDesktop.openExternal(url)
          })
        } catch (error) {
          await removeMcpServer(known.name).catch(() => {
            // Rollback is best-effort; the primary error/cancel wins.
          })
          throw error
        }

        triggerHaptic('submit')
        await respond({ server, status: 'installed', tools: (flow.tools ?? []).map(tool => tool.name) })

        return
      }

      const required = resolved.required_env.filter(env => env.required)

      if (required.some(env => !envDraft[env.name]?.trim())) {
        // Reveal the credential fields; the user approves again once filled.
        setEnvOpen(true)

        return
      }

      const res = await installMcpCatalogEntry(server, envDraft)

      // Git-backed entries clone in the background — poll to completion so a
      // non-zero exit surfaces as a real failure instead of a false success.
      if (res.background && res.action) {
        for (;;) {
          const status = throwIfCancelled(await getActionStatus(res.action, 1))

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
      await respond({ server, status: 'installed' })
    } catch (error) {
      // User cancel: the declined respond is already on the wire — the
      // abandoned flow just stops, nothing to report.
      if (error === CANCELLED || error instanceof McpOAuthCancelled) {
        return
      }

      notifyError(error, copy.failed(server))
      await respond({
        detail: error instanceof Error ? error.message : String(error),
        server,
        status: 'error'
      })
    } finally {
      setWorking(false)
    }
  }, [action, copy, entry, envDraft, respond, server])

  const title =
    action === 'enable'
      ? copy.enableTitle(prettyName(server))
      : action === 'authorize'
        ? copy.authorizeTitle(prettyName(server))
        : copy.installTitle(prettyName(server))

  const actionLabel =
    action === 'enable' ? copy.enableAction : action === 'authorize' ? copy.authorizeAction : copy.installAction

  // What connecting actually means — the endpoint that will be contacted.
  // VS Code's trust dialog links the config it's about to trust; same idea.
  // Directory servers know their URL statically; catalog entries state their
  // provenance (the reviewed manifest carries the transport).
  const known = directoryEntry(server)
  const sourceLine = action === 'install' ? (known?.url ?? copy.catalogSource) : null
  const brand = brandFor(server)

  const trailingIcon = brand ? (
    <brand.Icon aria-hidden className="mt-px size-4 shrink-0" style={brandGlyphStyle(brand)} />
  ) : (
    <Codicon className={ICON_CLASS} name="plug" size="1rem" />
  )

  // ⌘/Ctrl+Enter → approve, Esc → decline/cancel. Same accelerators, same
  // guard shape as the approval bar (tool/approval.tsx). Unlike approve, Esc
  // stays live while a flow is in flight — that's the cancel path. Stands
  // down whenever a focusable control has focus (clarify's rule): a keystroke
  // meant for the composer, a popover, or the card's own credential fields
  // must never silently approve an install or throw away typed input.
  useEffect(() => {
    if (!ready) {
      return
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented) {
        return
      }

      const active = document.activeElement as HTMLElement | null

      if (
        active &&
        (active.isContentEditable || active.matches('a[href], button, input, select, textarea, [role="button"]'))
      ) {
        return
      }

      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        if (!working) {
          event.preventDefault()
          void approve()
        }
      } else if (event.key === 'Escape') {
        event.preventDefault()
        decline()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [approve, decline, ready, working])

  if (!ready) {
    return (
      <div className={cn(SHELL_CLASS, 'my-1.5 flex items-center gap-2')} data-slot="mcp-setup-inline">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
        <span className="text-(--ui-text-tertiary)">{title}</span>
      </div>
    )
  }

  return (
    <div className={cn(SHELL_CLASS, 'my-1.5 grid gap-1.5')} data-slot="mcp-setup-inline">
      <SetupLine trailing={trailingIcon}>
        <span className="font-medium leading-(--conversation-line-height)">{title}</span>
        {reason ? <p className="mt-0.5 text-(--ui-text-secondary)">{reason}</p> : null}
        {sourceLine && <p className="mt-0.5 truncate text-[0.6875rem] text-(--ui-text-tertiary)">{sourceLine}</p>}
      </SetupLine>
      {envOpen && entry && entry.required_env.length > 0 && (
        <div className="grid gap-2" data-slot="mcp-setup-env">
          <p className="text-[0.6875rem] text-(--ui-text-tertiary)">{copy.envRequired}</p>
          {entry.required_env.map(env => (
            <label className="grid gap-1" key={env.name}>
              <span className="text-[0.6875rem] text-(--ui-text-secondary)">
                {env.prompt || env.name}
                {env.required ? ' *' : ''}
              </span>
              <Input
                className="h-7 text-xs"
                onChange={event => setEnvDraft(prev => ({ ...prev, [env.name]: event.currentTarget.value }))}
                type="password"
                value={envDraft[env.name] ?? ''}
              />
            </label>
          ))}
        </div>
      )}
      {/* Same strip as the tool approval bar (tool/approval.tsx): a bordered
          primary-tinted action plus a quiet ghost decline, with the matching
          keyboard hints. One consent vocabulary across the transcript. */}
      <div className="flex items-center gap-2.5">
        <div className="inline-flex h-6 items-stretch overflow-hidden rounded-md border border-primary/25 bg-primary/10 text-primary">
          <Button
            className="h-full gap-1 rounded-none px-2 text-xs font-medium text-primary hover:bg-primary/15 hover:text-primary"
            disabled={working}
            onClick={() => void approve()}
            size="xs"
            variant="ghost"
          >
            {working ? <Loader2 className="size-3 animate-spin" /> : actionLabel}
            {!working && <span className="text-[0.625rem] text-primary/60">{isMac ? '⌘⏎' : 'Ctrl⏎'}</span>}
          </Button>
        </div>
        {/* Never disabled: while a flow is in flight this is the cancel —
            a stuck OAuth tab or hung install must always have a way out. */}
        <Button
          className="h-6 gap-1.5 rounded-md px-1.5 text-xs font-normal text-(--ui-text-tertiary) hover:text-foreground"
          onClick={decline}
          size="xs"
          variant="ghost"
        >
          {working ? t.common.cancel : copy.decline}
          <span className="text-[0.625rem] opacity-55">Esc</span>
        </Button>
      </div>
    </div>
  )
}
