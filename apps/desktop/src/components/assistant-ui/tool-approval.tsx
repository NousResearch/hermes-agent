'use client'

import { useStore } from '@nanostores/react'
import { type FC, useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { triggerHaptic } from '@/lib/haptics'
import { ChevronDown, Loader2 } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $approvalRequest, type ApprovalRequest, clearApprovalRequest } from '@/store/prompts'

import type { ToolPart } from './tool-fallback-model'

// Inline approval control. Rendered as a compact button strip
// under the pending tool row that raised the approval (the row already shows
// the command, so the strip deliberately doesn't repeat it) instead of as a
// modal overlay.
//
// Binding is POSITIONAL, not command-matched: the desktop `tool.start` payload
// carries no structured args (only tool_id/name/context — see
// tui_gateway/server.py::_on_tool_start), so we cannot join the approval to the
// row by command string. But `approval.request` only ever fires from the
// `terminal` / `execute_code` guards and the agent thread blocks on exactly one
// approval at a time, so the single pending row of those tools IS the row that
// raised it. The command/description text comes from `$approvalRequest` (the
// event payload), which is the only place that data reliably exists.
export const APPROVAL_TOOLS = new Set(['terminal', 'execute_code'])

// Canonical gateway choices (ui-tui/src/components/prompts.tsx).
type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'

export const PendingToolApproval: FC<{ part: ToolPart }> = ({ part }) => {
  const request = useStore($approvalRequest)

  if (!request || !APPROVAL_TOOLS.has(part.toolName)) {
    return null
  }

  return <ApprovalBar request={request} />
}

const isMac = typeof navigator !== 'undefined' && /Mac|iP(hone|ad|od)/.test(navigator.platform)

const ApprovalBar: FC<{ request: ApprovalRequest }> = ({ request }) => {
  const gateway = useStore($gateway)
  const [submitting, setSubmitting] = useState<ApprovalChoice | null>(null)
  // "Always allow" persists the pattern to ~/.hermes/config.yaml permanently, so
  // it goes through a confirm step rather than firing straight from the menu.
  const [confirmAlways, setConfirmAlways] = useState(false)
  const busy = submitting !== null

  const respond = useCallback(
    async (choice: ApprovalChoice) => {
      // Another bar (or the keyboard path) may have already resolved this
      // approval; the atom is the single source of truth, so bail if it's gone.
      if (busy || !$approvalRequest.get()) {
        return
      }

      if (!gateway) {
        notifyError(new Error('Hermes gateway is not connected'), 'Could not send approval response')

        return
      }

      setSubmitting(choice)

      try {
        await gateway.request<{ resolved?: boolean }>('approval.respond', {
          choice,
          session_id: request.sessionId ?? undefined
        })
        triggerHaptic(choice === 'deny' ? 'cancel' : 'submit')
        clearApprovalRequest()
      } catch (error) {
        notifyError(error, 'Could not send approval response')
        setSubmitting(null)
      }
    },
    [busy, gateway, request.sessionId]
  )

  // ⌘/Ctrl+Enter → Run, Esc → Reject.
  // While the confirm dialog is open it owns the keyboard (Esc closes it), so
  // the strip-level shortcuts stand down to avoid denying the whole approval.
  useEffect(() => {
    if (confirmAlways) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        void respond('once')
      } else if (event.key === 'Escape') {
        event.preventDefault()
        void respond('deny')
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [confirmAlways, respond])

  return (
    <div className="mt-1 flex items-center gap-2.5 ps-5" data-slot="tool-approval-inline">
      <div className="inline-flex h-6 items-stretch overflow-hidden rounded-md border border-primary/25 bg-primary/10 text-primary">
        <Button
          className="h-full gap-1 rounded-none px-2 text-xs font-medium text-primary hover:bg-primary/15 hover:text-primary"
          disabled={busy}
          onClick={() => void respond('once')}
          size="xs"
          variant="ghost"
        >
          {submitting === 'once' ? <Loader2 className="size-3 animate-spin" /> : 'Run'}
          {submitting !== 'once' && <span className="text-[0.625rem] text-primary/60">{isMac ? '⌘⏎' : 'Ctrl⏎'}</span>}
        </Button>
        <span aria-hidden className="w-px self-stretch bg-primary/20" />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              aria-label="More approval options"
              className="h-full w-5 rounded-none px-0 text-primary hover:bg-primary/15 hover:text-primary"
              disabled={busy}
              size="xs"
              variant="ghost"
            >
              <ChevronDown className="size-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-44">
            <DropdownMenuItem onSelect={() => void respond('session')}>Allow this session</DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => {
                // Defer one tick so the menu fully unmounts before the dialog
                // mounts — otherwise Radix's focus-return races the dialog and
                // dismisses it via onInteractOutside.
                setTimeout(() => setConfirmAlways(true), 0)
              }}
            >
              Always allow…
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => void respond('deny')} variant="destructive">
              Reject
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <Button
        className="h-6 gap-1.5 rounded-md px-1.5 text-xs font-normal text-(--ui-text-tertiary) hover:text-foreground"
        disabled={busy}
        onClick={() => void respond('deny')}
        size="xs"
        variant="ghost"
      >
        {submitting === 'deny' ? <Loader2 className="size-3 animate-spin" /> : 'Reject'}
        {submitting !== 'deny' && <span className="text-[0.625rem] opacity-55">Esc</span>}
      </Button>

      <Dialog onOpenChange={setConfirmAlways} open={confirmAlways}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Always allow this command?</DialogTitle>
            <DialogDescription>
              This adds the “{request.description}” pattern to your permanent allowlist (
              <code className="font-mono text-xs">~/.hermes/config.yaml</code>). Hermes won’t ask again for commands
              like this — in this session or any future one.
            </DialogDescription>
          </DialogHeader>

          {request.command.trim() && (
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-words rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-2.5 py-1.5 font-mono text-xs leading-snug text-foreground">
              {request.command.trim()}
            </pre>
          )}

          <DialogFooter>
            <Button onClick={() => setConfirmAlways(false)} size="sm" variant="ghost">
              Cancel
            </Button>
            <Button
              onClick={() => {
                setConfirmAlways(false)
                void respond('always')
              }}
              size="sm"
              variant="destructive"
            >
              Always allow
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
