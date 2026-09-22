import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'

import {
  type CommandOperation,
  type CommandPlan,
  commandPlanIntegrityError,
  type CommandPlanningSnapshot
} from '../command-broker'
import type { EntityIdentity } from '../model'

export interface CommandConfirmationProps {
  disabledReason?: string
  getLatestSnapshot: () => CommandPlanningSnapshot
  onCancel: () => void
  onConfirm: (plan: CommandPlan) => void
  open: boolean
  plan: CommandPlan
  submitting?: boolean
}

const OPERATION_LABELS: Readonly<Record<CommandOperation, string>> = {
  'change-task-state': 'Change task state',
  'dispatch-task': 'Dispatch task',
  'inspect-evidence': 'Inspect evidence',
  'interrupt-session': 'Interrupt session',
  'interrupt-subagent': 'Interrupt subagent',
  'open-session': 'Open session',
  'reassign-task': 'Reassign task',
  'reclaim-task': 'Reclaim task',
  'retry-task': 'Retry task',
  'send-guidance': 'Send guidance',
  'terminate-run': 'Terminate run'
}

export function commandPlanConfirmationError(
  plan: CommandPlan,
  latestSnapshot: CommandPlanningSnapshot
): string | undefined {
  if (!plan.confirmation) {
    return 'This command does not require disruptive confirmation.'
  }

  const integrityError = commandPlanIntegrityError(plan, latestSnapshot)

  return integrityError ? `Confirmation unavailable: ${integrityError}` : undefined
}

function latestValidationError(
  plan: CommandPlan,
  getLatestSnapshot: () => CommandPlanningSnapshot
): string | undefined {
  try {
    return commandPlanConfirmationError(plan, getLatestSnapshot())
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    return `Confirmation unavailable: latest target revalidation failed: ${message}`
  }
}

function IdentityRows({ identity }: { identity: EntityIdentity }) {
  const rows: readonly [string, string | undefined][] = [
    ['Connection', identity.connectionId],
    ['Profile', identity.profile],
    ['Session', identity.kind === 'session' || identity.kind === 'subagent' ? identity.sessionId : undefined],
    ['Subagent', identity.kind === 'subagent' ? identity.subagentId : undefined],
    ['Board', identity.kind === 'kanban' ? identity.board : undefined],
    ['Task', identity.kind === 'kanban' ? identity.taskId : undefined],
    ['Run', identity.kind === 'kanban' ? identity.runId : undefined],
    ['Worker', identity.kind === 'kanban' ? identity.workerId : undefined]
  ]

  return (
    <dl>
      {rows.flatMap(([name, value]) =>
        value ? [<dt key={`${name}-term`}>{name}</dt>, <dd key={`${name}-value`}>{value}</dd>] : []
      )}
    </dl>
  )
}

export function CommandConfirmation({
  disabledReason,
  getLatestSnapshot,
  onCancel,
  onConfirm,
  open,
  plan,
  submitting = false
}: CommandConfirmationProps) {
  const cancelRef = useRef<HTMLButtonElement>(null)
  const [clickError, setClickError] = useState<{ digest: string; message: string } | undefined>()
  const operationLabel = OPERATION_LABELS[plan.operation]
  const validationError = latestValidationError(plan, getLatestSnapshot)
  const safeOperationLabel = validationError ? 'command' : operationLabel
  const externalReason = disabledReason?.trim() || undefined

  const unavailableReason =
    validationError ?? (clickError?.digest === plan.digest ? clickError.message : undefined) ?? externalReason

  const disabled = submitting || Boolean(unavailableReason)

  return (
    <Dialog onOpenChange={next => !next && !submitting && onCancel()} open={open}>
      <DialogContent
        className="max-w-lg"
        onOpenAutoFocus={event => {
          event.preventDefault()
          cancelRef.current?.focus()
        }}
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle>Confirm Lunar City command</DialogTitle>
          <DialogDescription>
            Verify the exact owner and identity. A successful request is not completion until authoritative readback.
          </DialogDescription>
        </DialogHeader>

        {validationError ? null : (
          <>
            <section aria-label="Exact target identity">
              <h3>Exact target identity</h3>
              <IdentityRows identity={plan.identity} />
              <dl>
                {plan.context.canonicalProjectId ? (
                  <>
                    <dt>Canonical project</dt>
                    <dd>{plan.context.canonicalProjectId}</dd>
                  </>
                ) : null}
                {plan.context.repositoryId ? (
                  <>
                    <dt>Repository</dt>
                    <dd>{plan.context.repositoryId}</dd>
                  </>
                ) : null}
                <dt>Current state</dt>
                <dd>{plan.context.currentState}</dd>
              </dl>
            </section>

            <section aria-label="Requested operation">
              <h3>Requested operation</h3>
              <p>{operationLabel}</p>
            </section>

            <section aria-label="Expected consequence">
              <h3>Expected consequence</h3>
              <p>{plan.consequence}</p>
            </section>
          </>
        )}

        {unavailableReason ? (
          <p aria-live="polite" role="status">
            {unavailableReason}
          </p>
        ) : null}

        <DialogFooter>
          <Button disabled={submitting} onClick={onCancel} ref={cancelRef} type="button" variant="ghost">
            Cancel command
          </Button>
          <Button
            aria-label={`Confirm ${safeOperationLabel}`}
            disabled={disabled}
            onClick={() => {
              if (disabled) {
                return
              }

              const latestError = latestValidationError(plan, getLatestSnapshot)

              if (latestError) {
                setClickError({ digest: plan.digest, message: latestError })

                return
              }

              onConfirm(plan)
            }}
            type="button"
            variant="destructive"
          >
            {submitting ? 'Sending once…' : `Confirm ${safeOperationLabel}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
