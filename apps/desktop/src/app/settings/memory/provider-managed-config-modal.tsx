import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogClose, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { getMemoryProviderConfig, runMemoryProviderAction } from '@/hermes'
import { ExternalLink } from '@/lib/external-link'
import { AlertCircle, CheckCircle2, Loader2, Play, RefreshCw, Save, SlidersHorizontal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify } from '@/store/notifications'
import type { MemoryProviderConfig, MemoryProviderConfigAction, MemoryProviderField } from '@/types/hermes'

import { ListRow } from '../primitives'

import { FieldControl, FieldTitle } from './field-control'
import { conditionsMatch } from './provider-config-conditions'

type Feedback = { message: string; tone: 'error' | 'success' }

function seedValues(config: MemoryProviderConfig): Record<string, string> {
  return Object.fromEntries(config.fields.map(field => [field.key, field.kind === 'secret' ? '' : field.value]))
}

function errorMessage(error: unknown): string {
  if (!(error instanceof Error)) {
    return typeof error === 'string' ? error : 'The operation failed.'
  }

  const jsonStart = error.message.indexOf('{')

  if (jsonStart >= 0) {
    try {
      const parsed = JSON.parse(error.message.slice(jsonStart)) as { detail?: unknown }

      if (typeof parsed.detail === 'string') {
        return parsed.detail
      }
    } catch {
      // Preserve the bridge error when the response body is not JSON.
    }
  }

  return error.message.replace(/^\d{3}:\s*/, '')
}

function isConflict(error: unknown): boolean {
  return (
    (typeof error === 'object' && error !== null && 'statusCode' in error && error.statusCode === 409) ||
    (error instanceof Error && /^409:\s*/.test(error.message))
  )
}

function FieldLink({ field }: { field: MemoryProviderField }) {
  if (!field.help_url) {
    return null
  }

  return (
    <ExternalLink
      className="text-xs font-medium text-primary decoration-primary/30 hover:decoration-primary/70"
      href={field.help_url}
      showExternalIcon={false}
    >
      {field.help_label || 'Learn more'}
    </ExternalLink>
  )
}

function FeedbackNotice({ feedback }: { feedback: Feedback }) {
  const Icon = feedback.tone === 'error' ? AlertCircle : CheckCircle2

  return (
    <div
      className={cn(
        'flex items-start gap-2 rounded-md border px-3 py-2 text-xs leading-5',
        feedback.tone === 'error'
          ? 'border-destructive/40 bg-destructive/10 text-destructive'
          : 'border-primary/30 bg-primary/8 text-foreground'
      )}
      role={feedback.tone === 'error' ? 'alert' : 'status'}
    >
      <Icon className="mt-0.5 size-3.5 shrink-0" />
      <span>{feedback.message}</span>
    </div>
  )
}

export function ProviderManagedConfigModal({
  config,
  profile = null,
  provider,
  open,
  onOpenChange,
  onSaved
}: {
  config: MemoryProviderConfig
  profile?: null | string
  provider: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved: () => Promise<void> | void
}) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [options, setOptions] = useState<Record<string, MemoryProviderField['options']>>({})
  const [saving, setSaving] = useState(false)
  const [runningAction, setRunningAction] = useState('')
  const [feedback, setFeedback] = useState<Feedback | null>(null)
  const [invalidFields, setInvalidFields] = useState<Set<string>>(new Set())
  const [confirmOverwrite, setConfirmOverwrite] = useState(false)
  const [initialized, setInitialized] = useState(false)
  const actionInFlight = useRef(false)
  const saveInFlight = useRef(false)

  useEffect(() => {
    if (!open) {
      setInitialized(false)

      return
    }

    if (initialized) {
      return
    }

    setValues(seedValues(config))
    setOptions(Object.fromEntries(config.fields.map(field => [field.key, field.options])))
    setFeedback(null)
    setInvalidFields(new Set())
    setConfirmOverwrite(false)
    setInitialized(true)
  }, [config, initialized, open])

  const fields = useMemo(
    () =>
      config.fields
        .filter(field => conditionsMatch(field.visible_when, values))
        .map(field => ({ ...field, options: options[field.key] ?? field.options })),
    [config.fields, options, values]
  )

  const busy = saving || Boolean(runningAction)

  async function reloadDynamicOptions() {
    if (actionInFlight.current || saveInFlight.current) {
      return
    }

    actionInFlight.current = true
    setRunningAction('refresh-options')
    setFeedback(null)

    try {
      const next = await getMemoryProviderConfig(provider, profile)

      setOptions(Object.fromEntries(next.fields.map(field => [field.key, field.options])))
      setValues(current => {
        const updated = { ...current }

        for (const field of next.fields.filter(candidate => candidate.dynamic_options)) {
          if (!field.options.some(option => option.value === updated[field.key])) {
            updated[field.key] = field.value || field.options[0]?.value || ''
          }
        }

        return updated
      })
    } catch (error) {
      setFeedback({ message: errorMessage(error), tone: 'error' })
    } finally {
      actionInFlight.current = false
      setRunningAction('')
    }
  }

  async function runAction(action: MemoryProviderConfigAction) {
    if (actionInFlight.current || saveInFlight.current) {
      return
    }

    actionInFlight.current = true
    setRunningAction(action.name)
    setFeedback(null)
    const payload = Object.fromEntries(action.payload_fields.map(key => [key, values[key] ?? '']))

    try {
      const result = await runMemoryProviderAction<{ message?: string; ok?: boolean }>(
        provider,
        action.name,
        payload,
        profile
      )

      if (result.ok === false) {
        setFeedback({ message: result.message || `${action.label} failed.`, tone: 'error' })

        return
      }

      setFeedback({ message: result.message || `${action.label} completed.`, tone: 'success' })

      if (action.refresh_after) {
        await onSaved()
      }
    } catch (error) {
      setFeedback({ message: errorMessage(error), tone: 'error' })
    } finally {
      actionInFlight.current = false
      setRunningAction('')
    }
  }

  async function save(overwrite = false) {
    if (saveInFlight.current || actionInFlight.current || !config.submit_action) {
      return
    }

    const missing = new Set(
      fields.filter(field => field.required && !values[field.key]?.trim()).map(field => field.key)
    )

    setInvalidFields(missing)

    if (missing.size > 0) {
      setFeedback({ message: 'Complete the required fields before saving.', tone: 'error' })

      return
    }

    saveInFlight.current = true
    setSaving(true)
    setFeedback(null)

    try {
      await runMemoryProviderAction(provider, config.submit_action, { overwrite, values }, profile)

      notify({
        kind: 'success',
        title: `${config.label} setup saved`,
        message: 'This setup is active now. New messages in existing and new chats will use it.'
      })

      await onSaved()
      setConfirmOverwrite(false)
      onOpenChange(false)
    } catch (error) {
      if (isConflict(error) && !overwrite) {
        setConfirmOverwrite(true)
      } else {
        setFeedback({ message: errorMessage(error), tone: 'error' })

        if (overwrite) {
          throw new Error(errorMessage(error))
        }
      }
    } finally {
      saveInFlight.current = false
      setSaving(false)
    }
  }

  return (
    <>
      <Dialog onOpenChange={value => !busy && onOpenChange(value)} open={open}>
        <DialogContent bodyClassName="grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden" className="max-w-2xl">
          <DialogHeader>
            <DialogTitle icon={SlidersHorizontal}>Configure {config.label}</DialogTitle>
          </DialogHeader>

          <div
            className="dt-portal-scrollbar min-h-0 min-w-0 overflow-y-auto overscroll-contain"
            data-slot="provider-managed-config-scroll"
          >
            <div className="min-w-0">
              {fields.map(field => {
                const fieldActions = (config.actions ?? []).filter(
                  action => action.after_field === field.key && conditionsMatch(action.visible_when, values)
                )

                const selectedOption =
                  field.kind === 'select' ? field.options.find(option => option.value === values[field.key]) : undefined

                const title = (
                  <div className="flex min-w-0 items-center justify-between gap-3">
                    <label htmlFor={field.key}>
                      <FieldTitle field={field} />
                    </label>
                    <FieldLink field={field} />
                  </div>
                )

                const control = (
                  <div className="grid min-w-0 gap-2">
                    <FieldControl
                      controlId={field.key}
                      field={field}
                      invalid={invalidFields.has(field.key)}
                      onChange={value => {
                        setValues(current => ({ ...current, [field.key]: value }))
                        setFeedback(null)

                        setInvalidFields(current => {
                          const next = new Set(current)
                          next.delete(field.key)

                          return next
                        })
                      }}
                      value={values[field.key] ?? ''}
                    />
                    {selectedOption?.description ? (
                      <div
                        className="truncate font-mono text-[0.68rem] text-muted-foreground/60"
                        title={selectedOption.description}
                      >
                        {selectedOption.description}
                      </div>
                    ) : null}
                    {fieldActions.map(action => (
                      <Button
                        className="justify-self-start"
                        disabled={busy}
                        key={action.name}
                        onClick={() => void runAction(action)}
                        size="sm"
                        type="button"
                        variant="secondary"
                      >
                        {runningAction === action.name ? (
                          <Loader2 className="size-3.5 animate-spin" />
                        ) : (
                          <Play className="size-3.5" />
                        )}
                        {action.label}
                      </Button>
                    ))}
                  </div>
                )

                return (
                  <div className="border-b border-border/40 last:border-b-0" key={field.key}>
                    {field.kind === 'segmented' ? (
                      <ListRow
                        below={<div className="mt-2">{control}</div>}
                        description={field.description}
                        title={title}
                        wide
                      />
                    ) : (
                      <ListRow
                        action={control}
                        description={field.description}
                        title={
                          <div className="flex min-w-0 items-center justify-between gap-3">
                            {title}
                            {field.dynamic_options ? (
                              <Button
                                disabled={busy}
                                onClick={() => void reloadDynamicOptions()}
                                size="sm"
                                type="button"
                                variant="ghost"
                              >
                                {runningAction === 'refresh-options' ? (
                                  <Loader2 className="size-3.5 animate-spin" />
                                ) : (
                                  <RefreshCw className="size-3.5" />
                                )}
                                Refresh
                              </Button>
                            ) : null}
                          </div>
                        }
                      />
                    )}
                  </div>
                )
              })}
            </div>

            {feedback ? <FeedbackNotice feedback={feedback} /> : null}
          </div>

          <DialogFooter>
            <DialogClose asChild>
              <Button disabled={busy} size="sm" type="button" variant="ghost">
                Cancel
              </Button>
            </DialogClose>
            <Button disabled={busy} onClick={() => void save()} size="sm">
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
              {config.submit_label || 'Save setup'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        busyLabel="Replacing..."
        confirmLabel="Replace profile"
        description="A saved profile with this name has different settings. Replace it with the values in this form?"
        doneLabel="Replaced"
        onClose={() => setConfirmOverwrite(false)}
        onConfirm={() => save(true)}
        open={confirmOverwrite}
        title={`${config.label} profile already exists`}
      />
    </>
  )
}
