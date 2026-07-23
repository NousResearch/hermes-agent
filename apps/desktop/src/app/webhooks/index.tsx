import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import {
  createWebhook,
  deleteWebhook,
  enableWebhooks,
  getWebhooks,
  setWebhookEnabled,
  type WebhookRoute,
  type WebhooksResponse
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, Copy, Globe, Plus, RefreshCw, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $profileScope } from '@/store/profile'
import { runGatewayRestart } from '@/store/system-actions'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

const DELIVER_OPTIONS: readonly string[] = ['log', 'telegram', 'discord', 'slack', 'email', 'github_comment']

interface CreatedWebhook {
  secret: string
  url: string
}

function CopyButton({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false)

  const onCopy = useCallback(() => {
    navigator.clipboard
      .writeText(value)
      .then(() => {
        setCopied(true)
        window.setTimeout(() => setCopied(false), 1500)
      })
      .catch(() => {})
  }, [value])

  return (
    <Button aria-label={label} onClick={onCopy} size="icon-sm" title={label} variant="ghost">
      {copied ? <Check /> : <Copy />}
    </Button>
  )
}

interface WebhooksViewProps {
  onClose: () => void
}

export function WebhooksView({ onClose }: WebhooksViewProps) {
  const { t } = useI18n()
  const w = t.webhooks
  // Re-load when the active profile changes so REST routes to the right backend.
  const profileScope = useStore($profileScope)

  const [data, setData] = useState<WebhooksResponse | null>(null)
  const [query, setQuery] = useState('')
  const [enabling, setEnabling] = useState(false)
  const [restartNeeded, setRestartNeeded] = useState(false)
  const [restartError, setRestartError] = useState<null | string>(null)
  const [restarting, setRestarting] = useState(false)
  const [togglingName, setTogglingName] = useState<null | string>(null)

  const [createOpen, setCreateOpen] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [events, setEvents] = useState('')
  const [deliver, setDeliver] = useState('log')
  const [deliverOnly, setDeliverOnly] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [skills, setSkills] = useState('')
  const [creating, setCreating] = useState(false)
  const [created, setCreated] = useState<CreatedWebhook | null>(null)

  const [pendingDelete, setPendingDelete] = useState<null | string>(null)
  const [deleting, setDeleting] = useState(false)

  const enabled = data?.enabled ?? false
  const subscriptions = useMemo(() => data?.subscriptions ?? [], [data])

  const loadWebhooks = useCallback(
    async (silent = false) => {
      try {
        setData(await getWebhooks())
      } catch (err) {
        if (!silent) {
          notifyError(err, w.loadFailed)
        }
      }
    },
    [w.loadFailed]
  )

  useRefreshHotkey(() => void loadWebhooks())

  useEffect(() => {
    void loadWebhooks()
    // profileScope drives a re-home; a fresh load rebinds to the active backend.
     
  }, [loadWebhooks, profileScope])

  const restartGatewayNow = useCallback(async () => {
    setRestarting(true)

    try {
      await runGatewayRestart()
      setRestartNeeded(false)
      setRestartError(null)
      // Give the receiver a moment to bind before re-reading state.
      window.setTimeout(() => void loadWebhooks(true), 4000)
    } catch (err) {
      setRestartNeeded(true)
      setRestartError(String(err))
      notifyError(err, w.restartFailed(''))
    } finally {
      setRestarting(false)
    }
  }, [loadWebhooks, w])

  const handleEnable = useCallback(async () => {
    setEnabling(true)
    setRestartNeeded(false)
    setRestartError(null)

    try {
      const result = await enableWebhooks()
      await loadWebhooks(true)

      if (result.restart_started) {
        notify({ kind: 'success', message: w.enabledRestarting })
        window.setTimeout(() => void loadWebhooks(true), 4000)
      } else {
        const detail = result.restart_error ? `: ${result.restart_error}` : '.'
        setRestartNeeded(true)
        setRestartError(w.restartFailed(detail))
        notify({ kind: 'error', message: w.restartFailed(detail) })
      }
    } catch (err) {
      notifyError(err, w.restartFailed(''))
    } finally {
      setEnabling(false)
    }
  }, [loadWebhooks, w])

  const resetForm = useCallback(() => {
    setName('')
    setDescription('')
    setEvents('')
    setDeliver('log')
    setDeliverOnly(false)
    setPrompt('')
    setSkills('')
  }, [])

  const closeCreate = useCallback(() => {
    if (creating) {
      return
    }

    setCreateOpen(false)
    setCreated(null)
  }, [creating])

  const handleCreate = useCallback(async () => {
    if (!name.trim()) {
      notify({ kind: 'error', message: w.nameRequired })

      return
    }

    setCreating(true)

    try {
      const eventsList = events
        .split(',')
        .map(e => e.trim())
        .filter(Boolean)

      const skillsList = skills
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)

      const res = await createWebhook({
        deliver,
        deliver_only: deliverOnly,
        description: description.trim() || undefined,
        events: eventsList.length ? eventsList : undefined,
        name: name.trim(),
        prompt: prompt.trim() || undefined,
        skills: skillsList.length ? skillsList : undefined
      })

      notify({ kind: 'success', message: w.created })
      setCreated({ secret: res.secret, url: res.url })
      resetForm()
      void loadWebhooks(true)
    } catch (err) {
      notifyError(err, w.createFailed(''))
    } finally {
      setCreating(false)
    }
  }, [deliver, deliverOnly, description, events, loadWebhooks, name, prompt, resetForm, skills, w])

  const handleToggle = useCallback(
    async (subName: string, nextEnabled: boolean) => {
      setTogglingName(subName)
      // Optimistic paint; an authoritative reload gets the last word.
      setData(current =>
        current
          ? {
              ...current,
              subscriptions: current.subscriptions.map(s =>
                s.name === subName ? { ...s, enabled: nextEnabled } : s
              )
            }
          : current
      )

      try {
        await setWebhookEnabled(subName, nextEnabled)
        notify({ kind: 'success', message: nextEnabled ? w.enabled(subName) : w.disabled(subName) })
        void loadWebhooks(true)
      } catch (err) {
        await loadWebhooks(true)
        notifyError(err, w.toggleFailed(subName))
      } finally {
        setTogglingName(null)
      }
    },
    [loadWebhooks, w]
  )

  const handleDelete = useCallback(async () => {
    if (!pendingDelete) {
      return
    }

    setDeleting(true)

    try {
      await deleteWebhook(pendingDelete)
      notify({ kind: 'success', message: `${pendingDelete}` })
      setPendingDelete(null)
      void loadWebhooks(true)
    } catch (err) {
      notifyError(err, w.deleteFailed(pendingDelete))
    } finally {
      setDeleting(false)
    }
  }, [loadWebhooks, pendingDelete, w])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()

    if (!q) {
      return subscriptions
    }

    return subscriptions.filter(s =>
      [s.name, s.description, s.deliver, ...s.events].filter(Boolean).some(v => v.toLowerCase().includes(q))
    )
  }, [query, subscriptions])

  const headerActions = (
    <div className="flex items-center gap-1.5">
      {subscriptions.length > 0 && (
        <SearchField
          aria-label={w.search}
          containerClassName="w-44"
          onChange={setQuery}
          placeholder={w.search}
          value={query}
        />
      )}
      <Button aria-label={t.commandCenter.refresh} onClick={() => void loadWebhooks()} size="icon-sm" variant="ghost">
        <RefreshCw />
      </Button>
      <Button
        disabled={!enabled || enabling}
        onClick={() => {
          setCreated(null)
          setCreateOpen(true)
        }}
        size="sm"
      >
        <Plus />
        {w.newSubscription}
      </Button>
    </div>
  )

  return (
    <Panel onClose={onClose}>
      <PanelHeader actions={headerActions} subtitle={w.hint} title={w.subscriptions(subscriptions.length)} />
      {!data ? (
        <PageLoader label={w.loading} />
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto overscroll-contain pb-2">
          {!enabled && (
            <div className="flex flex-col gap-3 rounded-md border border-amber-500/40 bg-amber-500/5 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-3">
                <Globe className="mt-0.5 size-5 shrink-0 text-amber-500" />
                <div className="flex flex-col gap-1">
                  <span className="text-sm font-medium">{w.disabledTitle}</span>
                  <span className="text-xs text-(--ui-text-tertiary)">{w.disabledBody}</span>
                </div>
              </div>
              <Button className="shrink-0" disabled={enabling} onClick={() => void handleEnable()} size="sm">
                <Globe />
                {enabling ? w.enabling : w.enable}
              </Button>
            </div>
          )}

          {restartNeeded && (
            <div className="flex flex-col gap-3 rounded-md border border-amber-500/40 bg-amber-500/5 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-2 text-sm">
                <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-500" />
                <span>{restartError ?? w.restartNeeded}</span>
              </div>
              <Button
                className="shrink-0"
                disabled={restarting}
                onClick={() => void restartGatewayNow()}
                size="sm"
                variant="secondary"
              >
                <RefreshCw />
                {restarting ? w.restartingGateway : w.restartGateway}
              </Button>
            </div>
          )}

          {visible.length === 0 ? (
            <PanelEmpty description={w.empty} icon="globe" title={w.subscriptions(0)} />
          ) : (
            <ul className="flex flex-col gap-2">
              {visible.map(sub => (
                <WebhookRow
                  key={sub.name}
                  onDelete={() => setPendingDelete(sub.name)}
                  onToggle={() => void handleToggle(sub.name, !sub.enabled)}
                  sub={sub}
                  toggling={togglingName === sub.name}
                />
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Create subscription dialog */}
      <Dialog onOpenChange={open => !open && closeCreate()} open={createOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{created ? w.createdTitle : w.newSubscription}</DialogTitle>
            {created && <DialogDescription>{w.createdSecretHint}</DialogDescription>}
          </DialogHeader>

          {created ? (
            <div className="grid gap-4">
              <div className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">{w.webhookUrl}</span>
                <div className="flex items-center gap-2 rounded-md border border-border bg-background/40 px-3 py-2">
                  <span className="min-w-0 flex-1 truncate font-mono text-xs">{created.url}</span>
                  <CopyButton label={w.copy} value={created.url} />
                </div>
              </div>
              <div className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">{w.secretOnce}</span>
                <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2">
                  <span className="min-w-0 flex-1 truncate font-mono text-xs">{created.secret}</span>
                  <CopyButton label={w.copy} value={created.secret} />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={closeCreate} size="sm">
                  {w.done}
                </Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="grid gap-4">
              <Field htmlFor="webhook-name" label={w.fieldName}>
                <Input
                  autoFocus
                  id="webhook-name"
                  onChange={e => setName(e.target.value)}
                  placeholder={w.fieldNamePlaceholder}
                  value={name}
                />
              </Field>
              <Field htmlFor="webhook-description" label={w.fieldDescription}>
                <Input
                  id="webhook-description"
                  onChange={e => setDescription(e.target.value)}
                  placeholder={w.fieldDescriptionPlaceholder}
                  value={description}
                />
              </Field>
              <Field htmlFor="webhook-events" label={w.fieldEvents}>
                <Input
                  id="webhook-events"
                  onChange={e => setEvents(e.target.value)}
                  placeholder={w.fieldEventsPlaceholder}
                  value={events}
                />
              </Field>
              <Field htmlFor="webhook-skills" label={w.fieldSkills}>
                <Input
                  id="webhook-skills"
                  onChange={e => setSkills(e.target.value)}
                  placeholder={w.fieldSkillsPlaceholder}
                  value={skills}
                />
              </Field>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field htmlFor="webhook-deliver" label={w.fieldDeliver}>
                  <Select onValueChange={setDeliver} value={deliver}>
                    <SelectTrigger className="h-9 rounded-md" id="webhook-deliver">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {DELIVER_OPTIONS.map(opt => (
                        <SelectItem key={opt} value={opt}>
                          {w.deliverOptions[opt] ?? opt}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                <div className="grid gap-1.5">
                  <span className="text-xs font-medium text-muted-foreground">{w.fieldDeliverOnly}</span>
                  <label className="flex h-9 items-center gap-2 text-sm text-muted-foreground">
                    <input
                      checked={deliverOnly}
                      onChange={e => setDeliverOnly(e.target.checked)}
                      type="checkbox"
                    />
                    {w.fieldDeliverOnlyHint}
                  </label>
                </div>
              </div>
              <Field htmlFor="webhook-prompt" label={w.fieldPrompt}>
                <Textarea
                  className="min-h-[80px]"
                  id="webhook-prompt"
                  onChange={e => setPrompt(e.target.value)}
                  placeholder={w.fieldPromptPlaceholder}
                  value={prompt}
                />
              </Field>
              <DialogFooter>
                <Button disabled={creating} onClick={() => void handleCreate()} size="sm">
                  {creating ? w.creating : w.create}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete confirm dialog */}
      <Dialog onOpenChange={open => !open && !deleting && setPendingDelete(null)} open={pendingDelete !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{w.deleteTitle}</DialogTitle>
            <DialogDescription>
              {pendingDelete ? w.deleteDescription(pendingDelete) : w.deleteGeneric}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button disabled={deleting} onClick={() => setPendingDelete(null)} size="sm" variant="secondary">
              {t.common.cancel}
            </Button>
            <Button disabled={deleting} onClick={() => void handleDelete()} size="sm" variant="destructive">
              {w.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Panel>
  )
}

function WebhookRow({
  onDelete,
  onToggle,
  sub,
  toggling
}: {
  onDelete: () => void
  onToggle: () => void
  sub: WebhookRoute
  toggling: boolean
}) {
  const { t } = useI18n()
  const w = t.webhooks

  return (
    <li className="flex items-start gap-4 rounded-md border border-border p-4">
      <div className={cn('min-w-0 flex-1', !sub.enabled && 'opacity-60')}>
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <span className="truncate text-sm font-medium">{sub.name}</span>
          <Badge variant="outline">{sub.deliver}</Badge>
          {sub.deliver_only && <Badge variant="muted">{w.deliverOnly}</Badge>}
          {!sub.enabled && <Badge variant="muted">{t.messaging.states.disabled}</Badge>}
        </div>

        {sub.description && <p className="mb-2 text-xs text-muted-foreground">{sub.description}</p>}

        <div className="mb-2 flex flex-wrap items-center gap-1">
          {sub.events.length === 0 ? (
            <Badge variant="muted">{w.all}</Badge>
          ) : (
            sub.events.map(evt => (
              <Badge key={evt} variant="muted">
                {evt}
              </Badge>
            ))
          )}
        </div>

        {sub.skills.length > 0 && (
          <div className="mb-2 flex flex-wrap items-center gap-1">
            {sub.skills.map(skill => (
              <Badge key={skill} variant="outline">
                {skill}
              </Badge>
            ))}
          </div>
        )}

        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <span className="min-w-0 truncate font-mono">{sub.url}</span>
          <CopyButton label={w.copy} value={sub.url} />
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <Button disabled={toggling} onClick={onToggle} size="sm" variant="ghost">
          {sub.enabled ? w.disableRow : w.enableRow}
        </Button>
        <Button aria-label={w.delete} onClick={onDelete} size="icon-sm" title={w.delete} variant="ghost">
          <Trash2 />
        </Button>
      </div>
    </li>
  )
}

function Field({
  children,
  htmlFor,
  label
}: {
  children: React.ReactNode
  htmlFor: string
  label: string
}) {
  return (
    <div className="grid gap-1.5">
      <label className="text-xs font-medium text-muted-foreground" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
    </div>
  )
}
