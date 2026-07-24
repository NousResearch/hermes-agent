import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
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
import { AlertTriangle, Globe, Plus, RefreshCw, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { $profileScope } from '@/store/profile'
import { runGatewayRestart } from '@/store/system-actions'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import {
  Panel,
  PanelAddButton,
  PanelBlock,
  PanelBody,
  PanelDetail,
  PanelEmpty,
  PanelHeader,
  PanelList,
  PanelListRow,
  PanelMeta,
  PanelPill,
  PanelRowMenu,
  PanelSectionLabel
} from '../overlays/panel'

const DELIVER_OPTIONS: readonly string[] = ['log', 'telegram', 'discord', 'slack', 'email', 'github_comment']

interface CreatedWebhook {
  secret: string
  url: string
}

interface WebhooksViewProps {
  onClose: () => void
}

export function WebhooksView({ onClose }: WebhooksViewProps) {
  const { t } = useI18n()
  const w = t.webhooks
  // Re-load when the active profile changes so REST routes to the right backend.
  const profileScope = useStore($profileScope)
  const queryClient = useQueryClient()
  const queryKey = useMemo(() => ['webhooks', profileScope] as const, [profileScope])

  const [query, setQuery] = useState('')
  const [enabling, setEnabling] = useState(false)
  const [restartNeeded, setRestartNeeded] = useState(false)
  const [restartError, setRestartError] = useState<null | string>(null)
  const [restarting, setRestarting] = useState(false)
  const [togglingName, setTogglingName] = useState<null | string>(null)
  // Master/detail: the subscription whose config fills the right pane.
  const [selectedName, setSelectedName] = useState<null | string>(null)

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

  const {
    data,
    error,
    isPending: loading,
    refetch
  } = useQuery({
    queryKey,
    queryFn: getWebhooks
  })

  // React Query v5 dropped useQuery onError; surface a load failure toast once
  // per error object instead.
  useEffect(() => {
    if (error) {
      notifyError(error, w.loadFailed)
    }
  }, [error, w.loadFailed])

  const enabled = data?.enabled ?? false
  const subscriptions = useMemo(() => data?.subscriptions ?? [], [data])

  // Pull fresh backend truth into the cache. `silent` swallows the error toast
  // for post-mutation reconciles (the mutation already reported success/failure).
  const reload = useCallback(
    async (silent = false) => {
      try {
        await queryClient.invalidateQueries({ queryKey })
      } catch (err) {
        if (!silent) {
          notifyError(err, w.loadFailed)
        }
      }
    },
    [queryClient, queryKey, w.loadFailed]
  )

  useRefreshHotkey(() => void refetch())

  const restartGatewayNow = useCallback(async () => {
    setRestarting(true)

    try {
      await runGatewayRestart()
      setRestartNeeded(false)
      setRestartError(null)
      // Give the receiver a moment to bind before re-reading state.
      window.setTimeout(() => void reload(true), 4000)
    } catch (err) {
      setRestartNeeded(true)
      setRestartError(String(err))
      notifyError(err, w.restartFailed(''))
    } finally {
      setRestarting(false)
    }
  }, [reload, w])

  const handleEnable = useCallback(async () => {
    setEnabling(true)
    setRestartNeeded(false)
    setRestartError(null)

    try {
      const result = await enableWebhooks()
      await reload(true)

      if (result.restart_started) {
        notify({ kind: 'success', message: w.enabledRestarting })
        window.setTimeout(() => void reload(true), 4000)
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
  }, [reload, w])

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
      void reload(true)
    } catch (err) {
      notifyError(err, w.createFailed(''))
    } finally {
      setCreating(false)
    }
  }, [deliver, deliverOnly, description, events, name, prompt, reload, resetForm, skills, w])

  const handleToggle = useCallback(
    async (subName: string, nextEnabled: boolean) => {
      setTogglingName(subName)
      // Optimistic cache paint; the invalidate below lets backend truth win.
      queryClient.setQueryData<WebhooksResponse>(queryKey, current =>
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
        void reload(true)
      } catch (err) {
        await reload(true)
        notifyError(err, w.toggleFailed(subName))
      } finally {
        setTogglingName(null)
      }
    },
    [queryClient, queryKey, reload, w]
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
      void reload(true)
    } catch (err) {
      notifyError(err, w.deleteFailed(pendingDelete))
    } finally {
      setDeleting(false)
    }
  }, [pendingDelete, reload, w])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()

    if (!q) {
      return subscriptions
    }

    return subscriptions.filter(s =>
      [s.name, s.description, s.deliver, ...s.events].filter(Boolean).some(v => v.toLowerCase().includes(q))
    )
  }, [query, subscriptions])

  // Detail always reflects a concrete sub: the explicitly selected one, else the
  // first visible row, so the right pane is never empty while subs exist.
  const selectedSub = useMemo(
    () => visible.find(s => s.name === selectedName) ?? visible[0] ?? null,
    [visible, selectedName]
  )

  const banners = (
    <>
      {!enabled && (
        <Alert className="mb-4" variant="warning">
          <Globe />
          <AlertTitle>{w.disabledTitle}</AlertTitle>
          <AlertDescription>
            <p>{w.disabledBody}</p>
            <Button className="mt-1" disabled={enabling} onClick={() => void handleEnable()} size="sm">
              <Globe />
              {enabling ? w.enabling : w.enable}
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {restartNeeded && (
        <Alert className="mb-4" variant="warning">
          <AlertTriangle />
          <AlertDescription>
            <p>{restartError ?? w.restartNeeded}</p>
            <Button
              className="mt-1"
              disabled={restarting}
              onClick={() => void restartGatewayNow()}
              size="sm"
              variant="secondary"
            >
              <RefreshCw />
              {restarting ? w.restartingGateway : w.restartGateway}
            </Button>
          </AlertDescription>
        </Alert>
      )}
    </>
  )

  return (
    <Panel onClose={onClose}>
      {loading ? (
        <PageLoader label={w.loading} />
      ) : subscriptions.length === 0 ? (
        <>
          {banners}
          <PanelEmpty
            action={
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
            }
            description={w.empty}
            icon="globe"
          />
        </>
      ) : (
        <>
          <PanelHeader subtitle={w.hint} title={w.subscriptions(subscriptions.length)} />
          {banners}
          <PanelBody>
            <PanelList
              onSearchChange={setQuery}
              searchLabel={w.search}
              searchPlaceholder={w.search}
              searchValue={query}
            >
              {visible.map(sub => (
                <PanelListRow
                  active={selectedSub?.name === sub.name}
                  dotClassName={sub.enabled ? 'bg-emerald-500' : 'bg-muted-foreground/50'}
                  key={sub.name}
                  menu={
                    <PanelRowMenu
                      items={[
                        {
                          icon: sub.enabled ? 'circle-slash' : 'check',
                          label: sub.enabled ? w.disableRow : w.enableRow,
                          onSelect: () => void handleToggle(sub.name, !sub.enabled)
                        },
                        { icon: 'trash', label: w.delete, onSelect: () => setPendingDelete(sub.name), tone: 'danger' }
                      ]}
                    />
                  }
                  onSelect={() => setSelectedName(sub.name)}
                  title={sub.name}
                />
              ))}
              {visible.length === 0 && (
                <p className="px-2 py-4 text-center text-xs text-muted-foreground">{w.empty}</p>
              )}
              <PanelAddButton
                label={w.newSubscription}
                onClick={() => {
                  setCreated(null)
                  setCreateOpen(true)
                }}
              />
            </PanelList>

            {selectedSub ? (
              <WebhookDetail
                onDelete={() => setPendingDelete(selectedSub.name)}
                onToggle={() => void handleToggle(selectedSub.name, !selectedSub.enabled)}
                sub={selectedSub}
                toggling={togglingName === selectedSub.name}
              />
            ) : (
              <PanelEmpty description={w.empty} icon="search" />
            )}
          </PanelBody>
        </>
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
                  <CopyButton appearance="icon" buttonSize="icon-sm" label={w.copy} text={created.url} />
                </div>
              </div>
              <div className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">{w.secretOnce}</span>
                <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2">
                  <span className="min-w-0 flex-1 truncate font-mono text-xs">{created.secret}</span>
                  <CopyButton appearance="icon" buttonSize="icon-sm" label={w.copy} text={created.secret} />
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
                  <label className="flex items-start gap-2 text-sm text-muted-foreground">
                    <Checkbox
                      checked={deliverOnly}
                      className="mt-[3px] shrink-0"
                      onCheckedChange={value => setDeliverOnly(value === true)}
                    />
                    <span className="leading-snug">{w.fieldDeliverOnlyHint}</span>
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

function WebhookDetail({
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
    <PanelDetail>
      <header className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <h3 className="text-[0.95rem] font-semibold tracking-tight text-foreground">{sub.name}</h3>
            <PanelPill tone={sub.enabled ? 'good' : 'muted'}>
              {sub.enabled ? t.messaging.states.enabled : t.messaging.states.disabled}
            </PanelPill>
            {sub.deliver_only && <PanelPill tone="warn">{w.deliverOnly}</PanelPill>}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Switch
              aria-label={sub.enabled ? w.disableRow : w.enableRow}
              checked={sub.enabled}
              disabled={toggling}
              onCheckedChange={onToggle}
              size="xs"
            />
            <Button
              aria-label={w.delete}
              className="text-muted-foreground hover:bg-(--ui-row-hover-background) hover:text-destructive"
              onClick={onDelete}
              size="icon-sm"
              title={w.delete}
              variant="ghost"
            >
              <Trash2 />
            </Button>
          </div>
        </div>

        <PanelMeta
          rows={[
            { label: w.fieldDeliver, value: w.deliverOptions[sub.deliver] ?? sub.deliver },
            {
              label: w.fieldEvents,
              value:
                sub.events.length === 0 ? (
                  w.all
                ) : (
                  <span className="flex flex-wrap gap-1">
                    {sub.events.map(evt => (
                      <PanelPill key={evt}>{evt}</PanelPill>
                    ))}
                  </span>
                )
            },
            ...(sub.skills.length > 0
              ? [
                  {
                    label: w.fieldSkills,
                    value: (
                      <span className="flex flex-wrap gap-1">
                        {sub.skills.map(skill => (
                          <PanelPill key={skill}>{skill}</PanelPill>
                        ))}
                      </span>
                    )
                  }
                ]
              : [])
          ]}
        />

        <div className="flex items-center gap-1 rounded bg-foreground/5 px-2.5 py-1.5 text-[0.7rem] text-muted-foreground">
          <span className="min-w-0 flex-1 truncate font-mono text-foreground/80">{sub.url}</span>
          <CopyButton appearance="icon" buttonSize="icon-sm" label={w.copy} text={sub.url} />
        </div>
      </header>

      {sub.description ? (
        <div className="space-y-1.5">
          <PanelSectionLabel>{w.fieldDescription}</PanelSectionLabel>
          <p className="text-xs leading-relaxed text-foreground/80">{sub.description}</p>
        </div>
      ) : null}

      {sub.prompt ? (
        <div className="space-y-1.5">
          <PanelSectionLabel>{w.fieldPrompt}</PanelSectionLabel>
          <PanelBlock>{sub.prompt}</PanelBlock>
        </div>
      ) : null}
    </PanelDetail>
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
