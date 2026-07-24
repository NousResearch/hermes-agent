import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
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
import { AlertTriangle, Globe, Plus, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
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
import { ListRow, ToggleRow } from '../settings/primitives'

const DELIVER_OPTIONS: readonly string[] = ['log', 'telegram', 'discord', 'slack', 'email', 'github_comment']

interface CreatedWebhook {
  secret: string
  url: string
}

// One affordance for "value + CopyButton": flat, token-backed (no border, no
// raw literals). DESIGN.md Principle 1 (flat, not boxed) + 4 (tokens, not
// literals). Reused by the detail URL row and the create-result URL/secret so
// there is a single copyable-value chrome in this file.
function CopyValueRow({ copyLabel, mono = true, value }: { copyLabel: string; mono?: boolean; value: string }) {
  return (
    <div className="flex items-center gap-1 rounded bg-foreground/5 px-2.5 py-1.5 text-[0.7rem]">
      <span className={cn('min-w-0 flex-1 truncate text-foreground/80', mono && 'font-mono')}>{value}</span>
      <CopyButton appearance="icon" buttonSize="icon-sm" label={copyLabel} text={value} />
    </div>
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
  const queryClient = useQueryClient()
  const queryKey = useMemo(() => ['webhooks', profileScope] as const, [profileScope])

  const [query, setQuery] = useState('')
  const [enabling, setEnabling] = useState(false)
  const [restartNeeded, setRestartNeeded] = useState(false)
  const [restartError, setRestartError] = useState<null | string>(null)
  const [restarting, setRestarting] = useState(false)
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
      // Optimistic cache paint; the invalidate below lets backend truth win.
      queryClient.setQueryData<WebhooksResponse>(queryKey, current =>
        current
          ? {
              ...current,
              subscriptions: current.subscriptions.map(s => (s.name === subName ? { ...s, enabled: nextEnabled } : s))
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
      }
    },
    [queryClient, queryKey, reload, w]
  )

  // ConfirmDialog owns the pending→done→close beat; throw to surface its inline
  // error and keep the dialog open. Success toast matches the cron delete idiom
  // (title + name).
  const handleDelete = useCallback(async () => {
    if (!pendingDelete) {
      return
    }

    try {
      await deleteWebhook(pendingDelete)
      notify({ kind: 'success', title: w.deleted, message: pendingDelete })
      void reload(true)
    } catch (err) {
      notifyError(err, w.deleteFailed(pendingDelete))
      throw err
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
              {visible.length === 0 && <p className="px-2 py-4 text-center text-xs text-muted-foreground">{w.empty}</p>}
              <PanelAddButton
                label={w.newSubscription}
                onClick={() => {
                  setCreated(null)
                  setCreateOpen(true)
                }}
              />
            </PanelList>

            {selectedSub ? <WebhookDetail sub={selectedSub} /> : <PanelEmpty description={w.empty} icon="search" />}
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
            <div className="grid gap-2">
              <ListRow action={<CopyValueRow copyLabel={w.copy} value={created.url} />} title={w.webhookUrl} wide />
              <ListRow action={<CopyValueRow copyLabel={w.copy} value={created.secret} />} title={w.secretOnce} wide />
              <DialogFooter>
                <Button onClick={closeCreate} size="sm">
                  {w.done}
                </Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="grid gap-1">
              <div className="grid grid-cols-2 gap-4">
                <ListRow
                  action={
                    <Input
                      autoFocus
                      id="webhook-name"
                      onChange={e => setName(e.target.value)}
                      placeholder={w.fieldNamePlaceholder}
                      value={name}
                    />
                  }
                  title={<label htmlFor="webhook-name">{w.fieldName}</label>}
                  wide
                />
                <ListRow
                  action={
                    <Input
                      id="webhook-description"
                      onChange={e => setDescription(e.target.value)}
                      placeholder={w.fieldDescriptionPlaceholder}
                      value={description}
                    />
                  }
                  title={<label htmlFor="webhook-description">{w.fieldDescription}</label>}
                  wide
                />
              </div>
              <ListRow
                action={
                  <Textarea
                    className="min-h-[80px]"
                    id="webhook-prompt"
                    onChange={e => setPrompt(e.target.value)}
                    placeholder={w.fieldPromptPlaceholder}
                    value={prompt}
                  />
                }
                title={<label htmlFor="webhook-prompt">{w.fieldPrompt}</label>}
                wide
              />
              <div className="grid grid-cols-2 gap-4">
                <ListRow
                  action={
                    <Input
                      id="webhook-events"
                      onChange={e => setEvents(e.target.value)}
                      placeholder={w.fieldEventsPlaceholder}
                      value={events}
                    />
                  }
                  title={<label htmlFor="webhook-events">{w.fieldEvents}</label>}
                  wide
                />
                <ListRow
                  action={
                    <Input
                      id="webhook-skills"
                      onChange={e => setSkills(e.target.value)}
                      placeholder={w.fieldSkillsPlaceholder}
                      value={skills}
                    />
                  }
                  title={<label htmlFor="webhook-skills">{w.fieldSkills}</label>}
                  wide
                />
              </div>
              <div className="grid grid-cols-2 items-start gap-4">
                <ListRow
                  action={
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
                  }
                  title={<label htmlFor="webhook-deliver">{w.fieldDeliver}</label>}
                  wide
                />
                <ToggleRow checked={deliverOnly} label={w.fieldDeliverOnly} onChange={setDeliverOnly} />
              </div>
              <DialogFooter>
                <Button disabled={creating} onClick={() => void handleCreate()} size="sm">
                  {creating ? w.creating : w.create}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        busyLabel={w.deleting}
        cancelLabel={t.common.cancel}
        confirmLabel={w.delete}
        description={
          pendingDelete ? (
            <>
              {w.deleteDescPrefix}
              <span className="font-medium text-foreground">{pendingDelete}</span>
              {w.deleteDescSuffix}
            </>
          ) : null
        }
        destructive
        onClose={() => setPendingDelete(null)}
        onConfirm={handleDelete}
        open={pendingDelete !== null}
        title={w.deleteTitle}
      />
    </Panel>
  )
}

function WebhookDetail({ sub }: { sub: WebhookRoute }) {
  const { t } = useI18n()
  const w = t.webhooks

  return (
    <PanelDetail>
      <header className="space-y-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <h3 className="text-[0.95rem] font-semibold tracking-tight text-foreground">{sub.name}</h3>
          <PanelPill tone={sub.enabled ? 'good' : 'muted'}>
            {sub.enabled ? t.messaging.states.enabled : t.messaging.states.disabled}
          </PanelPill>
          {sub.deliver_only && <PanelPill tone="warn">{w.deliverOnly}</PanelPill>}
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

        <CopyValueRow copyLabel={w.copy} value={sub.url} />
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
