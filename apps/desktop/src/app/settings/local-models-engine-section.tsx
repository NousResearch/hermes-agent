import { useQuery, useQueryClient } from '@tanstack/react-query'
import { type Dispatch, type ReactElement, type SetStateAction, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import {
  addLocalEngine,
  deleteLocalEngine,
  getLocalEngines,
  type LocalEngineFields,
  probeLocalEngine,
  selectLocalEngine,
  updateLocalEngine
} from '@/hermes'
import { Cpu, FolderOpen, Loader2, Pencil, Plus, Trash2 } from '@/lib/icons'
import { localModelsKey, localModelsRequestScope, refreshLocalModels } from '@/store/local-runtime-jobs'
import { notify, notifyError } from '@/store/notifications'
import type { LocalEngine } from '@/types/hermes'

import { useLocalModelsActionScope } from './local-models-actions'
import { ListRow, Pill, SettingsSection } from './primitives'

type LocalModelsCopy = ReturnType<typeof useLocalModelsActionScope>['copy']

interface EngineFormProps {
  busy: boolean
  copy: LocalModelsCopy
  draft: EngineDraft | null
  onBrowse: () => void
  onCancel: () => void
  onSave: () => void
  onProbe: () => void
  probeLine: string | null
  probing: boolean
  setDraft: Dispatch<SetStateAction<EngineDraft | null>>
}

function EngineForm({ busy, copy, draft, onBrowse, onCancel, onSave, onProbe, probeLine, probing, setDraft }: EngineFormProps): ReactElement {
  return (
    <div className="mb-1 grid gap-3 rounded-lg border border-border bg-card p-4">
      <div className="text-sm font-medium text-foreground">{draft?.id ? copy.engineEditTitle : copy.engineNewTitle}</div>
      <div className="grid gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">{copy.engineNameLabel}</span>
        <Input
          maxLength={64}
          onChange={event => setDraft(draft => (draft ? { ...draft, name: event.target.value } : draft))}
          placeholder={copy.engineNamePlaceholder}
          value={draft?.name ?? ''}
        />
      </div>
      <div className="grid gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">{copy.enginePathLabel}</span>
        <div className="flex items-center gap-2">
          <Input
            className="font-mono"
            onChange={event =>
              setDraft(draft => (draft ? { ...draft, path: event.target.value } : draft))
            }
            placeholder={copy.enginePathPlaceholder}
            value={draft?.path ?? ''}
          />
          <Button onClick={onBrowse} size="xs" type="button" variant="outline">
            <FolderOpen className="size-3.5" />
            {copy.engineBrowse}
          </Button>
        </div>
        <span className="text-[11px] leading-4 text-muted-foreground/80">{copy.engineRemoteHint}</span>
      </div>
      <div className="grid gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">{copy.engineBackendLabel}</span>
        <Select
          onValueChange={value => setDraft(draft => (draft ? { ...draft, backend: value } : draft))}
          value={draft?.backend ?? 'auto'}
        >
          <SelectTrigger className="w-full max-w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ENGINE_BACKENDS.map(backend => (
              <SelectItem key={backend} value={backend}>
                {backend}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">{copy.engineArgsLabel}</span>
        <Textarea
          className="min-h-16 font-mono"
          onChange={event => setDraft(draft => (draft ? { ...draft, args: event.target.value } : draft))}
          placeholder="--n-gpu-layers 99"
          value={draft?.args ?? ''}
        />
        <span className="text-[11px] leading-4 text-muted-foreground/80">{copy.engineArgsHint}</span>
      </div>
      <div className="grid gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">{copy.engineEnvLabel}</span>
        <Textarea
          className="min-h-16 font-mono"
          onChange={event => setDraft(draft => (draft ? { ...draft, env: event.target.value } : draft))}
          placeholder="GGML_VK_VISIBLE_DEVICES=0"
          value={draft?.env ?? ''}
        />
        <span className="text-[11px] leading-4 text-muted-foreground/80">{copy.engineEnvHint}</span>
      </div>
      {probeLine ? (
        <div className="max-h-20 overflow-y-auto whitespace-pre-wrap break-all rounded-md border border-border bg-background/60 px-3 py-2 font-mono text-[11px] leading-4 text-muted-foreground">
          {probeLine}
        </div>
      ) : null}
      <div className="flex items-center gap-2">
        <Button disabled={busy || probing || !draft?.path.trim()} onClick={onProbe} variant="outline">
          {probing ? <Loader2 className="size-4 animate-spin" /> : null}
          {probing ? copy.engineTesting : copy.engineTest}
        </Button>
        <Button
          disabled={busy || !draft?.name.trim() || !draft?.path.trim()}
          onClick={onSave}
        >
          {copy.engineSave}
        </Button>
        <Button
          disabled={busy}
            onClick={onCancel}
          variant="ghost"
        >
          {copy.engineCancel}
        </Button>
      </div>
    </div>
  )
}

// Engine registry management (Settings → Providers → Local Models). The
// picker chooses which llama.cpp binary answers the local endpoint; below it
// users register their own builds (path + launch args + env). The `managed`
// engine — Hermes' own PM-installed build, owned by the install/update
// flows — stays in the list but is never editable or deletable.
//
// Mutating routes bounce the runtime and write config under the dashboard's
// mutation lock; there are no jobs, so each action just refreshes the
// engines query and the pane's status polls the rest. An older gateway
// without the registry answers 404 — the whole section stays invisible.

const ENGINE_BACKENDS = ['auto', 'cpu', 'cuda', 'hip', 'metal', 'vulkan'] as const

interface EngineDraft {
  args: string
  backend: string
  env: string
  // null when registering a new engine; the registry id when editing.
  id: string | null
  name: string
  path: string
}

function engineToDraft(engine: LocalEngine): EngineDraft {
  return {
    args: engine.args.join('\n'),
    backend: engine.backend || 'auto',
    env: Object.entries(engine.env)
      .map(([key, value]) => `${key}=${value}`)
      .join('\n'),
    id: engine.id,
    name: engine.name,
    path: engine.path
  }
}

function draftToFields(draft: EngineDraft): LocalEngineFields {
  const env: Record<string, string> = {}

  for (const line of draft.env.split('\n')) {
    const eq = line.indexOf('=')
    const key = line.slice(0, eq).trim()

    // Same rule as the registry: real identifier, non-empty key. Values keep
    // their inner spacing; only the ends are trimmed.
    if (eq > 0 && /^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      env[key] = line.slice(eq + 1).trim()
    }
  }

  return {
    args: draft.args.split('\n').map(line => line.trim()).filter(Boolean),
    backend: draft.backend,
    env,
    name: draft.name.trim(),
    path: draft.path.trim()
  }
}

export function LocalModelsEngineSection(): ReactElement | null {
  const { client, copy, owner } = useLocalModelsActionScope()
  const queryClient = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [draft, setDraft] = useState<EngineDraft | null>(null)
  const [probeLine, setProbeLine] = useState<string | null>(null)
  const [probing, setProbing] = useState(false)

  const enginesQuery = useQuery({
    queryFn: () => getLocalEngines(localModelsRequestScope(owner)),
    queryKey: localModelsKey(owner, 'engines'),
    retry: false
  })

  // Pre-registry gateway (404): the section quietly opts out.
  if (enginesQuery.isError || !enginesQuery.data) {
    return null
  }

  const engines = enginesQuery.data.engines
  const active = engines.find(engine => engine.active) ?? null

  const invalidate = () => queryClient.invalidateQueries({ queryKey: localModelsKey(owner, 'engines') })

  async function handleSelect(engineId: string): Promise<void> {
    if (busy || engineId === active?.id) {
      return
    }

    const name = engines.find(engine => engine.id === engineId)?.name ?? engineId
    setBusy(true)

    try {
      await selectLocalEngine(engineId, localModelsRequestScope(owner))
      notify({ message: copy.engineSelectedToast(name), kind: 'success', title: copy.engineTitle })
      invalidate()
      refreshLocalModels(owner, client)
    } catch (err) {
      notifyError(err, copy.engineTitle)
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete(engine: LocalEngine): Promise<void> {
    if (busy) {
      return
    }

    setBusy(true)

    try {
      await deleteLocalEngine(engine.id, localModelsRequestScope(owner))
      notify({ message: copy.engineRemovedToast(engine.name), kind: 'info', title: copy.engineTitle })

      if (draft?.id === engine.id) {
        setDraft(null)
      }

      invalidate()
      refreshLocalModels(owner, client)
    } catch (err) {
      notifyError(err, copy.engineTitle)
    } finally {
      setBusy(false)
    }
  }

  async function handleProbe(): Promise<void> {
    if (!draft || probing) {
      return
    }

    const fields = draftToFields(draft)

    if (!draft.id && !fields.path) {
      return
    }

    setProbing(true)
    setProbeLine(null)

    try {
      // A filled path always wins: the form probes what is on screen, even
      // when editing a saved engine whose path was just changed.
      const result = await probeLocalEngine(
        fields.path ? { path: fields.path } : { engineId: draft.id ?? '' },
        localModelsRequestScope(owner)
      )

      setProbeLine(copy.engineTestOk(result.version))
    } catch (err) {
      setProbeLine(err instanceof Error ? err.message : String(err))
    } finally {
      setProbing(false)
    }
  }

  async function handleSave(): Promise<void> {
    if (!draft || busy) {
      return
    }

    const fields = draftToFields(draft)

    if (!fields.name || !fields.path) {
      return
    }

    setBusy(true)

    try {
      if (draft.id) {
        await updateLocalEngine(draft.id, fields, localModelsRequestScope(owner))
        notify({ message: copy.engineUpdatedToast, kind: 'success', title: copy.engineTitle })
      } else {
        await addLocalEngine(fields, localModelsRequestScope(owner))
        notify({ message: copy.engineAddedToast, kind: 'success', title: copy.engineTitle })
      }

      setDraft(null)
      setProbeLine(null)
      invalidate()
      refreshLocalModels(owner, client)
    } catch (err) {
      notifyError(err, copy.engineTitle)
    } finally {
      setBusy(false)
    }
  }

  async function handleBrowse(): Promise<void> {
    const paths = await window.hermesDesktop.selectPaths({
      multiple: false,
      title: copy.enginePathLabel
    })

    if (paths.length > 0 && draft) {
      setDraft({ ...draft, path: paths[0] })
      setProbeLine(null)
    }
  }

  return (
    <SettingsSection icon={Cpu} title={copy.engineTitle}>
      {draft ? (
          <EngineForm
            busy={busy}
            copy={copy}
            draft={draft}
            onBrowse={handleBrowse}
            onCancel={() => {
              setDraft(null)
              setProbeLine(null)
            }}
            onProbe={handleProbe}
            onSave={handleSave}
            probeLine={probeLine}
            probing={probing}
            setDraft={setDraft}
          />
        ) : null}
      <ListRow
        action={
          <Select onValueChange={value => void handleSelect(value)} value={active?.id ?? undefined}>
            <SelectTrigger className="h-8 w-52" disabled={busy}>
              <SelectValue placeholder={active?.name ?? ''} />
            </SelectTrigger>
            <SelectContent align="end">
              {engines.map(engine => (
                <SelectItem disabled={!engine.installed && !engine.active} key={engine.id} value={engine.id}>
                  {engine.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
        description={copy.enginePickerDetail}
        title={copy.enginePickerLabel}
      />
      {engines.map(engine => (
        <ListRow
          action={
            engine.kind === 'managed' ? null : (
              <span className="flex items-center gap-1">
                <Button
                  aria-label={copy.engineEditTitle}
                  disabled={busy}
                  onClick={() => {
                    setDraft(engineToDraft(engine))
                    setProbeLine(null)
                  }}
                  size="icon-xs"
                  type="button"
                  variant="ghost"
                >
                  <Pencil className="size-3.5" />
                </Button>
                <Button
                  aria-label={copy.engineRemoveLabel}
                  disabled={busy}
                  onClick={() => void handleDelete(engine)}
                  size="icon-xs"
                  type="button"
                  variant="ghost"
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </span>
            )
          }
          description={
            engine.kind === 'managed'
              ? copy.engineManagedNote
              : engine.installed
                ? engine.path
                : <span className="text-destructive">{copy.engineMissingNote}</span>
          }
          key={engine.id}
          title={
            <span className="flex items-center gap-2">
              {engine.name}
              {engine.active ? <Pill tone="success">{copy.engineActivePill}</Pill> : null}
              {engine.kind === 'managed' ? <Pill tone="muted">{copy.engineManagedPill}</Pill> : null}
            </span>
          }
        />
      ))}
      <ListRow
        action={
          <Button
            disabled={busy || Boolean(draft)}
            onClick={() => {
              setDraft({ args: '', backend: 'auto', env: '', id: null, name: '', path: '' })
              setProbeLine(null)
            }}
            size="xs"
            type="button"
            variant="outline"
          >
            <Plus className="size-3.5" />
            {copy.engineAddLabel}
          </Button>
        }
        description={copy.engineAddDetail}
        title={copy.engineAddLabel}
      />
    </SettingsSection>
  )
}
