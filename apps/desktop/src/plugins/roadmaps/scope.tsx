/**
 * Roadmaps plugin — scope bar.
 *
 * Read-only active profile (Tip) + project id input + roadmap selector with
 * copy buttons and the + / ⋮ input flows (inline roadmap create + rename, and
 * the roadmap management menu). The roadmaps REST API has no projects list,
 * so the project is a free-form identifier the operator pastes; the roadmap
 * selector is fed by the live roadmaps list. Selection resets happen here:
 * a project change clears the roadmap and node selection; a roadmap change
 * clears the node selection.
 */

import {
  Button,
  Codicon,
  ConfirmDialog,
  CopyButton,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Tip,
  useQueryClient
} from '@hermes/plugin-sdk'
import { useCallback, useState } from 'react'

import { archiveRoadmap, createRoadmap, roadmapsListKey, updateRoadmap } from './api'
import { mutationErrorCopy, plural, rpcError, validateRoadmapTitle } from './data'
import type { RoadmapListItem } from './types'

/** Compact inline error: stable guidance by code only. */
function FormError({ error }: { error: { code: number | null; hint?: string } | null }) {
  if (!error) {return null}
  const ec = mutationErrorCopy(error)

  if (!ec) {return null}

  return (
    <div className="flex items-start gap-1.5 rounded-[3px] bg-destructive/10 px-2 py-1 text-xs text-destructive">
      <Codicon className="mt-px shrink-0" name="error" size="0.75rem" />
      <span>{`${ec.hint}${ec.code != null ? ` (code ${ec.code})` : ''}`}</span>
    </div>
  )
}

function RoadmapCreateForm({
  profile,
  projectId,
  actor,
  onCreated,
  onCancel
}: {
  profile: string
  projectId: string
  actor: string
  onCreated: (id: string) => void
  onCancel: () => void
}) {
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ code: number | null; hint?: string } | null>(null)
  const queryClient = useQueryClient()

  const submit = useCallback(async () => {
    if (busy) {return}
    const trimmed = title.trim()

    if (!validateRoadmapTitle(trimmed)) {
      setError({ code: null, hint: 'Roadmap title must be non-empty, at most 200 characters, and free of control characters.' })

      return
    }

    setBusy(true)
    setError(null)

    try {
      const res = await createRoadmap(profile, projectId, { actor, title: trimmed })
      // Wait for the authoritative refresh so the new roadmap is in the
      // selector BEFORE the scope selection points at it (selection hygiene).
      await queryClient.invalidateQueries({ queryKey: roadmapsListKey(profile, projectId) })
      onCreated(res?.roadmap_id ?? '')
    } catch (err) {
      setError({ code: rpcError(err).code, hint: (err as { hint?: string })?.hint })
    } finally {
      setBusy(false)
    }
  }, [actor, busy, onCreated, profile, projectId, queryClient, title])

  return (
    <div className="flex flex-col gap-1 px-0.5">
      <div className="flex items-center gap-1.5">
        <Input
          aria-label="New roadmap title"
          autoFocus
          className="h-6 w-48 px-1.5 text-xs"
          disabled={busy}
          onChange={(ev) => setTitle(ev.target.value)}
          onKeyDown={(ev) => {
            if (ev.key === 'Enter') {void submit()}

            if (ev.key === 'Escape') {onCancel()}
          }}
          placeholder="Roadmap title…"
          value={title}
        />
        <Button disabled={busy || title.trim() === ''} onClick={() => void submit()} size="xs" type="button" variant="secondary">
          Create
        </Button>
        <Button disabled={busy} onClick={onCancel} size="xs" type="button" variant="ghost">
          Cancel
        </Button>
      </div>
      <FormError error={error} />
    </div>
  )
}

function RoadmapRenameForm({
  profile,
  projectId,
  roadmapId,
  currentTitle,
  expectedVersion,
  actor,
  onRenamed,
  onCancel
}: {
  profile: string
  projectId: string
  roadmapId: string
  currentTitle: string
  expectedVersion: number
  actor: string
  onRenamed: () => void
  onCancel: () => void
}) {
  const [title, setTitle] = useState(currentTitle)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ code: number | null; hint?: string } | null>(null)
  const queryClient = useQueryClient()

  const submit = useCallback(async () => {
    if (busy) {return}
    const trimmed = title.trim()

    if (!validateRoadmapTitle(trimmed)) {
      setError({ code: null, hint: 'Roadmap title must be non-empty, at most 200 characters, and free of control characters.' })

      return
    }

    setBusy(true)
    setError(null)

    try {
      await updateRoadmap(profile, projectId, roadmapId, { actor, expected_version: expectedVersion, title: trimmed })
      await queryClient.invalidateQueries({ queryKey: roadmapsListKey(profile, projectId) })
      onRenamed()
    } catch (err) {
      setError({ code: rpcError(err).code, hint: (err as { hint?: string })?.hint })
    } finally {
      setBusy(false)
    }
  }, [actor, busy, expectedVersion, onRenamed, profile, projectId, queryClient, roadmapId, title])

  return (
    <div className="flex flex-col gap-1 px-0.5">
      <div className="flex items-center gap-1.5">
        <Input
          aria-label="Rename roadmap"
          autoFocus
          className="h-6 w-48 px-1.5 text-xs"
          disabled={busy}
          onChange={(ev) => setTitle(ev.target.value)}
          onKeyDown={(ev) => {
            if (ev.key === 'Enter') {void submit()}

            if (ev.key === 'Escape') {onCancel()}
          }}
          placeholder="Roadmap title…"
          value={title}
        />
        <Button disabled={busy || title.trim() === ''} onClick={() => void submit()} size="xs" type="button" variant="secondary">
          Save
        </Button>
        <Button disabled={busy} onClick={onCancel} size="xs" type="button" variant="ghost">
          Cancel
        </Button>
      </div>
      <FormError error={error} />
    </div>
  )
}

/** Roadmap ⋮ menu: Rename (inline form), Copy ID (menu item), Archive (confirm). */
function RoadmapMenu({
  profile,
  projectId,
  roadmapId,
  roadmapTitle,
  expectedVersion,
  actor,
  onRequestRename,
  onArchived
}: {
  profile: string
  projectId: string
  roadmapId: string
  roadmapTitle: string
  expectedVersion: number
  actor: string
  onRequestRename: () => void
  onArchived: () => void
}) {
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const queryClient = useQueryClient()

  const archive = useCallback(async () => {
    if (busy) {return}
    setBusy(true)

    try {
      await archiveRoadmap(profile, projectId, roadmapId, { actor, expected_version: expectedVersion })
      await queryClient.invalidateQueries({ queryKey: roadmapsListKey(profile, projectId) })
      setConfirmOpen(false)
      onArchived()
    } catch (err) {
      // Throw the GENERIC hint so ConfirmDialog surfaces stable guidance.
      const ec = mutationErrorCopy({ code: rpcError(err).code })
      throw new Error(ec?.hint ?? 'Archive failed.')
    } finally {
      setBusy(false)
    }
  }, [actor, busy, expectedVersion, onArchived, profile, projectId, queryClient, roadmapId])

  const hasRoadmap = roadmapId !== ''

  return (
    <div className="flex items-center gap-1.5">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label="Roadmap actions"
            className="data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground"
            disabled={!hasRoadmap}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="ellipsis" size="0.8rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44" sideOffset={4}>
          <DropdownMenuItem disabled={!hasRoadmap} onSelect={onRequestRename}>
            <Codicon name="edit" size="0.75rem" />
            Rename
          </DropdownMenuItem>
          <CopyButton appearance="menu-item" disabled={!hasRoadmap} label="Copy ID" text={roadmapId} />
          <DropdownMenuSeparator />
          <DropdownMenuItem disabled={!hasRoadmap} onSelect={() => setConfirmOpen(true)} variant="destructive">
            <Codicon name="archive" size="0.75rem" />
            Archive
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ConfirmDialog
        cancelLabel="Cancel"
        confirmLabel="Archive"
        description={`Archive "${roadmapTitle}"? The roadmap leaves the selector; its data stays on the backend.`}
        destructive
        onClose={() => {
          if (!busy) {setConfirmOpen(false)}
        }}
        onConfirm={archive}
        open={confirmOpen}
        title="Archive roadmap"
      />
    </div>
  )
}

export function ScopeBar({
  profile,
  projectId,
  onProjectChange,
  roadmapId,
  setRoadmapId,
  setSelectedNodeId,
  roadmapOptions,
  compact,
  actor
}: {
  profile: string
  projectId: string
  onProjectChange: (v: string) => void
  roadmapId: string
  setRoadmapId: (v: string) => void
  setSelectedNodeId: (v: string) => void
  roadmapOptions: RoadmapListItem[]
  compact: boolean
  actor: string
}) {
  const [roadmapCreateOpen, setRoadmapCreateOpen] = useState(false)
  const [roadmapRenameOpen, setRoadmapRenameOpen] = useState(false)

  const selectRoadmap = (v: string) => {
    setRoadmapId(v)
    setSelectedNodeId('')
  }

  const selectedRoadmap = roadmapOptions.find((r) => r.roadmap_id === roadmapId) ?? null
  const roadmapTitle = selectedRoadmap?.title || roadmapId
  const roadmapActiveVersion = Number(selectedRoadmap?.active_version) || 0

  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-2 px-0.5">
        <Tip label="Active profile (read-only)">
          <span className="inline-flex items-center gap-1.5 rounded-[3px] bg-(--ui-bg-quaternary) px-2 py-1 font-mono text-[0.625rem] text-(--ui-text-secondary)">
            <Codicon name="account" size="0.7rem" />
            {profile}
          </span>
        </Tip>

        <div className="flex items-center gap-1">
          <span className="text-[0.625rem] text-(--ui-text-tertiary)">Project</span>
          <Input
            aria-label="Project id"
            className="h-7 w-40 font-mono text-xs"
            onChange={(ev) => onProjectChange(ev.target.value)}
            placeholder="project id…"
            spellCheck={false}
            value={projectId}
          />
          {projectId !== '' ? (
            <CopyButton appearance="icon" buttonSize="icon-xs" buttonVariant="ghost" label="Copy project ID" text={projectId} title="Copy project ID" />
          ) : null}
        </div>

        <div className="flex items-center gap-1">
          <span className="text-[0.625rem] text-(--ui-text-tertiary)">Roadmap</span>
          <Select disabled={projectId === ''} onValueChange={selectRoadmap} value={roadmapId}>
            <SelectTrigger aria-label="Roadmap" className="h-7 w-48 text-xs">
              <SelectValue placeholder={projectId === '' ? '—' : 'select…'} />
            </SelectTrigger>
            <SelectContent>
              {roadmapOptions.length === 0 ? (
                <SelectItem disabled value="__none__">
                  No roadmaps
                </SelectItem>
              ) : (
                roadmapOptions.map((r) => (
                  <SelectItem key={r.roadmap_id} value={r.roadmap_id}>
                    <span className="block min-w-0 truncate">{r.title || r.roadmap_id}</span>
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          <Button
            aria-label="Create roadmap"
            disabled={projectId === ''}
            onClick={() => setRoadmapCreateOpen((v) => !v)}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="add" size="0.8rem" />
          </Button>
          <RoadmapMenu
            actor={actor}
            expectedVersion={roadmapActiveVersion}
            onArchived={() => {
              setRoadmapRenameOpen(false)
              setRoadmapCreateOpen(false)
            }}
            onRequestRename={() => setRoadmapRenameOpen(true)}
            profile={profile}
            projectId={projectId}
            roadmapId={roadmapId}
            roadmapTitle={roadmapTitle}
          />
        </div>

        {!compact ? (
          <span className="ml-auto text-[0.625rem] text-(--ui-text-tertiary)">
            {`${plural(roadmapOptions.length, 'roadmap')} · profile ${profile}`}
          </span>
        ) : null}
      </div>

      {roadmapCreateOpen ? (
        <RoadmapCreateForm
          actor={actor}
          onCancel={() => setRoadmapCreateOpen(false)}
          onCreated={(id) => {
            selectRoadmap(id)
            setRoadmapCreateOpen(false)
          }}
          profile={profile}
          projectId={projectId}
        />
      ) : null}

      {roadmapRenameOpen ? (
        <RoadmapRenameForm
          actor={actor}
          currentTitle={roadmapTitle}
          expectedVersion={roadmapActiveVersion}
          onCancel={() => setRoadmapRenameOpen(false)}
          onRenamed={() => setRoadmapRenameOpen(false)}
          profile={profile}
          projectId={projectId}
          roadmapId={roadmapId}
        />
      ) : null}
    </div>
  )
}
