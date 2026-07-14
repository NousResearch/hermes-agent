import { useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
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
import { useI18n } from '@/i18n'
import { AlertTriangle, Loader2, Plus } from '@/lib/icons'
import {
  archiveWorkspaceDoc,
  listWorkspaceDocs,
  readWorkspaceDoc,
  writeWorkspaceDoc,
  type WorkspaceDocDetail,
  type WorkspaceDocStatus,
  type WorkspaceDocSummary,
  type WorkspaceDocType
} from '@/lib/workspace-docs'

// Templates offered from "New document" — a fixed subset of WorkspaceDocType
// (skill/memory/prompt authoring belongs to other surfaces; this panel only
// covers the doc types explicitly scoped for this slice).
const CREATE_TEMPLATES: { docType: WorkspaceDocType; body: string }[] = [
  { docType: 'generic-md', body: '' },
  { docType: 'runbook', body: '## Steps\n\n1. \n' },
  { docType: 'skill-template', body: '## When to use\n\n## Steps\n' },
  { docType: 'memory-note', body: '## Context\n\n## Notes\n' }
]

const STATUS_OPTIONS: WorkspaceDocStatus[] = ['draft', 'ready', 'archived']

function slugify(title: string): string {
  const slug = title
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  return slug || 'document'
}

function uniqueDocPath(title: string, existing: WorkspaceDocSummary[]): string {
  const base = slugify(title)
  const taken = new Set(existing.map(doc => doc.path))
  let candidate = `${base}.md`
  let suffix = 2

  while (taken.has(candidate)) {
    candidate = `${base}-${suffix}.md`
    suffix += 1
  }

  return candidate
}

function tagsToInput(tags?: string[]): string {
  return (tags ?? []).join(', ')
}

function inputToTags(value: string): string[] {
  return value
    .split(',')
    .map(tag => tag.trim())
    .filter(Boolean)
}

function statusBadgeVariant(status?: WorkspaceDocStatus): 'default' | 'muted' | 'outline' {
  if (status === 'ready') {
    return 'default'
  }

  if (status === 'archived') {
    return 'outline'
  }

  return 'muted'
}

type View = 'list' | 'create' | 'detail'

// Minimal library UI for the inert workspace documents under
// `<project.path>/.hermes/docs` — list, create-from-template, edit, and
// archive. No MDX rendering, no import/export/promote actions; saves and
// archives go straight through `@/lib/workspace-docs`.
export function WorkspaceDocumentsPanel({
  open,
  onOpenChange,
  workspaceRoot,
  label
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  workspaceRoot: string
  label: string
}) {
  const { t } = useI18n()
  const d = t.workspaceDocs

  const [view, setView] = useState<View>('list')
  const [docs, setDocs] = useState<WorkspaceDocSummary[]>([])
  const [listState, setListState] = useState<'error' | 'idle' | 'loading'>('loading')
  const [listError, setListError] = useState<null | string>(null)

  const [createDocType, setCreateDocType] = useState<WorkspaceDocType>('generic-md')
  const [createTitle, setCreateTitle] = useState('')
  const [createState, setCreateState] = useState<'error' | 'idle' | 'saving'>('idle')
  const [createError, setCreateError] = useState<null | string>(null)

  const [selected, setSelected] = useState<null | WorkspaceDocDetail>(null)
  const [readError, setReadError] = useState<null | string>(null)
  const [titleInput, setTitleInput] = useState('')
  const [statusInput, setStatusInput] = useState<WorkspaceDocStatus>('draft')
  const [descriptionInput, setDescriptionInput] = useState('')
  const [tagsInput, setTagsInput] = useState('')
  const [bodyInput, setBodyInput] = useState('')
  const [saveState, setSaveState] = useState<'error' | 'idle' | 'saving'>('idle')
  const [saveError, setSaveError] = useState<null | string>(null)
  const [archiveState, setArchiveState] = useState<'error' | 'idle' | 'saving'>('idle')
  const [archiveError, setArchiveError] = useState<null | string>(null)

  const dirty = useMemo(() => {
    if (!selected) {
      return false
    }

    return (
      titleInput !== selected.frontmatter.title ||
      statusInput !== selected.frontmatter.status ||
      descriptionInput !== (selected.frontmatter.description ?? '') ||
      tagsInput !== tagsToInput(selected.frontmatter.tags) ||
      bodyInput !== selected.body
    )
  }, [selected, titleInput, statusInput, descriptionInput, tagsInput, bodyInput])

  async function loadDocs() {
    setListState('loading')
    setListError(null)

    try {
      const result = await listWorkspaceDocs(workspaceRoot)
      setDocs(result)
      setListState('idle')
    } catch (err) {
      setListState('error')
      setListError(err instanceof Error ? err.message : d.loadFailed)
    }
  }

  useEffect(() => {
    if (!open) {
      return
    }

    setView('list')
    setSelected(null)
    void loadDocs()
    // Only reload when the panel opens for a (possibly different) project —
    // in-panel navigation manages its own refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, workspaceRoot])

  function openCreate() {
    setCreateDocType('generic-md')
    setCreateTitle('')
    setCreateError(null)
    setCreateState('idle')
    setView('create')
  }

  function loadDetail(detail: WorkspaceDocDetail) {
    setSelected(detail)
    setTitleInput(detail.frontmatter.title)
    setStatusInput(detail.frontmatter.status)
    setDescriptionInput(detail.frontmatter.description ?? '')
    setTagsInput(tagsToInput(detail.frontmatter.tags))
    setBodyInput(detail.body)
    setSaveState('idle')
    setSaveError(null)
    setArchiveState('idle')
    setArchiveError(null)
    setView('detail')
  }

  async function openDoc(doc: WorkspaceDocSummary) {
    setReadError(null)

    try {
      const detail = await readWorkspaceDoc(workspaceRoot, doc.path)
      loadDetail(detail)
    } catch (err) {
      setReadError(err instanceof Error ? err.message : d.readFailed)
    }
  }

  async function handleCreate() {
    setCreateState('saving')
    setCreateError(null)

    const title = createTitle.trim() || d.titlePlaceholder
    const path = uniqueDocPath(title, docs)
    const template = CREATE_TEMPLATES.find(entry => entry.docType === createDocType) ?? CREATE_TEMPLATES[0]

    try {
      await writeWorkspaceDoc(workspaceRoot, path, { docType: createDocType, title, status: 'draft', tags: [] }, template.body)
      const detail = await readWorkspaceDoc(workspaceRoot, path)
      await loadDocs()
      loadDetail(detail)
    } catch (err) {
      setCreateState('error')
      setCreateError(err instanceof Error ? err.message : d.saveFailed)
    }
  }

  async function handleSave() {
    if (!selected) {
      return
    }

    setSaveState('saving')
    setSaveError(null)

    try {
      const result = await writeWorkspaceDoc(
        workspaceRoot,
        selected.path,
        {
          docType: selected.frontmatter.docType,
          title: titleInput.trim() || selected.frontmatter.title,
          status: statusInput,
          applyState: selected.frontmatter.applyState,
          description: descriptionInput.trim() || null,
          tags: inputToTags(tagsInput),
          workspaceId: selected.frontmatter.workspaceId,
          createdAt: selected.frontmatter.createdAt
        },
        bodyInput
      )

      setSelected({ ...selected, frontmatter: result.frontmatter, body: bodyInput })
      setSaveState('idle')
      await loadDocs()
    } catch (err) {
      setSaveState('error')
      setSaveError(err instanceof Error ? err.message : d.saveFailed)
    }
  }

  async function handleArchive() {
    if (!selected) {
      return
    }

    setArchiveState('saving')
    setArchiveError(null)

    try {
      const result = await archiveWorkspaceDoc(workspaceRoot, selected.path)
      setSelected({ ...selected, frontmatter: result.frontmatter })
      setStatusInput(result.frontmatter.status)
      setArchiveState('idle')
      await loadDocs()
    } catch (err) {
      setArchiveState('error')
      setArchiveError(err instanceof Error ? err.message : d.archiveFailed)
    }
  }

  const isArchived = selected?.frontmatter.status === 'archived'

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-lg" onClick={event => event.stopPropagation()}>
        {view === 'list' && (
          <>
            <DialogHeader>
              <DialogTitle>{d.title}</DialogTitle>
              <DialogDescription>{d.description(label)}</DialogDescription>
            </DialogHeader>

            {listState === 'loading' && (
              <div className="flex items-center gap-2 py-6 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" />
                {d.loading}
              </div>
            )}

            {listState === 'error' && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{listError ?? d.loadFailed}</span>
              </div>
            )}

            {listState === 'idle' && docs.length === 0 && (
              <div className="flex flex-col items-center gap-1 py-6 text-center text-xs text-muted-foreground">
                <span>{d.empty}</span>
                <span>{d.emptyHint}</span>
              </div>
            )}

            {listState === 'idle' && docs.length > 0 && (
              <div className="flex max-h-80 flex-col gap-1 overflow-y-auto">
                {docs.map(doc => (
                  <button
                    className="flex w-full items-center justify-between gap-2 rounded-md border border-transparent px-2.5 py-1.5 text-left text-xs hover:border-(--ui-stroke-secondary) hover:bg-(--chrome-action-hover) disabled:cursor-not-allowed disabled:opacity-60"
                    disabled={!doc.valid}
                    key={doc.path}
                    onClick={() => void openDoc(doc)}
                    type="button"
                  >
                    <span className="flex min-w-0 flex-col gap-0.5">
                      <span className="truncate font-medium text-foreground">{doc.title ?? doc.path}</span>
                      <span className="truncate text-(--ui-text-tertiary)">
                        {doc.docType ? d.docType[doc.docType] : doc.path}
                        {doc.error ? ` — ${doc.error}` : ''}
                      </span>
                    </span>
                    <span className="flex shrink-0 items-center gap-1.5">
                      {!doc.valid && (
                        <Badge variant="destructive">
                          <AlertTriangle className="size-3" />
                          {d.invalid}
                        </Badge>
                      )}
                      {doc.status && <Badge variant={statusBadgeVariant(doc.status)}>{d.status[doc.status]}</Badge>}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {readError && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{readError}</span>
              </div>
            )}

            <DialogFooter>
              <Button onClick={openCreate} variant="secondary">
                <Plus className="size-3.5" />
                {d.newButton}
              </Button>
            </DialogFooter>
          </>
        )}

        {view === 'create' && (
          <>
            <DialogHeader>
              <DialogTitle>{d.createTitle}</DialogTitle>
              <DialogDescription>{d.createDesc}</DialogDescription>
            </DialogHeader>

            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-(--ui-text-secondary)">{d.fieldTitle}</label>
                <Input
                  onChange={event => setCreateTitle(event.target.value)}
                  placeholder={d.titlePlaceholder}
                  value={createTitle}
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                {CREATE_TEMPLATES.map(template => (
                  <button
                    className={`rounded-md border px-2.5 py-2 text-left text-xs transition ${
                      createDocType === template.docType
                        ? 'border-primary bg-primary/10 text-foreground'
                        : 'border-(--ui-stroke-secondary) text-(--ui-text-secondary) hover:bg-(--chrome-action-hover)'
                    }`}
                    key={template.docType}
                    onClick={() => setCreateDocType(template.docType)}
                    type="button"
                  >
                    {d.docType[template.docType]}
                  </button>
                ))}
              </div>
            </div>

            {createState === 'error' && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{createError ?? d.saveFailed}</span>
              </div>
            )}

            <DialogFooter>
              <Button disabled={createState === 'saving'} onClick={() => setView('list')} variant="ghost">
                {d.back}
              </Button>
              <Button disabled={createState === 'saving'} onClick={() => void handleCreate()}>
                {createState === 'saving' ? <Loader2 className="size-3.5 animate-spin" /> : null}
                {d.newButton}
              </Button>
            </DialogFooter>
          </>
        )}

        {view === 'detail' && selected && (
          <>
            <DialogHeader>
              <DialogTitle>{titleInput || d.titlePlaceholder}</DialogTitle>
              <DialogDescription>
                {d.docType[selected.frontmatter.docType]}
                {dirty ? ` · ${d.dirtyHint}` : ''}
              </DialogDescription>
            </DialogHeader>

            <div className="flex max-h-[60vh] flex-col gap-3 overflow-y-auto pr-1">
              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-(--ui-text-secondary)">{d.fieldTitle}</label>
                <Input onChange={event => setTitleInput(event.target.value)} value={titleInput} />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-(--ui-text-secondary)">{d.fieldStatus}</label>
                <Select
                  onValueChange={value => setStatusInput(value as WorkspaceDocStatus)}
                  value={statusInput}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STATUS_OPTIONS.map(status => (
                      <SelectItem key={status} value={status}>
                        {d.status[status]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-(--ui-text-secondary)">{d.fieldDescription}</label>
                <Input onChange={event => setDescriptionInput(event.target.value)} value={descriptionInput} />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-(--ui-text-secondary)">{d.fieldTags}</label>
                <Input
                  onChange={event => setTagsInput(event.target.value)}
                  placeholder={d.tagsPlaceholder}
                  value={tagsInput}
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-(--ui-text-secondary)">{d.fieldBody}</label>
                <Textarea
                  className="min-h-40"
                  onChange={event => setBodyInput(event.target.value)}
                  value={bodyInput}
                />
              </div>
            </div>

            {saveState === 'error' && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{saveError ?? d.saveFailed}</span>
              </div>
            )}

            {archiveState === 'error' && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{archiveError ?? d.archiveFailed}</span>
              </div>
            )}

            <DialogFooter className="sm:justify-between">
              <Button onClick={() => setView('list')} variant="ghost">
                {d.back}
              </Button>
              <div className="flex gap-2">
                <Button
                  disabled={isArchived || archiveState === 'saving'}
                  onClick={() => void handleArchive()}
                  variant="secondary"
                >
                  {archiveState === 'saving' ? <Loader2 className="size-3.5 animate-spin" /> : null}
                  {d.archive}
                </Button>
                <Button disabled={!dirty || saveState === 'saving'} onClick={() => void handleSave()}>
                  {saveState === 'saving' ? <Loader2 className="size-3.5 animate-spin" /> : null}
                  {d.save}
                </Button>
              </div>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
