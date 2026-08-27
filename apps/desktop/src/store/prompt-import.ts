/**
 * Hot-import / PE-save path for prompt templates (local dogfood).
 *
 * Agent or scripts write a job JSON; the desktop store merges it live — no
 * rebuild, no seed edit. Static dedupe only (normalize + token overlap); no LLM.
 */
import {
  $promptTemplates,
  addFolder,
  addTemplate,
  type PromptTemplate,
  repairTree
} from './prompt-templates'

export const PROMPT_IMPORT_MARK_DUP = '〔重复〕'
export const PROMPT_IMPORT_MARK_NEAR = '〔相近〕'

export interface PromptImportTemplateIn {
  description?: string
  folder?: string
  label: string
  text: string
}

export interface PromptImportJob {
  /** Optional folder description when creating a named folder. */
  folder?: string
  folderDescription?: string
  /** merge (default) appends; replace wipes the store first. */
  op?: 'merge' | 'replace'
  source?: string
  templates?: PromptImportTemplateIn[]
  v?: number
}

export interface PromptImportResult {
  created: number
  foldersEnsured: string[]
  near: number
  skippedDup: number
}

function setList(next: PromptTemplate[]): void {
  $promptTemplates.set(repairTree(next))
}

/** Strip import marks and whitespace noise for comparison. */
export function normalizePromptLabel(label: string): string {
  return label
    .replace(/\s*〔[^〕]+〕\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

export function normalizePromptText(text: string): string {
  return text
    .replace(/\u00a0/g, ' ')
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+/g, ' ')
    .trim()
    .toLowerCase()
}

function tokenize(s: string): Set<string> {
  const out = new Set<string>()
  // Latin/num runs length>1
  for (const m of s.matchAll(/[a-z0-9]{2,}/gi)) {
    out.add(m[0].toLowerCase())
  }
  // CJK: unigrams + bigrams so short titles still overlap
  const cjk = [...s].filter(ch => /[\u4e00-\u9fff]/.test(ch))
  for (const ch of cjk) {
    out.add(ch)
  }
  for (let i = 0; i < cjk.length - 1; i++) {
    out.add(cjk[i] + cjk[i + 1])
  }
  return out
}

/** Jaccard on tokens; 0..1. Cheap, no model. */
export function tokenJaccard(a: string, b: string): number {
  const A = tokenize(a)
  const B = tokenize(b)

  if (A.size === 0 && B.size === 0) {
    return 1
  }

  if (A.size === 0 || B.size === 0) {
    return 0
  }

  let inter = 0

  for (const t of A) {
    if (B.has(t)) {
      inter += 1
    }
  }

  return inter / (A.size + B.size - inter)
}

function stripMark(label: string): string {
  return label.replace(/\s*〔[^〕]+〕\s*/g, ' ').replace(/\s+/g, ' ').trim()
}

function withMark(label: string, mark: string): string {
  const base = stripMark(label)

  if (!base) {
    return mark
  }

  if (base.includes(mark)) {
    return base
  }

  return `${base}  ${mark}`
}

function ensureFolderByLabel(label: string, description = ''): string {
  const list = $promptTemplates.get()
  const existing = list.find(
    n => n.kind === 'folder' && n.parentId === null && normalizePromptLabel(n.label) === normalizePromptLabel(label)
  )

  if (existing) {
    return existing.id
  }

  return addFolder(label || 'Folder', null).id
}

/**
 * Apply one import job against the live store.
 * - exact normalized text vs existing template → skip (skippedDup++)
 * - label jaccard ≥ 0.5 or text jaccard ≥ 0.72 → keep, mark 〔相近〕 / 〔重复〕
 */
export function applyPromptImportJob(job: PromptImportJob): PromptImportResult {
  const result: PromptImportResult = { created: 0, foldersEnsured: [], near: 0, skippedDup: 0 }

  if (job.op === 'replace') {
    setList([])
  }

  const templates = Array.isArray(job.templates) ? job.templates : []
  const defaultFolder = typeof job.folder === 'string' && job.folder.trim() ? job.folder.trim() : null

  if (defaultFolder) {
    const id = ensureFolderByLabel(defaultFolder, job.folderDescription || '')
    if (!result.foldersEnsured.includes(defaultFolder)) {
      result.foldersEnsured.push(defaultFolder)
    }
    void id
  }

  for (const raw of templates) {
    if (!raw || typeof raw.label !== 'string' || typeof raw.text !== 'string') {
      continue
    }

    const label = stripMark(raw.label).trim()
    const text = raw.text
    const description = typeof raw.description === 'string' ? raw.description : ''
    const folderLabel =
      typeof raw.folder === 'string' && raw.folder.trim()
        ? raw.folder.trim()
        : defaultFolder

    if (!label && !text.trim()) {
      continue
    }

    const normText = normalizePromptText(text)
    const normLabel = normalizePromptLabel(label)
    const existing = $promptTemplates.get().filter(n => n.kind === 'template')

    const exact = existing.find(n => normalizePromptText(n.text) === normText && normText.length > 0)

    if (exact) {
      result.skippedDup += 1
      continue
    }

    let mark: string | null = null
    let bestText = 0
    let bestLabel = 0

    for (const n of existing) {
      bestText = Math.max(bestText, tokenJaccard(normText, normalizePromptText(n.text)))
      bestLabel = Math.max(bestLabel, tokenJaccard(normLabel, normalizePromptLabel(n.label)))
    }

    if (bestText >= 0.85) {
      mark = PROMPT_IMPORT_MARK_DUP
      result.near += 1
    } else if (bestText >= 0.72 || bestLabel >= 0.5) {
      mark = PROMPT_IMPORT_MARK_NEAR
      result.near += 1
    }

    let parentId: string | null = null

    if (folderLabel) {
      parentId = ensureFolderByLabel(folderLabel, folderLabel === defaultFolder ? job.folderDescription || '' : '')
      if (!result.foldersEnsured.includes(folderLabel)) {
        result.foldersEnsured.push(folderLabel)
      }
    }

    const finalLabel = mark ? withMark(label || 'untitled', mark) : label || 'untitled'
    const finalDesc =
      mark && description && !description.includes('〔与库内')
        ? `〔与库内${mark === PROMPT_IMPORT_MARK_DUP ? '重复' : '相近'}〕${description}`
        : description

    addTemplate(finalLabel, finalDesc, text, parentId)
    result.created += 1
  }

  return result
}

/** Parse unknown JSON into a job; returns null if unusable. */
export function parsePromptImportJob(raw: unknown): PromptImportJob | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null
  }

  const o = raw as Record<string, unknown>
  const op = o.op === 'replace' ? 'replace' : 'merge'
  const templates = Array.isArray(o.templates) ? o.templates : []
  const cleaned: PromptImportTemplateIn[] = []

  for (const item of templates) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const t = item as Record<string, unknown>

    if (typeof t.label !== 'string' || typeof t.text !== 'string') {
      continue
    }

    cleaned.push({
      label: t.label,
      text: t.text,
      description: typeof t.description === 'string' ? t.description : '',
      folder: typeof t.folder === 'string' ? t.folder : undefined
    })
  }

  // replace with empty templates is valid (clear)
  if (op === 'merge' && cleaned.length === 0 && typeof o.folder !== 'string') {
    return null
  }

  return {
    v: typeof o.v === 'number' ? o.v : 1,
    op,
    source: typeof o.source === 'string' ? o.source : undefined,
    folder: typeof o.folder === 'string' ? o.folder : undefined,
    folderDescription: typeof o.folderDescription === 'string' ? o.folderDescription : undefined,
    templates: cleaned
  }
}
