import { atom, computed } from 'nanostores'

import { sha256Text } from '@/app/chat/right-rail/sandboxed-html-approval'
import { getStatus } from '@/hermes'
import {
  desktopFsCacheKey,
  isDesktopFsRemoteMode,
  readDesktopDir,
  readDesktopFileText
} from '@/lib/desktop-fs'
import { readJson, writeJson } from '@/lib/storage'

import {
  GENERATED_VIEW_HTML_MAX_BYTES,
  GENERATED_VIEW_ID_PATTERN,
  GENERATED_VIEW_MANIFEST_MAX_BYTES,
  generatedViewApprovalSource,
  generatedViewEntryPath,
  type GeneratedViewManifest,
  generatedViewPathIsContained,
  parseGeneratedViewManifest
} from './manifest'

export interface GeneratedViewDocument {
  connectionKey: string
  digest: string
  directory: string
  entryPath: string
  html: string
  manifest: GeneratedViewManifest
  manifestPath: string
}

export interface GeneratedViewProblem {
  id?: string
  message: string
  path: string
}

const OPEN_VIEWS_KEY = 'hermes.desktop.generatedViews.open.v1'
const DISCOVERY_INTERVAL_MS = 5_000

export const $generatedViews = atom<GeneratedViewDocument[]>([])
export const $generatedViewProblems = atom<GeneratedViewProblem[]>([])

let activeOpenScope: string | null = null

function readOpenRecord(): Record<string, string[]> {
  const parsed = readJson<unknown>(OPEN_VIEWS_KEY)

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(parsed).flatMap(([key, ids]) =>
      Array.isArray(ids)
        ? [[key, ids.filter((id): id is string => typeof id === 'string' && GENERATED_VIEW_ID_PATTERN.test(id))]]
        : []
    )
  )
}

export const $openGeneratedViewIds = atom<string[]>([])

export const $openGeneratedViews = computed(
  [$generatedViews, $openGeneratedViewIds],
  (views, ids) => ids.flatMap(id => views.find(view => view.manifest.id === id) ?? [])
)

function generatedViewOpenScope(connectionKey: string, root: string): string {
  // The local API base URL is established after the shell starts and can change
  // during that first connection handshake. The canonical Hermes home is the
  // durable local identity; remote profiles retain their full connection key.
  return `${isDesktopFsRemoteMode() ? connectionKey : 'local'}\u0000${root}`
}

function setActiveOpenScope(scope: string) {
  if (activeOpenScope === scope) {
    const persisted = readOpenRecord()[scope] ?? []

    if ($openGeneratedViewIds.get().length === 0 && persisted.length > 0) {
      $openGeneratedViewIds.set(persisted)
    }

    return
  }

  activeOpenScope = scope
  $openGeneratedViewIds.set(readOpenRecord()[scope] ?? [])
}

function saveOpenIds(ids: string[]) {
  if (!activeOpenScope) {
    return
  }

  const record = readOpenRecord()
  const next = [...new Set(ids)].filter(id => GENERATED_VIEW_ID_PATTERN.test(id))

  if (next.length > 0) {
    record[activeOpenScope] = next
  } else {
    delete record[activeOpenScope]
  }

  $openGeneratedViewIds.set(next)
  writeJson(OPEN_VIEWS_KEY, Object.keys(record).length > 0 ? record : null)
}

export function openGeneratedView(id: string): void {
  if (!GENERATED_VIEW_ID_PATTERN.test(id) || !$generatedViews.get().some(view => view.manifest.id === id)) {
    return
  }

  const current = $openGeneratedViewIds.get()

  if (!current.includes(id)) {
    saveOpenIds([...current, id])
  }
}

export function closeGeneratedView(id: string): void {
  saveOpenIds($openGeneratedViewIds.get().filter(openId => openId !== id))
}

function utf8Bytes(value: string): number {
  return new TextEncoder().encode(value).byteLength
}

async function readTextFile(path: string, maxBytes: number, label: string) {
  const result = await readDesktopFileText(path)

  if (result.binary) {
    throw new Error(`${label} is binary`)
  }

  if (result.truncated) {
    throw new Error(`${label} was truncated`)
  }

  if (!result.path || typeof result.path !== 'string' || typeof result.text !== 'string') {
    throw new Error(`${label} is unreadable`)
  }

  const actualBytes = utf8Bytes(result.text)

  if ((result.byteSize ?? actualBytes) > maxBytes || actualBytes > maxBytes) {
    throw new Error(`${label} exceeds ${maxBytes} bytes`)
  }

  return { path: result.path, text: result.text }
}

async function loadGeneratedView(directoryPath: string, directoryName: string, connectionKey: string) {
  if (!GENERATED_VIEW_ID_PATTERN.test(directoryName)) {
    throw new Error('generated-view directory name is invalid')
  }

  const manifestRequestPath = `${directoryPath.replace(/[\\/]+$/, '')}/view.json`
  const manifestFile = await readTextFile(manifestRequestPath, GENERATED_VIEW_MANIFEST_MAX_BYTES, 'view.json')
  const manifest = parseGeneratedViewManifest(manifestFile.text, directoryName)
  const canonicalDirectory = manifestFile.path.replace(/[\\/][^\\/]+$/, '')
  const entryRequestPath = generatedViewEntryPath(canonicalDirectory, manifest.entry)
  const entryFile = await readTextFile(entryRequestPath, GENERATED_VIEW_HTML_MAX_BYTES, manifest.entry)

  if (!generatedViewPathIsContained(canonicalDirectory, entryFile.path)) {
    throw new Error('view.json entry resolved outside its generated-view directory')
  }

  const digest = await sha256Text(generatedViewApprovalSource(manifest, entryFile.text))

  return {
    connectionKey,
    digest,
    directory: canonicalDirectory,
    entryPath: entryFile.path,
    html: entryFile.text,
    manifest,
    manifestPath: manifestFile.path
  } satisfies GeneratedViewDocument
}

let scanPromise: Promise<void> | null = null
const watches = new Map<string, string>()

async function reconcileWatches(views: GeneratedViewDocument[]): Promise<void> {
  const desktop = window.hermesDesktop

  if (!desktop?.watchPreviewFile || isDesktopFsRemoteMode()) {
    return
  }

  const wanted = new Set(views.flatMap(view => [view.manifestPath, view.entryPath]))

  for (const [path, watchId] of watches) {
    if (!wanted.has(path)) {
      watches.delete(path)
      void desktop.stopPreviewFileWatch(watchId)
    }
  }

  for (const path of wanted) {
    if (watches.has(path)) {
      continue
    }

    try {
      watches.set(path, (await desktop.watchPreviewFile(path)).id)
    } catch {
      // Visible-tab polling remains the fallback for an unwatchable file.
    }
  }
}

async function scanGeneratedViews(): Promise<void> {
  const connectionKey = desktopFsCacheKey()

  try {
    const { hermes_home } = await getStatus()
    const root = `${hermes_home.replace(/[\\/]+$/, '')}/generated-views`
    setActiveOpenScope(generatedViewOpenScope(connectionKey, root))
    const documentConnectionKey = isDesktopFsRemoteMode() ? connectionKey : 'local:'
    const { entries, error } = await readDesktopDir(root)

    if (error) {
      throw new Error(error)
    }

    const settled = await Promise.allSettled(
      entries
        .filter(entry => entry.isDirectory)
        .map(async entry => ({
          id: entry.name,
          path: entry.path,
          view: await loadGeneratedView(entry.path, entry.name, documentConnectionKey)
        }))
    )

    const views: GeneratedViewDocument[] = []
    const problems: GeneratedViewProblem[] = []

    settled.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        views.push(result.value.view)
      } else {
        const entry = entries.filter(item => item.isDirectory)[index]
        problems.push({
          id: entry?.name,
          message: result.reason instanceof Error ? result.reason.message : String(result.reason),
          path: entry?.path ?? root
        })
      }
    })

    views.sort((left, right) => left.manifest.title.localeCompare(right.manifest.title))
    const validIds = new Set(views.map(view => view.manifest.id))
    const openIds = $openGeneratedViewIds.get()
    const retainedOpenIds = openIds.filter(id => validIds.has(id))

    if (retainedOpenIds.length !== openIds.length) {
      saveOpenIds(retainedOpenIds)
    }

    $generatedViews.set(views)
    $generatedViewProblems.set(problems)
    await reconcileWatches(views)
  } catch (error) {
    $generatedViewProblems.set([
      {
        message: error instanceof Error ? error.message : String(error),
        path: 'generated-views'
      }
    ])
  }
}

/** Rescan now; concurrent callers share one filesystem pass. */
export function discoverGeneratedViews(): Promise<void> {
  if (!scanPromise) {
    scanPromise = scanGeneratedViews().finally(() => {
      scanPromise = null
    })
  }

  return scanPromise
}

let watching = false

/** Start content watching plus a slow visible-tab directory reconciliation. */
export function watchGeneratedViewDocuments(): Promise<void> {
  if (watching) {
    return discoverGeneratedViews()
  }

  watching = true
  window.hermesDesktop?.onPreviewFileChanged(({ id }) => {
    if ([...watches.values()].includes(id)) {
      void discoverGeneratedViews()
    }
  })
  const initialDiscovery = discoverGeneratedViews()
  window.setInterval(() => {
    if (document.visibilityState === 'visible') {
      void discoverGeneratedViews()
    }
  }, DISCOVERY_INTERVAL_MS)

  return initialDiscovery
}
