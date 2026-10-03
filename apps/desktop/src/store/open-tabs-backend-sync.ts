/**
 * Push and pull the open tab strip against the connected backend.
 *
 * Local connection-scoped storage still restores tabs when this device
 * switches gateways. The backend file is what a *different* device reads, so
 * the strip follows the sessions instead of the laptop. A failed GET never
 * PUTs: an old backend or a blip must not be treated as "no tabs".
 */

import { hermesApi, profileScoped } from '@/api/client'
import {
  canonicalizeOpenTabs,
  decideOpenTabSync,
  type OpenTabsDocument
} from '@/lib/open-tabs-sync'
import { readJson, writeJson } from '@/lib/storage'

import { $activeGatewayProfile } from './profile'
import { $connection } from './session'
import {
  adoptRemoteOpenTabs,
  currentOpenSessionTabs,
  openTabsSyncScopeKey,
  setOpenTabsPersistListener
} from './session-states'
import { isBrowserWindow, isSecondaryWindow } from './windows'

const SYNC_KEY = 'hermes.desktop.openTabsSync.v1'
const PUSH_DELAY_MS = 400

const hydratedScopes = new Set<string>()
let mutationGen = 0
let hydrateToken = 0
let gatewayOpen = false
let pushTimer: ReturnType<typeof setTimeout> | null = null
let installed = false

function appliedRevision(scope: string): null | number {
  const parsed = readJson<Record<string, unknown>>(SYNC_KEY)
  const value = parsed?.[scope]

  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
}

function rememberRevision(scope: string, revision: number): void {
  const parsed = readJson<Record<string, number>>(SYNC_KEY) ?? {}

  writeJson(SYNC_KEY, { ...parsed, [scope]: revision })
}

function clearPushTimer(): void {
  if (pushTimer) {
    clearTimeout(pushTimer)
    pushTimer = null
  }
}

async function readRemote(): Promise<null | OpenTabsDocument> {
  try {
    const document = await hermesApi<OpenTabsDocument>({
      ...profileScoped(),
      path: '/api/desktop/open-tabs',
      timeoutMs: 15_000
    })

    if (!document || !Number.isInteger(document.revision) || !Array.isArray(document.tiles)) {
      return null
    }

    return document
  } catch {
    return null
  }
}

async function flushPush(scope: string): Promise<void> {
  if (!gatewayOpen || scope !== openTabsSyncScopeKey() || isSecondaryWindow() || isBrowserWindow()) {
    return
  }

  const base = appliedRevision(scope) ?? 0

  try {
    const saved = await hermesApi<OpenTabsDocument>({
      ...profileScoped(),
      body: { base_revision: base, tiles: canonicalizeOpenTabs(currentOpenSessionTabs()) },
      method: 'PUT',
      path: '/api/desktop/open-tabs',
      timeoutMs: 15_000
    })

    if (scope === openTabsSyncScopeKey() && Number.isInteger(saved?.revision)) {
      rememberRevision(scope, saved.revision)
    }
  } catch {
    const current = await readRemote()

    if (!current || scope !== openTabsSyncScopeKey() || current.revision === base) {
      return
    }

    adoptRemoteOpenTabs(canonicalizeOpenTabs(current.tiles))
    rememberRevision(scope, current.revision)
  }
}

function schedulePush(): void {
  const scope = openTabsSyncScopeKey()

  if (!hydratedScopes.has(scope)) {
    return
  }

  clearPushTimer()
  pushTimer = setTimeout(() => {
    pushTimer = null
    void flushPush(scope)
  }, PUSH_DELAY_MS)
}

export async function syncOpenTabsFromBackend(): Promise<void> {
  if (!gatewayOpen || isSecondaryWindow() || isBrowserWindow()) {
    return
  }

  const scope = openTabsSyncScopeKey()
  const gen = mutationGen
  const token = ++hydrateToken
  const remote = await readRemote()

  if (token !== hydrateToken || scope !== openTabsSyncScopeKey()) {
    return
  }

  if (mutationGen !== gen) {
    hydratedScopes.add(scope)
    schedulePush()

    return
  }

  const decision = decideOpenTabSync({
    localAppliedRevision: appliedRevision(scope),
    localTiles: currentOpenSessionTabs(),
    remote
  })

  if (decision.kind === 'stay') {
    return
  }

  hydratedScopes.add(scope)

  if (decision.kind === 'adopt') {
    adoptRemoteOpenTabs(decision.tiles)
    rememberRevision(scope, decision.revision)

    return
  }

  if (decision.kind === 'noop') {
    rememberRevision(scope, decision.revision)

    return
  }

  await flushPush(scope)
}

export function setOpenTabsSyncGateway(open: boolean): void {
  gatewayOpen = open

  if (!open) {
    clearPushTimer()
    hydrateToken += 1

    return
  }

  void syncOpenTabsFromBackend()
}

export function installOpenTabsBackendSync(): void {
  if (installed || isSecondaryWindow() || isBrowserWindow()) {
    return
  }

  installed = true
  setOpenTabsPersistListener(() => {
    mutationGen += 1
    schedulePush()
  })

  const onScope = () => {
    if (gatewayOpen) {
      void syncOpenTabsFromBackend()
    }
  }

  $activeGatewayProfile.subscribe(onScope)
  $connection.subscribe((connection: unknown) => {
    if (connection) {
      onScope()
    }
  })
}

