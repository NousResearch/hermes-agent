import { randomUuid } from '@hermes/shared'
import { atom } from 'nanostores'

import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { readFileDataUrlForAttach } from '@/app/session/hooks/use-prompt-actions/utils'
import { desktopSessionCreateParams } from '@/app/session/hooks/use-session-actions/session-create-request'

import type { DurableSendOwner, DurableSendRecord, DurableSendTarget } from '../../electron/durable-send-types'

import {
  type ComposerAttachment,
  isFreshDraftScope,
  savedDraftAt,
  stashSessionDraft,
  takeSessionDraft
} from './composer'
import { leaseGatewayForAgent, requestGatewayForAgent } from './gateway'
import { resolveNewChatOwnerRoute } from './profile'
import { $projectScope } from './project-scope'
import { resolveNewSessionCwd } from './projects'
import { $currentCwd, $newChatWorkspaceTarget } from './session'
import { isSessionOwnerRoute, type SessionOwnerRoute } from './session-request-router'
import { isBotChatSession, knownOwnerForSession } from './session-states'

export const $durableOwnerVersion = atom(0)
export const $durableSends = atom<DurableSendRecord[]>([])
const CACHE = 'hermes.durableSend.owners.v1'
const RETIRE = 'hermes.durableSend.retire.v1:'
const owners = new Map<string, DurableSendOwner>()

const key = (route: Pick<DurableSendOwner, 'connectionId' | 'profile'>) =>
  JSON.stringify([route.connectionId, route.profile])

let started = false
let refreshing = false

export function durableComposerRoute(scope: string): SessionOwnerRoute | null {
  if (isBotChatSession(scope)) {
    return null
  }

  const route = isFreshDraftScope(scope) ? resolveNewChatOwnerRoute() : knownOwnerForSession(scope)

  return isSessionOwnerRoute(route) && (!route.targetProfile || route.targetProfile === route.profile) ? route : null
}

export function durableComposerOwner(scope: string): DurableSendOwner | undefined {
  if (!window.hermesDesktop?.durableSend) {
    return undefined
  }

  const route = durableComposerRoute(scope)

  return route ? owners.get(key(route)) : undefined
}

export async function refreshDurableSends(): Promise<void> {
  const api = window.hermesDesktop?.durableSend

  if (!api) {
    return
  }

  const records = await api.list()

  if (api.retired) {
    const ids = Object.keys(localStorage).filter(k => k.startsWith(RETIRE)).map(k => k.slice(RETIRE.length))

    for (let start = 0; start < ids.length; start += 128) {
      const retired = await api.retired(ids.slice(start, start + 128))

      for (const [id, state] of Object.entries(retired)) {
        const markerKey = RETIRE + id

        try {
          const marker = JSON.parse(localStorage.getItem(markerKey) || 'null')
          const draft = takeSessionDraft(marker.scope)

          if (state === 'confirmed' && draft.text === marker.text && savedDraftAt(marker.scope) === marker.savedAt &&
              Array.isArray(marker.occurrences) && draft.attachments.every(file => marker.occurrences.some(
                (sent: {id: string; occurrenceId?: string}) => sent.id === file.id && sent.occurrenceId === file.occurrenceId))) {
            stashSessionDraft(marker.scope, '', [])
          }

          localStorage.removeItem(markerKey)
        } catch { /* Never clear a different draft on malformed handoff. */ }
      }
    }
  }

  // A process can close after native commit but before clearing the DOM/stash. Resolve that
  // exact handoff before mounting the editor on restart; never resurrect a queued send as a draft.
  for (const record of records) {
    if (record.state === 'staging') {
      continue
    }

    const markerKey = RETIRE + record.intent.id
    const raw = localStorage.getItem(markerKey)

    if (!raw) {
      continue
    }

    try {
      const marker = JSON.parse(raw)
      const scope = 'sessionId' in record.intent.target ? record.intent.target.sessionId : record.intent.target.draftId
      const draft = takeSessionDraft(scope)

      if (
        record.state !== 'cancelled' &&
        marker.scope === scope &&
        draft.text === marker.text &&
        savedDraftAt(scope) === marker.savedAt &&
        Array.isArray(marker.occurrences) &&
        draft.attachments.every(file =>
          marker.occurrences.some(
            (sent: { id: string; occurrenceId?: string }) =>
              sent.id === file.id && sent.occurrenceId === file.occurrenceId
          )
        )
      ) {
        stashSessionDraft(scope, '', [])
      }

      localStorage.removeItem(markerKey)
    } catch {
      /* Leave an unreadable handoff for recovery, never overwrite another draft. */
    }
  }

  $durableSends.set(records)
}

export async function warmDurableOwner(scope: string): Promise<void> {
  const api = window.hermesDesktop?.durableSend
  const route = durableComposerRoute(scope)

  if (!api || !route) {
    return
  }

  try {
    const owner = await api.negotiate({ connectionId: route.connectionId, profile: route.profile })
    owners.set(key(route), owner)
    $durableOwnerVersion.set($durableOwnerVersion.get() + 1)
    localStorage.setItem(CACHE, JSON.stringify([...owners.entries()]))
  } catch {
    // A previously authenticated identity may queue offline. Every delivery re-negotiates it.
    // An unrecognized/legacy owner stays on the existing fail-closed submission path.
  }
}

export function startDurableSends(): void {
  const api = window.hermesDesktop?.durableSend

  if (!api || started) {
    return
  }

  started = true

  try {
    for (const [cacheKey, owner] of JSON.parse(localStorage.getItem(CACHE) || '[]')) {
      if (
        owner &&
        ['connectionId', 'profile', 'gatewayId', 'principalId'].every(k => typeof owner[k] === 'string') &&
        key(owner) === cacheKey
      ) {
        owners.set(cacheKey, owner)
      }
    }
  } catch {
    /* Broken hints never authorize a delivery. */
  }

  $durableOwnerVersion.set($durableOwnerVersion.get() + 1)

  const leases = new Map<string, {
    route: string; lease: Awaited<ReturnType<typeof leaseGatewayForAgent>>; timer: ReturnType<typeof setTimeout>
  }>()

  const release = (id: string) => {
    const entry = leases.get(id)

    if (entry) { clearTimeout(entry.timer); entry.lease.release(); leases.delete(id) }
  }

  window.addEventListener('pagehide', () => { for (const id of leases.keys()) {release(id)} })
  api.onChange?.(() => { void refreshDurableSends() })
  api.onTransportRequest(async ({ route, method, params, leaseId }) => {
    if (method === 'transport.release') {
      if (leaseId && leases.get(leaseId)?.route === key(route)) {release(leaseId)}

      return null
    }

    if (method === 'transport.acquire') {
      if (!leaseId || leases.has(leaseId) || leases.size >= 8) {throw new Error('Invalid transport lease')}
      const lease = await leaseGatewayForAgent(route.connectionId, route.profile)
      leases.set(leaseId, { route: key(route), lease, timer: setTimeout(() => release(leaseId), 60_000) })

      return null
    }

    if (
      !['durable.describe', 'durable.create', 'session.resume', 'durable.upload', 'durable.submit', 'durable.cancel'].includes(method)
    ) {
      throw new Error('Durable transport method denied')
    }

    if (leaseId) {
      const entry = leases.get(leaseId)

      if (!entry || entry.route !== key(route)) {throw new Error('Transport lease unavailable')}
      clearTimeout(entry.timer)
      entry.timer = setTimeout(() => release(leaseId), 60_000)

      return entry.lease.request(method, { ...params, profile: route.profile })
    }

    if (method !== 'durable.describe') {throw new Error('Transport lease required')}

    return requestGatewayForAgent(route.connectionId, route.profile, method, { ...params, profile: route.profile })
  })

  const tick = async () => {
    if (refreshing) {
      return
    }

    refreshing = true

    try {
      await refreshDurableSends()
      await api.pump()
      await refreshDurableSends()
    } catch {
      /* Next reconnect/poll resumes the durable state, never a new identity. */
    } finally {
      refreshing = false
    }
  }

  window.addEventListener('online', () => {
    void tick()
  })
  window.setInterval(() => {
    void tick()
  }, 2000)
  void tick()
}

/** Snapshot synchronously, then copy original bytes before any composer-clear or network send. */
export async function enqueueDurableComposer(
  scope: string,
  text: string,
  attachments: ComposerAttachment[],
  owner: DurableSendOwner,
  draftText = text
): Promise<DurableSendRecord> {
  const api = window.hermesDesktop!.durableSend!
  const route = durableComposerRoute(scope)

  if (!route || key(route) !== key(owner)) {
    throw new Error('Destino da mensagem mudou')
  }

  const id = randomUuid()
  const savedAt = savedDraftAt(scope)
  let target: DurableSendTarget

  if (!isFreshDraftScope(scope)) {
    target = { sessionId: scope }
  } else {
    const existing = $durableSends
      .get()
      .find(
        entry =>
          !('sessionId' in entry.intent.target) &&
          entry.intent.target.draftId === scope &&
          key(entry.intent.owner) === key(owner) &&
          entry.intent.owner.gatewayId === owner.gatewayId &&
          entry.intent.owner.principalId === owner.principalId
      )

    if (existing) {
      target = existing.intent.target
    } else {
      const workspace = $newChatWorkspaceTarget.get()

      const cwd =
        workspace === null || (workspace === undefined && $projectScope.get() === NO_PROJECT_ID)
          ? ''
          : typeof workspace === 'string'
            ? workspace.trim()
            : $currentCwd.get().trim() || resolveNewSessionCwd()

      const params = await desktopSessionCreateParams(cwd, route, route.profile, false, true, false)
      target = {
        draftId: scope,
        createRequestId: randomUuid(),
        params: params as Record<string, string | number | boolean>
      }
    }
  }

  if (attachments.length > 32) {
    throw new Error('O limite é 32 anexos por mensagem')
  }

  const staged = []
  let totalBytes = 0

  for (const attachment of attachments) {
    if (!['file', 'image'].includes(attachment.kind) || !attachment.path || attachment.stagedUpload) {
      throw new Error('Este anexo não permite cópia local durável. A mensagem não foi enviada.')
    }

    const data = await readFileDataUrlForAttach(attachment.durableSourcePath || attachment.path)

    if (!data || !/^data:[^,]*;base64,/.test(data)) {
      throw new Error('Não foi possível preservar o arquivo')
    }

    const comma = data.indexOf(',')
    const mediaType = attachment.mime || data.slice(5, data.indexOf(';')) || 'application/octet-stream'
    const encoded = data.slice(comma + 1)
    const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0))
    totalBytes += bytes.length

    if (bytes.length > 64 * 1024 * 1024 || totalBytes > 256 * 1024 * 1024) {
      throw new Error('Limite de anexos excedido (64 MiB por arquivo; 256 MiB por mensagem)')
    }

    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte =>
      byte.toString(16).padStart(2, '0')
    ).join('')

    staged.push({ bytes, file: { id: randomUuid(), name: attachment.label, mediaType, size: bytes.length, sha256 } })
  }

  await api.begin({ id, owner, target, text, files: staged.map(entry => entry.file) })

  try {
    for (const { bytes, file } of staged) {
      for (let offset = 0; offset < bytes.length; offset += 1024 * 1024) {
        await api.append(id, file.id, offset, bytes.subarray(offset, offset + 1024 * 1024))
      }
    }

    localStorage.setItem(
      RETIRE + id,
      JSON.stringify({
        scope,
        text: draftText,
        savedAt,
        occurrences: attachments.map(({ id, occurrenceId }) => ({ id, occurrenceId }))
      })
    )
    const record = await api.commit(id)
    $durableSends.set([...$durableSends.get().filter(entry => entry.intent.id !== id), record])

    return record
  } catch (error) {
    await api.cancel(id).catch(() => {})
    throw error
  }
}
