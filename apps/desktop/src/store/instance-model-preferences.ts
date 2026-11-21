import type { ModelOptionProvider } from '@hermes/shared'
import { atom } from 'nanostores'

import { $apiRequestScope, hermesApi } from '@/api/client'
import {
  type InstanceModelSnapshot,
  parseInstanceModels,
  type PreferenceField,
  preferenceMutation
} from '@/lib/instance-model-preferences'
import { storedString, storedStringArray } from '@/lib/storage'

import { $favoriteModels, favoriteModelKey, setFavoriteWriter } from './favorite-models'
import {
  $knownModels,
  $visibleModels,
  collapseModelFamilies,
  resolveVisibleKeys,
  setVisibilityWriter
} from './model-visibility'

interface PreferenceView {
  scope: string
  status: 'loading' | 'ready' | 'legacy' | 'error'
  busy: boolean
  error: string | null
  snapshot: InstanceModelSnapshot | null
}
export const $instanceModels = atom<PreferenceView>({
  scope: '',
  status: 'loading',
  busy: false,
  error: null,
  snapshot: null
})
let legacyFavorites = storedStringArray('hermes.desktop.favorite-models')
let legacyVisible = storedString('hermes.desktop.visible-models')
let legacyKnown = storedString('hermes.desktop.known-models')
let generation = 0
let stopCurrent: (() => void) | null = null
const failure = (conflict = false) => (conflict ? 'HR-MODEL-002' : 'HR-MODEL-001')

function project(snapshot: InstanceModelSnapshot): void {
  $favoriteModels.set(
    snapshot.rows
      .filter(r => r.favorite)
      .sort((a, b) => (BigInt(a.favoriteRevision) < BigInt(b.favoriteRevision) ? -1 : 1))
      .map(r => favoriteModelKey(r.provider, r.model))
  )
  const overrides = snapshot.rows.filter(r => r.hidden !== null)
  $knownModels.set(new Set(overrides.filter(r => r.model).map(r => favoriteModelKey(r.provider, r.model))))
  $visibleModels.set(
    overrides.length
      ? new Set(
          overrides
            .filter(r => (r.model ? r.hidden === false : r.hidden === true))
            .map(r => favoriteModelKey(r.provider, r.model))
        )
      : null
  )
}

function restoredSet(raw: string | null): Set<string> | null {
  if (!raw) {
    return null
  }

  try {
    const values: unknown = JSON.parse(raw)

    return Array.isArray(values) ? new Set(values.filter((v): v is string => typeof v === 'string')) : null
  } catch {
    return null
  }
}

function legacyVisibilityChoice(
  visible: Set<string> | null,
  known: Set<string> | null,
  key: string,
  model: string
): boolean | undefined {
  if (!visible) {return undefined}

  if (!model) {return visible.has(key) ? true : undefined}

  if (known && !known.has(key) && !visible.has(key)) {return undefined}

  return !visible.has(key)
}

function untouched(snapshot: InstanceModelSnapshot, provider: string, model: string, field: PreferenceField): boolean {
  const existing = snapshot.rows.find(row => row.provider === provider && row.model === model)

  return !existing || existing[field === 'favorite' ? 'favoriteRevision' : 'hiddenRevision'] === '0'
}

export function bindInstanceModels(connectionId?: string, profile?: string): () => void {
  stopCurrent?.()
  const token = ++generation
  const ambient = $apiRequestScope.get()
  const route = { connectionId: connectionId ?? ambient.connectionId, profile: profile ?? ambient.profile }
  const scope = JSON.stringify(route)
  let stopped = false
  let reading = false
  let serial = 0
  const active = () => !stopped && token === generation

  const request = (method: string, body?: unknown) =>
    hermesApi<unknown>({ ...route, method, path: '/api/model/preferences', body })

  const publish = (snapshot: InstanceModelSnapshot) => {
    if (!active()) {
      return
    }

    const previous = $instanceModels.get().snapshot

    if (previous?.instanceId === snapshot.instanceId && BigInt(previous.revision) > BigInt(snapshot.revision)) {
      return
    }

    $instanceModels.set({
      ...$instanceModels.get(),
      status: 'ready',
      snapshot,
      error: $instanceModels.get().error === 'HR-MODEL-001' ? null : $instanceModels.get().error
    })
    setFavoriteWriter(favoriteWriter)
    setVisibilityWriter(visibilityWriter)
    project(snapshot)
  }

  const refresh = async () => {
    if (!active() || reading || $instanceModels.get().busy) {
      return
    }

    reading = true
    const ticket = ++serial

    try {
      const value = parseInstanceModels(await request('GET'))

      if (ticket === serial) {
        publish(value)
      }
    } catch (error) {
      if (!active() || ticket !== serial) {
        return
      }

      // Only identified capability absence enables local legacy writes.
      const status =
        Number((error as { statusCode?: unknown })?.statusCode) ||
        Number(/(?:^|Error: |HTTP )([1-5][0-9]{2})(?=:|$|\s)/.exec(String(error))?.[1])

      const missing = status === 404 || status === 405
      $instanceModels.set({
        ...$instanceModels.get(),
        status: missing ? 'legacy' : 'error',
        error: missing ? null : failure()
      })

      if (missing) {
        legacyFavorites = storedStringArray('hermes.desktop.favorite-models')
        legacyVisible = storedString('hermes.desktop.visible-models')
        legacyKnown = storedString('hermes.desktop.known-models')
        setFavoriteWriter(null)
        setVisibilityWriter(null)
        $favoriteModels.set(legacyFavorites)
        $visibleModels.set(restoredSet(legacyVisible))
        $knownModels.set(restoredSet(legacyKnown))
      }
    } finally {
      reading = false
    }
  }

  const write = async (
    changes: { provider: string; model: string; field: PreferenceField; value: boolean | null; importing?: boolean }[]
  ) => {
    const initial = $instanceModels.get()

    if (!active() || initial.busy || !initial.snapshot) {
      return
    }

    ++serial
    $instanceModels.set({ ...initial, busy: true, error: null })

    try {
      for (const c of changes) {
        if (!active()) {
          return
        }

        const before = $instanceModels.get().snapshot!

        const updated = parseInstanceModels(
          await request('POST', preferenceMutation(before, c.provider, c.model, c.field, c.value, c.importing))
        )

        if (
          updated.instanceId !== before.instanceId ||
          $instanceModels.get().snapshot?.instanceId !== before.instanceId
        ) {
          throw new Error('Instance changed')
        }

        publish(updated)
      }
    } catch (error) {
      if (active()) {
        $instanceModels.set({ ...$instanceModels.get(), error: failure(/409/.test(String(error))) })
      }
    } finally {
      if (active()) {
        $instanceModels.set({ ...$instanceModels.get(), busy: false })
        void refresh()
      }
    }
  }

  $instanceModels.set({ scope, status: 'loading', busy: false, error: null, snapshot: null })
  $favoriteModels.set([])
  $visibleModels.set(null)
  $knownModels.set(null)

  const favoriteWriter = (provider: string, model: string, value: boolean) => {
    void write([{ provider, model, field: 'favorite', value }])
  }

  const visibilityWriter = (keys: Set<string> | null, providers: readonly ModelOptionProvider[]) => {
    const snapshot = $instanceModels.get().snapshot

    if (!snapshot) {
      return
    }

    if (keys === null) {
      void write(
        snapshot.rows
          .filter(r => r.hidden !== null)
          .map(r => ({ provider: r.provider, model: r.model, field: 'hidden', value: null }))
      )

      return
    }

    const before = resolveVisibleKeys($visibleModels.get(), providers)
    const changes: { provider: string; model: string; field: PreferenceField; value: boolean }[] = []

    for (const p of providers) {
      for (const model of ['', ...collapseModelFamilies(p.models ?? []).map(f => f.id)]) {
        const key = favoriteModelKey(p.slug, model)

        if (keys.has(key) !== before.has(key)) {
          changes.push({ provider: p.slug, model, field: 'hidden', value: model ? !keys.has(key) : keys.has(key) })
        }
      }
    }

    void write(changes)
  }

  setFavoriteWriter(favoriteWriter)
  setVisibilityWriter(visibilityWriter)

  importCurrent = providers => {
    const snapshot = $instanceModels.get().snapshot

    if (!snapshot || !active()) {
      return
    }

    const catalog = providers.flatMap(p =>
      ['', ...collapseModelFamilies(p.models ?? []).map(f => f.id)].map(model => ({
        provider: p.slug,
        model,
        key: favoriteModelKey(p.slug, model)
      }))
    )

    const visible = restoredSet(legacyVisible)
    const known = restoredSet(legacyKnown)
    const keys = new Set([...legacyFavorites, ...(known ?? []), ...(visible ?? [])])

    if (visible && !known) {
      for (const r of catalog) {
        if (r.model) {
          keys.add(r.key)
        }
      }
    }

    const changes: { provider: string; model: string; field: PreferenceField; value: boolean; importing: boolean }[] =
      []

    for (const key of keys) {
      const matches = catalog.filter(r => r.key === key)

      if (matches.length !== 1) {
        continue
      }

      const { provider, model } = matches[0]

      if (model && legacyFavorites.includes(key) && untouched(snapshot, provider, model, 'favorite')) {
        changes.push({ provider, model, field: 'favorite', value: true, importing: true })
      }

      const hidden = legacyVisibilityChoice(visible, known, key, model)

      if (hidden !== undefined && untouched(snapshot, provider, model, 'hidden')) {
        changes.push({ provider, model, field: 'hidden', value: hidden, importing: true })
      }
    }

    void write(changes)
  }

  retryCurrent = () => {
    void refresh()
  }

  void refresh()

  const timer = setInterval(() => {
    if (document.visibilityState === 'visible') {
      void refresh()
    }
  }, 5000)

  const stop = () => {
    stopped = true
    clearInterval(timer)

    if (token === generation) {
      setFavoriteWriter(null)
      setVisibilityWriter(null)
    }
  }

  stopCurrent = stop

  return stop
}

let importCurrent: (providers: readonly ModelOptionProvider[]) => void = () => {}

let retryCurrent = () => {}
export const importPreviousPreferences = (providers: readonly ModelOptionProvider[]) => importCurrent(providers)
export const refreshInstanceModels = () => retryCurrent()
export const hasPreviousPreferences = () => legacyFavorites.length > 0 || legacyVisible !== null
