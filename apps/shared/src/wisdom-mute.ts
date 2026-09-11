export type WisdomMuteDuration = '1_day' | '1_week' | '30_days' | 'forever' | null
export type WisdomMuteSync = 'pending' | 'syncing' | 'synced' | 'failed' | 'conflict' | 'expired'

export interface WisdomMuteSnapshot {
  organization_id: string
  gateway_available: boolean
  mute: null | {
    org_id: string
    muted: boolean
    duration: WisdomMuteDuration
    muted_until: string | null
    forever: boolean
    revision: number
  }
  sync: null | {
    mutation_id: string
    requested_duration: WisdomMuteDuration
    requested_until: number | null
    preference_sync: WisdomMuteSync
    request_expires_at: number
  }
}

export interface WisdomMuteControl {
  id: string
  expires_at: number
  organization_id: string
  mute: NonNullable<WisdomMuteSnapshot['mute']>
  sync: WisdomMuteSnapshot['sync']
}

export interface WisdomMuteState {
  snapshot: WisdomMuteSnapshot | null
  control: WisdomMuteControl | null
  choice: WisdomMuteDuration | undefined
  busy: boolean
  error: boolean
  retryingChoice: boolean
}

export const initialWisdomMuteState: WisdomMuteState = {
  snapshot: null,
  control: null,
  choice: undefined,
  busy: false,
  error: false,
  retryingChoice: false
}

export interface WisdomMuteCopy {
  title: string
  scope: string
  on: string
  muted: string
  day: string
  week: string
  month: string
  forever: string
  pending: string
  failed: string
  conflict: string
  expired: string
}

/** Shared state machine: transport retries keep the original native choice. */
export function createWisdomMuteController(deps: {
  prepare: () => Promise<WisdomMuteControl>
  read: () => Promise<WisdomMuteSnapshot>
  choose: (id: string, duration: WisdomMuteDuration) => Promise<WisdomMuteSnapshot>
  changed: (state: WisdomMuteState) => void
  now?: () => number
}) {
  let state = { ...initialWisdomMuteState }
  let generation = 0
  let disposed = false
  let attempt: { id: string; duration: WisdomMuteDuration; org: string } | null = null
  const now = deps.now ?? Date.now

  const emit = (patch: Partial<WisdomMuteState>) => {
    state = { ...state, ...patch }

    if (!disposed) {
      deps.changed(state)
    }
  }

  const current = (request: number) => !disposed && generation === request

  const refresh = async () => {
    if (disposed || state.busy) {
      return
    }

    const request = ++generation
    attempt = null
    emit({ busy: true, error: false, retryingChoice: false, control: null, choice: undefined })

    try {
      const control = await deps.prepare()

      if (!current(request)) {
        return
      }

      if (control.organization_id !== control.mute.org_id || control.expires_at * 1000 <= now()) {
        throw new Error('Invalid preference control')
      }

      emit({
        control,
        snapshot: {
          organization_id: control.organization_id,
          gateway_available: true,
          mute: control.mute,
          sync: control.sync
        }
      })
    } catch {
      if (current(request)) {
        emit({ error: true })
      }
    } finally {
      if (current(request)) {
        emit({ busy: false })
      }
    }
  }

  return {
    refresh,
    select(choice: WisdomMuteDuration) {
      if (!disposed && !state.busy && !attempt && state.control) {
        emit({ choice })
      }
    },
    async apply() {
      if (disposed || state.busy) {
        return
      }

      if (!attempt) {
        if (state.choice === undefined || !state.control) {
          return
        }

        if (state.control.expires_at * 1000 <= now()) {
          emit({ control: null, error: true, choice: undefined })

          return
        }

        attempt = { id: state.control.id, duration: state.choice, org: state.control.organization_id }
      }

      const selected = attempt
      const request = ++generation
      emit({ busy: true, error: false })

      try {
        const snapshot = await deps.choose(selected.id, selected.duration)

        if (!current(request)) {
          return
        }

        if (snapshot.organization_id !== selected.org || (snapshot.mute && snapshot.mute.org_id !== selected.org)) {
          throw new Error('Preference identity changed')
        }

        attempt = null
        emit({ snapshot, control: null, choice: undefined, retryingChoice: false, busy: false })
        await refresh()
      } catch {
        if (current(request)) {
          emit({ busy: false, error: true, retryingChoice: true })
        }
      }
    },
    async poll() {
      if (disposed || state.busy || attempt || !state.snapshot) {
        return
      }

      const request = ++generation

      try {
        const snapshot = await deps.read()

        if (!current(request)) {
          return
        }

        if (
          snapshot.organization_id !== state.snapshot?.organization_id ||
          (snapshot.mute && snapshot.mute.org_id !== snapshot.organization_id)
        ) {
          emit({ snapshot: null, control: null, choice: undefined, error: true })

          return
        }

        if (!snapshot.gateway_available || !snapshot.mute) {
          emit({ error: true })

          return
        }

        const stale =
          state.control &&
          (state.control.mute.revision !== snapshot.mute.revision || state.control.expires_at * 1000 <= now())

        emit({ snapshot, error: false, ...(stale ? { control: null, choice: undefined } : {}) })
      } catch {
        if (current(request)) {
          emit({ error: true })
        }
      }
    },
    dispose() {
      disposed = true
      generation++
    }
  }
}
