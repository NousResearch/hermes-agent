import { ModelLogo } from '@/components/ui/model-logo'
import type { SessionInfo } from '@/hermes'
import { modelBrand } from '@/lib/model-brand'
import { useStoresSelector } from '@/lib/use-session-slice'
import { $activeSessionId, sessionMatchesStoredId } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'

/** Subscribe only the glyph to its model author, never the row to transcript churn. */
export function SessionModelLogo({ session }: { session: SessionInfo }) {
  const brand = useStoresSelector([$sessionStates, $activeSessionId, $sessionTiles], () => {
    const states = $sessionStates.get()

    const matches = Object.entries(states).filter(
      ([id, state]) =>
        !id.startsWith('read-only:') && state.storedSessionId && sessionMatchesStoredId(session, state.storedSessionId)
    )

    // A reopened conversation can retain several runtime snapshots. Prefer
    // the bound view, not object insertion order or the global composer model.
    const active = matches.find(([id]) => id === $activeSessionId.get())

    const tiled = matches.find(([id]) =>
      $sessionTiles.get().some(tile => tile.runtimeId === id && sessionMatchesStoredId(session, tile.storedSessionId))
    )

    const live = active ?? tiled ?? (matches.length === 1 ? matches[0] : undefined)

    // session.info reports switches (including a queued choice) and the latest
    // reported fallback. The backend may report fallback only at turn end.
    return modelBrand(live?.[1].model || session.model)
  })

  return <ModelLogo brand={brand} />
}
