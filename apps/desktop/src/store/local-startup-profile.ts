import { withTimeout } from '@/lib/with-timeout'

const STARTUP_PROFILE_READ_TIMEOUT_MS = 5_000

export async function readLocalStartupProfile(): Promise<string | undefined> {
  try {
    return await withTimeout(
      (async () => {
        const desktop = window.hermesDesktop
        const saved = (await desktop?.profile?.get?.())?.profile?.trim()

        if (!saved) {
          return undefined
        }

        const config = await desktop?.getConnectionConfig?.(saved)

        return !config || config.mode === 'local' ? saved : undefined
      })(),
      STARTUP_PROFILE_READ_TIMEOUT_MS,
      'Timed out reading the local startup profile'
    )
  } catch {
    // Older/unavailable IPC keeps the existing bounded source-map fallback.
  }
}
