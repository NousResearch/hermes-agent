import { translateNow } from '@/i18n'
import { withTimeout } from '@/lib/with-timeout'
import { refreshConnectionsRegistry } from '@/store/connections'
import { notifyError } from '@/store/notifications'
import { applyEverythingUpdate, setUpdateOverlayOpen } from '@/store/updates'

/** The launch action always means all registered targets, including during
 * cold boot before a remote primary has connected. Registry reads are local;
 * never wait for that primary to become reachable before updating other hosts. */
export async function updateAllFromLauncher(): Promise<void> {
  try {
    const registry = await withTimeout(
      refreshConnectionsRegistry(),
      10_000,
      translateNow('updates.everythingFanoutFailedTitle')
    )

    if (!registry) {
      throw new Error(translateNow('updates.everythingFanoutFailedTitle'))
    }

    setUpdateOverlayOpen(true)
    await applyEverythingUpdate()
  } catch (error) {
    notifyError(error, translateNow('updates.everythingFanoutFailedTitle'))
  }
}
