/**
 * Deferred post-connect list refreshes (#C3, t_406dbcb3).
 *
 * `finishPrimaryConnect` used to run the config and sessions refreshes AWAITED
 * and serially inside every reconnect completion (the ladder's and the wake's),
 * so the composer waited on two full list round trips after the socket was
 * already usable. Only the socket and the replay barrier are load-bearing:
 * the transcript gates on the barrier, and the lists are ordinary reads whose
 * stores publish when they land.
 *
 * They are now ONE SHOT per reconnect episode: `schedule()` claims the episode,
 * returns synchronously (it must never be awaited on the reconnect path), and
 * the pair fires when the primary generation reaches application-usable — first
 * server response AND every replay barrier settled
 * (`JsonRpcGatewayClient.whenApplicationUsable`, the same condition C1's phase
 * emission reports, so the refresh and the telemetry cannot disagree).
 *
 * Episode dedup matters because a wake burst finishes more than one generation:
 * the successor and the ladder's dial both run `finishPrimaryConnect` for the
 * same episode, and a burst must refresh once, not once per socket. A finish
 * that arrives after the episode already settled (token null) is a straggler of
 * a claimed burst and is also dropped; the claim itself already scheduled the
 * pair on a live socket.
 */
import { currentReconnectEpisodeToken } from '@/store/desktop-metrics'

export interface PostConnectRefreshHost {
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
  /** Override the episode token source (tests); defaults to the live reconnect episode. */
  currentEpisode?: () => object | null
  /** Resolves when the window's CURRENT primary generation is application-usable. */
  applicationUsable: () => Promise<void>
}

export interface PostConnectRefreshScheduler {
  /** Claim this episode's deferred refresh pair. Synchronous by contract. */
  schedule: () => void
  /** Stop the pending claim from firing (effect teardown). */
  dispose: () => void
}

export interface PostConnectRefreshCallbacks {
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
}

/**
 * Bind the scheduler to the boot hook's live inputs: the callbacks ref (the
 * latest props, read at fire time) and the window's current primary gateway
 * (read at fire time — the wake path REPLACES the generation being claimed).
 */
export function createBootPostConnectRefreshes(
  callbacks: { current: PostConnectRefreshCallbacks },
  primary: () => { whenApplicationUsable: () => Promise<void> },
): PostConnectRefreshScheduler {
  return createPostConnectRefreshScheduler({
    refreshHermesConfig: () => callbacks.current.refreshHermesConfig(),
    refreshSessions: () => callbacks.current.refreshSessions(),
    applicationUsable: () => primary().whenApplicationUsable(),
  })
}

export function createPostConnectRefreshScheduler(host: PostConnectRefreshHost): PostConnectRefreshScheduler {
  const currentEpisode = host.currentEpisode ?? currentReconnectEpisodeToken
  let claimedEpisode: object | null = null
  let hasClaimed = false
  // One pending claim; a newer claim replaces it (the older waiter becomes a no-op).
  let pendingClaim = 0
  let claimSeq = 0
  let disposed = false

  const firePair = async () => {
    await host.refreshHermesConfig().catch(() => undefined)
    await host.refreshSessions().catch(() => undefined)
  }

  return {
    schedule() {
      if (disposed) {
        return
      }

      const episode = currentEpisode()

      // Same episode already claimed, or a post-settle straggler of one that was.
      if (episode === null ? hasClaimed : episode === claimedEpisode) {
        return
      }

      claimedEpisode = episode
      hasClaimed = true

      const claim = ++claimSeq

      pendingClaim = claim

      host
        .applicationUsable()
        .then(() => {
          if (disposed || pendingClaim !== claim) {
            return
          }

          pendingClaim = 0
          void firePair()
        })
        .catch(() => {
          // applicationUsable never rejects today; if it ever does, the next
          // episode's finish re-claims and the lists refresh then.
        })
    },
    dispose() {
      disposed = true
      pendingClaim = 0
    }
  }
}
