// Wait for a profile switch, but never longer than the wake budget.
//
// ensureGatewayProfile awaits the store's dial, and HermesGateway.connect() has
// no dial timeout of its own: a backend that accepts the socket and then never
// completes the handshake leaves this promise pending for the life of the
// window. That is not merely a slow open. waitForFocusedSessionHydration arms
// the only timer on this path, and it is armed AFTER this await returns - so an
// unbounded activation means the wake never settles at all, and the pane wedges
// with no error, no Retry and no timeout (#89556: `ws accepted` in the gateway
// log with no matching `ws closed`).
//
// The activation gets its OWN budget rather than sharing the hydration one. A
// cold profile backend can legitimately spend most of the hydration budget
// painting a large transcript - that race is already tight enough to lose
// (#89617) - so charging activation to the same clock would turn a wedge into a
// regression. The trade is that a wake that is slow in BOTH phases can now take
// up to twice the budget before it surfaces; that is a maintainer call and is
// called out in the PR rather than buried here.
// The caller supplies the dial itself, because WHICH backend to open is a
// routing decision (a workspace switch moves chrome; a plain bot navigation
// only opens the gateway) while the deadline enforced here is the same either
// way.
export async function awaitProfileActivation(
  dial: () => Promise<void>,
  targetProfile: string,
  timeoutMs: number
): Promise<void> {
  const activation = dial()
  let timer: number | undefined

  try {
    await Promise.race([
      activation,
      new Promise<never>((_resolve, reject) => {
        // Same message shape as the hydration timeout on purpose: openSession's
        // catch keys the core stranded-session surface off this prefix, and a
        // wedged dial wants exactly that surface. The phase is distinguished in
        // the [bot-wake] support log, not in the user-facing string.
        timer = window.setTimeout(
          () => reject(new Error(`Timed out loading ${targetProfile}'s session history.`)),
          timeoutMs
        )
      })
    ])
  } finally {
    if (timer !== undefined) {
      window.clearTimeout(timer)
    }
  }

  // No extra catch on the abandoned dial: an in-flight activation has no
  // cancellation handle and keeps running after the budget expires, but
  // Promise.race subscribes to every input, so a rejection that lands after the
  // race has settled is already handled and cannot escape as an unhandled
  // rejection. An explicit `activation.catch()` here was dead code - verified
  // by mutation: removing it changed no test outcome.
}
