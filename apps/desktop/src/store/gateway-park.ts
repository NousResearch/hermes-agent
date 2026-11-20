/**
 * Parked-while-hidden latch (#C2a) for the POOL half of a window's sockets.
 *
 * A hidden window is not a user: its sockets may die while nobody is looking
 * (a sleep, a keeper's ping cut, an OS-discarded transport), and every redial
 * ladder would otherwise re-mint a ticket and re-dial in the background — the
 * reconnect storm a backgrounded app must not pay for. Parked means the
 * registry's ladders stand down at both the ARM point (`scheduleReconnect`)
 * and the FIRE point (a timer armed before the park); the window's wake path is
 * what reopens sockets, in one dial, once the user is back.
 *
 * Leaf module by design: the registry consults the latch, and nothing here
 * reaches back into the registry.
 */
let parked = false

export function parkSecondaryReconnects(): void {
  parked = true
}

export function unparkSecondaryReconnects(): void {
  parked = false
}

export function secondaryReconnectsParked(): boolean {
  return parked
}
