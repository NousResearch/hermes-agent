'use strict'

/**
 * In-process port reservation pool for the desktop backend launcher.
 *
 * pickPort() probes a localhost port with a throwaway server and closes it
 * before the real bind happens in a separate Python child. Between that probe
 * and the child's bind there is a TOCTOU window: a second concurrent spawn
 * (the primary backend racing a pool backend) can be handed the SAME port, and
 * one then dies with EADDRINUSE ("address already in use" -> "Object has been
 * destroyed" boot loop). Reserving the chosen port in THIS process until the
 * child exits closes that window.
 *
 * The OS bind remains the source of truth; this only deconflicts racers inside
 * this process — it can't stop a foreign squatter, which the probe + the
 * EADDRINUSE self-heal still cover.
 *
 * The pool is dependency-injected (the availability probe is passed in) and
 * free of Electron/Node socket I/O, so it is unit-tested without real sockets
 * (see port-pool.test.cjs).
 */
class PortPool {
  /**
   * @param {number} floor   inclusive lowest port to hand out
   * @param {number} ceiling inclusive highest port to hand out
   */
  constructor(floor, ceiling) {
    this.floor = floor
    this.ceiling = ceiling
    this._reserved = new Set()
  }

  /** @returns {boolean} whether `port` is currently reserved in-process. */
  has(port) {
    return this._reserved.has(port)
  }

  /** Release a previously reserved port. No-op if it was not reserved. */
  release(port) {
    this._reserved.delete(port)
  }

  /** Drop all reservations. */
  clear() {
    this._reserved.clear()
  }

  /** @returns {number} count of currently reserved ports. */
  get size() {
    return this._reserved.size
  }

  /**
   * Reserve and return the lowest port in [floor, ceiling] that is neither
   * already reserved in-process nor rejected by `isAvailable(port)`, or null
   * if every port is taken. `isAvailable` may be sync (boolean) or async
   * (Promise<boolean>); it is awaited either way.
   *
   * @param {(port: number) => boolean | Promise<boolean>} isAvailable
   * @returns {Promise<number|null>}
   */
  async reserve(isAvailable) {
    for (let port = this.floor; port <= this.ceiling; port += 1) {
      if (this._reserved.has(port)) continue
      if (!(await isAvailable(port))) continue
      this._reserved.add(port)
      return port
    }
    return null
  }
}

module.exports = { PortPool }
