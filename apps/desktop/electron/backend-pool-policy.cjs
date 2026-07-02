'use strict'

// Pure helpers for Desktop's per-profile backend pool. The Electron main process
// owns the actual child processes; these helpers keep the lifecycle policy small
// and testable so the idle reaper cannot accidentally kill a backend that still
// has a renderer-owned WebSocket.

function nowMs() {
  return Date.now()
}

function createPoolBackendEntry(options = {}) {
  const now = Number.isFinite(options.now) ? options.now : nowMs()
  return {
    process: null,
    port: null,
    token: null,
    connectionPromise: null,
    lastActiveAt: now,
    rendererLeaseCount: 0
  }
}

function normalizeLeaseCount(entry) {
  if (!entry) return 0
  const count = Number(entry.rendererLeaseCount)
  if (!Number.isFinite(count) || count < 0) {
    entry.rendererLeaseCount = 0
    return 0
  }
  entry.rendererLeaseCount = Math.trunc(count)
  return entry.rendererLeaseCount
}

function touchPoolBackendEntry(entry, options = {}) {
  if (!entry) return 0
  const now = Number.isFinite(options.now) ? options.now : nowMs()
  entry.lastActiveAt = now
  return normalizeLeaseCount(entry)
}

function retainPoolBackendEntry(entry, options = {}) {
  if (!entry) return 0
  touchPoolBackendEntry(entry, options)
  entry.rendererLeaseCount = normalizeLeaseCount(entry) + 1
  return entry.rendererLeaseCount
}

function releasePoolBackendEntry(entry, options = {}) {
  if (!entry) return 0
  if (options.touch !== false) {
    touchPoolBackendEntry(entry, options)
  }
  entry.rendererLeaseCount = Math.max(0, normalizeLeaseCount(entry) - 1)
  return entry.rendererLeaseCount
}

function poolBackendHasRendererLease(entry) {
  return normalizeLeaseCount(entry) > 0
}

function poolBackendIdleAgeMs(entry, now) {
  const last = Number(entry?.lastActiveAt)
  if (!Number.isFinite(last)) return Infinity
  return now - last
}

function isPoolBackendReapable(entry, options = {}) {
  const now = Number.isFinite(options.now) ? options.now : nowMs()
  const idleMs = Math.max(0, Number(options.idleMs) || 0)
  if (poolBackendHasRendererLease(entry)) return false
  return poolBackendIdleAgeMs(entry, now) > idleMs
}

function isPoolBackendEvictable(entry, options = {}) {
  const now = Number.isFinite(options.now) ? options.now : nowMs()
  const freshMs = Math.max(0, Number(options.freshMs) || 0)
  if (poolBackendHasRendererLease(entry)) return false
  return poolBackendIdleAgeMs(entry, now) > freshMs
}

module.exports = {
  createPoolBackendEntry,
  isPoolBackendEvictable,
  isPoolBackendReapable,
  poolBackendHasRendererLease,
  releasePoolBackendEntry,
  retainPoolBackendEntry,
  touchPoolBackendEntry
}
