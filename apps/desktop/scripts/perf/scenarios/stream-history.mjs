// Streaming into an ALREADY-LONG transcript. Same measurement as `stream`, but
// the history is mounted and allowed to settle before the recorders start, so
// what it captures is the per-delta cost that scales with transcript length —
// the regression reported in #69120.
//
// Report-only (tier: manual): the number depends on how much history the host
// can mount, so it is not gated against the committed baseline.

import stream from './stream.mjs'

export default {
  name: 'stream-history',
  tier: 'manual',
  description: 'Streaming cost with a long settled transcript already mounted.',
  run(cdp, opts = {}) {
    return stream.run(cdp, { ...opts, historyTurns: Number(opts.historyTurns ?? 200) })
  }
}
