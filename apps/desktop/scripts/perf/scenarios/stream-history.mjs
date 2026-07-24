import stream from './stream.mjs'

export default {
  name: 'stream-history',
  tier: 'manual',
  description: 'Synthetic token stream over a preloaded long transcript.',
  run(cdp, opts = {}) {
    return stream.run(cdp, {
      ...opts,
      historyTurns: opts.historyTurns ?? 200
    })
  }
}
