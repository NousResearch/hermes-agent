// Cold start — launch → renderer → interactive. Unlike the other scenarios this
// measures the LAUNCH itself, so it can't run against an already-up instance:
// the runner spawns a fresh isolated instance per run (requires --spawn) and
// reads the timings/boot-marks the launcher captures. Registered here so it's a
// known name with a baseline entry; the actual measurement lives in run.mjs.
//
// Metrics (lower is better):
//   spawn_to_cdp_ms     process spawn → CDP page target reachable (electron/V8 up)
//   spawn_to_driver_ms  process spawn → renderer mounted + perf driver present
//   fcp_ms              renderer nav start → first contentful paint
export default {
  name: 'cold-start',
  tier: 'cold',
  description: 'Launch → first paint → interactive (fresh spawn per run).',
  run() {
    throw new Error('cold-start is measured by the runner via fresh spawns; use `--spawn`.')
  }
}
