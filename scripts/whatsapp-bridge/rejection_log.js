/**
 * Timestamped, counted wrapper around the bridge's `ignored` admission
 * events (allowlist misses, self-chat mismatches, etc).
 *
 * Before this, the `ignored` JSON lines written to bridge.log carried no
 * `ts` field, so an operator staring at thousands of identical rejections
 * could not correlate them with "the message I sent at 3pm" or with a
 * config change. Nothing counted them either, so a deployment rejecting
 * every single inbound message looked the same as one rejecting none.
 *
 * `record()` still returns a plain object for `console.log(JSON.stringify(...))`
 * at the call site — the event shape on stdout is unchanged except for the
 * added `ts`. `snapshot()` feeds the bridge's `/health` endpoint so the
 * Python adapter can surface rejection volume without tailing bridge.log.
 */

export function createRejectionLog() {
  const counts = Object.create(null);

  function record(reason, fields = {}) {
    counts[reason] = (counts[reason] || 0) + 1;
    return { event: 'ignored', reason, ts: Date.now(), ...fields };
  }

  function snapshot() {
    return { ...counts };
  }

  return { record, snapshot };
}
