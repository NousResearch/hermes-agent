// Replay scheduling pressure without changing Date.now() results:
// NODE_OPTIONS="--require=$PWD/evals/desktop-reap-clock-pressure.cjs" \
//   npm run test:desktop:platforms -w apps/desktop -- electron/backend-ownership.test.ts
// The old 1ms fixtures can expire before their first record is processed.
const now = Date.now.bind(Date)
const gate = new Int32Array(new SharedArrayBuffer(4))
Date.now = function () {
  if (new Error().stack.includes('/electron/backend-ownership.ts:')) {
    Atomics.wait(gate, 0, 0, 5)
  }
  return now()
}
