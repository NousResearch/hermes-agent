// Proof harness for the hitch diagnostics (U5, R5): cause a hitch on purpose,
// in a known process, and check the exported bundle blames that process.
//
// Every other scenario here measures. This one ASSERTS. The capture stack
// (renderer ring → main controller → gateway ring → sanitized bundle →
// classifier) is only worth anything if its answer is right, and the only way
// to know that is to hand it a hitch whose true cause is already known:
//
//   control       nothing at all                        → not gateway-bound
//   renderer      3 x 400ms busy loops in the page      → renderer-bound
//   gateway       one 6s event-loop block in the gateway → gateway-bound
//   gateway-floor 3 x 600ms blocks — each one BELOW the  → gateway-bound
//                 gateway's 5s "event loop stalled" log
//                 threshold, so this is exactly the CF-1
//                 blind spot the ring was added to close
//   combined      both at once                          → both labels
//
// Each case runs a real capture through the app's own IPC — the scenario calls
// `window.hermesDesktop.diagnosticsCapture.start()` / `.stop()` in the page,
// which is the same path the Diagnostics settings section uses — and reads the
// labels off `stop()`'s export result. A case FAILs on the labels, and the
// scenario throws at the end if any case failed, so the harness exits nonzero.
//
// The gateway block needs the gateway's guarded test hook, which is mounted
// only when the backend process has HERMES_DIAGNOSTICS_TEST_HOOKS=1. That env
// var is set for this scenario by `startIsolatedInstance` (see the
// `gatewayTestHooks` flag below), so it requires --spawn:
//
//   node scripts/perf/run.mjs hitch-classify --spawn
//   node scripts/perf/run.mjs hitch-classify --spawn --prod

import { sleep } from '../lib/cdp.mjs'

/** The gateway's test hook. Absent from a normally started gateway. */
const BLOCK_LOOP_PATH = '/api/diagnostics/test/block-loop'

/** How long after arming before injecting. The gateway's CF-1 heartbeat only
 *  tightens to its armed 20Hz cadence on its NEXT tick, and that tick is up to
 *  the production 2.0s interval away — inject sooner and the drift is measured
 *  against the slow tick (or missed entirely). */
const ARM_SETTLE_MS = 3000

/** Drain time after the last injection: one armed heartbeat tick to record the
 *  drift, and one frame plus the LoAF queue for the renderer's side. */
const DRAIN_MS = 1500

const START = `(async () => JSON.stringify(await window.hermesDesktop.diagnosticsCapture.start()))()`
const STOP = `(async () => JSON.stringify(await window.hermesDesktop.diagnosticsCapture.stop()))()`
const STATUS = `(async () => JSON.stringify(await window.hermesDesktop.diagnosticsCapture.status()))()`

/** Busy-loop the renderer's main thread inside a real rendering frame.
 *
 *  Deliberately a rAF callback and not a bare timer: a Long Animation Frame
 *  entry — the renderer capture's only source of `long_frame` events — is
 *  reported for a *frame*, so the block has to happen inside one and be
 *  followed by work the engine must render. A timer fallback keeps the case
 *  from hanging if rAF never fires (a throttled/hidden renderer would be a
 *  finding of its own, reported as a short burst rather than a timeout). */
const rendererBurst = (ms, count) => `
  (async () => {
    const spent = []
    for (let i = 0; i < ${count}; i++) {
      await new Promise(resolve => {
        let ran = false
        const run = () => {
          if (ran) return
          ran = true
          const t0 = performance.now()
          while (performance.now() - t0 < ${ms}) {}
          // Force style + layout so the blocked frame is a rendering frame.
          document.body.style.setProperty('--perf-hitch', String(i))
          void document.body.offsetHeight
          spent.push(Math.round(performance.now() - t0))
          resolve()
        }
        const timer = setTimeout(run, 1200)
        requestAnimationFrame(() => { clearTimeout(timer); run() })
      })
      await new Promise(r => setTimeout(r, 300))
    }
    return spent.join(',')
  })()
`

/** Block the GATEWAY's event loop via its guarded test hook, over the app's own
 *  authenticated backend channel (`hermes:api` → main → gateway). The response
 *  cannot arrive until the loop it is blocking runs again, so the timeout has
 *  to clear the block itself with room to spare. */
const gatewayBlock = (seconds, count) => `
  (async () => {
    const out = []
    for (let i = 0; i < ${count}; i++) {
      try {
        out.push(await window.hermesDesktop.api({
          path: ${JSON.stringify(BLOCK_LOOP_PATH)},
          method: 'POST',
          body: { seconds: ${seconds} },
          timeoutMs: ${Math.round(seconds * 1000) + 15000}
        }))
      } catch (error) {
        return 'error:' + (error && error.message ? error.message : String(error))
      }
      await new Promise(r => setTimeout(r, 400))
    }
    return JSON.stringify(out)
  })()
`

async function evalJson(cdp, expression) {
  const raw = await cdp.eval(expression)

  return raw === undefined || raw === null ? null : JSON.parse(raw)
}

/** Run one case: arm a capture, inject, stop, and read the classification. */
async function capture(cdp, inject) {
  const started = await evalJson(cdp, START)

  if (!started?.captureId) {
    throw new Error('diagnosticsCapture.start() returned no captureId — is this build missing the U3 controller?')
  }

  let injected = null
  let stopped = null

  try {
    await sleep(ARM_SETTLE_MS)
    injected = await inject()
    await sleep(DRAIN_MS)
    stopped = await evalJson(cdp, STOP)
  } catch (error) {
    // A throw mid-injection must not leave the app (and the gateway's 20Hz
    // armed heartbeat) running for the rest of the harness run.
    await evalJson(cdp, STOP).catch(() => null)

    throw error
  }

  if (!stopped) {
    throw new Error(`capture ${started.captureId} exported nothing — stop() returned null`)
  }

  return { ...stopped, injected }
}

const gatewayStream = result => result.streams?.find(s => s.name === 'gateway') ?? null

/** Turn a case's export into a PASS/FAIL row. `expect` lists labels that must
 *  be present; `reject` lists labels that must not be. */
function check(name, result, { expect, reject = [] }) {
  const labels = result.labels ?? []
  const missing = expect.filter(label => !labels.includes(label))
  const unwanted = reject.filter(label => labels.includes(label))
  const gateway = gatewayStream(result)
  const problems = [
    ...missing.map(label => `missing ${label}`),
    ...unwanted.map(label => `unexpected ${label}`),
    ...(expect.includes('gateway-bound') && gateway?.absent ? [`gateway stream absent (${gateway.absent})`] : [])
  ]

  const ok = problems.length === 0
  const streams = (result.streams ?? []).map(s => `${s.name}=${s.absent ? `absent:${s.absent}` : s.events}`).join(' ')

  console.log(
    `   ${ok ? '✓ PASS' : '✗ FAIL'}  ${name.padEnd(14)} ` +
      `primary=${String(result.primary)} labels=[${labels.join(', ')}] ${streams}` +
      (ok ? '' : `\n            ${problems.join('; ')}`)
  )

  return {
    name,
    ok,
    labels,
    primary: result.primary,
    problems,
    streams: result.streams,
    injected: result.injected,
    // Where the sanitized bundle landed, for a reviewer who wants to read the
    // raw JSONL behind a label (inside the instance's temp user-data dir, so it
    // only exists until teardown).
    directory: result.directory
  }
}

export default {
  name: 'hitch-classify',
  // Not 'ci': it needs a live local gateway (with the test hook enabled) and it
  // asserts rather than measures, so there is nothing to gate on a baseline.
  tier: 'report',
  // Asks the runner to start the isolated instance's backend with the gateway
  // diagnostics test hooks enabled. Nothing else in the harness sets it.
  gatewayTestHooks: true,
  description: 'Synthetic renderer / gateway hitches, asserted against the capture bundle classification.',
  async run(cdp, opts = {}) {
    const burstMs = Number(opts.burstMs ?? 400)
    const bursts = Number(opts.bursts ?? 3)
    const blockSeconds = Number(opts.blockSeconds ?? 6)
    const floorSeconds = Number(opts.floorSeconds ?? 0.6)
    const floorBlocks = Number(opts.floorBlocks ?? 3)

    await cdp.send('Runtime.enable')

    const hasCapture = await cdp.eval('!!(window.hermesDesktop && window.hermesDesktop.diagnosticsCapture)')

    if (!hasCapture) {
      throw new Error(
        'window.hermesDesktop.diagnosticsCapture is missing — the diagnostics capture IPC (U3/U4) is not in this build.'
      )
    }

    const status = await evalJson(cdp, STATUS)

    if (status?.armed) {
      await evalJson(cdp, STOP)
    }

    const renderer = () => cdp.eval(rendererBurst(burstMs, bursts))
    const gateway = (seconds, count) =>
      cdp.eval(gatewayBlock(seconds, count)).then(answer => {
        if (typeof answer === 'string' && answer.startsWith('error:')) {
          throw new Error(
            `gateway block hook failed (${answer.slice(6)}). ${BLOCK_LOOP_PATH} is mounted only when the ` +
              'backend process has HERMES_DIAGNOSTICS_TEST_HOOKS=1 — run this scenario with --spawn so the ' +
              'harness sets it.'
          )
        }

        return answer
      })

    console.log('\n[hitch-classify] injected hitches vs. the bundle classification')

    const rows = []

    // 0. Control: a capture of the same length with NOTHING injected. It is
    //    what makes the rest readable — gateway-bound must be a consequence of
    //    blocking the gateway and of nothing else, and whatever labels the idle
    //    app earns on its own are printed here rather than being quietly
    //    inherited by the injected cases.
    rows.push(
      check('control', await capture(cdp, async () => null), {
        expect: [],
        reject: ['gateway-bound']
      })
    )

    // 1. Renderer only. The gateway is untouched, so a gateway-bound label here
    //    would mean the classifier is reading the wrong stream.
    rows.push(
      check('renderer', await capture(cdp, renderer), {
        expect: ['renderer-bound'],
        reject: ['gateway-bound']
      })
    )

    // 2. Gateway only, well past the 5s log threshold.
    rows.push(
      check('gateway', await capture(cdp, () => gateway(blockSeconds, 1)), {
        expect: ['gateway-bound']
      })
    )

    // 3. Gateway only, every block BELOW the 5s log threshold — the CF-1 blind
    //    spot. The ring's 0.25s capture floor is what makes this classifiable.
    rows.push(
      check('gateway-floor', await capture(cdp, () => gateway(floorSeconds, floorBlocks)), {
        expect: ['gateway-bound']
      })
    )

    // 4. Both at once: the classifier must report both rather than pick one.
    rows.push(
      check(
        'combined',
        await capture(cdp, async () => {
          const both = await Promise.all([gateway(blockSeconds, 1), renderer()])

          return both.join(' | ')
        }),
        { expect: ['gateway-bound', 'renderer-bound'] }
      )
    )

    const failed = rows.filter(r => !r.ok)

    if (failed.length) {
      throw new Error(
        `hitch-classify: ${failed.length}/${rows.length} case(s) misclassified — ` +
          failed.map(r => `${r.name}: ${r.problems.join(', ')}`).join(' | ')
      )
    }

    return {
      metrics: {
        cases: rows.length,
        cases_failed: failed.length
      },
      detail: {
        burstMs,
        bursts,
        blockSeconds,
        floorSeconds,
        floorBlocks,
        cases: rows
      }
    }
  }
}
