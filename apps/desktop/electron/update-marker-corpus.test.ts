/**
 * Shared update-marker corpus consumer (A7 rule 7):
 * `tests/fixtures/update_marker_corpus.json` is authoritative for parsing and
 * judging in every language. Every `judge` case runs through the Desktop's
 * judge with the case's live table, our pid/ct and the corpus clock.
 *
 * The corpus `release` cases do not apply here: Electron never releases,
 * reclaims or rewrites a marker (A7 rule 3) — the checkout script helper does,
 * under the sidecar lock. The second half of this file proves that every
 * marker export leaves the bytes of every corpus marker untouched.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import path from 'path'

import { afterEach, describe, test } from 'vitest'

import {
  claimBridgeMarker,
  inspectUpdateMarker,
  markerPath,
  readLiveUpdateMarker,
  updateHandoffConflict,
  waitForHandoffClaim,
  writeUpdateMarker
} from './update-marker'
import { type JudgeEnv, judgeMarkerText } from './update-marker-judge'
import { cleanupMarkerFixtures, tmpHome } from './update-marker.test-helpers'

interface JudgeCase {
  name: string
  text: string
  live: Record<string, number | null>
  our_pid?: number
  our_ct?: number
  expect: { verdict: string; owner: number | null; run: string | null }
}

interface Corpus {
  schema: number
  now: number
  own_ct_epsilon: number
  ct_tolerance: number
  v1_max_age: number
  our_pid: number
  our_ct: number
  judge: JudgeCase[]
  release: unknown[]
}

const CORPUS_PATH = path.resolve(__dirname, '..', '..', '..', 'tests', 'fixtures', 'update_marker_corpus.json')
const corpus: Corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, 'utf8'))

afterEach(cleanupMarkerFixtures)

function caseEnv(c: JudgeCase): JudgeEnv {
  const ourPid = c.our_pid ?? corpus.our_pid
  const ourCt = c.our_ct ?? corpus.our_ct

  return {
    ourPid,
    ourCt: () => ourCt,
    isAlive: pid => Object.prototype.hasOwnProperty.call(c.live, String(pid)),
    createTime: pid => c.live[String(pid)] ?? null,
    nowS: corpus.now
  }
}

describe('update marker corpus (A7 rule 7)', () => {
  test('the corpus constants are the Desktop constants', async () => {
    const judge = await import('./update-marker-judge')

    assert.equal(corpus.schema, 1)
    assert.equal(judge.OWN_CT_EPSILON_S, corpus.own_ct_epsilon)
    assert.equal(judge.CREATE_TIME_TOLERANCE_S, corpus.ct_tolerance)
    assert.equal(judge.V1_MAX_AGE_S, corpus.v1_max_age)
    assert.ok(corpus.judge.length >= 40, `corpus has ${corpus.judge.length} judge cases`)
  })

  test.each(corpus.judge.map(c => [c.name, c] as const))('judge: %s', async (_name, c) => {
    const got = await judgeMarkerText(c.text, caseEnv(c))

    assert.deepEqual({ verdict: got.verdict, owner: got.owner, run: got.run }, c.expect)
  })
})

/**
 * A process table for the HOST probes: nothing named in a corpus marker is
 * alive (kill(0) => ESRCH) except via this injected table, so the real
 * readers judge corpus bytes without touching live host pids.
 */
function hostDeps(c: JudgeCase) {
  const ownPid = c.our_pid ?? corpus.our_pid

  return {
    ownPid,
    now: () => corpus.now * 1000,
    kill: ((pid: number) => {
      if (pid === ownPid || Object.prototype.hasOwnProperty.call(c.live, String(pid))) {
        return true
      }

      throw Object.assign(new Error('no such process'), { code: 'ESRCH' })
    }) as typeof process.kill,
    processState: () => null,
    createTime: (pid: number) => (pid === ownPid ? (c.our_ct ?? corpus.our_ct) : (c.live[String(pid)] ?? null))
  }
}

describe('Electron never mutates a marker (A7 rule 3)', () => {
  test.each(corpus.judge.map(c => [c.name, c] as const))('every export leaves the bytes: %s', async (_name, c) => {
    const home = tmpHome('corpus-ro')
    const file = markerPath(home)
    fs.writeFileSync(file, c.text)
    const before = fs.readFileSync(file)
    const deps = hostDeps(c)

    await inspectUpdateMarker(home, deps)
    await readLiveUpdateMarker(home, deps)
    await updateHandoffConflict(home, deps)
    await claimBridgeMarker(home, { ...deps, pid: deps.ownPid, startedAt: corpus.now, runId: 'desk-1-a-0000' })
    await writeUpdateMarker(home, 4343, { ...deps, startedAt: corpus.now })
    await waitForHandoffClaim(home, deps.ownPid, { ...deps, timeoutMs: 0, runId: 'desk-1-a-0000' })
    await waitForHandoffClaim(home, deps.ownPid, { ...deps, timeoutMs: 0, startedAt: corpus.now })

    assert.ok(fs.readFileSync(file).equals(before), 'marker bytes unchanged')
    assert.deepEqual(fs.readdirSync(home), ['.hermes-update-in-progress'], 'no tmp litter, no sidecar')
  })

  test('no marker module unlinks, renames, truncates or overwrites a path', () => {
    for (const name of ['update-marker.ts', 'update-marker-judge.ts', 'update-marker-gate.ts']) {
      const source = fs.readFileSync(path.join(__dirname, name), 'utf8')

      for (const mutator of [
        'unlinkSync',
        'unlink(',
        'renameSync',
        'rename(',
        'truncate',
        'writeFileSync',
        "'w'",
        "'w+'",
        "'a'"
      ]) {
        assert.ok(!source.includes(mutator), `${name} must not call ${mutator}`)
      }
    }
  })
})
