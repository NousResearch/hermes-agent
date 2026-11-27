import assert from 'node:assert/strict'

import { test } from 'vitest'

import { waitForHermesReady } from './backend-health'

const codes = ['ENOBUFS', 'EMFILE', 'ENFILE', 'EADDRNOTAVAIL']

test.each([false, true])(
  'readiness stops on resource exhaustion (wrapped=%s) without spending the boot polling budget',
  async wrapped => {
    for (const code of codes) {
      for (const alreadyBound of [false, true]) {
        for (const legacy of [false, true]) {
          const transportError = Object.assign(new Error(`connect ${code} 127.0.0.1:2802`), { code })
          const error = wrapped ? new TypeError('fetch failed', { cause: transportError }) : transportError
          let time = 0
          let probes = 0
          let sleeps = 0

          const fail = async () => {
            probes += 1
            throw error
          }

          await assert.rejects(
            waitForHermesReady('http://127.0.0.1:2802', {
              alreadyBound,
              fetchPublicJson: legacy
                ? async () => {
                    throw new Error('404: missing')
                  }
                : fail,
              fetchJson: fail,
              now: () => time,
              sleep: async ms => {
                sleeps += 1
                time += ms
              }
            }),
            caught => caught === error
          )
          assert.equal(probes, 1, `${code}: one failed probe, including legacy status`)
          assert.equal(sleeps, 0, `${code}: no retry while the host cannot open sockets`)
        }
      }
    }
  }
)
