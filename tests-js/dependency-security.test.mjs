import assert from 'node:assert/strict'
import http from 'node:http'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'

import { test } from 'vitest'

const desktop = createRequire(resolve(import.meta.dirname, '../apps/desktop/package.json'))
const fromWaitOn = createRequire(desktop.resolve('wait-on'))

test('wait-on Joi consumer validates multiple email addresses', () => {
  const joi = fromWaitOn('joi')
  const schema = joi.string().email({ multiple: true })
  assert.equal(schema.validate('one@example.com, two@example.com').error, undefined)
  assert.ok(schema.validate('one@example.com, invalid').error)
})

test('wait-on Axios consumer strips authorization across a redirect domain', async () => {
  const axios = fromWaitOn('axios')
  const destination = http.createServer((request, response) => {
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify(request.headers))
  })
  const origin = http.createServer((_request, response) => {
    response.writeHead(302, { Location: `http://localhost:${destination.address().port}/final` })
    response.end()
  })
  await Promise.all([
    new Promise(resolve => destination.listen(0, resolve)),
    new Promise(resolve => origin.listen(0, '127.0.0.1', resolve)),
  ])
  try {
    const response = await axios.get(`http://127.0.0.1:${origin.address().port}/`, {
      headers: { Authorization: 'synthetic-test-value' }, proxy: false,
    })
    assert.equal(response.data.authorization, undefined)
  } finally {
    await Promise.all([
      new Promise(resolve => origin.close(resolve)),
      new Promise(resolve => destination.close(resolve)),
    ])
  }
})
