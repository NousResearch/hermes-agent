import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createBoundedLinkTitleQueue,
  fetchPinnedLinkTitle,
  linkTitleTargetIsPublic,
  resolvePublicLinkTitleTarget
} from './link-title-fetch'

function fakeCurl(responses: string[], calls: string[][]) {
  return (args: string[]) => {
    calls.push(args)
    const response = Buffer.from(responses.shift() ?? '')
    let onData: ((chunk: Buffer) => void) | undefined
    let onClose: (() => void) | undefined

    queueMicrotask(() => {
      onData?.(response)
      onClose?.()
    })

    return {
      on(event: 'close' | 'error', listener: () => void) {
        if (event === 'close') {
          onClose = listener
        }
      },
      stdout: {
        on(_event: 'data', listener: (chunk: Buffer) => void) {
          onData = listener
        }
      }
    }
  }
}

test('rejects private, link-local, multicast, and malformed link-title targets', async () => {
  await assert.rejects(() => resolvePublicLinkTitleTarget('http://127.0.0.1/admin'))
  await assert.rejects(() => resolvePublicLinkTitleTarget('http://169.254.169.254/latest/meta-data'))
  await assert.rejects(() => resolvePublicLinkTitleTarget('http://224.0.0.1/'))
  await assert.rejects(() => resolvePublicLinkTitleTarget('file:///etc/passwd'))

  assert.equal(linkTitleTargetIsPublic('https://example.com'), true)
  assert.equal(linkTitleTargetIsPublic('http://localhost:3000'), false)
})

test('pins every DNS answer for a public target and rejects mixed answers', async () => {
  const target = await resolvePublicLinkTitleTarget('https://public.example:8443/path', {
    lookup: async () => [
      { address: '1.1.1.1', family: 4 },
      { address: '2606:4700:4700::1111', family: 6 }
    ]
  })

  assert.deepEqual(target.curlResolveArgs, [
    '--resolve',
    'public.example:8443:1.1.1.1',
    '--resolve',
    'public.example:8443:[2606:4700:4700::1111]'
  ])

  await assert.rejects(() =>
    resolvePublicLinkTitleTarget('https://mixed.example/', {
      lookup: async () => [
        { address: '1.1.1.1', family: 4 },
        { address: '127.0.0.1', family: 4 }
      ]
    })
  )
})

test('pins each public redirect hop and preserves a public page title response', async () => {
  const calls: string[][] = []
  const lookups: string[] = []
  const title = await fetchPinnedLinkTitle('https://first.example/start', {
    lookup: async hostname => {
      lookups.push(hostname)
      return hostname === 'first.example' ? [{ address: '1.1.1.1', family: 4 }] : [{ address: '8.8.8.8', family: 4 }]
    },
    spawnCurl: fakeCurl(
      [
        'HTTP/1.1 302 Found\r\nLocation: https://second.example/final\r\n\r\n',
        'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<title>Public Preview</title>'
      ],
      calls
    ),
    userAgent: 'test-agent'
  })

  assert.equal(title, '<title>Public Preview</title>')
  assert.deepEqual(lookups, ['first.example', 'second.example'])
  assert.equal(calls.length, 2)
  assert.equal(calls[0]?.includes('--location'), false)
  assert.ok(calls[0]?.includes('--max-filesize'))
  assert.ok(calls[0]?.includes(String(96 * 1024)))
  assert.ok(calls[0]?.includes('first.example:443:1.1.1.1'))
  assert.ok(calls[1]?.includes('second.example:443:8.8.8.8'))
})

test('rejects a redirect to a private target before a second curl child starts', async () => {
  const calls: string[][] = []
  const title = await fetchPinnedLinkTitle('https://public.example/start', {
    lookup: async () => [{ address: '1.1.1.1', family: 4 }],
    spawnCurl: fakeCurl(['HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/admin\r\n\r\n'], calls),
    userAgent: 'test-agent'
  })

  assert.equal(title, '')
  assert.equal(calls.length, 1)
})

test('caps the complete title-resolution pipeline while allowing queued work to drain', async () => {
  const started: string[] = []
  let finishFirst: ((title: string) => void) | undefined
  const queue = createBoundedLinkTitleQueue(
    rawUrl => {
      started.push(rawUrl)
      return new Promise(resolve => {
        finishFirst = resolve
      })
    },
    { maxConcurrent: 1, maxQueued: 1 }
  )

  const first = queue('https://first.example')
  const second = queue('https://second.example')
  const dropped = queue('https://third.example')

  assert.deepEqual(started, ['https://first.example'])
  assert.equal(await dropped, '')
  finishFirst?.('First title')
  assert.equal(await first, 'First title')
  await new Promise(resolve => setImmediate(resolve))
  assert.deepEqual(started, ['https://first.example', 'https://second.example'])
  finishFirst?.('Second title')
  assert.equal(await second, 'Second title')
})
