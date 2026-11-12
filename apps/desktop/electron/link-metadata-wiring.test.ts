import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

// The guard helpers are tested on their own; this pins how link-metadata.ts
// wires them: the curl hop dials the vetted answer, a refused hop never reaches
// the hidden window, and the hidden window's hook refuses names that resolve
// into private space.

type Hook = (details: { resourceType: string; url: string }, callback: (result: { cancel: boolean }) => void) => void

const h = vi.hoisted(() => ({
  answers: {} as Record<string, string[]>,
  curl: [] as string[][],
  hook: null as Hook | null,
  loaded: [] as string[],
  replies: {} as Record<string, string>
}))

vi.mock('node:dns/promises', () => {
  const lookup = async (host: string) => {
    const list = h.answers[host]

    if (!list) {
      throw Object.assign(new Error(`ENOTFOUND ${host}`), { code: 'ENOTFOUND' })
    }

    return list.map(address => ({ address, family: address.includes(':') ? 6 : 4 }))
  }

  return { default: { lookup }, lookup }
})

vi.mock('node:child_process', async importOriginal => ({
  ...(await importOriginal<object>()),
  spawn: (_command: string, args: string[]) => {
    h.curl.push(args)
    const child = Object.assign(new EventEmitter(), { stdout: new EventEmitter() })

    setImmediate(() => {
      const reply = h.replies[args[args.length - 1]]

      if (reply) {
        child.stdout.emit('data', Buffer.from(reply))
      }

      child.emit('close', 0)
    })

    return child
  }
}))

vi.mock('electron', () => {
  class FakeWindow {
    webContents = Object.assign(new EventEmitter(), {
      getTitle: () => '',
      isDestroyed: () => false,
      setAudioMuted: () => undefined,
      setUserAgent: () => undefined,
      setWindowOpenHandler: () => undefined
    })

    destroy() {}
    hide() {}
    isDestroyed() {
      return false
    }
    setOpacity() {}

    loadURL(url: string) {
      h.loaded.push(url)
      setImmediate(() => this.webContents.emit('did-fail-load', {}, -2, '', url, true))

      return Promise.resolve()
    }
  }

  return {
    app: { getPath: () => '/nonexistent-hermes-test', isReady: () => true },
    BrowserWindow: FakeWindow,
    net: {},
    session: {
      fromPartition: () => ({
        on: () => undefined,
        webRequest: { onBeforeRequest: (hook: Hook) => (h.hook = hook) }
      })
    }
  }
})

const { fetchLinkTitle } = await import('./link-metadata')

const curlReply = (status: number, location: string) =>
  `\nhermes-content-type:text/html\nhermes-url-effective:x\nhermes-url-redirect:${location}\nhermes-http-code:${status}`

const hookVerdict = (url: string) =>
  new Promise<boolean>(resolve => h.hook!({ resourceType: 'mainFrame', url }, ({ cancel }) => resolve(cancel)))

describe('link-metadata wiring', () => {
  beforeEach(() => {
    h.curl.length = 0
    h.loaded.length = 0

    // A proxy in the host's env would make the curl hop take --proxy instead of the --resolve pin.
    for (const name of ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']) {
      vi.stubEnv(name, '')
    }
  })

  afterEach(() => vi.unstubAllEnvs())

  test('a curl hop dials the vetted answer, and a refused redirect never reaches the hidden window', async () => {
    h.answers['hop.wiring.test'] = ['93.184.216.34']
    h.answers['lan.wiring.test'] = ['10.0.0.7']
    h.replies['http://hop.wiring.test/start'] = curlReply(302, 'http://lan.wiring.test/secret')

    expect(await fetchLinkTitle('http://hop.wiring.test/start')).toBe('')
    expect(h.curl.map(args => args[args.length - 1])).toEqual(['http://hop.wiring.test/start'])
    expect(h.curl[0]).toEqual(expect.arrayContaining(['--resolve', 'hop.wiring.test:80:93.184.216.34', '--globoff']))
    expect(h.loaded).toEqual([])
  })

  test("the hidden window's hook cancels names that resolve privately and admits public ones", async () => {
    h.answers['pub.wiring.test'] = ['93.184.216.34']
    h.answers['rebind.wiring.test'] = ['169.254.169.254']

    // curl finds no title, so the renderer tier loads the page and installs the hook.
    expect(await fetchLinkTitle('http://pub.wiring.test/')).toBe('')
    expect(h.loaded).toEqual(['http://pub.wiring.test/'])

    expect(await hookVerdict('http://rebind.wiring.test/')).toBe(true)
    expect(await hookVerdict('http://127.0.0.1:8080/')).toBe(true)
    expect(await hookVerdict('http://pub.wiring.test/next')).toBe(false)
  })
})
