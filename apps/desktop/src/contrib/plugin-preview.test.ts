import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registerPreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'
import { $rightRailActiveTabId } from '@/store/layout'
import {
  $previewScriptConsentQueue,
  $previewScriptGrants,
  answerPreviewScriptConsent,
  grantPreviewScript,
  revokePreviewScript
} from '@/store/plugin-preview-grants'
import { closeRightRail, openPreview } from '@/store/preview'

import { createPluginContext } from './plugin'
import { dropPlugin, publishPlugin } from './plugins-store'

const CAPABILITY = 'preview.executeJavaScript'

// A stand-in guest page: evaluates the injected source the way the webview
// does (script completion value) against a page whose origin the test moves.
let pageOrigin = 'https://mail.example.com'
let afterProbe: (() => void) | null = null
let ran: string[] = []

async function fakeGuest(code: string): Promise<unknown> {
  ran.push(code)
  const page = { document: { title: 'Inbox (3)' }, window: { location: { origin: pageOrigin } } }
  const result = new Function('window', 'document', `return eval(${JSON.stringify(code)})`)(page.window, page.document)

  if (code === 'window.location.origin' && afterProbe) {
    afterProbe()
    afterProbe = null
  }

  return result
}

let unbindRunner: (() => void) | undefined
let pluginSeq = 0

function pluginWith(capabilities: string[]) {
  const id = `page-reader-${++pluginSeq}`

  publishPlugin({ capabilities, id, kind: 'disk', name: 'Page Reader', status: 'loaded' })

  return { ctx: createPluginContext(id, undefined, capabilities), id }
}

function openPage() {
  openPreview({ kind: 'url', label: 'Browser', source: pageOrigin, url: `${pageOrigin}/inbox` })
  unbindRunner = registerPreviewScriptRunner($rightRailActiveTabId.get()!, fakeGuest)
}

const userCodeRuns = () => ran.filter(code => code.includes('document.title')).length

beforeEach(() => {
  closeRightRail()
  window.localStorage.clear()
  pageOrigin = 'https://mail.example.com'
  afterProbe = null
  ran = []
  $previewScriptGrants.set({})
  $previewScriptConsentQueue.set([])
})

afterEach(() => {
  unbindRunner?.()
  unbindRunner = undefined

  for (let n = 1; n <= pluginSeq; n++) {
    dropPlugin(`page-reader-${n}`)
  }
})

describe('ctx.preview.executeJavaScript', () => {
  it('refuses a plugin that did not declare the capability, without asking or running', async () => {
    openPage()
    const { ctx } = pluginWith([])

    expect(ctx.preview.isAvailable()).toBe(false)
    await expect(ctx.preview.executeJavaScript('document.title')).rejects.toThrow(/did not declare/)
    expect($previewScriptConsentQueue.get()).toEqual([])
    expect(ran).toEqual([])
  })

  it('asks once naming the plugin and origin; Allow runs the script and persists the grant', async () => {
    openPage()
    const { ctx, id } = pluginWith([CAPABILITY])

    expect(ctx.preview.isAvailable()).toBe(true)
    const first = ctx.preview.executeJavaScript('document.title')

    await expect.poll(() => $previewScriptConsentQueue.get().length).toBe(1)
    const [request] = $previewScriptConsentQueue.get()
    expect(request).toMatchObject({ origin: 'https://mail.example.com', pluginId: id, pluginName: 'Page Reader' })
    expect(userCodeRuns()).toBe(0)

    answerPreviewScriptConsent(request.id, true)
    await expect(first).resolves.toBe('Inbox (3)')
    expect($previewScriptGrants.get()[id]).toEqual(['https://mail.example.com'])

    await expect(ctx.preview.executeJavaScript('document.title')).resolves.toBe('Inbox (3)')
    expect($previewScriptConsentQueue.get()).toEqual([])
  })

  it("Don't allow refuses without running the script and does not re-ask this run", async () => {
    openPage()
    const { ctx } = pluginWith([CAPABILITY])
    const first = ctx.preview.executeJavaScript('document.title')

    await expect.poll(() => $previewScriptConsentQueue.get().length).toBe(1)
    answerPreviewScriptConsent($previewScriptConsentQueue.get()[0].id, false)

    await expect(first).rejects.toThrow(/not allowed/)
    await expect(ctx.preview.executeJavaScript('document.title')).rejects.toThrow(/not allowed/)
    expect($previewScriptConsentQueue.get()).toEqual([])
    expect(userCodeRuns()).toBe(0)
  })

  it('asks again after the grant is revoked', async () => {
    openPage()
    const { ctx, id } = pluginWith([CAPABILITY])

    grantPreviewScript(id, 'https://mail.example.com')
    await expect(ctx.preview.executeJavaScript('document.title')).resolves.toBe('Inbox (3)')

    revokePreviewScript(id, 'https://mail.example.com')
    void ctx.preview.executeJavaScript('document.title').catch(() => {})

    await expect.poll(() => $previewScriptConsentQueue.get().length).toBe(1)
    expect(userCodeRuns()).toBe(1)
  })

  it('scopes a grant to its origin', async () => {
    pageOrigin = 'https://bank.example.com'
    openPage()
    const { ctx, id } = pluginWith([CAPABILITY])

    grantPreviewScript(id, 'https://mail.example.com')
    void ctx.preview.executeJavaScript('document.title').catch(() => {})

    await expect.poll(() => $previewScriptConsentQueue.get().length).toBe(1)
    expect($previewScriptConsentQueue.get()[0].origin).toBe('https://bank.example.com')
    expect(userCodeRuns()).toBe(0)
  })

  it('refuses when the page navigated to another origin between the check and the run', async () => {
    openPage()
    const { ctx, id } = pluginWith([CAPABILITY])

    grantPreviewScript(id, 'https://mail.example.com')

    afterProbe = () => {
      pageOrigin = 'https://bank.example.com'
    }

    await expect(ctx.preview.executeJavaScript('document.title')).rejects.toThrow(/changed origin/)
  })

  it('refuses opaque and non-web origins without asking', async () => {
    pageOrigin = 'null'
    openPage()
    const { ctx } = pluginWith([CAPABILITY])

    await expect(ctx.preview.executeJavaScript('document.title')).rejects.toThrow(/web page/)
    expect($previewScriptConsentQueue.get()).toEqual([])
  })

  it('rejects when no preview page is open', async () => {
    const { ctx } = pluginWith([CAPABILITY])

    expect(ctx.preview.isAvailable()).toBe(false)
    await expect(ctx.preview.executeJavaScript('document.title')).rejects.toThrow('No preview page is open.')
  })
})
