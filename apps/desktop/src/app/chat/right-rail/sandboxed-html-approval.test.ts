import { beforeEach, describe, expect, it } from 'vitest'

import {
  approveSandboxedHtml,
  buildSandboxedHtmlDocument,
  isSandboxedHtmlApproved,
  SANDBOXED_HTML_CSP,
  sandboxedHtmlApprovalIdentity,
  sha256Text
} from './sandboxed-html-approval'

describe('sandboxed HTML approvals', () => {
  beforeEach(() => window.localStorage.clear())

  it('binds approval to a collision-free connection and canonical path identity', async () => {
    const windows = sandboxedHtmlApprovalIdentity('remote:work:https://gateway', 'C:\\work\\view.html')
    const posix = sandboxedHtmlApprovalIdentity('remote:work:https://gateway', '/C:/work/view.html')
    const otherConnection = sandboxedHtmlApprovalIdentity('local:', 'C:\\work\\view.html')
    const digest = await sha256Text('<p>approved</p>')

    expect(new Set([windows, posix, otherConnection])).toHaveLength(3)
    approveSandboxedHtml(windows, digest)

    expect(isSandboxedHtmlApproved(windows, digest)).toBe(true)
    expect(isSandboxedHtmlApproved(windows, await sha256Text('<p>changed</p>'))).toBe(false)
    expect(isSandboxedHtmlApproved(posix, digest)).toBe(false)
    expect(isSandboxedHtmlApproved(otherConnection, digest)).toBe(false)
  })

  it('injects the host CSP before authored head content', () => {
    const sourceDocument = buildSandboxedHtmlDocument(
      '<html><head><script>window.started = true</script></head></html>'
    )

    const policyIndex = sourceDocument.indexOf(SANDBOXED_HTML_CSP)
    const scriptIndex = sourceDocument.indexOf('<script>')

    expect(policyIndex).toBeGreaterThan(0)
    expect(policyIndex).toBeLessThan(scriptIndex)
    expect(sourceDocument).toContain('data-hermes-sandbox-policy="true"')
  })
})
