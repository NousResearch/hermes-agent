import { readJson, writeJson } from '@/lib/storage'

export const SANDBOXED_HTML_APPROVALS_KEY = 'hermes.desktop.sandboxedHtmlApprovals.v1'
export const SANDBOXED_HTML_MAX_BYTES = 512 * 1024
export const SANDBOXED_HTML_CSP =
  "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; media-src data: blob:; object-src 'none'; frame-src 'none'; child-src 'none'; base-uri 'none'; form-action 'none'"
export const SANDBOXED_HTML_PERMISSIONS =
  "camera 'none'; microphone 'none'; geolocation 'none'; display-capture 'none'; clipboard-read 'none'; clipboard-write 'none'; payment 'none'; usb 'none'; serial 'none'; hid 'none'; fullscreen 'none'"

const MAX_APPROVALS = 200

interface ApprovalStore {
  approvals: Record<string, string>
  order: string[]
  version: 1
}

function readApprovalStore(): ApprovalStore {
  const parsed = readJson<Partial<ApprovalStore>>(SANDBOXED_HTML_APPROVALS_KEY)

  if (!parsed || parsed.version !== 1 || !parsed.approvals || typeof parsed.approvals !== 'object') {
    return { approvals: {}, order: [], version: 1 }
  }

  const approvals = Object.fromEntries(
    Object.entries(parsed.approvals).filter(
      (entry): entry is [string, string] =>
        typeof entry[0] === 'string' && typeof entry[1] === 'string' && /^[a-f0-9]{64}$/.test(entry[1])
    )
  )

  const order = Array.isArray(parsed.order)
    ? parsed.order.filter((identity): identity is string => typeof identity === 'string' && identity in approvals)
    : []

  return { approvals, order, version: 1 }
}

export function sandboxedHtmlApprovalIdentity(connectionKey: string, canonicalPath: string): string {
  return JSON.stringify([connectionKey, canonicalPath])
}

export function isSandboxedHtmlApproved(identity: string, digest: string): boolean {
  return readApprovalStore().approvals[identity] === digest
}

export function approveSandboxedHtml(identity: string, digest: string): void {
  const store = readApprovalStore()
  const order = [...store.order.filter(item => item !== identity), identity].slice(-MAX_APPROVALS)
  const keep = new Set(order)

  const approvals = Object.fromEntries(
    Object.entries({ ...store.approvals, [identity]: digest }).filter(([item]) => keep.has(item))
  )

  writeJson(SANDBOXED_HTML_APPROVALS_KEY, { approvals, order, version: 1 })
}

export async function sha256Text(source: string): Promise<string> {
  const bytes = new TextEncoder().encode(source)
  const digest = await crypto.subtle.digest('SHA-256', bytes)

  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')
}

export function buildSandboxedHtmlDocument(source: string): string {
  const document = new DOMParser().parseFromString(source, 'text/html')
  const csp = document.createElement('meta')
  const referrer = document.createElement('meta')

  csp.httpEquiv = 'Content-Security-Policy'
  csp.content = SANDBOXED_HTML_CSP
  csp.dataset.hermesSandboxPolicy = 'true'
  referrer.name = 'referrer'
  referrer.content = 'no-referrer'
  document.head.prepend(referrer)
  document.head.prepend(csp)

  return `<!doctype html>\n${document.documentElement.outerHTML}`
}
