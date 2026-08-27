/**
 * Inline per-profile MCP setup: the `mcp.servers.*` RPC wrapper, its
 * feature-detect, and the button a capability row renders.
 *
 * Shared leaf: the advanced profile editor and the create dialog both render
 * the button, so it lives below both.
 */

import { Button, host, Input, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

// -- inline MCP setup (per-profile), driven by the mcp.servers.* gateway RPCs --
// Feature-detected: if the gateway predates those RPCs the setup button hides
// and the row falls back to the "run hermes mcp / Settings" hint. profile is
// the target bot's profile name (its config is what we write).

/** Body of an `mcp.servers.*` reply. Some gateway builds wrap it in a second
 *  `result` envelope, which every call site below unwraps — hence the
 *  self-reference. */
interface McpServerPayload {
  auth_url?: string
  error?: string
  error_message?: string
  ok?: boolean
  result?: McpServerPayload
  session_id?: string
  status?: string
  verification_url?: string
}

/** `mcpRpc`'s outcome. `unsupported` separates an older gateway that doesn't
 *  know the method from a real failure. */
interface McpRpcResult {
  error?: string
  ok: boolean
  result?: McpServerPayload
  unsupported?: boolean
}

async function mcpRpc(method: string, params: Record<string, unknown>): Promise<McpRpcResult> {
  // Returns { ok, result } or { ok:false, unsupported:true } when the gateway
  // doesn't know the method (older backend) vs a real error.
  try {
    const res = await host.request<McpServerPayload>(method, params)
    return {
      ok: true,
      result: res
    }
  } catch (err: any) {
    const msg = String((err && err.message) || err || '')
    if (/unknown method/i.test(msg)) {
      return {
        ok: false,
        unsupported: true
      }
    }
    return {
      ok: false,
      error: msg
    }
  }
}

// Probe whether the new lifecycle RPCs exist on this gateway (cached per session).
let _mcpRpcSupported: boolean | null = null
async function mcpSetupSupported(): Promise<boolean> {
  if (_mcpRpcSupported !== null) {
    return _mcpRpcSupported
  }
  const r = await mcpRpc('mcp.servers.list', {})
  _mcpRpcSupported = !(r.ok === false && r.unsupported)
  return _mcpRpcSupported
}

/** One row of the capability pane's MCP list (catalog entry or installed server). */
interface McpCatalogEntry {
  auth?: null | string
  fromCatalog?: boolean
  installed?: boolean
  name: string
  requires?: string[]
}

/** The capability scope the Edit Profile / New Bot panes hand down — the SDK's
 *  `ProfileScope`: a bare profile name, or a connection-qualified scope for a
 *  source-scoped bot. */
type McpSetupScope = null | string | undefined | { connectionId?: null | string; profile?: null | string }

interface McpSetupButtonProps {
  ensureProfile?: () => Promise<null | string>
  entry: McpCatalogEntry
  onDone?: () => void
  // TODO(bot-mode-types): Edit Profile passes `botBackendProfileScope(...)`, which
  // is a `{ connectionId, profile }` OBJECT for every source-scoped bot. That
  // object is forwarded verbatim as the `profile` param of mcp.servers.add /
  // set_api_key / test / oauth.*, where the gateway expects a profile NAME
  // string — and mcpRpc goes through host.request, so the connection isn't
  // routed either. Typed as-written.
  profile: McpSetupScope
}

export function McpSetupButton({ profile, entry, onDone, ensureProfile }: McpSetupButtonProps) {
  const { t } = useI18n()
  // entry: { name, requires:[env keys], auth?, fromCatalog, installed }
  // profile may be null at first (New Bot: the profile isn't created yet).
  // ensureProfile() lazily creates it on the first setup action and returns the
  // slug, so OAuth / API-key setup works DURING creation, not only in Edit.
  const [phase, setPhase] = useState<'busy' | 'done' | 'error' | 'idle' | 'keys' | 'oauth'>('idle') // idle | keys | oauth | busy | done | error
  const [supported, setSupported] = useState<boolean | null>(null)
  const [keyValues, setKeyValues] = useState<Record<string, string>>({})
  const [message, setMessage] = useState('')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const profileRef = useRef<McpSetupScope>(profile || null)
  useEffect(() => {
    if (profile) {
      profileRef.current = profile
    }
  }, [profile])

  // Resolve the target profile, creating it on demand for the New Bot flow.
  const resolveProfile = async () => {
    if (profileRef.current) {
      return profileRef.current
    }
    if (ensureProfile) {
      const created = await ensureProfile()
      if (created) {
        profileRef.current = created
      }
      return created
    }
    return null
  }
  useEffect(() => {
    let alive = true
    mcpSetupSupported().then(ok => {
      if (alive) setSupported(ok)
    })
    return () => {
      alive = false
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [])
  const isOAuth = (entry.auth || '').toLowerCase() === 'oauth'
  const requires = entry.requires || []
  const beginKeys = async () => {
    // Ensure the server exists in the target profile first (add from catalog).
    setPhase('busy')
    setMessage('')
    const profile = await resolveProfile()
    if (!profile) {
      setPhase('idle')
      return
    }
    if (entry.fromCatalog && !entry.installed) {
      const add = await mcpRpc('mcp.servers.add', {
        profile,
        name: entry.name,
        preset: entry.name
      })
      if (!add.ok) {
        setPhase('error')
        setMessage(add.error || 'Could not add server')
        return
      }
    }
    setPhase(isOAuth ? 'oauth' : 'keys')
  }
  const submitKeys = async () => {
    setPhase('busy')
    const profile = profileRef.current
    if (!profile) {
      setPhase('error')
      setMessage('No target profile')
      return
    }
    for (const k of requires) {
      const val = (keyValues[k] || '').trim()
      if (!val) {
        continue
      }
      const r = await mcpRpc('mcp.servers.set_api_key', {
        profile,
        name: entry.name,
        env_var: k,
        value: val
      })
      if (!r.ok) {
        setPhase('error')
        setMessage(r.error || 'Failed to set ' + k)
        return
      }
    }
    // Verify via test.
    const t = await mcpRpc('mcp.servers.test', {
      profile,
      name: entry.name
    })
    if (t.ok && t.result && (t.result.ok || (t.result.result && t.result.result.ok))) {
      setPhase('done')
      host.notify({
        kind: 'success',
        message: entry.name + ' configured'
      })
      onDone && onDone()
    } else {
      setPhase('error')
      setMessage(
        (t.result && (t.result.error || (t.result.result && t.result.result.error))) || 'Server test failed after setup'
      )
    }
  }
  const beginOAuth = async () => {
    // A second click (retry, impatient double-click) must not orphan the
    // previous poll interval — an overwritten pollRef leaks a 2s poller that
    // runs until unmount and can flip phase from a stale OAuth session.
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    setPhase('busy')
    setMessage('')
    const profile = await resolveProfile()
    if (!profile) {
      setPhase('idle')
      return
    }
    if (entry.fromCatalog && !entry.installed) {
      const add = await mcpRpc('mcp.servers.add', {
        profile,
        name: entry.name,
        preset: entry.name
      })
      if (!add.ok) {
        setPhase('error')
        setMessage(add.error || 'Could not add server')
        return
      }
    }
    const start = await mcpRpc('mcp.servers.oauth.start', {
      profile,
      name: entry.name
    })
    const payload = start.result && (start.result.result || start.result)
    const authUrl = payload && (payload.auth_url || payload.verification_url)
    const sessionId = payload && payload.session_id
    if (!start.ok || !authUrl || !sessionId) {
      setPhase('error')
      setMessage(start.error || 'Could not start OAuth')
      return
    }
    // Open the auth URL in the native browser, same as provider OAuth.
    // TODO(bot-mode-types): the plugin SDK's `host` has no `openExternal`, so this
    // branch is dead and the window bridge / window.open fallbacks are the only
    // live paths. `ctx.os.openExternal` is the real verb. Kept as-written under
    // ts-expect-error, which reports itself as unused the day the SDK grows one.
    try {
      // @ts-expect-error TODO(bot-mode-types): not on the SDK host, branch is dead
      if (host.openExternal) {
        // @ts-expect-error TODO(bot-mode-types): not on the SDK host, branch is dead
        host.openExternal(authUrl)
      } else if (typeof window !== 'undefined' && window.hermesDesktop && window.hermesDesktop.openExternal) {
        window.hermesDesktop.openExternal(authUrl)
      } else {
        window.open(authUrl, '_blank')
      }
    } catch {
      /* fall through to poll; user can open the URL from the toast */
    }
    setPhase('oauth')
    setMessage('Complete sign-in in your browser...')
    pollRef.current = setInterval(async () => {
      const poll = await mcpRpc('mcp.servers.oauth.poll', {
        profile,
        name: entry.name,
        session_id: sessionId
      })
      const pd = poll.result && (poll.result.result || poll.result)
      const status = pd && pd.status
      if (status === 'approved') {
        clearInterval(pollRef.current!)
        pollRef.current = null
        setPhase('done')
        host.notify({
          kind: 'success',
          message: entry.name + ' authenticated'
        })
        onDone && onDone()
      } else if (status === 'error') {
        clearInterval(pollRef.current!)
        pollRef.current = null
        setPhase('error')
        setMessage((pd && pd.error_message) || 'OAuth failed')
      }
    }, 2000)
  }
  if (supported === false) {
    return (
      <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">
        {'needs setup (' + requires.join(', ') + ') \u2014 restart the gateway to enable in-app setup'}
      </span>
    )
  }
  if (phase === 'done') {
    return <span className="ml-1.5 text-[0.65rem] text-(--ui-success)">set up ✓</span>
  }
  if (phase === 'keys') {
    return (
      <div className="mt-1 grid gap-1">
        {requires.map(k => (
          <Input
            key={k}
            type="password"
            className="h-6 text-[0.7rem]"
            placeholder={k}
            value={keyValues[k] || ''}
            onChange={e =>
              setKeyValues(prev => ({
                ...prev,
                [k]: e.target.value
              }))
            }
          />
        ))}
        <div className="flex gap-1">
          <Button size="xs" variant="secondary" onClick={() => void submitKeys()}>
            Save & test
          </Button>
          <Button size="xs" variant="ghost" onClick={() => setPhase('idle')}>
            {t.common.cancel}
          </Button>
        </div>
      </div>
    )
  }
  if (phase === 'oauth') {
    return <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">{message || 'Authorizing\u2026'}</span>
  }
  if (phase === 'busy') {
    return <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">Working…</span>
  }
  if (phase === 'error') {
    return (
      <span className="ml-1.5 text-[0.65rem] text-(--ui-danger,#f87171)">
        {(message || 'Setup failed') + ' '}
        <Button className="underline" onClick={() => setPhase('idle')} size="inline" variant="link">
          retry
        </Button>
      </span>
    )
  }
  // idle
  return (
    <Button
      className="ml-1.5 text-[0.65rem] text-(--ui-accent) underline"
      onClick={() => void (isOAuth ? beginOAuth() : beginKeys())}
      size="inline"
      variant="link"
    >
      {isOAuth ? 'Sign in\u2026' : 'Set up\u2026'}
    </Button>
  )
}
