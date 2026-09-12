import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ConnectorCard,
  type ConnectorCardCopy,
  type ConnectorCardProps,
  type ConnectorCardSubject
} from './connector-card'
import { connectorLogoSource } from './connector-logo'

afterEach(cleanup)

const COPY: ConnectorCardCopy = {
  connectAction: 'Connect',
  decline: 'Not now',
  envRequired: 'Fill in the required credentials first',
  grantAction: 'Grant access',
  retryAction: 'Retry',
  stateConnected: 'Connected',
  stateDeclined: 'Skipped',
  stateDisabled: 'currently off',
  stateFailed: 'Failed',
  stateNeedsAuth: 'signed out',
  toolCount: count => (count === 1 ? '1 tool' : `${count} tools`),
  trustCommunity: 'unreviewed',
  trustCommunityTip: host => `Nobody has verified who runs ${host}`,
  trustVerified: publisher => `verified · ${publisher}`,
  trustVerifiedTip: publisher => `The publisher proved it owns ${publisher}`
}

const LINEAR: ConnectorCardSubject = {
  description: 'Issues and projects.',
  homepage: 'https://linear.app',
  name: 'linear',
  title: 'Linear',
  trust: 'catalog'
}

function renderCard(overrides: Partial<ConnectorCardProps> = {}) {
  const onConnect = vi.fn()
  const onDismiss = vi.fn()

  render(
    <ConnectorCard
      connector={LINEAR}
      copy={COPY}
      onConnect={onConnect}
      onDismiss={onDismiss}
      state="not_configured"
      {...overrides}
    />
  )

  return { onConnect, onDismiss }
}

describe('the resting offer', () => {
  it('leads with what the thing is, not with a question about it', () => {
    renderCard()

    // Scoped to a span: the brand glyph is an <svg> carrying its own
    // <title>Linear</title>, which is the mark's accessible name rather than
    // the card's copy.
    expect(screen.getByText('Linear', { selector: 'span' })).toBeTruthy()
    expect(screen.getByText('Issues and projects.')).toBeTruthy()
  })

  it('offers a connect and a way out', () => {
    const { onConnect, onDismiss } = renderCard()

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
    expect(onConnect).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
    expect(onDismiss).toHaveBeenCalledOnce()
  })

  it('says how a configured connector currently stands', () => {
    renderCard({ state: 'needs_auth' })

    expect(screen.getByText('signed out')).toBeTruthy()
  })
})

describe('while it is working', () => {
  it('shows the phase instead of the resting state, because the browser tab that just took focus is otherwise unexplained', () => {
    renderCard({ phase: 'Signing in…', state: 'needs_auth' })

    expect(screen.getByText('Signing in…')).toBeTruthy()
    expect(screen.queryByText('signed out')).toBeNull()
  })

  it('holds its own action but never the way out', () => {
    // While a connect is in flight, decline is the escape from a stuck
    // sign-in tab or a hung install.
    renderCard({ phase: 'Installing…' })

    expect(screen.getByRole('button', { name: 'Not now' }).hasAttribute('disabled')).toBe(false)
  })

  it('holds its action while a sibling is mid-flight, so two sign-in tabs never race for focus', () => {
    renderCard({ otherBusy: true })

    expect(screen.getByRole('button', { name: 'Connect' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Not now' }).hasAttribute('disabled')).toBe(false)
  })
})

describe('once it is answered', () => {
  it('collapses to a single line reporting what came of it', () => {
    renderCard({ outcome: { status: 'connected', tools: ['a', 'b', 'c'] } })

    expect(screen.getByText('Connected · 3 tools')).toBeTruthy()
    // The offer is spent; the space belongs to whatever is still asking.
    expect(screen.queryByRole('button', { name: 'Connect' })).toBeNull()
  })

  it('collapses the same way when waved off, still naming the thing', () => {
    renderCard({ dismissed: true })

    expect(screen.getByText('Linear', { selector: 'span' })).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Not now' })).toBeNull()
  })

  it('does not claim a tool count it does not have', () => {
    renderCard({ outcome: { status: 'connected' } })

    expect(screen.getByText('Connected')).toBeTruthy()
  })
})

describe('after a failure', () => {
  it('stays live with the reason and a retry', () => {
    renderCard({ outcome: { detail: 'Connection refused', status: 'error' } })

    expect(screen.getByText('Connection refused')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('asks for access rather than a retry when the credential was refused', () => {
    // Repeating a refused grant just gets refused again; the fix is to ask.
    renderCard({ outcome: { detail: 'insufficient scope', needsAuth: true, status: 'error' } })

    expect(screen.getByRole('button', { name: 'Grant access' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('brings the setup steps back, because the failure was usually caused by one of them', () => {
    const withSetup: ConnectorCardSubject = { ...LINEAR, setup: ['Enable the API'] }

    renderCard({ connector: withSetup, outcome: { status: 'error' }, state: 'connected' })

    expect(screen.getByText('Enable the API')).toBeTruthy()
  })
})

describe('the work only the user can do', () => {
  it('numbers the steps, because their order matters', () => {
    renderCard({ connector: { ...LINEAR, setup: ['Create a project', 'Enable the API'] } })

    expect(screen.getByText('1.')).toBeTruthy()
    expect(screen.getByText('2.')).toBeTruthy()
  })

  it('turns a markdown link into a real link, because the whole cost of a step is finding the page', () => {
    renderCard({ connector: { ...LINEAR, setup: ['Create a token at [Settings](https://linear.app/settings/api)'] } })

    const link = screen.getByRole('link', { name: 'Settings' })

    expect(link.getAttribute('href')).toBe('https://linear.app/settings/api')
  })

  it('does not repeat console instructions at a connector that already works', () => {
    renderCard({ connector: { ...LINEAR, setup: ['Enable the API'] }, state: 'disabled' })

    expect(screen.queryByText('Enable the API')).toBeNull()
  })
})

describe('credentials', () => {
  const withEnv: ConnectorCardSubject = {
    ...LINEAR,
    requiredEnv: [{ name: 'LINEAR_API_KEY', prompt: 'API key', required: true }]
  }

  it('stays out of the way until the card asks for it', () => {
    renderCard({ connector: withEnv })

    expect(screen.queryByText('API key *')).toBeNull()
  })

  it('reports each keystroke so the caller owns the draft', () => {
    const onEnvChange = vi.fn()

    renderCard({ connector: withEnv, envOpen: true, onEnvChange })

    fireEvent.change(screen.getByLabelText('API key *'), { target: { value: 'lin_abc' } })
    expect(onEnvChange).toHaveBeenCalledWith('LINEAR_API_KEY', 'lin_abc')
  })

  it('masks the value, since these are secrets', () => {
    renderCard({ connector: withEnv, envOpen: true })

    expect(screen.getByLabelText('API key *').getAttribute('type')).toBe('password')
  })
})

describe('how much the source vouches for it', () => {
  it('says nothing for a reviewed source, so the badge that matters is not trained away', () => {
    renderCard()

    expect(screen.queryByText('unreviewed')).toBeNull()
    expect(screen.queryByText(/verified/)).toBeNull()
  })

  it('names the domain a publisher proved it owns, rather than claiming trust', () => {
    renderCard({ connector: { ...LINEAR, publisher: 'notion.com', trust: 'verified' } })

    expect(screen.getByText('verified · notion.com')).toBeTruthy()
  })

  it('flags a publisher nobody has vouched for', () => {
    renderCard({ connector: { ...LINEAR, trust: 'community', url: 'https://random.test/mcp' } })

    expect(screen.getByText('unreviewed')).toBeTruthy()
  })
})

describe('where a mark is read from', () => {
  it('prefers the product site over the endpoint it talks to', () => {
    expect(
      connectorLogoSource({ homepage: 'https://linear.app', name: 'linear', url: 'https://mcp.linear.app/sse' })
    ).toBe('https://linear.app')
  })

  it('falls back to the endpoint, then to the docs', () => {
    expect(connectorLogoSource({ name: 'linear', url: 'https://mcp.linear.app/sse' })).toBe('https://mcp.linear.app')
    expect(connectorLogoSource({ docs: 'https://docs.stripe.com/x', name: 'stripe' })).toBe('https://docs.stripe.com')
  })

  it('reads the origin, never the path, so an endpoint is not fetched just to draw a logo', () => {
    expect(connectorLogoSource({ name: 'acme', url: 'https://acme.test/deep/mcp?token=1' })).toBe('https://acme.test')
  })

  it('refuses a code host, because a bridge published on GitHub is not GitHub', () => {
    expect(connectorLogoSource({ name: 'n8n-bridge', url: 'https://github.com/someone/n8n-mcp' })).toBe('')
  })

  it('refuses a private host, which has no logo to find and should not be named aloud', () => {
    expect(connectorLogoSource({ name: 'unreal-engine', url: 'http://127.0.0.1:8000/mcp' })).toBe('')
  })

  it('shrugs at something that is not a URL at all', () => {
    expect(connectorLogoSource({ docs: 'see the README', name: 'local' })).toBe('')
  })
})
