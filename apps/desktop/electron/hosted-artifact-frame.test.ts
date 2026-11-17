import { describe, expect, it } from 'vitest'

import {
  HOSTED_ARTIFACT_FRAME_DOMAIN,
  hostedArtifactFrameNavigation,
  hostedArtifactFrameOrigin,
  hostedArtifactFramePolicy,
  isHostedArtifactId,
  isHostedArtifactRequestingUrl
} from './hosted-artifact-frame'

const ID = '01j9z8y7x6w5v4t3s2r1q0p9n8'
const OTHER = '01j9z8y7x6w5v4t3s2r1q0p9n7'
const DOMAIN = 'frame.example'
const PROD = { dev: false, frameDomain: DOMAIN }
const DEV = { dev: true, frameDomain: DOMAIN }

describe('hosted artifact frame origin', () => {
  it('ships with no production frame domain, so packaged builds never frame', () => {
    expect(HOSTED_ARTIFACT_FRAME_DOMAIN).toBe('')
    expect(hostedArtifactFrameOrigin(`https://${ID}.${DOMAIN}/__av?t=x`, { dev: false })).toBeNull()
    expect(hostedArtifactFrameOrigin(`https://${ID}.example.com/__av?t=x`, { dev: false })).toBeNull()
  })

  it('accepts exactly https://<artifact id>.<frame domain>', () => {
    expect(hostedArtifactFrameOrigin(`https://${ID}.${DOMAIN}/__av?t=abc`, PROD)).toBe(`https://${ID}.${DOMAIN}`)
    expect(hostedArtifactFrameOrigin(`https://${ID}.${DOMAIN}/__av?t=abc`, PROD, ID)).toBe(`https://${ID}.${DOMAIN}`)
  })

  it.each([
    ['http on the production domain', `http://${ID}.${DOMAIN}/__av?t=x`],
    ['an explicit port on https', `https://${ID}.${DOMAIN}:8443/__av?t=x`],
    ['two labels above the domain', `https://a.${ID}.${DOMAIN}/__av?t=x`],
    ['the bare frame domain', `https://${DOMAIN}/__av?t=x`],
    ['a lookalike suffix', `https://${ID}.${DOMAIN}.evil.test/__av?t=x`],
    ['a label glued to the domain', `https://${ID}x${DOMAIN}/__av?t=x`],
    ['a label that is not an artifact id', `https://notanid.${DOMAIN}/__av?t=x`],
    ['an id with a letter Crockford excludes', `https://${ID.slice(0, 25)}u.${DOMAIN}/__av?t=x`],
    ['an id one character short', `https://${ID.slice(1)}.${DOMAIN}/__av?t=x`],
    ['credentials in the URL', `https://user:pw@${ID}.${DOMAIN}/__av?t=x`],
    ['the development domain in a packaged build', `http://${ID}.artifacts.localhost:8787/__av?t=x`],
    ['a non-web scheme', `javascript:alert(1)//${ID}.${DOMAIN}`],
    ['garbage', 'not a url']
  ])('rejects %s', (_label, url) => {
    expect(hostedArtifactFrameOrigin(url, PROD)).toBeNull()
  })

  it('refuses a content URL on another artifact than the result names', () => {
    expect(hostedArtifactFrameOrigin(`https://${OTHER}.${DOMAIN}/__av?t=x`, PROD, ID)).toBeNull()
  })

  it('frames the local artifacts server only in development builds, over http on any port', () => {
    expect(hostedArtifactFrameOrigin(`http://${ID}.artifacts.localhost:8787/__av?t=x`, DEV)).toBe(
      `http://${ID}.artifacts.localhost:8787`
    )
    expect(hostedArtifactFrameOrigin(`https://${ID}.artifacts.localhost/__av?t=x`, DEV)).toBeNull()
    expect(hostedArtifactFrameOrigin(`http://${ID}.artifacts.localhost.evil.test:8787/`, DEV)).toBeNull()
  })

  it('knows an artifact id when it sees one', () => {
    expect(isHostedArtifactId(ID)).toBe(true)
    expect(isHostedArtifactId(ID.toUpperCase())).toBe(false)
    expect(isHostedArtifactId(`${ID}0`)).toBe(false)
    expect(isHostedArtifactId(42)).toBe(false)
  })
})

describe('hosted artifact frame navigation', () => {
  const artifactOrigin = `https://${ID}.${DOMAIN}`

  it('keeps an artifact frame on its own origin', () => {
    expect(hostedArtifactFrameNavigation(artifactOrigin, `${artifactOrigin}/page.html`, PROD)).toBe('allow')
    expect(hostedArtifactFrameNavigation(artifactOrigin, 'https://evil.test/', PROD)).toBe('deny')
    expect(hostedArtifactFrameNavigation(artifactOrigin, `https://${OTHER}.${DOMAIN}/`, PROD)).toBe('deny')
    expect(hostedArtifactFrameNavigation(artifactOrigin, 'about:blank', PROD)).toBe('deny')
  })

  it('lets a frame enter an artifact origin only when this build may frame it', () => {
    expect(hostedArtifactFrameNavigation('file://', `${artifactOrigin}/__av?t=x`, PROD)).toBe('allow')
    expect(hostedArtifactFrameNavigation(null, `https://notanid.${DOMAIN}/`, PROD)).toBe('deny')
    expect(hostedArtifactFrameNavigation(null, `http://${ID}.artifacts.localhost:8787/__av?t=x`, { dev: false })).toBe(
      'deny'
    )
    expect(hostedArtifactFrameNavigation(null, `http://${ID}.artifacts.localhost:8787/__av?t=x`, { dev: true })).toBe(
      'allow'
    )
  })

  it('leaves every other navigation to the existing rules', () => {
    expect(
      hostedArtifactFrameNavigation('https://hermes-agent.nousresearch.com', 'https://example.com/', PROD)
    ).toBeNull()
    expect(hostedArtifactFrameNavigation(null, 'https://www.youtube-nocookie.com/embed/x', PROD)).toBeNull()
  })
})

describe('hosted artifact permissions', () => {
  it('flags every request from the artifact namespace', () => {
    expect(isHostedArtifactRequestingUrl(`https://${ID}.${DOMAIN}/`, PROD)).toBe(true)
    expect(isHostedArtifactRequestingUrl(`https://anything.${DOMAIN}/`, PROD)).toBe(true)
    expect(isHostedArtifactRequestingUrl(`http://${ID}.artifacts.localhost:8787`, { dev: false })).toBe(true)
    expect(isHostedArtifactRequestingUrl('https://hermes-agent.nousresearch.com/', PROD)).toBe(false)
    expect(isHostedArtifactRequestingUrl(undefined, PROD)).toBe(false)
  })
})

describe('hosted artifact frame policy of a build', () => {
  it('honours the development frame domain only in unpackaged runs', () => {
    const dev = `http://${ID}.artifacts.localhost:8787/__av?t=x`

    expect(hostedArtifactFrameOrigin(dev, hostedArtifactFramePolicy(true))).toBeNull()
    expect(hostedArtifactFrameOrigin(dev, hostedArtifactFramePolicy(false))).toBe(
      `http://${ID}.artifacts.localhost:8787`
    )
  })
})
