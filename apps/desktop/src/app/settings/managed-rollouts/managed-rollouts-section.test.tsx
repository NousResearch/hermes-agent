import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { managedRolloutsEn } from '@/i18n/managed-rollouts'
import { _resetManagedRolloutsForTests } from '@/store/managed-rollouts'
import { rolloutSnapshot } from '@/store/managed-rollouts.test-fixtures'

import { ManagedRolloutsSection } from './managed-rollouts-section'

describe('managed rollout section', () => {
  afterEach(() => {
    _resetManagedRolloutsForTests()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
  })

  it('fails closed when the reviewed bridge is absent', () => {
    render(<ManagedRolloutsSection history={[]} onSelect={() => undefined} />)
    expect(screen.queryByRole('region', { name: 'Managed rollouts' })).toBeNull()
  })

  it('uses the nested connections bridge and surfaces unsupported capability state', async () => {
    const capabilities = vi.fn().mockResolvedValue({
      protocol: 1,
      available: false,
      reason: 'trusted-assurance-provider-unavailable',
      maxConcurrency: 0,
      maxInstallations: 0
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: { managedRollouts: { capabilities } } }
    })

    render(<ManagedRolloutsSection history={[]} onSelect={() => undefined} />)

    expect(screen.getByRole('region', { name: 'Managed rollouts' })).toBeTruthy()
    await waitFor(() => expect(capabilities).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(screen.getByRole('alert', { name: 'trusted-assurance-provider-unavailable' })).toBeTruthy())
    expect(screen.getByRole('status').textContent).toContain(managedRolloutsEn.warnings.unavailable)
    expect(screen.getByText(/Rollout admission unavailable.*0 installations/i)).toBeTruthy()
    expect(screen.queryByRole('button', { name: /start rollout/i })).toBeNull()
  })

  it('renders observed-inventory safety copy and revision in Arabic', async () => {
    const inventory = vi.fn().mockResolvedValue({ inventoryRevision: 'inventory-ar-1', capturedMono: 100, observations: [] })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: { managedRollouts: {
        capabilities: vi.fn().mockResolvedValue({ protocol: 1, available: true, reason: null, maxConcurrency: 1, maxInstallations: 500 }),
        activeRevision: vi.fn().mockResolvedValue(null),
        inventory
      } } }
    })

    render(<I18nProvider configClient={null} initialLocale="ar"><ManagedRolloutsSection /></I18nProvider>)

    await waitFor(() => expect(inventory).toHaveBeenCalledTimes(1))
    expect(screen.getByText('تعكس قائمة الأجهزة الحالة المرصودة فقط. لا يعني تحديد هدف وحده أنه مؤهل أو أن تحديثه مُصرَّح به.')).toBeTruthy()
    expect(screen.getByText('مراجعة القائمة: inventory-ar-1؛ وقت الالتقاط وفق الساعة الرتيبة: 100.')).toBeTruthy()
    expect(screen.queryByText(/Inventory is observed state/)).toBeNull()
  })

  it('keeps bridge inventory visible while rollout admission capacity is unavailable', async () => {
    const inventory = vi.fn().mockResolvedValue({
      inventoryRevision: 'inventory-readonly', capturedMono: 321,
      observations: [{
        installId: 'a'.repeat(32), connectionId: '11111111-1111-4111-8111-111111111111',
        aliasConnectionIds: [], codeRoot: '/srv/hermes', repositoryId: 'github.com/NousResearch/hermes-agent',
        headSha: 'b'.repeat(40), requiredScopeIds: [],
        source: { connectionId: '11111111-1111-4111-8111-111111111111', verifiedHostKeyFingerprint: 'host-a' }
      }]
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: { managedRollouts: {
        capabilities: vi.fn().mockResolvedValue({ protocol: 1, available: false, reason: 'measured-ssh-capacity-unavailable', maxConcurrency: 0, maxInstallations: 0 }),
        activeRevision: vi.fn().mockResolvedValue(null), inventory
      } } }
    })

    render(<ManagedRolloutsSection />)
    await waitFor(() => expect(inventory).toHaveBeenCalledTimes(1))
    expect(screen.getByText('b'.repeat(40))).toBeTruthy()
    expect(screen.getByText(/Rollout admission unavailable.*0 installations/i)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    expect(screen.getByRole('button', { name: /review selected target/i })).toHaveProperty('disabled', true)
  })

  it('loads main-owned inventory and refuses target review when the reviewed manifest is unavailable', async () => {
    const inventory = vi.fn().mockResolvedValue({
      inventoryRevision: 'inventory-1',
      capturedMono: 100,
      observations: [{
        installId: 'a'.repeat(32),
        connectionId: '11111111-1111-4111-8111-111111111111',
        aliasConnectionIds: [],
        codeRoot: '/srv/hermes',
        repositoryId: 'github.com/NousResearch/hermes-agent',
        headSha: 'b'.repeat(40),
        requiredScopeIds: ['main'],
        source: {
          connectionId: '11111111-1111-4111-8111-111111111111',
          connectionConfigRevision: 'config-1',
          verifiedHostKeyFingerprint: 'host-key-a',
          remoteUser: 'tester',
          port: 22,
          configuredProfile: 'default',
          configuredCodePath: '/srv/hermes'
        }
      }]
    })

    const resolveTarget = vi.fn().mockRejectedValue(new Error('review-manifest-unavailable'))
    const start = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: { managedRollouts: {
        capabilities: vi.fn().mockResolvedValue({ protocol: 1, available: true, reason: null, maxConcurrency: 1, maxInstallations: 500 }),
        activeRevision: vi.fn().mockResolvedValue(null),
        inventory,
        resolveTarget,
        start
      } } }
    })

    render(<ManagedRolloutsSection />)
    await waitFor(() => expect(inventory).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    fireEvent.click(screen.getByRole('button', { name: /review selected target/i }))

    await waitFor(() => expect(resolveTarget).toHaveBeenCalledWith({
      connectionIds: ['11111111-1111-4111-8111-111111111111'],
      inventoryRevision: 'inventory-1',
      retryOf: null
    }))
    expect(screen.getByRole('alert').textContent).toContain('review-manifest-unavailable')
    expect(start).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: /start rollout/i })).toBeNull()
  })

  it('submits the main-owned resolution to preflight and starts only with its returned token and request id', async () => {
    const installId = 'a'.repeat(32)
    const connectionId = '11111111-1111-4111-8111-111111111111'
    const target = { repositoryId: 'github.com/NousResearch/hermes-agent', branch: 'main', sha: 'b'.repeat(40), protocol: 1 }

    const plan = {
      target,
      inventoryRevision: 'inventory-2',
      waves: [[installId]],
      concurrency: 1,
      promotionPolicy: 'manual',
      rows: [{
        installId,
        connectionId,
        installationFingerprint: 'c'.repeat(64),
        sourceFingerprint: 'd'.repeat(64),
        admittedHead: 'e'.repeat(40),
        requiredScopeIds: [],
        eligible: true
      }],
      retryOf: null,
      exclusions: []
    }

    const inventory = vi.fn().mockResolvedValue({
      inventoryRevision: 'inventory-2',
      capturedMono: 100,
      observations: [{
        installId,
        connectionId,
        aliasConnectionIds: [],
        codeRoot: '/srv/hermes',
        repositoryId: target.repositoryId,
        headSha: 'e'.repeat(40),
        requiredScopeIds: [],
        source: {
          connectionId,
          connectionConfigRevision: 'config-2',
          verifiedHostKeyFingerprint: 'host-key-a',
          remoteUser: 'tester',
          port: 22,
          configuredProfile: 'default',
          configuredCodePath: '/srv/hermes'
        }
      }]
    })

    const resolveTarget = vi.fn().mockResolvedValue({
      resolutionId: 'reviewed-resolution',
      expiresAt: Date.now() + 60_000,
      inventoryRevision: 'inventory-2',
      target,
      plan
    })

    const preflight = vi.fn().mockResolvedValue({
      ok: true,
      token: 'main-owned-review-token',
      requestId: '22222222-2222-4222-8222-222222222222',
      rolloutId: '33333333-3333-4333-8333-333333333333',
      expiresAt: Date.now() + 60_000,
      planDigest: 'f'.repeat(64),
      canonicalPlan: plan,
      changes: [],
      blockers: []
    })

    const start = vi.fn().mockResolvedValue({ ok: true, id: '33333333-3333-4333-8333-333333333333', revision: 1, phase: 'running' })
    const get = vi.fn().mockResolvedValue(rolloutSnapshot(1, { id: '33333333-3333-4333-8333-333333333333' }))
    const updateManaged = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: {
        updateManaged,
        managedRollouts: {
          capabilities: vi.fn().mockResolvedValue({ protocol: 1, available: true, reason: null, maxConcurrency: 1, maxInstallations: 500 }),
          activeRevision: vi.fn().mockResolvedValue(null),
          inventory,
          resolveTarget,
          preflight,
          start,
          get
        }
      } }
    })

    render(<ManagedRolloutsSection />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Select' })).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    expect(updateManaged).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /review selected target/i }))
    await waitFor(() => expect(screen.getByRole('region', { name: 'Managed rollout configuration' })).toBeTruthy())
    expect(resolveTarget).toHaveBeenCalledWith({ connectionIds: [connectionId], inventoryRevision: 'inventory-2', retryOf: null })
    fireEvent.click(screen.getByRole('button', { name: 'Continue to preflight' }))
    await waitFor(() => expect(preflight).toHaveBeenCalledWith({
      inventoryRevision: 'inventory-2',
      targetResolutionId: 'reviewed-resolution',
      waves: [[installId]],
      concurrency: 1,
      promotionPolicy: 'manual',
      retryOf: null
    }))
    expect(within(screen.getByRole('region', { name: 'Managed rollout preflight review' })).getByText(new RegExp(target.sha))).toBeTruthy()
    expect(start).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('checkbox', { name: /confirm this exact preflight review/i }))
    fireEvent.click(screen.getByRole('button', { name: 'Start rollout' }))
    await waitFor(() => expect(start).toHaveBeenCalledWith({
      token: 'main-owned-review-token',
      requestId: '22222222-2222-4222-8222-222222222222'
    }))
    await waitFor(() => expect(screen.getByRole('region', { name: 'Active managed rollout' })).toBeTruthy())
    expect(updateManaged).not.toHaveBeenCalled()
  })

  it('runs unpinned individual preparation only after confirmation and refreshes inventory before any review', async () => {
    const connectionId = '11111111-1111-4111-8111-111111111111'

    const inventory = vi.fn().mockResolvedValue({
      inventoryRevision: 'inventory-prep',
      capturedMono: 200,
      observations: [{
        installId: 'a'.repeat(32),
        connectionId,
        aliasConnectionIds: [],
        codeRoot: '/srv/hermes',
        repositoryId: 'github.com/NousResearch/hermes-agent',
        headSha: 'b'.repeat(40),
        requiredScopeIds: [],
        source: {
          connectionId,
          connectionConfigRevision: 'config-prep',
          verifiedHostKeyFingerprint: 'host-key-a',
          remoteUser: 'tester',
          port: 22,
          configuredProfile: 'default',
          configuredCodePath: '/srv/hermes'
        }
      }]
    })

    const updateManaged = vi.fn().mockResolvedValue({
      connectionId,
      correlationId: 'preparation-correlation',
      ok: true,
      updateOk: true,
      restoreOk: true,
      outcome: 'updated',
      exitCode: 0,
      receipt: null,
      scopes: []
    })

    const resolveTarget = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { connections: {
        updateManaged,
        managedRollouts: {
          capabilities: vi.fn().mockResolvedValue({ protocol: 1, available: false, reason: 'review-manifest-unavailable', maxConcurrency: 0, maxInstallations: 0 }),
          activeRevision: vi.fn().mockResolvedValue(null),
          inventory,
          resolveTarget
        }
      } }
    })

    render(<ManagedRolloutsSection />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Select' })).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    expect(screen.getByRole('button', { name: /review selected target/i })).toHaveProperty('disabled', true)
    const prepare = screen.getByRole('button', { name: 'Prepare selected targets' })
    expect(prepare).toHaveProperty('disabled', true)
    expect(updateManaged).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('checkbox', { name: /confirm these targets/i }))
    fireEvent.click(prepare)
    await waitFor(() => expect(updateManaged).toHaveBeenCalledWith(connectionId))
    await waitFor(() => expect(inventory).toHaveBeenCalledTimes(2))
    expect(resolveTarget).not.toHaveBeenCalled()
    expect(screen.getByText(/Unpinned preparation finished/)).toBeTruthy()
  })
})
