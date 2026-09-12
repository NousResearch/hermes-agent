import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $freeTierContinuation, continuationRequired, refreshContinuation } from '@/store/free-tier-continuation'
import type { ContinuationTarget } from '@/store/free-tier-continuation'
import { FreeTierReadyPanel } from '.'

const target: ContinuationTarget = { owner: { connectionId: 'remote-a', profile: 'writer' }, sessionId: 'chat-a' }
const status = (required: boolean) => ({
  has_guest: true, enabled: true, available: true, notice_pending: false, model: 'nous/welcome', label: 'Nous',
  tool_calls_used: 10, tool_call_cap: 10, capped: true, continuation_required: required
})

afterEach(() => { cleanup(); $freeTierContinuation.set({}) })

it('the ready variant offers all three existing setup doors without a bypass', () => {
  const onSignIn = vi.fn(), onLocal = vi.fn(), onProviders = vi.fn(), onDismiss = vi.fn()
  render(<FreeTierReadyPanel leaving={false} onDismiss={onDismiss} continuation={{ onSignIn, onLocal, onProviders }} />)
  fireEvent.click(screen.getByRole('button', { name: 'Sign in / create account' }))
  fireEvent.click(screen.getByRole('button', { name: 'Use local model' }))
  fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))
  expect(onSignIn).toHaveBeenCalledOnce()
  expect(onLocal).toHaveBeenCalledOnce()
  expect(onProviders).toHaveBeenCalledOnce()
  expect(onDismiss).not.toHaveBeenCalled()
  expect(screen.queryByRole('button', { name: 'Begin' })).toBeNull()
})

it('route verdicts isolate same-name profiles and reject stale clears', async () => {
  const other = { ...target, owner: { ...target.owner, connectionId: 'remote-b' } }
  let resolveOld!: (value: unknown) => void
  const older = refreshContinuation(target, () => new Promise(resolve => { resolveOld = value => resolve(value as never) }))
  await refreshContinuation(target, async () => status(true) as never)
  await refreshContinuation(other, async () => status(false) as never)
  resolveOld(status(false))
  await older
  expect(continuationRequired(target)).toBe(true)
  expect(continuationRequired(other)).toBe(false)
  // A failed re-read or a cancel/choice is not proof a provider was configured.
  await refreshContinuation(target, async () => { throw new Error('offline') })
  expect(continuationRequired(target)).toBe(true)
  await refreshContinuation(target, async () => status(false) as never)
  expect(continuationRequired(target)).toBe(false)
})
