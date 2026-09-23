import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { managedRolloutsAr } from '@/i18n/managed-rollouts'

import { ActiveRollout } from './active-rollout'
import { PreflightReview } from './preflight-review'
import { RecoveryPanel } from './recovery-panel'
import { RolloutConfig } from './rollout-config'
import { RolloutControls } from './rollout-controls'
import { RolloutSummary } from './rollout-summary'
import { WavePreview } from './wave-preview'

const draft = { mode: 'manual' as const, concurrency: 1, canaryInstallId: 'install-a', selectedInstallIds: ['install-a'] }

describe('managed rollout Arabic regions', () => {
  it('names every below-fold landmark and its review controls in Arabic', () => {
    render(<I18nProvider configClient={null} initialLocale="ar">
      <RolloutConfig draft={draft} onChange={() => undefined} onContinue={() => undefined} />
      <PreflightReview compatible={false} draft={draft} onStart={() => undefined} reviewedToken={null} />
      <WavePreview draft={draft} planner={() => [['install-a']]} />
      <ActiveRollout draft={draft} state={{ phase: 'running', completed: 0, total: 1, receipt: null, readiness: null, canaryGate: 'pending', restartRequired: false }} />
      <RolloutControls onCommand={() => undefined} onVerify={() => undefined} phase="running" />
      <RecoveryPanel onExclude={() => undefined} onRecheck={() => undefined} onRecover={() => undefined} onRetry={() => undefined} onStop={() => undefined} target={{ installId: 'install-a', phase: 'unverified', unknown: true, fenced: true, reason: null }} />
      <RolloutSummary summary={{ phase: 'stopped', excluded: 0, unresolved: 1, archived: false, reason: null }} />
    </I18nProvider>)

    for (const [key, label] of Object.entries(managedRolloutsAr.sections)) {
      if (['fleet', 'preparation', 'inventory', 'history'].includes(key)) {continue}
      expect(screen.getByRole(key === 'controls' ? 'group' : 'region', { name: label })).toBeTruthy()
    }

    expect(screen.getByRole('button', { name: managedRolloutsAr.actions.continueToPreflight })).toBeTruthy()
    expect(screen.getByRole('button', { name: managedRolloutsAr.actions.start })).toBeTruthy()
    expect(screen.getByRole('checkbox', { name: managedRolloutsAr.a11y.confirmPreflight })).toBeTruthy()
    expect(screen.getByRole('textbox', { name: managedRolloutsAr.labels.exclusionReason })).toBeTruthy()
    expect(screen.getByText(managedRolloutsAr.descriptions.recoveryActions)).toBeTruthy()
    expect(screen.queryByText(/Recheck is read-only/)).toBeNull()
  })
})
