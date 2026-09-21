import { beforeEach, expect, test, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getActionStatus: vi.fn(),
  installSkillFromHub: vi.fn(),
  scanSkillHub: vi.fn(),
  uninstallSkillFromHub: vi.fn(),
  updateSkillsFromHub: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/lib/slash-completion-cache', () => ({ invalidateSlashCompletions: vi.fn() }))
vi.mock('@/store/activity', () => ({ upsertDesktopActionTask: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: { subscribe: vi.fn(), get: () => 'default' },
  normalizeProfileKey: (v: string | null | undefined) => v || 'default'
}))

import { HubInstallBlockedError, notifyHubActionFailed, parseInstallBlocked } from './hub-actions'
import { $notifications, clearNotifications } from './notifications'

beforeEach(() => clearNotifications())

// The CLI's blocked-install tail is the only signal the Desktop gets; the toast
// must turn it into a cause and a next step — `--force` does not exist on the
// Desktop route, so it must never be the remedy shown.

// Current CLI (`hermes_cli/skills_hub.py::_scan_block_message`): "Not installed:"
// label + plain sentence; the "never installs unverified" clause marks the
// hard-block (unverified source) case.
test('the current CLI "Not installed:" tail parses into findings + trust', () => {
  const hardBlock = [
    'Scan: 3 findings. Verdict: DANGEROUS',
    "Not installed: the security scan found 3 high-risk pattern(s) in 'org/skill' (listed above). " +
      'Hermes never installs unverified skills with high-risk findings, even with --force. ' +
      'Review the findings or ask the author to fix them; to read the skill without installing, run `hermes skills inspect org/skill`.'
  ]

  expect(parseInstallBlocked(hardBlock)).toEqual({ findings: 3, unverified: true })

  // Console wrapping can split the sentence across log lines.
  const wrapped = [
    'Not installed: the security scan found 1',
    "high-risk pattern(s) in 'org/skill' (listed above). Re-run with --force to install anyway."
  ]

  expect(parseInstallBlocked(wrapped)).toEqual({ findings: 1, unverified: false })

  // No count in the sentence when the scanner reported a verdict without findings.
  const uncounted = ["Not installed: the security scan found high-risk patterns in 'org/skill' (listed above). Re-run with --force to install anyway."]
  expect(parseInstallBlocked(uncounted)).toEqual({ findings: 0, unverified: false })
})

test('the legacy "Installation blocked:" tail still parses and toasts a plain explanation', () => {
  const lines = [
    'Quarantined to quarantine/abc',
    'Scan: 2 findings. Verdict: CAUTION',
    'Installation blocked: Blocked (community source + caution verdict, 2 findings). Use --force to override.'
  ]

  expect(parseInstallBlocked(lines)).toEqual({ findings: 2, unverified: true })
  expect(parseInstallBlocked(['Failed to spawn skills install'])).toBeNull()

  notifyHubActionFailed(new HubInstallBlockedError('org/skill', 2, true, lines.join('\n')), 'Skill action failed', 'skill')

  const toast = $notifications.get()[0]
  expect(toast?.title).toMatch(/Couldn't install skill/)
  expect(toast?.message).toMatch(/2 items to review/)
  expect(toast?.message).toMatch(/unverified source/)
  expect(toast?.message).not.toMatch(/--force/)
  expect(toast?.action?.label).toBe('View scan')
  expect(toast?.detail).toContain('Installation blocked')
})

test('a non-block failure keeps the generic summary', () => {
  notifyHubActionFailed(new Error('network down'), 'Skill action failed')

  expect($notifications.get()[0]?.title).toBe('Skill action failed')
  expect($notifications.get()[0]?.message).toBe('network down')
})
