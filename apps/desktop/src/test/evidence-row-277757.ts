import type { TodoItem } from '@/lib/todos'

export const LEGACY_TODO_CARRIER_277757 = [
  '[Your active task list was preserved across context compression]',
  '- [ ] s12. Session 20260829_223026_c651fc: fold todo reconciliation into the single renderer-foundation PR (pending)',
  '- [ ] s09. Session 20260829_162810_cf60c3: finish terminal-path todo cleanup and renderer-foundation PR (pending)',
  '- [ ] s11. Session 20260829_215641_a9fba1: preserve RCA and route the optimistic user-bubble defect without an RCA PR (pending)',
  '- [ ] goal. Goal-control correction pushed to PR #98331 at 48450910e0eb; current CI has Python and Windows failures requiring diagnosis (pending)',
  '- [ ] integrate. Integrate all six currently CI-green Hermes PR heads into a current-upstream fork-integration line and verify exact remote state; no release/install — remote publication complete, canonical activation pending external restart authority (pending)',
  '  - [ ] int-canonical. Reconcile the clean canonical fork-integration checkout from e24e0372 to published 1540d425; requires externally controlled Hermes stop/reset/restart because live-source guard blocks in-process reset (pending)',
  '- [ ] verify. Read back every pushed branch and PR, verify CI/state, and report unresolved human or external blockers (pending)',
  '- [>] session-audit. Audit top-level campaign-related sessions missed by parent-session provenance (in_progress)',
  '  - [ ] session-audit-inventory. Inventory unarchived top-level sessions in the campaign time window (pending)',
  '  - [ ] session-audit-classify. Classify candidates using transcript, branch, worktree, and outcome evidence (pending)',
  '  - [ ] session-audit-report. Verify candidate scope and report exact cleanup recommendation without mutating sessions (pending)',
  '',
  '[Skills pruned during compression — reload before acting on these tasks]',
  "The task list above crossed the compression boundary verbatim, but the skill instructions that governed it were pruned. Before executing any preserved task that depends on these skills, reload them first: skill_view(name='software-development/systematic-debugging'); skill_view(name='software-development/desktop-ui-engineering'); skill_view(name='autonomous-ai-agents/coding-agent-handoff-recovery'); skill_view(name='diagnosing-bugs'); skill_view(name='devops/fleet-helmsman-integration'); skill_view(name='autonomous-ai-agents/hermes-agent'); skill_view(name='devops/hermes-windows-gateway-operations'); skill_view(name='github/hermes-fork-integration'); skill_view(name='github/github-pr-workflow'); skill_view(name='github/git-repository-reconciliation'); skill_view(name='github/upstream-feature-porting'); skill_view(name='devops/kanban-operations'); skill_view(name='critical-study'); skill_view(name='code-review'); skill_view(name='structured-adversarial-review'); skill_view(name='hermes-oneshot'); skill_view(name='devops/wsl-interop'); skill_view(name='autonomous-ai-agents/claude-code'); skill_view(name='design-taste-frontend'); skill_view(name='productivity/session-librarian'). After reloading, re-check that each pending task is still justified — findings recorded before the boundary may have invalidated it."
].join('\n')

export const LEGACY_TODOS_277757: TodoItem[] = [
  {
    content: 'Session 20260829_223026_c651fc: fold todo reconciliation into the single renderer-foundation PR',
    id: 's12',
    status: 'pending'
  },
  {
    content: 'Session 20260829_162810_cf60c3: finish terminal-path todo cleanup and renderer-foundation PR',
    id: 's09',
    status: 'pending'
  },
  {
    content:
      'Session 20260829_215641_a9fba1: preserve RCA and route the optimistic user-bubble defect without an RCA PR',
    id: 's11',
    status: 'pending'
  },
  {
    content:
      'Goal-control correction pushed to PR #98331 at 48450910e0eb; current CI has Python and Windows failures requiring diagnosis',
    id: 'goal',
    status: 'pending'
  },
  {
    content:
      'Integrate all six currently CI-green Hermes PR heads into a current-upstream fork-integration line and verify exact remote state; no release/install — remote publication complete, canonical activation pending external restart authority',
    id: 'integrate',
    status: 'pending'
  },
  {
    content:
      'Reconcile the clean canonical fork-integration checkout from e24e0372 to published 1540d425; requires externally controlled Hermes stop/reset/restart because live-source guard blocks in-process reset',
    id: 'int-canonical',
    parent: 'integrate',
    status: 'pending'
  },
  {
    content: 'Read back every pushed branch and PR, verify CI/state, and report unresolved human or external blockers',
    id: 'verify',
    status: 'pending'
  },
  {
    content: 'Audit top-level campaign-related sessions missed by parent-session provenance',
    id: 'session-audit',
    status: 'in_progress'
  },
  {
    content: 'Inventory unarchived top-level sessions in the campaign time window',
    id: 'session-audit-inventory',
    parent: 'session-audit',
    status: 'pending'
  },
  {
    content: 'Classify candidates using transcript, branch, worktree, and outcome evidence',
    id: 'session-audit-classify',
    parent: 'session-audit',
    status: 'pending'
  },
  {
    content: 'Verify candidate scope and report exact cleanup recommendation without mutating sessions',
    id: 'session-audit-report',
    parent: 'session-audit',
    status: 'pending'
  }
]

export const legacyEvidenceMessage277757 = {
  content: LEGACY_TODO_CARRIER_277757,
  display_kind: null,
  display_metadata: null,
  id: 277757,
  role: 'user' as const,
  session_id: '20260829_233907_98a1d7e3'
}
