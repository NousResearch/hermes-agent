#!/usr/bin/env node
/**
 * Surgical Orchestration CLI Entry Point
 *
 * Usage:
 *   npx tsx surgical-orchestration.ts <build-plan.json>
 *   npx tsx surgical-orchestration.ts --init
 *   npx tsx surgical-orchestration.ts --dry-run <build-plan.json>
 *
 * The orchestrator engine is a library with no subagent runtime of its own.
 * This file supplies one via the SubagentDispatcher injection point.
 */

import {
  OrchestrationEngine,
  ORCHESTRATOR_CONFIG,
  type BuildPlan,
  type SubagentDispatcher,
  type SubagentResult,
} from './orchestrator.js';
import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

// ============================================================================
// Hermes delegate_task Integration
// ============================================================================

/**
 * Dispatcher backed by the Hermes `delegate_task` tool.
 *
 * `delegate_task` is only injected into an agent session, not into a plain
 * Node process, so this resolves it off globalThis at call time and fails
 * loudly when absent. It must never invent a COMPLETED result: a fabricated
 * debrief would be hashed into the loop registry and mark unverified work
 * VERIFIED.
 */
type DelegateTaskFn = (args: {
  goal: string;
  context: string;
  role: 'leaf' | 'orchestrator';
}) => Promise<Array<{ summary: string }>>;

export const hermesDispatcher: SubagentDispatcher = async (agentId, payload) => {
  const delegateTask = (globalThis as { delegate_task?: DelegateTaskFn }).delegate_task;
  if (typeof delegateTask !== 'function') {
    throw new Error(
      `[NO_HERMES_RUNTIME] delegate_task is unavailable for '${agentId}'. ` +
        'Run this orchestrator inside a Hermes agent session, or pass --dry-run.',
    );
  }

  const results = await delegateTask({
    goal: payload.instructions,
    context: [
      `Mission: ${payload.missionId}`,
      `Role: ${payload.role}`,
      `Scope (do not touch anything outside this folder): ${payload.allowedFolderScope}`,
      `Ledger: ${payload.contextSummary}`,
    ].join('\n'),
    role: 'leaf',
  });

  return parseSubagentResult(results[0]?.summary ?? '');
};

/**
 * Parses a subagent's final JSON debrief.
 *
 * Subagents wrap JSON in prose and fences, so this extracts the last balanced
 * object rather than trusting the whole summary to be JSON. Anything
 * unparseable is FAILED — never a silent pass.
 */
export function parseSubagentResult(summary: string): SubagentResult {
  const failed = (reason: string): SubagentResult => ({
    status: 'FAILED',
    filesModified: [],
    debrief: reason,
    selfAudit: 'Unparseable subagent output; treated as failure by the orchestrator.',
  });

  const fenced = summary.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = (fenced?.[1] ?? summary).trim();
  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start === -1 || end <= start) {
    return failed('Subagent returned no JSON object.');
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(candidate.slice(start, end + 1)) as Record<string, unknown>;
  } catch (err) {
    return failed(`Subagent JSON did not parse: ${(err as Error).message}`);
  }

  const status = parsed.status === 'COMPLETED' ? 'COMPLETED' : 'FAILED';
  const rawFiles = parsed.files_modified ?? parsed.filesModified;
  return {
    status,
    filesModified: Array.isArray(rawFiles) ? rawFiles.map(String) : [],
    debrief: String(parsed.debrief ?? ''),
    selfAudit: String(parsed.self_audit ?? parsed.selfAudit ?? ''),
  };
}

/**
 * Offline dispatcher for `--dry-run`: proves the state machine, concurrency cap
 * and loop guard without spending a single subagent. It returns a debrief that
 * is UNIQUE per mission so the hash registry is exercised honestly.
 */
export const dryRunDispatcher: SubagentDispatcher = async (agentId, payload) => {
  console.log(`[DRY-RUN] ${payload.role} ${payload.missionId} -> ${payload.allowedFolderScope}`);
  return {
    status: 'COMPLETED',
    filesModified: [],
    debrief: `DRY RUN: no edits made for ${payload.missionId} (${agentId}).`,
    selfAudit: 'Dry run — no work attempted, no claim of correctness.',
  };
};

// ============================================================================
// CLI
// ============================================================================

const SAMPLE_PLAN: BuildPlan = {
  description: 'Sample build plan for Surgical Orchestration',
  changes: [
    { filePath: 'src/components/auth/LoginForm.tsx', description: 'Add JWT token refresh logic', type: 'modify' },
    { filePath: 'src/components/auth/RegisterForm.tsx', description: 'Add password strength meter', type: 'modify' },
    { filePath: 'src/services/payment/StripeClient.ts', description: 'Implement webhook signature verification', type: 'add' },
    { filePath: 'src/services/payment/PaymentProcessor.ts', description: 'Add idempotency key support', type: 'modify' },
  ],
};

function printHelp(): void {
  console.log('Surgical Orchestration Engine (Hermes Integration)');
  console.log('');
  console.log('Usage:');
  console.log('  npx tsx surgical-orchestration.ts <build-plan.json>   Execute build plan');
  console.log('  npx tsx surgical-orchestration.ts --dry-run <plan>    Walk the state machine, spawn nothing');
  console.log('  npx tsx surgical-orchestration.ts --init             Generate sample build plan');
  console.log('');
  console.log('Configuration:');
  console.log(`  MAX_CONCURRENCY: ${ORCHESTRATOR_CONFIG.MAX_CONCURRENCY}`);
  console.log(`  MAX_REVISION_CYCLES: ${ORCHESTRATOR_CONFIG.MAX_REVISION_CYCLES}`);
  console.log(`  SUBAGENT_TIMEOUT_MS: ${ORCHESTRATOR_CONFIG.SUBAGENT_TIMEOUT_MS}`);
  console.log(`  COMPACTION_TOKEN_THRESHOLD: ${ORCHESTRATOR_CONFIG.COMPACTION_TOKEN_THRESHOLD}`);
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    printHelp();
    return 0;
  }

  if (args[0] === '--init') {
    if (existsSync('build-plan.json')) {
      console.error('Refusing to overwrite existing build-plan.json');
      return 1;
    }
    writeFileSync('build-plan.json', JSON.stringify(SAMPLE_PLAN, null, 2));
    console.log('Created sample build-plan.json');
    return 0;
  }

  const dryRun = args[0] === '--dry-run';
  const planArg = dryRun ? args[1] : args[0];
  if (!planArg) {
    console.error('Missing <build-plan.json>');
    return 1;
  }

  const planPath = resolve(planArg);
  if (!existsSync(planPath)) {
    console.error(`Build plan not found: ${planPath}`);
    return 1;
  }

  let plan: BuildPlan;
  try {
    plan = JSON.parse(readFileSync(planPath, 'utf-8')) as BuildPlan;
  } catch (err) {
    console.error(`Build plan is not valid JSON: ${(err as Error).message}`);
    return 1;
  }
  if (!Array.isArray(plan.changes) || plan.changes.length === 0) {
    console.error('Build plan has no `changes` array.');
    return 1;
  }

  console.log('[ORCHESTRATOR] Starting Surgical Orchestration...');
  console.log(`Plan: ${plan.description}`);
  console.log(`Changes: ${plan.changes.length} files`);

  const engine = new OrchestrationEngine(plan, dryRun ? dryRunDispatcher : hermesDispatcher);

  engine.on('test_fixer_requested', ({ context }: { context: string }) => {
    console.log('\n[TEST-FIXER] Context prepared for test-fixer subagent:');
    console.log(`${context.substring(0, 500)}...`);
  });

  try {
    const result = await engine.run();
    console.log('\n[ORCHESTRATOR] Final Result:', result.success ? 'SUCCESS' : 'FAILED');
    console.log(`Overall Status: ${result.jobCard.overallStatus}`);

    if (result.testResults) {
      console.log(`Playwright: ${result.testResults.passed ? 'PASSED' : 'FAILED'}`);
      if (result.testResults.failingSpecs?.length) {
        console.log(`Failing specs: ${result.testResults.failingSpecs.join(', ')}`);
      }
    }
    return result.success ? 0 : 1;
  } catch (err) {
    console.error('[ORCHESTRATOR] Fatal error:', err);
    return 1;
  }
}

main().then((code) => {
  process.exitCode = code;
});
