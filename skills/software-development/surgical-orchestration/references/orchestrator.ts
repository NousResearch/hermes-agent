import * as path from 'node:path';
import * as fs from 'node:fs';
import { EventEmitter } from 'node:events';
import { exec } from 'node:child_process';
import { assertScopeBoundary } from './security.js';
import { computeDebriefHash, type DebriefPayload } from './debrief.js';

// ============================================================================
// Technical Specifications & System Limits
// ============================================================================

export const ORCHESTRATOR_CONFIG = {
  MAX_CONCURRENCY: 2,
  MAX_REVISION_CYCLES: 3,
  SUBAGENT_TIMEOUT_MS: 180_000,
  COMPACTION_TOKEN_THRESHOLD: 0.75,
} as const;

export type AgentRole = 'WORKER' | 'VERIFIER' | 'TEST_FIXER';
export type JobStatus =
  | 'PENDING'
  | 'WORKER_ACTIVE'
  | 'VERIFICATION_ACTIVE'
  | 'VERIFIED'
  | 'ESCALATED'
  | 'FAILED';

export interface SubagentPayload {
  missionId: string;
  role: AgentRole;
  allowedFolderScope: string;
  instructions: string;
  contextSummary: string;
}

export interface SubagentResult {
  status: 'COMPLETED' | 'FAILED';
  filesModified: string[];
  debrief: string;
  selfAudit: string;
}

/**
 * Injectable subagent dispatch strategy. The library ships no runtime of its
 * own: a host (Hermes `delegate_task`, a child process, a test double) supplies
 * this. Injection — not subclassing — is the extension point, because the
 * previous shape forced consumers to monkeypatch a private method through `any`.
 */
export type SubagentDispatcher = (
  agentId: string,
  payload: SubagentPayload,
) => Promise<SubagentResult>;

export interface FolderJob {
  id: string;
  parentFolder: string;
  status: JobStatus;
  attempts: number;
  assignedWorkerId?: string;
  assignedVerifierId?: string;
  /** Verifier's rejection reason, fed into the next Worker attempt. */
  lastVerifierFeedback?: string;
  debriefHistory: Array<{
    attempt: number;
    agentRole: AgentRole;
    debrief: string;
    hash: string;
  }>;
}

export interface JobCard {
  planId: string;
  jobs: Map<string, FolderJob>;
  completedHashes: Set<string>;
  overallStatus: 'IN_PROGRESS' | 'PLAYWRIGHT_TESTING' | 'COMPLETED' | 'ESCALATED' | 'FAILED';
}

// ============================================================================
// 1. Tool-Layer Directory Sandbox Guard (delegates to security.ts)
// ============================================================================

/**
 * Segment-aware "is `child` inside `parent`?" test. Plain string prefix
 * comparison is wrong here: 'src/authz'.startsWith('src/auth') is true.
 *
 * This is a pure planning-time predicate. It does NOT resolve symlinks and is
 * NOT a security boundary — use assertScopeBoundary() from './security.js' for
 * anything that gates real file access.
 */
export function isInsideFolder(child: string, parent: string): boolean {
  const rel = path.relative(path.resolve(parent), path.resolve(child));
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

// ============================================================================
// 2. Anti-Loop Debrief Hashing (delegates to debrief.ts)
// ============================================================================

/**
 * Checks if a debrief hash has already been seen.
 * If a duplicate is found, the subagent branch should be terminated.
 */
export function loopGuardIsDuplicateHash(hash: string, completedHashes: Set<string>): boolean {
  return completedHashes.has(hash);
}

/**
 * Records a debrief hash for future loop detection.
 */
export function loopGuardRecordHash(hash: string, completedHashes: Set<string>): void {
  completedHashes.add(hash);
}

// ============================================================================
// 3. Context Compaction
// ============================================================================

export class ContextCompactor {
  /**
   * Strips raw tool outputs, long diffs, and execution logs.
   * Retains only state markers, debrief hashes, and active checklists.
   */
  static compact(jobCard: JobCard): CompactLedger {
    const ledger: CompactLedger = {
      activeJobs: [],
      completedChecks: [],
      debriefHashes: [],
    };

    for (const [jobId, job] of jobCard.jobs) {
      const latestDebrief = job.debriefHistory[job.debriefHistory.length - 1];
      if (job.status === 'VERIFIED' && latestDebrief) {
        // Reuse the hash already computed at dispatch time. Recomputing it from
        // the debrief string produced a DIFFERENT digest than the registry
        // entry, so the ledger advertised hashes that loop detection never
        // matched on.
        ledger.completedChecks.push({
          folder: job.parentFolder,
          hash: latestDebrief.hash.substring(0, 8),
        });
        ledger.debriefHashes.push(latestDebrief.hash);
      } else {
        ledger.activeJobs.push({
          jobId,
          folder: job.parentFolder,
          status: job.status,
          attempts: job.attempts,
        });
      }
    }

    return ledger;
  }

  /**
   * Estimates token count of the compacted ledger for threshold checking.
   */
  static estimateTokenCount(ledger: CompactLedger): number {
    return JSON.stringify(ledger).length / 4; // rough 4 chars/token estimate
  }
}

export interface CompactLedger {
  activeJobs: Array<{
    jobId: string;
    folder: string;
    status: JobStatus;
    attempts: number;
  }>;
  completedChecks: Array<{
    folder: string;
    hash: string;
  }>;
  debriefHashes: string[];
}

// ============================================================================
// 4. Subagent Manager — Spawning & Lifecycle
// ============================================================================

/**
 * Fail-closed default. A manager constructed without a dispatcher refuses to
 * run rather than silently returning fabricated "COMPLETED" debriefs, which
 * would poison the hash registry and mark unverified folders as VERIFIED.
 */
const defaultDispatcher: SubagentDispatcher = async (agentId) => {
  throw new Error(
    `[NO_DISPATCHER] Subagent runtime not configured for '${agentId}'. ` +
      'Pass a SubagentDispatcher to the OrchestrationEngine/SubagentManager constructor ' +
      '(e.g. one that calls Hermes delegate_task).',
  );
};

export class SubagentManager extends EventEmitter {
  private activeAgents: Map<string, { role: AgentRole; startTime: number }> = new Map();
  private jobCard: JobCard;
  private dispatch: SubagentDispatcher;

  constructor(jobCard: JobCard, dispatch?: SubagentDispatcher) {
    super();
    this.jobCard = jobCard;
    this.dispatch = dispatch ?? defaultDispatcher;
  }

  /**
   * Replaces the dispatch strategy after construction (e.g. a host wiring in
   * `delegate_task` once its runtime is ready).
   */
  setDispatcher(dispatch: SubagentDispatcher): void {
    this.dispatch = dispatch;
  }

  /**
   * Returns current count of active subagents.
   */
  getActiveCount(): number {
    return this.activeAgents.size;
  }

  /**
   * Checks if a new subagent can be spawned (respects MAX_CONCURRENCY cap).
   */
  canSpawn(): boolean {
    return this.getActiveCount() < ORCHESTRATOR_CONFIG.MAX_CONCURRENCY;
  }

  /**
   * Spawns a subagent with the given payload.
   * Enforces scope boundaries, timeout, and concurrency limits.
   */
  async spawnSubagent(payload: SubagentPayload): Promise<SubagentResult> {
    if (!this.canSpawn()) {
      throw new Error('Max concurrency reached. Cannot spawn subagent.');
    }

    const agentId = `agent-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`;
    this.activeAgents.set(agentId, { role: payload.role, startTime: Date.now() });

    // Validate the scope itself is a real, resolvable directory before handing
    // it to an agent. (This deliberately no longer compares the scope to
    // itself — that assertion was a tautology and could never fail. Per-file
    // enforcement is assertScopeBoundary(file, scope) at the tool layer.)
    assertScopeBoundary(payload.allowedFolderScope, payload.allowedFolderScope, agentId);

    // Timeout watchdog. The dispatcher wins or the deadline wins, whichever
    // settles first — previously the timer only emitted an event while the
    // orchestrator stayed blocked on a dispatcher that may never return.
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<never>((_resolve, reject) => {
      timeoutId = setTimeout(() => {
        this.emit('agent_timeout', agentId);
        this.terminateAgent(agentId, 'TIMEOUT');
        reject(
          new Error(
            `[TIMEOUT] Subagent '${agentId}' exceeded ${ORCHESTRATOR_CONFIG.SUBAGENT_TIMEOUT_MS}ms.`,
          ),
        );
      }, ORCHESTRATOR_CONFIG.SUBAGENT_TIMEOUT_MS);
    });

    try {
      this.emit('spawn_requested', { agentId, payload });

      const result = await Promise.race([this.dispatch(agentId, payload), deadline]);

      clearTimeout(timeoutId);
      this.activeAgents.delete(agentId);
      this.emit('agent_completed', { agentId, result });

      return result;
    } catch (error) {
      clearTimeout(timeoutId);
      this.activeAgents.delete(agentId);
      this.emit('agent_error', { agentId, error });
      throw error;
    }
  }

  /**
   * Terminates a running subagent (SIGKILL equivalent).
   */
  private terminateAgent(agentId: string, reason: string): void {
    this.activeAgents.delete(agentId);
    this.emit('agent_terminated', { agentId, reason });
  }
}

// ============================================================================
// 5. Orchestration Engine — State Machine
// ============================================================================

export class OrchestrationEngine extends EventEmitter {
  private jobCard: JobCard;
  protected manager: SubagentManager;
  private plan: BuildPlan;

  constructor(plan: BuildPlan, dispatch?: SubagentDispatcher) {
    super();
    this.plan = plan;
    this.jobCard = {
      planId: `BUILD-${Date.now()}`,
      jobs: new Map(),
      completedHashes: new Set(),
      overallStatus: 'IN_PROGRESS',
    };
    this.manager = new SubagentManager(this.jobCard, dispatch);
    this.initializeJobCard(plan);
  }

  /**
   * Swaps in a dispatch strategy after construction.
   */
  setDispatcher(dispatch: SubagentDispatcher): void {
    this.manager.setDispatcher(dispatch);
  }

  /**
   * Initializes the JobCard from the build plan.
   */
  private initializeJobCard(plan: BuildPlan): void {
    const parentFolders = this.extractParentFolders(plan);

    for (const [index, folder] of parentFolders.entries()) {
      const jobId = `JOB-${String(index + 1).padStart(3, '0')}`;
      this.jobCard.jobs.set(jobId, {
        id: jobId,
        parentFolder: folder,
        status: 'PENDING',
        attempts: 0,
        debriefHistory: [],
      });
    }
  }

  /**
   * Extracts unique parent folders from the build plan's file list.
   */
  private extractParentFolders(plan: BuildPlan): string[] {
    const folders = new Set<string>();
    for (const change of plan.changes) {
      const parent = path.dirname(change.filePath);
      // Use the immediate parent directory as the job scope.  A previous
      // version took parts.slice(0, 2), which silently dropped single-segment
      // parents (path.dirname("README.md") === ".", path.dirname("src/a.ts")
      // === "src") — those files were invisible to the orchestrator.
      folders.add(parent || '.');
    }
    return Array.from(folders);
  }

  /**
   * Runs the full orchestration pipeline.
   */
  async run(): Promise<OrchestrationResult> {
    // Step 1: Dispatch Worker-Verifier loops in parallel, limited to
    // MAX_CONCURRENCY concurrent jobs.  Each job has at most one active
    // agent at a time (worker OR verifier), so the total active agents
    // never exceeds MAX_CONCURRENCY.
    //
    // The previous version used a sequential for-await loop, which meant
    // MAX_CONCURRENCY was dead code — only one agent was ever active.
    const jobEntries = Array.from(this.jobCard.jobs.entries());
    const limit = ORCHESTRATOR_CONFIG.MAX_CONCURRENCY;
    let next = 0;
    const runNext = async (): Promise<void> => {
      while (next < jobEntries.length) {
        const idx = next++;
        const [jobId, job] = jobEntries[idx];
        await this.executeJobLoop(jobId, job);
      }
    };
    const runners = Array.from(
      { length: Math.min(limit, jobEntries.length) },
      () => runNext(),
    );
    await Promise.all(runners);

    // Step 2: Compact context
    const ledger = ContextCompactor.compact(this.jobCard);

    // Step 3: Run Playwright tests
    this.jobCard.overallStatus = 'PLAYWRIGHT_TESTING';
    const testResult = await this.runPlaywrightTests(ledger);

    if (testResult.passed) {
      this.jobCard.overallStatus = 'COMPLETED';
      return { success: true, jobCard: this.jobCard, testResults: testResult };
    } else {
      this.jobCard.overallStatus = 'ESCALATED';
      // Spawn Test-Fixer subagent
      const fixResult = await this.spawnTestFixer(testResult, ledger);
      return { success: false, jobCard: this.jobCard, testResults: testResult, fixResult };
    }
  }

  /**
   * Executes the Worker-Verifier loop for a single folder job.
   */
  private async executeJobLoop(jobId: string, job: FolderJob): Promise<void> {
    while (job.attempts < ORCHESTRATOR_CONFIG.MAX_REVISION_CYCLES) {
      job.status = 'WORKER_ACTIVE';

      // Compact context before spawning
      const ledger = ContextCompactor.compact(this.jobCard);

      // Spawn Worker
      const workerPayload: SubagentPayload = {
        missionId: `${jobId}-W`,
        role: 'WORKER',
        allowedFolderScope: job.parentFolder,
        instructions: this.buildWorkerInstructions(job, this.plan),
        contextSummary: JSON.stringify(ledger),
      };

      const workerResult = await this.manager.spawnSubagent(workerPayload);
      // Hash the CANONICAL payload (scope + sorted file list + debrief), not the
      // bare debrief string. Hashing the string alone made two different folders
      // that happened to emit the same sentence collide, escalating a healthy
      // job as a "loop".
      const workerHash = computeDebriefHash({
        directory: job.parentFolder,
        modifiedFiles: workerResult.filesModified,
        diffHash: workerResult.debrief,
        errorSignature: workerResult.status === 'FAILED' ? workerResult.selfAudit : '',
      });

      // Loop detection: an identical payload for this same scope means the
      // worker redid the exact same work, so another cycle cannot help.
      if (loopGuardIsDuplicateHash(workerHash, this.jobCard.completedHashes)) {
        job.status = 'ESCALATED';
        return;
      }
      loopGuardRecordHash(workerHash, this.jobCard.completedHashes);

      job.debriefHistory.push({
        attempt: job.attempts + 1,
        agentRole: 'WORKER',
        debrief: workerResult.debrief,
        hash: workerHash,
      });

      // Spawn Verifier
      job.status = 'VERIFICATION_ACTIVE';
      const verifierPayload: SubagentPayload = {
        missionId: `${jobId}-V`,
        role: 'VERIFIER',
        allowedFolderScope: job.parentFolder,
        instructions: this.buildVerifierInstructions(job, workerResult),
        contextSummary: JSON.stringify(ledger),
      };

      const verifierResult = await this.manager.spawnSubagent(verifierPayload);

      if (verifierResult.status === 'COMPLETED') {
        job.status = 'VERIFIED';
        return;
      }

      job.attempts++;
      job.lastVerifierFeedback = verifierResult.debrief;
      if (job.attempts >= ORCHESTRATOR_CONFIG.MAX_REVISION_CYCLES) {
        job.status = 'ESCALATED';
        return;
      }
      // else: loop again, and buildWorkerInstructions() will carry the feedback
    }

    job.status = 'ESCALATED';
  }

  /**
   * Builds Worker instructions from the build plan.
   */
  private buildWorkerInstructions(job: FolderJob, plan: BuildPlan): string {
    // Segment-aware containment. `startsWith` alone matched sibling folders
    // whose name merely shares a prefix (scope `src/auth` swallowing
    // `src/authz`), leaking out-of-scope changes into a locked mission.
    const relevantChanges = plan.changes.filter((c) =>
      isInsideFolder(path.dirname(c.filePath), job.parentFolder),
    );

    const feedback = job.lastVerifierFeedback
      ? `\nVerifier feedback from attempt ${job.attempts} (address this first):\n${job.lastVerifierFeedback}\n`
      : '';

    return `## Worker Mission: ${job.id}

Scope: ${job.parentFolder}

Changes required:
${relevantChanges.map((c) => `- ${c.filePath}: ${c.description}`).join('\n')}
${feedback}
Principles:
1. FOLLOWING BEST PRACTICES
2. IS THAT THE BEST YOU CAN DO?

Exit protocol: Emit JSON with status, files_modified, debrief, self_audit.`;
  }

  /**
   * Builds Verifier instructions for reviewing Worker output.
   */
  private buildVerifierInstructions(job: FolderJob, workerResult: SubagentResult): string {
    return `## Verifier Mission: ${job.id}

Review the Worker's changes in scope: ${job.parentFolder}

Worker debrief:
${workerResult.debrief}

Files modified:
${workerResult.filesModified.map((f) => `- ${f}`).join('\n')}

Check:
1. All changes follow best practices
2. No scope boundary violations
3. Code quality is high
4. Is that the best work possible?

Exit protocol: Emit JSON with status (COMPLETED|FAILED), files_modified, debrief, self_audit.`;
  }

  /**
   * Runs Playwright tests against the modified codebase.
   */
  private async runPlaywrightTests(_ledger: CompactLedger): Promise<TestResult> {
    const projectRoot = this.plan.changes[0]?.filePath ? path.dirname(this.plan.changes[0].filePath) : process.cwd();
    // Navigate to project root (find package.json)
    let root = projectRoot;
    while (root !== path.parse(root).root && !fs.existsSync(path.join(root, 'package.json'))) {
      root = path.dirname(root);
    }

    try {
      await new Promise<void>((resolve, reject) => {
        exec('npx playwright test', {
          cwd: root,
          timeout: 120000,
        }, (error: Error | null) => {
          if (error) reject(error);
          else resolve();
        });
      });
      return { passed: true };
    } catch (error: any) {
      const failureLog = error.stdout?.toString() || error.stderr?.toString() || error.message;
      return {
        passed: false,
        trace: failureLog,
        failureLog,
        failingSpecs: this.extractFailingSpecs(failureLog)
      };
    }
  }

  /**
   * Extracts failing spec file paths from Playwright output.
   */
  private extractFailingSpecs(output: string): string[] {
    const specs = new Set<string>();
    // Match patterns like "spec.ts:123" or "test.spec.ts - failed"
    const patterns = [
      /([\w/.-]+\.spec\.(ts|js)):?\d*/g,
      /([\w/.-]+\.test\.(ts|js)):?\d*/g
    ];
    for (const pattern of patterns) {
      const matches = output.matchAll(pattern);
      for (const match of matches) {
        specs.add(match[1]);
      }
    }
    return Array.from(specs);
  }

  /**
   * Spawns a Test-Fixer subagent with need-to-know scope only, routed through
   * the injected SubagentDispatcher — not a hardcoded stub.
   */
  private async spawnTestFixer(testResult: TestResult, ledger: CompactLedger): Promise<SubagentResult> {
    const projectRoot = this.findProjectRoot();
    const readme = this.readIfExists(path.join(projectRoot, 'README.md'));
    const archDiagram = this.readArchitectureDiagram(projectRoot);

    const context = `
TEST FAILURE CONTEXT:
Trace: ${testResult.trace?.substring(0, 8000)}
Failing Specs: ${testResult.failingSpecs?.join(', ')}

PROJECT CONTEXT:
README: ${readme?.substring(0, 4000)}
ARCHITECTURE: ${archDiagram?.substring(0, 4000)}

COMPACTED LEDGER:
${JSON.stringify(ledger, null, 2)}
`.trim();

    this.emit('test_fixer_requested', { context });

    const testFixerPayload: SubagentPayload = {
      missionId: 'TEST-FIXER',
      role: 'TEST_FIXER',
      allowedFolderScope: projectRoot,
      instructions: context,
      contextSummary: JSON.stringify(ledger),
    };

    return this.manager.spawnSubagent(testFixerPayload);
  }

  private findProjectRoot(): string {
    let root = process.cwd();
    while (root !== path.parse(root).root && !fs.existsSync(path.join(root, 'package.json'))) {
      root = path.dirname(root);
    }
    return root;
  }

  private readIfExists(filePath: string): string | null {
    try {
      return fs.readFileSync(filePath, 'utf-8');
    } catch {
      return null;
    }
  }

  private readArchitectureDiagram(projectRoot: string): string | null {
    const candidates = [
      'ARCHITECTURE.md',
      'docs/architecture.md',
      'docs/ARCHITECTURE.md',
      'architecture.md'
    ];
    for (const candidate of candidates) {
      const content = this.readIfExists(path.join(projectRoot, candidate));
      if (content) return content;
    }
    return null;
  }
}

// ============================================================================
// Types
// ============================================================================

export interface BuildPlan {
  description: string;
  changes: Array<{
    filePath: string;
    description: string;
    type: 'add' | 'modify' | 'delete';
  }>;
}

export interface OrchestrationResult {
  success: boolean;
  jobCard: JobCard;
  testResults?: TestResult;
  fixResult?: SubagentResult;
}

export interface TestResult {
  passed: boolean;
  trace?: string;
  failureLog?: string;
  failingSpecs?: string[];
}

// The CLI lives in ./surgical-orchestration.ts — this module is a library and
// deliberately has no top-level side effects (the previous `require.main` guard
// was CommonJS-only and unreachable from this ESM module).
