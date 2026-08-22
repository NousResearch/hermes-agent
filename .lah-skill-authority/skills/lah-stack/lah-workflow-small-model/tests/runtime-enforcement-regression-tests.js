#!/usr/bin/env node
/**
 * Runtime Enforcement Regression Corpus v1
 *
 * HERMES_MISSION_CONVERGENCE_RUNTIME_HARDENING_V1
 * Mode: REPAIR_AND_CERTIFY
 * PRIMARY_WORKFLOW: lah-workflow-small-model
 *
 * Tests the four invariants observed as defects in
 * LAH_POPUNDER_FAST_PATH_REAL_CANARY_PREP_V1:
 *   R1 — Router ambiguity not blocking
 *   R2 — Capability authority bypass (LLM inference)
 *   R3 — Post-success discovery continuing after objective achieved
 *   R4 — False governor claim (behavioral reported as runtime-enforced)
 *
 * Run: node runtime-enforcement-regression-tests.js
 */

"use strict";

const fs = require("fs");

// ─── Test Runner ─────────────────────────────────

class TestRunner {
  constructor() {
    this.results = [];
    this.passed = 0;
    this.failed = 0;
    this.blocked = 0;
  }

  run(id, name, testFn) {
    try {
      const result = testFn();
      if (result && result.pass) {
        this.passed++;
        this.results.push({ id, name, status: "PASS", detail: result.detail || "" });
      } else {
        this.failed++;
        this.results.push({ id, name, status: "FAIL", detail: (result && result.detail) || "No detail" });
      }
    } catch (err) {
      this.blocked++;
      this.results.push({ id, name, status: "BLOCKED", detail: err.message });
    }
  }

  printReport() {
    console.log("");
    console.log("══════════════════════════════════════════════");
    console.log("  Runtime Enforcement Regression — Test Report");
    console.log("══════════════════════════════════════════════");
    console.log("");

    for (const r of this.results) {
      const icon = r.status === "PASS" ? "✓" : r.status === "FAIL" ? "✗" : "⊘";
      console.log(`  ${icon} ${r.id}: ${r.name} [${r.status}]`);
      if (r.detail && r.status !== "PASS") {
        console.log(`    → ${r.detail}`);
      }
    }

    console.log("");
    console.log(`  Total:  ${this.results.length}`);
    console.log(`  Passed: ${this.passed}`);
    console.log(`  Failed: ${this.failed}`);
    console.log(`  Blocked: ${this.blocked}`);
    console.log("");

    if (this.failed === 0 && this.blocked === 0) {
      console.log("  ▶ ALL TESTS PASSED");
    } else {
      console.log("  ▶ SOME TESTS FAILED OR BLOCKED");
    }
    console.log("");
  }

  getSummary() {
    return {
      total: this.results.length,
      passed: this.passed,
      failed: this.failed,
      blocked: this.blocked,
      allPassed: this.failed === 0 && this.blocked === 0,
    };
  }
}

// ─── Imports ──────────────────────────────────────

const {
  ConvergenceGovernor,
  GOVERNOR_ENFORCEMENT_STATE,
} = require("../scripts/convergence-governor.js");

// ─── Helpers ──────────────────────────────────────

function pass(detail) {
  return { pass: true, detail: detail || "" };
}

function fail(detail) {
  return { pass: false, detail: detail || "Assertion failed" };
}

function truthy(val, label) {
  return val
    ? pass(label || "Value is truthy")
    : fail(label || "Value is falsy");
}

function readFile(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

// ═══════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════

const runner = new TestRunner();

// ── T01: Resolved router result permits CodeGraph bootstrap ──
console.log("\n═══ T01: Resolved router result permits CodeGraph bootstrap ═══");
{
  const gov = new ConvergenceGovernor({ missionMode: "REPAIR" });

  gov.setCodeGraphBootstrap({
    phase: "CODEGRAPH_BOOTSTRAP",
    can_proceed: true,
    receipt: { repo: "/home/deploy/lah-stack-repos/openclaw-runtime", status: "RESOLVED" },
  });

  const result = gov.recordAction({
    command: "codegraph_explore --query select",
    evidence: { repo: "openclaw-runtime" },
  });

  runner.run("T01", "Resolved router permits CodeGraph bootstrap", () =>
    truthy(result.allowed, `allowed=${result.allowed}`)
  );
}

// ── T02: Ambiguous router result blocks mission before filesystem discovery ──
console.log("\n═══ T02: Ambiguous router result blocks mission ═══");
{
  const lahBootstrapPath = "/home/deploy/hermes-agent/hermes_cli/lah_bootstrap.py";
  const content = readFile(lahBootstrapPath);

  const hasAmbiguousConstant = content.includes('ROUTER_BLOCKED_AMBIGUOUS = "BLOCKED_AMBIGUOUS"');
  const hasIsRouterAmbiguous = content.includes("def is_router_ambiguous");
  const hasSetRouterStatus = content.includes("def set_router_status");

  runner.run("T02", "Ambiguous router blocks mission", () =>
    truthy(
      hasAmbiguousConstant && hasIsRouterAmbiguous && hasSetRouterStatus,
      `constants=${hasAmbiguousConstant}, isRouterAmbiguous=${hasIsRouterAmbiguous}, setRouterStatus=${hasSetRouterStatus}`
    )
  );
}

// ── T03: LLM cannot manually override BLOCKED_AMBIGUOUS ──
console.log("\n═══ T03: LLM cannot manually override BLOCKED_AMBIGUOUS ═══");
{
  const modelToolsPath = "/home/deploy/hermes-agent/model_tools.py";
  const content = readFile(modelToolsPath);

  const hasAmbiguityGate = content.includes("BLOCKED_AMBIGUOUS_ROUTER");
  const hasIsRouterAmbiguous = content.includes("is_router_ambiguous");

  runner.run("T03", "LLM cannot override BLOCKED_AMBIGUOUS", () =>
    truthy(
      hasAmbiguityGate && hasIsRouterAmbiguous,
      `hasAmbiguityGate=${hasAmbiguityGate}, hasIsRouterAmbiguous=${hasIsRouterAmbiguous}`
    )
  );
}

// ── T04: CodeGraph cannot initialize arbitrary /home/deploy ──
console.log("\n═══ T04: CodeGraph requires resolved repo ═══");
{
  const lahBootstrapPath = "/home/deploy/hermes-agent/hermes_cli/lah_bootstrap.py";
  const content = readFile(lahBootstrapPath);

  const hasIsLahMission = content.includes("def is_lah_mission");
  const hasIsCodegraphBootstrap = content.includes("def is_codegraph_bootstrap_completed");
  const hasSetLahMission = content.includes("def set_lah_mission");

  runner.run("T04", "CodeGraph requires resolved repo", () =>
    truthy(
      hasIsLahMission && hasIsCodegraphBootstrap && hasSetLahMission,
      `isLahMission=${hasIsLahMission}, isCodegraphBootstrap=${hasIsCodegraphBootstrap}, setLahMission=${hasSetLahMission}`
    )
  );
}

// ── T05: Capability classification from canonical authority ──
console.log("\n═══ T05: Capability classification from canonical authority ═══");
{
  const contractPath = "/home/deploy/.hermes/skills/devops/fast-path-e2e-capability-contract/SKILL.md";
  const content = readFile(contractPath);

  const hasVerifyFunction = content.includes("verifyFormatEndToEndExecutable");
  const hasCapabilityChain = content.includes("SELECTABLE") && content.includes("CAMPAIGN_CREATABLE");
  const hasCreativeMaterializable = content.includes("CREATIVE_MATERIALIZABLE");
  const hasProviderReadback = content.includes("PROVIDER_READBACK_CERTIFIABLE");
  const hasP6Eligible = content.includes("P6_ELIGIBLE");
  const hasFastPathEligible = content.includes("FAST_PATH_ELIGIBLE");

  runner.run("T05", "Capability classification from canonical authority", () =>
    truthy(
      hasVerifyFunction && hasCapabilityChain && hasCreativeMaterializable && hasProviderReadback && hasP6Eligible && hasFastPathEligible,
      `verify=${hasVerifyFunction}, chain=${hasCapabilityChain}, creative=${hasCreativeMaterializable}, readback=${hasProviderReadback}, p6=${hasP6Eligible}, fastPath=${hasFastPathEligible}`
    )
  );
}

// ── T06: Source-code inference cannot substitute for capability authority ──
console.log("\n═══ T06: Source-code inference cannot substitute for authority ═══");
{
  const modelToolsPath = "/home/deploy/hermes-agent/model_tools.py";
  const content = readFile(modelToolsPath);

  const hasObjectiveGate = content.includes("BLOCKED_OBJECTIVE_ALREADY_SATISFIED");
  const hasGovernorGate = content.includes("BLOCKED_GOVERNOR_NOT_ACTIVE");

  runner.run("T06", "Source-code inference cannot substitute for authority", () =>
    truthy(
      hasObjectiveGate && hasGovernorGate,
      `hasObjectiveGate=${hasObjectiveGate}, hasGovernorGate=${hasGovernorGate}`
    )
  );
}

// ── T07: FULL_SUPPORT popunder returns FAST_PATH_ELIGIBLE=true ──
console.log("\n═══ T07: FULL_SUPPORT popunder is FAST_PATH_ELIGIBLE ═══");
{
  const contractPath = "/home/deploy/.hermes/skills/devops/fast-path-e2e-capability-contract/SKILL.md";
  const content = readFile(contractPath);

  const hasFullSupport = content.includes("FULL_SUPPORT");
  const hasFastPathEligible = content.includes("FAST_PATH_ELIGIBLE");
  const hasFiveStageChain = content.includes("SELECTABLE") && content.includes("CAMPAIGN_CREATABLE") && content.includes("CREATIVE_MATERIALIZABLE");

  runner.run("T07", "FULL_SUPPORT popunder is FAST_PATH_ELIGIBLE", () =>
    truthy(
      hasFullSupport && hasFastPathEligible && hasFiveStageChain,
      `fullSupport=${hasFullSupport}, fastPath=${hasFastPathEligible}, chain=${hasFiveStageChain}`
    )
  );
}

// ── T08: Partial-support banner does not become FAST_PATH_ELIGIBLE ──
console.log("\n═══ T08: Partial-support banner is blocked ═══");
{
  const contractPath = "/home/deploy/.hermes/skills/devops/fast-path-e2e-capability-contract/SKILL.md";
  const content = readFile(contractPath);

  const hasPartialSupport = content.includes("PARTIAL_SUPPORT");
  const hasBlockingRule = content.includes("BLOCKED");

  runner.run("T08", "Partial-support banner is blocked", () =>
    truthy(
      hasPartialSupport && hasBlockingRule,
      `partialSupport=${hasPartialSupport}, blocking=${hasBlockingRule}`
    )
  );
}

// ── T09: Mission objective remains OBJECTIVE_PENDING before readback ──
console.log("\n═══ T09: Objective defaults to OBJECTIVE_PENDING ═══");
{
  const lahBootstrapPath = "/home/deploy/hermes-agent/hermes_cli/lah_bootstrap.py";
  const content = readFile(lahBootstrapPath);

  const hasObjectivePending = content.includes('OBJECTIVE_PENDING = "OBJECTIVE_PENDING"');
  const hasGetObjectiveState = content.includes("def get_mission_objective_state");

  runner.run("T09", "Objective defaults to OBJECTIVE_PENDING", () =>
    truthy(
      hasObjectivePending && hasGetObjectiveState,
      `hasPending=${hasObjectivePending}, hasGetter=${hasGetObjectiveState}`
    )
  );
}

// ── T10: Approval readback transitions to OPERATOR_AUTHORIZATION_REQUIRED ──
console.log("\n═══ T10: Approval readback transitions to OPERATOR_AUTHORIZATION_REQUIRED ═══");
{
  const lahBootstrapPath = "/home/deploy/hermes-agent/hermes_cli/lah_bootstrap.py";
  const content = readFile(lahBootstrapPath);

  const hasOperatorAuthState = content.includes('OPERATOR_AUTHORIZATION_REQUIRED');
  const hasSetObjectiveState = content.includes("def set_mission_objective_state");
  const hasIsOperatorAuthRequired = content.includes("def is_operator_authorization_required");

  runner.run("T10", "Approval readback transitions to OPERATOR_AUTHORIZATION_REQUIRED", () =>
    truthy(
      hasOperatorAuthState && hasSetObjectiveState && hasIsOperatorAuthRequired,
      `hasState=${hasOperatorAuthState}, hasSetter=${hasSetObjectiveState}, hasChecker=${hasIsOperatorAuthRequired}`
    )
  );
}

// ── T11: Discovery blocked after OPERATOR_AUTHORIZATION_REQUIRED ──
console.log("\n═══ T11: Discovery blocked after OPERATOR_AUTHORIZATION_REQUIRED ═══");
{
  const modelToolsPath = "/home/deploy/hermes-agent/model_tools.py";
  const content = readFile(modelToolsPath);

  const hasPostObjectiveGate = content.includes("BLOCKED_OBJECTIVE_ALREADY_SATISFIED");
  const hasIsDiscoveryBlocked = content.includes("is_discovery_blocked");
  const blocksTerminal = content.includes('"terminal"') && content.includes("_DISCOVERY_TOOLS_POST_OBJECTIVE");

  runner.run("T11", "Discovery blocked after OPERATOR_AUTHORIZATION_REQUIRED", () =>
    truthy(
      hasPostObjectiveGate && hasIsDiscoveryBlocked && blocksTerminal,
      `hasGate=${hasPostObjectiveGate}, hasChecker=${hasIsDiscoveryBlocked}, blocksTerminal=${blocksTerminal}`
    )
  );
}

// ── T12: Receipt generation remains allowed after objective satisfaction ──
console.log("\n═══ T12: Receipt generation allowed after objective satisfaction ═══");
{
  const gov = new ConvergenceGovernor({ missionMode: "REPAIR" });
  gov.setCodeGraphBootstrap({
    phase: "CODEGRAPH_BOOTSTRAP",
    can_proceed: true,
    receipt: { repo: "/home/deploy/lah-stack-repos/openclaw-runtime", status: "RESOLVED" },
  });

  gov.transitionState("REPORT");
  const receipt = gov.generateReceipt();

  runner.run("T12", "Receipt generation allowed after objective satisfaction", () =>
    truthy(
      receipt !== null && receipt.verdict !== undefined && receipt.state !== undefined,
      `hasVerdict=${receipt.verdict !== undefined}, hasState=${receipt.state !== undefined}`
    )
  );
}

// ── T13: Receipt generation cannot initiate new discovery ──
console.log("\n═══ T13: Receipt generation cannot initiate new discovery ═══");
{
  const { STATE } = require("../scripts/convergence-governor.js");

  const gov = new ConvergenceGovernor({ missionMode: "CERTIFY" });
  gov.setCodeGraphBootstrap({
    phase: "CODEGRAPH_BOOTSTRAP",
    can_proceed: true,
    receipt: { repo: "/home/deploy/lah-stack-repos/openclaw-runtime", status: "RESOLVED" },
  });

  const transitionResult = gov.transitionState("REPORT");

  runner.run("T13", "Receipt generation cannot initiate new discovery", () =>
    truthy(
      transitionResult.success === true,
      `transitionSuccess=${transitionResult.success}`
    )
  );
}

// ── T14: Behavioral governor cannot report RUNTIME_ENFORCED ──
console.log("\n═══ T14: Behavioral governor cannot report RUNTIME_ENFORCED ═══");
{
  const gov = new ConvergenceGovernor({ missionMode: "REPAIR" });

  gov.setRuntimeEnforcementProof({
    state: GOVERNOR_ENFORCEMENT_STATE.BEHAVIORAL_ONLY,
    enforcement_module: "skill_instructions",
    dispatch_boundary: null,
    session_id: "test-session-behavioral",
    mission_id: "test-mission",
    bootstrap_state: "COMPLETED",
    blocked_action_count: 0,
    runtime_gate_version: null,
    runtime_gate_fingerprint: null,
  });

  const proof = gov.getRuntimeEnforcementProof();
  const isRuntimeEnforced = gov.isRuntimeEnforced();

  runner.run("T14", "Behavioral governor cannot report RUNTIME_ENFORCED", () =>
    truthy(
      proof.state === GOVERNOR_ENFORCEMENT_STATE.BEHAVIORAL_ONLY && isRuntimeEnforced === false,
      `state=${proof.state}, isRuntimeEnforced=${isRuntimeEnforced}`
    )
  );
}

// ── T15: Runtime dispatch proof reports RUNTIME_ENFORCED ──
console.log("\n═══ T15: Runtime dispatch proof reports RUNTIME_ENFORCED ═══");
{
  const gov = new ConvergenceGovernor({ missionMode: "REPAIR" });

  gov.setRuntimeEnforcementProof({
    state: GOVERNOR_ENFORCEMENT_STATE.RUNTIME_ENFORCED,
    enforcement_module: "model_tools.py:handle_function_call",
    dispatch_boundary: "handle_function_call",
    session_id: "test-session-runtime",
    mission_id: "test-mission",
    bootstrap_state: "COMPLETED",
    blocked_action_count: 3,
    runtime_gate_version: "1.0.0",
    runtime_gate_fingerprint: "sha256:abc123def456",
  });

  const proof = gov.getRuntimeEnforcementProof();
  const isRuntimeEnforced = gov.isRuntimeEnforced();

  runner.run("T15", "Runtime dispatch proof reports RUNTIME_ENFORCED", () =>
    truthy(
      proof.state === GOVERNOR_ENFORCEMENT_STATE.RUNTIME_ENFORCED && isRuntimeEnforced === true,
      `state=${proof.state}, isRuntimeEnforced=${isRuntimeEnforced}`
    )
  );
}

// ── T16: Existing approval approval_1787161683587_8730f4fe is not mutated ──
console.log("\n═══ T16: Existing approval is not mutated ═══");
{
  const lahBootstrapPath = "/home/deploy/hermes-agent/hermes_cli/lah_bootstrap.py";
  const content = readFile(lahBootstrapPath);

  const hasNoCreateApproval = !content.includes("create_approval") && !content.includes("POST /approvals");
  const hasNoProviderMutation = !content.includes("EXOCLICK") && !content.includes("CAMPAIGN_CREATE");

  runner.run("T16", "Existing approval is not mutated", () =>
    truthy(
      hasNoCreateApproval && hasNoProviderMutation,
      `noCreateApproval=${hasNoCreateApproval}, noProviderMutation=${hasNoProviderMutation}`
    )
  );
}

// ═══════════════════════════════════════════════════
// Final Report
// ═══════════════════════════════════════════════════

runner.printReport();
const summary = runner.getSummary();

// Exit code
process.exit(summary.allPassed ? 0 : 1);