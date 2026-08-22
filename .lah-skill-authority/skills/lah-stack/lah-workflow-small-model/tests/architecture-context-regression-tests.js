#!/usr/bin/env node
/**
 * Certified Architecture Context & Resume Packet — Regression Tests
 *
 * P13: 10 regression scenarios replaying recent Canary V3 startup.
 *
 * BEFORE (old behavior):
 *   skills → env → LAHB probes → 4000 probes → 18789 probes →
 *   find config → ps → route inspection → receipt inspection → API guessing → ...
 *
 * AFTER (expected):
 *   load context → verify fingerprint → GET exact approval → approve → execute
 *
 * Test cases:
 *   1. known LAHB URL rediscovery → BLOCK
 *   2. known port 4000 role rediscovery → BLOCK
 *   3. known port 18789 role rediscovery → BLOCK
 *   4. OPENCLAW_INTERNAL_URL rediscovery → BLOCK
 *   5. exact approval readback → ALLOW
 *   6. context fingerprint drift → targeted rediscovery ALLOW
 *   7. contradictory runtime evidence → targeted investigation ALLOW
 *   8. unrelated architecture search → BLOCK
 *   9. resume packet checkpoint honored
 *   10. next_action executed without orientation
 */

"use strict";

const assert = require("assert");
const path = require("path");

const { CertifiedArchitectureContext } = require("../scripts/certified-architecture-context");
const { MissionResumePacket } = require("../scripts/mission-resume-packet");
const { CertifiedStartupOrchestrator } = require("../scripts/startup-orchestrator");

// ─── Test Runner ──────────────────────────────────────────────

class RegressionTestRunner {
  constructor() {
    this.results = [];
    this.passed = 0;
    this.failed = 0;
    this.blocked = 0;
  }

  /**
   * Run a single test case.
   * @param {number} id - Test case number
   * @param {string} name - Test case name
   * @param {function} testFn - Async test function returning { pass: boolean, detail: string }
   */
  async run(id, name, testFn) {
    let result;
    try {
      result = await testFn();
    } catch (e) {
      result = { pass: false, detail: `ERROR: ${e.message}` };
    }

    const entry = { id, name, ...result };
    this.results.push(entry);

    if (result.pass) {
      this.passed++;
      console.log(`  ✅ P13-C${id}: ${name}`);
    } else if (result.blocked) {
      this.blocked++;
      console.log(`  🚫 P13-C${id}: ${name} — BLOCKED (expected)`);
    } else {
      this.failed++;
      console.log(`  ❌ P13-C${id}: ${name} — FAILED`);
      console.log(`     Detail: ${result.detail}`);
    }

    return entry;
  }

  /**
   * Run all regression tests.
   */
  async runAll() {
    console.log("\n═══ P13 Regression Tests: Certified Architecture Context & Resume Packet ═══\n");

    // Setup shared instances
    const context = new CertifiedArchitectureContext();
    context.load();

    // ── C1: known LAHB URL rediscovery → BLOCK ──
    await this.run(1, "known LAHB URL rediscovery → BLOCK", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: "LAHB",
          description: "find LAHB URL again",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: false }
      );

      const pass = result.blocked === true && result.allowed === false;
      return { pass, detail: `blocked=${result.blocked}, allowed=${result.allowed}, reason="${result.reason}"` };
    });

    // ── C2: known port 4000 role rediscovery → BLOCK ──
    await this.run(2, "known port 4000 role rediscovery → BLOCK", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: "PORT_4000",
          description: "probe port 4000 again",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: false }
      );

      const pass = result.blocked === true && result.allowed === false;
      return { pass, detail: `blocked=${result.blocked}, allowed=${result.allowed}, reason="${result.reason}"` };
    });

    // ── C3: known port 18789 role rediscovery → BLOCK ──
    await this.run(3, "known port 18789 role rediscovery → BLOCK", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: "PORT_18789",
          description: "probe port 18789 again",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: false }
      );

      const pass = result.blocked === true && result.allowed === false;
      return { pass, detail: `blocked=${result.blocked}, allowed=${result.allowed}, reason="${result.reason}"` };
    });

    // ── C4: OPENCLAW_INTERNAL_URL rediscovery → BLOCK ──
    await this.run(4, "OPENCLAW_INTERNAL_URL rediscovery → BLOCK", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: "OPENCLAW_INTERNAL_URL",
          description: "search OPENCLAW_INTERNAL_URL precedence again",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: false }
      );

      const pass = result.blocked === true && result.allowed === false;
      return { pass, detail: `blocked=${result.blocked}, allowed=${result.allowed}, reason="${result.reason}"` };
    });

    // ── C5: exact approval readback → ALLOW ──
    await this.run(5, "exact approval readback → ALLOW", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "exact_readback",
          factKey: "LAHB",
          description: "GET /approvals/:id for exact approval",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: false }
      );

      const pass = result.allowed === true && result.blocked === false;
      return { pass, detail: `allowed=${result.allowed}, blocked=${result.blocked}, reason="${result.reason}"` };
    });

    // ── C6: context fingerprint drift → targeted rediscovery ALLOW ──
    await this.run(6, "context fingerprint drift → targeted rediscovery ALLOW", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: "LAHB",
          description: "recheck LAHB URL after drift",
        },
        { contextFingerprintUnchanged: false, hasContradictoryEvidence: false, contextDrift: true }
      );

      const pass = result.allowed === true && result.blocked === false;
      return { pass, detail: `allowed=${result.allowed}, blocked=${result.blocked}, reason="${result.reason}"` };
    });

    // ── C7: contradictory runtime evidence → targeted investigation ALLOW ──
    await this.run(7, "contradictory runtime evidence → targeted investigation ALLOW", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: "PORT_4000",
          description: "inspect runtime service boundary after failure",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: true }
      );

      const pass = result.allowed === true && result.blocked === false;
      return { pass, detail: `allowed=${result.allowed}, blocked=${result.blocked}, reason="${result.reason}"` };
    });

    // ── C8: unrelated architecture search → BLOCK ──
    await this.run(8, "unrelated architecture search → BLOCK", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      const result = orchestrator.evaluateDiscovery(
        {
          type: "rediscover",
          factKey: null,
          description: "find .env files and grep for architecture",
        },
        { contextFingerprintUnchanged: true, hasContradictoryEvidence: false }
      );

      const pass = result.blocked === true && result.allowed === false;
      return { pass, detail: `blocked=${result.blocked}, allowed=${result.allowed}, reason="${result.reason}"` };
    });

    // ── C9: resume packet checkpoint honored ──
    await this.run(9, "resume packet checkpoint honored", () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      // Create a resume packet for a known mission
      const packet = orchestrator.resumePacket.create({
        mission_id: "LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3",
        mission_type: "BUILD_AND_CERTIFY",
        current_checkpoint: "PENDING_OPERATOR_APPROVAL",
        next_action: "APPROVE_AND_EXECUTE",
        known_facts: {
          LAHB_URL: "https://leanframeworklab.com",
          internal_runtime: "lah-openclaw-mvp:4000",
          execution_seam: "LAHB → governed runtime → compiler → ExoClick",
          banner_mapping_certified: true,
          approval_readback_verified: true,
        },
        resolved_blockers: [],
        blocking_unknowns: [],
        authorization_state: { operator_approved: false, approval_id: "approval_1787149585840_d8937b09" },
        provider_state: { campaign_id: null, status: "CREATED_PAUSED_READY_TO_PLAY" },
        approval_ids: ["approval_1787149585840_d8937b09"],
        campaign_ids: [],
        compiled_packet_ids: [],
        canonical_context_fingerprint: orchestrator.fingerprintValue || "test-fingerprint",
        forbidden_rediscovery: ["LAHB", "PORT_4000", "PORT_18789", "OPENCLAW_INTERNAL_URL", "CAMPAIGN_EXECUTION_PATH"],
        last_verified_at: new Date().toISOString(),
      });

      // Load it back
      const loadResult = orchestrator.resumePacket.load("LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3");

      const pass = loadResult.success && loadResult.packet.current_checkpoint === "PENDING_OPERATOR_APPROVAL" && loadResult.packet.next_action === "APPROVE_AND_EXECUTE";
      return { pass, detail: `checkpoint="${loadResult.packet?.current_checkpoint}", next_action="${loadResult.packet?.next_action}"` };
    });

    // ── C10: next_action executed without orientation ──
    await this.run(10, "next_action executed without orientation", async () => {
      const orchestrator = new CertifiedStartupOrchestrator();
      orchestrator.context.load();

      // Get the actual fingerprint from the context
      orchestrator.fingerprintValue = orchestrator.context.fingerprint.getCurrent();

      // Create a resume packet for a known mission with the correct fingerprint
      orchestrator.resumePacket.create({
        mission_id: "LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3",
        mission_type: "BUILD_AND_CERTIFY",
        current_checkpoint: "PENDING_OPERATOR_APPROVAL",
        next_action: "APPROVE_AND_EXECUTE",
        known_facts: {
          LAHB_URL: "https://leanframeworklab.com",
          internal_runtime: "lah-openclaw-mvp:4000",
          execution_seam: "LAHB → governed runtime → compiler → ExoClick",
          banner_mapping_certified: true,
          approval_readback_verified: true,
        },
        resolved_blockers: [],
        blocking_unknowns: [],
        authorization_state: { operator_approved: false, approval_id: "approval_1787149585840_d8937b09" },
        provider_state: { campaign_id: null, status: "CREATED_PAUSED_READY_TO_PLAY" },
        approval_ids: ["approval_1787149585840_d8937b09"],
        campaign_ids: [],
        compiled_packet_ids: [],
        canonical_context_fingerprint: orchestrator.fingerprintValue,
        forbidden_rediscovery: ["LAHB", "PORT_4000", "PORT_18789", "OPENCLAW_INTERNAL_URL", "CAMPAIGN_EXECUTION_PATH"],
        last_verified_at: new Date().toISOString(),
      });

      // Simulate startup with resume packet
      const startupResult = await orchestrator.startup({
        missionId: "LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3",
        missionType: "BUILD_AND_CERTIFY",
      });

      // The key assertion: when a resume packet exists and fingerprint is unchanged,
      // the orchestrator should resume directly (not go through ORIENT)
      const directResume = startupResult.result?.resume_packet_loaded === true;
      const correctNextAction = startupResult.result?.next_action === "APPROVE_AND_EXECUTE";

      return { pass: directResume === true && correctNextAction === true, detail: `phase="${startupResult.phase}", resume_packet_loaded=${directResume}, next_action="${startupResult.result?.next_action}", checkpoint="${startupResult.result?.checkpoint}"` };
    });

    // ── Summary ──
    console.log("\n═══ P13 Regression Summary ═══");
    console.log(`  Total: ${this.results.length}`);
    console.log(`  Passed: ${this.passed}`);
    console.log(`  Failed: ${this.failed}`);
    console.log(`  Blocked (expected): ${this.blocked}`);

    const allPass = this.failed === 0 && this.passed === this.results.length;
    console.log(`  Verdict: ${allPass ? "ALL PASS" : "SOME FAILED"}\n`);

    return { allPass, passed: this.passed, failed: this.failed, total: this.results.length, results: this.results };
  }
}

// ─── Run if executed directly ──────────────────────────────

if (require.main === module) {
  const runner = new RegressionTestRunner();
  runner.runAll().then((summary) => {
    process.exit(summary.allPass ? 0 : 1);
  });
}

module.exports = { RegressionTestRunner };