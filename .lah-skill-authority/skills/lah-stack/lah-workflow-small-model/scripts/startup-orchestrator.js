#!/usr/bin/env node
/**
 * Hermes Certified Startup Orchestrator v1
 *
 * Integrates Certified Architecture Context + Mission Resume Packet
 * + mandatory CodeGraph bootstrap into the lah-workflow-small-model
 * startup sequence.
 *
 * Startup order (P11 + CodeGraph mandatory bootstrap):
 *   1. LOAD_CONTEXT
 *   2. VERIFY_FINGERPRINT
 *   3. LOAD_RESUME_PACKET
 *   4. CODEGRAPH_BOOTSTRAP  ← NEW: mandatory CodeGraph freshness check
 *                              + mission context packet load
 *   5. IDENTIFY_NEXT_ACTION
 *   6. EXECUTE
 *
 * Only if:
 *   - context missing
 *   - context drift
 *   - blocking unknown not represented
 * may ORIENT/DISCOVER begin.
 *
 * @module startup-orchestrator
 * @version 1.1.0
 */

"use strict";

const { execFileSync } = require("child_process");
const { join } = require("path");
const { CertifiedArchitectureContext } = require("./certified-architecture-context");
const { MissionResumePacket } = require("./mission-resume-packet");

// ─── CodeGraph Bootstrap ────────────────────────────────────
// Paths are relative to the openclaw-runtime repo root,
// resolved from the skill directory via process.cwd() or
// a known relative path from the startup orchestrator.

const OPENCLAW_RUNTIME_ROOT = join(
  __dirname,
  "..",
  "..",
  "..",
  "..",
  "..",
  "openclaw-runtime",
  "lah-openclaw-mvp"
);

const CODEGRAPH_TOOLS_DIR = join(
  OPENCLAW_RUNTIME_ROOT,
  "tools",
  "codegraph"
);

const CODEGRAPH_DIR = join(OPENCLAW_RUNTIME_ROOT, ".codegraph");

/**
 * Run the CodeGraph freshness check via the existing CLI tool.
 * Returns { fresh: boolean, error?: string, output?: string }
 */
function checkCodeGraphFreshness() {
  try {
    const result = execFileSync(
      "node",
      [join(CODEGRAPH_TOOLS_DIR, "freshness-check.js"), "--repo", "openclaw"],
      { encoding: "utf8", timeout: 10000, stdio: ["ignore", "pipe", "pipe"] }
    );
    return { fresh: true, output: result };
  } catch (err) {
    // exit code 1 = missing or stale
    return { fresh: false, error: err.message, output: err.stdout || "" };
  }
}

/**
 * Run the CodeGraph mission context pack via the existing CLI tool.
 * Returns { success: boolean, packet?: object, error?: string }
 */
function loadCodeGraphContext(missionId) {
  try {
    const args = [
      join(CODEGRAPH_TOOLS_DIR, "mission-context-pack.js"),
      "--repo",
      "openclaw",
      "--mission",
      missionId || "unknown",
      "--json",
      "--compact",
    ];
    const result = execFileSync("node", args, {
      encoding: "utf8",
      timeout: 15000,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const packet = JSON.parse(result);
    return { success: true, packet };
  } catch (err) {
    return {
      success: false,
      error: err.message,
      output: err.stdout || "",
    };
  }
}

/**
 * Refresh the CodeGraph pack via the existing CLI tool.
 * Returns { success: boolean, error?: string }
 */
function refreshCodeGraphPack() {
  try {
    const result = execFileSync(
      "node",
      [join(CODEGRAPH_TOOLS_DIR, "refresh-pack.js"), "--repo", "openclaw"],
      { encoding: "utf8", timeout: 30000, stdio: ["ignore", "pipe", "pipe"] }
    );
    return { success: true, output: result };
  } catch (err) {
    return { success: false, error: err.message, output: err.stdout || "" };
  }
}

/**
 * lah_context_resolve — mandatory CodeGraph bootstrap for mission startup.
 *
 * Checks CodeGraph freshness, refreshes if stale, and loads the mission
 * context packet. Returns a structured receipt.
 *
 * @param {object} options
 * @param {string} [options.missionId] - Mission ID for context packet
 * @returns {{ phase: string, can_proceed: boolean, receipt?: object, error?: string }}
 */
function lah_context_resolve(options = {}) {
  const { missionId } = options;

  // Step 1: Check freshness
  const freshness = checkCodeGraphFreshness();

  if (!freshness.fresh) {
    // Step 2: Refresh if stale/missing
    const refresh = refreshCodeGraphPack();
    if (!refresh.success) {
      return {
        phase: "CODEGRAPH_BOOTSTRAP",
        can_proceed: false,
        error: `CodeGraph refresh failed: ${refresh.error}`,
        freshness: { fresh: false, reason: "refresh_failed" },
      };
    }
  }

  // Step 3: Load mission context packet
  const context = loadCodeGraphContext(missionId);

  if (!context.success) {
    return {
      phase: "CODEGRAPH_BOOTSTRAP",
      can_proceed: false,
      error: `CodeGraph context load failed: ${context.error}`,
      freshness: { fresh: freshness.fresh, reason: "context_load_failed" },
    };
  }

  return {
    phase: "CODEGRAPH_BOOTSTRAP",
    can_proceed: true,
    receipt: {
      fresh: freshness.fresh,
      packet: context.packet,
      mission_id: missionId || "unknown",
      loaded_at: new Date().toISOString(),
    },
  };
}

// ─── Startup Orchestrator ──────────────────────────────────────

class CertifiedStartupOrchestrator {
  constructor(options = {}) {
    this.context = new CertifiedArchitectureContext(options);
    this.resumePacket = new MissionResumePacket(options);
    this.fingerprintValue = null;
    this.startupPhase = "NOT_STARTED";
    this.startupActions = [];
    this.codegraphBootstrapReceipt = null;
    this.metrics = {
      startup_tool_calls: 0,
      certified_fact_rediscovery_attempts: 0,
      certified_fact_rediscovery_blocked: 0,
      architecture_context_load_ms: 0,
      fingerprint_check_ms: 0,
      resume_packet_load_ms: 0,
      codegraph_bootstrap_ms: 0,
      targeted_drift_discovery_calls: 0,
      startup_orientation_actions: 0,
      startup_orientation_blocked: 0,
    };
  }

  /**
   * Execute the certified startup sequence.
   *
   * @param {object} options
   * @param {string} [options.missionId] - Mission ID for resume packet lookup
   * @param {string} [options.missionType] - Mission type
   * @param {boolean} [options.forceOrientation] - Force full orientation even with context
   * @returns {{ phase: string, actions: string[], result: object }}
   */
  async startup(options = {}) {
    const { missionId, missionType, forceOrientation = false } = options;
    this.startupPhase = "STARTED";
    const actions = [];

    // ── Step 1: Load Certified Architecture Context ──
    this.startupPhase = "LOAD_CONTEXT";
    const loadResult = this.context.load();
    actions.push("LOAD_CONTEXT");
    this.metrics.startup_tool_calls++;
    this.metrics.architecture_context_load_ms = loadResult.load_ms || 0;

    if (!loadResult.success) {
      return {
        phase: "LOAD_CONTEXT_FAILED",
        actions,
        result: { error: loadResult.reason, can_proceed: false },
      };
    }

    // ── Step 2: Compare fingerprints ──
    this.startupPhase = "VERIFY_FINGERPRINT";
    // Stored fingerprint comes from resume packet (loaded in next step)
    // For now, check against stored fingerprint from context
    const storedFingerprint = this.context.getStoredFingerprint
      ? this.context.getStoredFingerprint()
      : null;
    const fingerprintResult = this.context.verifyFingerprint(storedFingerprint);
    actions.push("VERIFY_FINGERPRINT");
    this.metrics.startup_tool_calls++;
    this.metrics.fingerprint_check_ms = fingerprintResult.check_ms || 0;
    this.fingerprintValue = fingerprintResult.current;

    // ── Step 3: Load Mission Resume Packet if present ──
    this.startupPhase = "LOAD_RESUME_PACKET";
    const resumeResult = this.resumePacket.load(missionId);
    actions.push("LOAD_RESUME_PACKET");
    this.metrics.startup_tool_calls++;
    this.metrics.resume_packet_load_ms = resumeResult.load_ms || 0;

    // ── Step 4: Mandatory CodeGraph Bootstrap ──
    this.startupPhase = "CODEGRAPH_BOOTSTRAP";
    const cgStartTime = Date.now();
    const codegraphResult = lah_context_resolve({
      missionId,
    });
    this.metrics.codegraph_bootstrap_ms = Date.now() - cgStartTime;
    this.metrics.startup_tool_calls++;

    if (!codegraphResult.can_proceed) {
      // CodeGraph bootstrap failed — block mission start
      return {
        phase: "CODEGRAPH_BOOTSTRAP_FAILED",
        actions,
        result: {
          error: codegraphResult.error,
          can_proceed: false,
          codegraph_bootstrap_failed: true,
          freshness: codegraphResult.freshness,
        },
      };
    }

    this.codegraphBootstrapReceipt = codegraphResult.receipt;
    actions.push("CODEGRAPH_BOOTSTRAP");

    // ── Step 5: Identify next_action ──
    this.startupPhase = "IDENTIFY_NEXT_ACTION";

    // If resume packet exists and fingerprint is unchanged, resume directly
    if (resumeResult.success && fingerprintResult.unchanged && !forceOrientation) {
      const packet = resumeResult.packet;

      // P8: Honor checkpoint — do NOT inspect architecture
      actions.push("HONOR_CHECKPOINT");
      this.metrics.startup_tool_calls++;

      // Record startup action
      const budgetResult = this.context.recordStartupAction("LOAD_CONTEXT");
      if (!budgetResult.allowed) {
        this.metrics.startup_orientation_blocked++;
        return {
          phase: "STARTUP_BUDGET_EXHAUSTED",
          actions,
          result: { error: budgetResult.reason, can_proceed: false },
        };
      }

      const budgetResult2 = this.context.recordStartupAction("VERIFY_FINGERPRINT");
      if (!budgetResult2.allowed) {
        this.metrics.startup_orientation_blocked++;
        return {
          phase: "STARTUP_BUDGET_EXHAUSTED",
          actions,
          result: { error: budgetResult2.reason, can_proceed: false },
        };
      }

      const budgetResult3 = this.context.recordStartupAction("READ_EXACT_OBJECT");
      if (!budgetResult3.allowed) {
        this.metrics.startup_orientation_blocked++;
        return {
          phase: "STARTUP_BUDGET_EXHAUSTED",
          actions,
          result: { error: budgetResult3.reason, can_proceed: false },
        };
      }

      // P8: Resume directly — next_action from packet
      const nextAction = packet.next_action;

      return {
        phase: "RESUME_DIRECT",
        actions,
        result: {
          resume_packet_loaded: true,
          fingerprint_unchanged: true,
          checkpoint: packet.current_checkpoint,
          next_action: nextAction,
          approval_id: this.resumePacket.getApprovalId(),
          known_facts: packet.known_facts,
          can_proceed: true,
          startup_tool_calls: this.metrics.startup_tool_calls,
          startup_actions_count: actions.length,
        },
      };
    }

    // ── Context drift or no resume packet ──
    if (!fingerprintResult.unchanged) {
      // P3: Context drift — targeted rediscovery allowed only for changed components
      this.startupPhase = "CONTEXT_DRIFT";
      actions.push("TARGETED_DRIFT_DISCOVERY");
      this.metrics.targeted_drift_discovery_calls++;
      this.metrics.startup_tool_calls++;

      return {
        phase: "CONTEXT_DRIFT_DETECTED",
        actions,
        result: {
          fingerprint_unchanged: false,
          current_fingerprint: fingerprintResult.current,
          stored_fingerprint: fingerprintResult.stored,
          can_proceed: true,
          discovery_scope: "targeted_only_changed_components",
          startup_tool_calls: this.metrics.startup_tool_calls,
        },
      };
    }

    // ── No resume packet — normal startup with budget ──
    this.startupPhase = "ORIENT";
    actions.push("ORIENT");

    // P7: Check startup budget
    const budgetCheck = this.context.recordStartupAction("LOAD_CONTEXT");
    if (!budgetCheck.allowed) {
      this.metrics.startup_orientation_blocked++;
      return {
        phase: "STARTUP_ARCHAEOLOGY_BLOCKED",
        actions,
        result: { error: budgetCheck.reason, can_proceed: false },
      };
    }

    // If we get here with no resume packet, we need to orient
    // but are limited to 3 actions
    const remainingBudget = this.context.budget.remaining();

    return {
      phase: "ORIENTATION_PHASE",
      actions,
      result: {
        resume_packet_loaded: false,
        fingerprint_unchanged: true,
        remaining_budget: remainingBudget,
        can_proceed: remainingBudget > 0,
        startup_tool_calls: this.metrics.startup_tool_calls,
        startup_actions_count: actions.length,
      },
    };
  }

  /**
   * P5+P6: Evaluate a discovery action against the no-rediscovery gate.
   * @param {object} action - The discovery action
   * @param {object} options - Context options
   * @returns {{ allowed: boolean, reason: string, blocked: boolean }}
   */
  evaluateDiscovery(action, options = {}) {
    this.metrics.startup_tool_calls++;

    if (action.type === "rediscover") {
      this.metrics.certified_fact_rediscovery_attempts++;
    }

    const result = this.context.evaluateDiscovery(action, options);

    if (result.blocked) {
      this.metrics.certified_fact_rediscovery_blocked++;
    }

    return result;
  }

  /**
   * Get the current orchestrator state.
   */
  getState() {
    return {
      phase: this.startupPhase,
      fingerprint: this.fingerprintValue,
      actions: [...this.startupActions],
      metrics: { ...this.metrics },
      budget: this.context.budget.getState(),
      gate_metrics: this.context.gate.getMetrics(),
    };
  }

  /**
   * Generate a startup receipt.
   */
  generateReceipt() {
    return {
      verdict: "HERMES_CERTIFIED_STARTUP_ORCHESTRATOR_ACTIVE",
      phase: this.startupPhase,
      fingerprint: this.fingerprintValue,
      actions: [...this.startupActions],
      metrics: { ...this.metrics },
      budget: this.context.budget.getState(),
      gate_metrics: this.context.gate.getMetrics(),
      timestamp: new Date().toISOString(),
    };
  }
}

// ─── Export ──────────────────────────────────────────────

module.exports = {
  CertifiedStartupOrchestrator,
  CertifiedArchitectureContext,
  MissionResumePacket,
  lah_context_resolve,
};