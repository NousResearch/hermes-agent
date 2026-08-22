#!/usr/bin/env node
/**
 * Hermes Certified Architecture Context v1
 *
 * Machine-readable canonical operational context registry for the LAH Stack.
 * Eliminates repeated architecture rediscovery at mission startup.
 *
 * Implements:
 *   P1 — Certified Architecture Context registry
 *   P2 — Current certified facts seeding
 *   P3 — Architecture fingerprint (LAH_ARCHITECTURE_FINGERPRINT)
 *   P5 — No-rediscovery gate (CERTIFIED_FACT_REDISCOVERY_BLOCKED)
 *   P6 — Allowed revalidation vs rediscovery distinction
 *   P7 — Startup budget (max 3 orientation actions)
 *   P9 — Context update policy (incremental only)
 *   P10 — Contradiction handling (CERTIFIED_CONTEXT_CONTRADICTION)
 *   P14 — Metrics instrumentation
 *
 * @module certified-architecture-context
 * @version 1.0.0
 */

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

// ─── Constants ───────────────────────────────────────────────────────────

const CONTEXT_PATH = path.join(
  __dirname,
  "..",
  "data",
  "certified-architecture-context.json"
);

const FINGERPRINT_PATH = path.join(
  __dirname,
  "..",
  "data",
  "architecture-fingerprint.json"
);

const RESUME_PACKET_PATH = path.join(
  __dirname,
  "..",
  "data",
  "mission-resume-packet.json"
);

const STALENESS_POLICY = Object.freeze({
  // Facts that are stable and never stale unless contradicted
  permanent: "never_stale_without_contradiction",
  // Facts that should be re-verified on each mission
  session: "reverify_each_session",
  // Facts that are stale after 24 hours
  daily: "stale_after_24h",
  // Facts that are stale after 1 hour
  hourly: "stale_after_1h",
});

const STARTUP_BUDGET = Object.freeze({
  max_orientation_actions: 3,
  allowed_actions: [
    "LOAD_CONTEXT",
    "VERIFY_FINGERPRINT",
    "READ_EXACT_OBJECT",
  ],
});

// ─── Certified Facts Registry (P1+P2) ──────────────────────────────────

/**
 * Canonical operational context registry.
 * Seeded ONLY from already-certified evidence (P2).
 * Do NOT rediscover these during seeding if existing receipts/current
 * canonical evidence already certify them.
 */
const CERTIFIED_FACTS = Object.freeze({
  // ── Repo Ownership ──
  REPO_OWNERSHIP: {
    value: {
      workspace_clone: "/home/deploy/openclaw-runtime",
      canonical_checkout: "/home/deploy/lah-stack-repos/openclaw-runtime",
      remote: "github.com-lah-stack",
      router_status: "G0_UNRESOLVED_bypassed_direct_reading",
    },
    source_authority: "P0_authority_map.md (2026-08-18)",
    fingerprint: "a1b2c3d4-repo-ownership",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── LAHB ──
  LAHB: {
    value: {
      approval_authority: "LAHB (lah-brain)",
      canonical_url: "https://leanframeworklab.com",
      env_var: "LAHB_URL",
      default_url: "https://leanframeworklab.com",
      admin_api_key_env: "LAHB_ADMIN_API_KEY",
      interface: "REST API /approval_queue + /approvals/:id",
      role: "governance and approval authority",
    },
    source_authority: "P0_authority_map.md + lah-workflow-small-model SKILL.md",
    fingerprint: "e5f6a7b8-lahb-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── OpenClaw Runtime ──
  OPENCLAW_RUNTIME: {
    value: {
      governed_execution_runtime: "lah-openclaw-mvp container",
      role: "governed execution runtime for campaign operations",
      container_name: "lah-openclaw-mvp",
      code_location: "openclaw-runtime/",
      governance_boundary: "approvals in lah-brain (LAHB), execution in openclaw-runtime",
    },
    source_authority: "P0_authority_map.md + lah-workflow-small-model SKILL.md",
    fingerprint: "c9d0e1f2-openclaw-runtime",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── OPENCLAW_INTERNAL_URL ──
  OPENCLAW_INTERNAL_URL: {
    value: {
      url: "http://lah-openclaw-mvp:4000",
      preferred_over: "OPENCLAW_API_URL",
      role: "preferred execution URL for governed runtime",
      env_var: "OPENCLAW_INTERNAL_URL",
    },
    source_authority: ".env file (openclaw-runtime/lah-openclaw-mvp/.env) + P0_authority_map.md",
    fingerprint: "d3e4f5a6-openclaw-internal-url",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Port 4000 ──
  PORT_4000: {
    value: {
      port: 4000,
      role: "governed runtime (OpenClaw internal)",
      NOT_campaign_execution_authority: true,
      service: "lah-openclaw-mvp container",
    },
    source_authority: ".env PORT=4000 + P0_authority_map.md",
    fingerprint: "f7a8b9c0-port-4000",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Port 18789 ──
  PORT_18789: {
    value: {
      port: 18789,
      role: "OpenClaw browser gateway",
      NOT_campaign_execution_authority: true,
      NOT_governed_runtime: true,
    },
    source_authority: "P0_authority_map.md + memory directive 2026-08-13",
    fingerprint: "g1h2i3j4-port-18789",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Campaign Execution Path ──
  CAMPAIGN_EXECUTION_PATH: {
    value: {
      path: ["LAHB", "governed_runtime", "campaign_compiler", "launch_orchestration", "ExoClick"],
      description: "Canonical campaign execution transport",
      governed: true,
    },
    source_authority: "P0_authority_map.md + lah-workflow-small-model SKILL.md + canary-execution-flow.md",
    fingerprint: "k5l6m7n8-campaign-execution-path",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Compiler Authority ──
  COMPILER_AUTHORITY: {
    value: {
      canonical_file: "services/campaign-compiler/campaign-compiler.js",
      repo: "lah-openclaw-mvp",
      role: "campaign compilation authority",
      output: "compiled_packet with draft_id, draft_hash, status",
    },
    source_authority: "P0_authority_map.md + P0-P8 receipt",
    fingerprint: "o9p0q1r2-compiler-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Launcher Authority ──
  LAUNCHER_AUTHORITY: {
    value: {
      canonical_file: "services/launch-orchestration.js",
      repo: "lah-openclaw-mvp",
      role: "launch orchestration authority",
      function: "advanceLaunchSession()",
    },
    source_authority: "P0_authority_map.md",
    fingerprint: "s3t4u5v6-launcher-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── P6 Authority ──
  P6_AUTHORITY: {
    value: {
      canonical_file: "services/campaign-compiler/p6-provider-readback-certification.js",
      role: "provider certification authority (7/7 automatic checks)",
      certified_at: "2026-08-18",
    },
    source_authority: "P0-P8 receipt + P6 certification",
    fingerprint: "w7x8y9z0-p6-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── ExoClick Authority ──
  EXOCLICK_AUTHORITY: {
    value: {
      canonical_files: [
        "services/exoclick-client.js",
        "services/exoclick-normalizer.js",
        "services/exoclick-variation.js",
      ],
      role: "provider mutation and readback authority",
      api_base: "https://api.exoclick.com/v2",
      live_enabled_env: "EXOCLICK_LIVE_ENABLED",
      token_env: "EXOCLICK_API_TOKEN",
    },
    source_authority: "P0_authority_map.md + T04/T05 directives",
    fingerprint: "a1b2c3d4-exoclick-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Tracking Authority ──
  TRACKING_AUTHORITY: {
    value: {
      canonical_file: "lah-brain/src/campaign-factory/tracking-url.factory.js",
      function: "buildTrackingUrl()",
      contract: "Offer + geo + device + format + macros → tracking URL string",
      crakrevenue_builder: "services/crakrevenue-tracking-url-builder.js",
      redirect_gateway: "lah-brain/src/redirect-gateway.js",
      aff_sub_macro: "{clickid} → aff_sub",
      zone_id_macro: "{zone_id} → aff_sub2({zoneid})",
      campaign_id_macro: "{campaign_id} → aff_sub3({campaignid})",
    },
    source_authority: "P0_authority_map.md + T05 directive + redirect-gateway.js",
    fingerprint: "e5f6a7b8-tracking-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Safety Authority ──
  SAFETY_AUTHORITY: {
    value: {
      canonical_files: [
        "services/launch-safety-bind-phase.js",
        "services/launch-safety-gates.js",
        "services/safety/launch-safety-binding.js",
      ],
      role: "campaign + variation Safety authority",
      scope: "campaign-level and variation-level hard stops",
      binding: "bindCampaignToSafety() + evaluateSafetyCampaignBinding()",
    },
    source_authority: "P0_authority_map.md + lah-workflow-small-model SKILL.md",
    fingerprint: "i9j0k1l2-safety-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Format Authority ──
  FORMAT_AUTHORITY: {
    value: {
      canonical_file: "services/exoclick-maps.js",
      field: "FORMAT_ID",
      runtime_executable_formats: ["popunder", "banner", "native", "video"],
      description: "Format authority — maps catalog format names to ExoClick advertiser_ad_type IDs",
    },
    source_authority: "P0_authority_map.md + format-catalog-reconciliation.md",
    fingerprint: "m3n4o5p6-format-authority",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Approval Execution Seam ──
  APPROVAL_EXECUTION_SEAM: {
    value: {
      seam: "LAHB approval → executeApprovedAction → OpenClaw → ExoClick",
      approval_submission: "POST /approvals/submit with action_type: CAMPAIGN_CREATE_PAUSED",
      approval_function: "submitApproval(db, actionType, payload, correlationId)",
      execution_function: "executeApprovedAction() in openclaw-client.js",
      execution_endpoint: "POST {OPENCLAW_API_URL}/execute",
      known_gap: "OpenClaw does NOT expose /execute for campaign creation (404/400)",
      workaround: "Direct ExoClick API or dedicated campaign creation service needed",
    },
    source_authority: "canary-execution-flow.md + P0_authority_map.md + P9-P12 receipt",
    fingerprint: "q7r8s9t0-approval-execution-seam",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Canonical Runtime Services ──
  CANONICAL_RUNTIME_SERVICES: {
    value: {
      services: [
        { name: "exoclick-normalizer.js", role: "FORMAT_MAP normalization" },
        { name: "exoclick-client.js", role: "provider mutation (create, readback)" },
        { name: "exoclick-variation.js", role: "variation creation (multipart/form-data)" },
        { name: "exoclick-maps.js", role: "format/device/pricing ID resolution" },
        { name: "exoclick-preflight.js", role: "pre-launch validation" },
        { name: "campaign-compiler.js", role: "deterministic compilation" },
        { name: "compile-invariants.js", role: "safety + attribution contracts" },
        { name: "launch-orchestration.js", role: "state machine for launch sessions" },
        { name: "launch-materialization.js", role: "campaign + variation creation" },
        { name: "launch-safety-bind-phase.js", role: "safety binding" },
        { name: "execution-adapter.js", role: "campaign descriptor from arm" },
        { name: "campaign-factory-routing.js", role: "selection strategy routing" },
        { name: "tracking-url.factory.js", role: "canonical tracking URL builder" },
        { name: "crakrevenue-tracking-url-builder.js", role: "CR tracking + SubID1" },
        { name: "redirect-gateway.js", role: "/go/ token router" },
      ],
    },
    source_authority: "P0_authority_map.md + canary-execution-flow.md",
    fingerprint: "u1v2w3x4-runtime-services",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },

  // ── Campaign Execution Transport ──
  CAMPAIGN_EXECUTION_TRANSPORT: {
    value: {
      method: "governed execution through LAHB approval → OpenClaw → ExoClick",
      mutation_boundary: "LAHB approval is the gate; Hermes never triggers mutation without it",
      operator_approval_required: true,
      directive_2026_08_13: "PREPARE → INSPECT → remit trigger to operator",
    },
    source_authority: "lah-workflow-small-model SKILL.md + memory directive 2026-08-13",
    fingerprint: "y5z6a7b8-campaign-transport",
    certified_at: "2026-08-18T21:00:00Z",
    staleness_policy: STALENESS_POLICY.permanent,
  },
});

// ─── Architecture Fingerprint (P3) ──────────────────────────────────────

class ArchitectureFingerprint {
  constructor() {
    this.fingerprintPath = FINGERPRINT_PATH;
  }

  /**
   * Generate LAH_ARCHITECTURE_FINGERPRINT from certified facts.
   * SHA-256 of all fact fingerprints combined deterministically.
   * @returns {string} The aggregate fingerprint
   */
  generate() {
    const factKeys = Object.keys(CERTIFIED_FACTS).sort();
    const parts = factKeys.map((key) => {
      const fact = CERTIFIED_FACTS[key];
      return `${key}:${fact.fingerprint}`;
    });
    const aggregate = parts.join("|");
    return crypto.createHash("sha256").update(aggregate).digest("hex");
  }

  /**
   * Get the current fingerprint, generating if needed.
   * @returns {string}
   */
  getCurrent() {
    if (fs.existsSync(this.fingerprintPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.fingerprintPath, "utf8"));
        return data.fingerprint;
      } catch (_) {
        // Corrupt file — regenerate
      }
    }
    const fp = this.generate();
    this.persist(fp);
    return fp;
  }

  /**
   * Persist a fingerprint to disk.
   * @param {string} fingerprint
   */
  persist(fingerprint) {
    const dir = path.dirname(this.fingerprintPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      this.fingerprintPath,
      JSON.stringify(
        {
          fingerprint,
          generated_at: new Date().toISOString(),
          fact_count: Object.keys(CERTIFIED_FACTS).length,
          fact_keys: Object.keys(CERTIFIED_FACTS).sort(),
        },
        null,
        2
      )
    );
  }

  /**
   * Compare current fingerprint with a stored one.
   * @param {string} stored - The stored fingerprint to compare against
   * @returns {{ unchanged: boolean, current: string, stored: string }}
   */
  compare(stored) {
    const current = this.generate();
    return {
      unchanged: current === stored,
      current,
      stored,
    };
  }

  /**
   * Compute a per-fact fingerprint for incremental updates (P9).
   * @param {string} factKey - The fact key to fingerprint
   * @returns {string} SHA-256 of the fact's value
   */
  fingerprintFact(factKey) {
    const fact = CERTIFIED_FACTS[factKey];
    if (!fact) return null;
    return crypto.createHash("sha256").update(JSON.stringify(fact.value)).digest("hex");
  }
}

// ─── No-Rediscovery Gate (P5+P6) ───────────────────────────────────────

class NoRediscoveryGate {
  constructor() {
    this.certifiedFactKeys = new Set(Object.keys(CERTIFIED_FACTS));
    this.blockedAttempts = [];
    this.allowedValidations = [];
  }

  /**
   * Check if a discovery action should be blocked.
   *
   * BLOCK if:
   *   - context fingerprint unchanged AND
   *   - fact not stale AND
   *   - no contradictory runtime evidence exists AND
   *   - the action is a rediscovery of a certified fact
   *
   * ALLOW if:
   *   - exact approval readback (GET /approvals/:id)
   *   - exact campaign readback when relevant
   *   - architecture fingerprint comparison
   *   - one LAHB health/status check
   *   - one governed runtime health check
   *   - context drift detected
   *   - contradictory runtime evidence exists
   *
   * @param {object} action - The discovery action to evaluate
   * @param {string} action.type - Type of action (rediscover|validate|readback|health|drift_check)
   * @param {string} [action.factKey] - Which certified fact is being rediscovered
   * @param {string} [action.description] - Human description of what's being searched
   * @param {object} [options] - Context options
   * @param {boolean} [options.contextFingerprintUnchanged] - Whether context fingerprint is unchanged
   * @param {boolean} [options.hasContradictoryEvidence] - Whether contradictory runtime evidence exists
   * @param {boolean} [options.contextDrift] - Whether context drift was detected
   * @returns {{ allowed: boolean, reason: string, blocked: boolean }}
   */
  evaluate(action, options = {}) {
    const {
      contextFingerprintUnchanged = true,
      hasContradictoryEvidence = false,
      contextDrift = false,
    } = options;

    const result = {
      allowed: false,
      reason: "",
      blocked: false,
      action_type: action.type,
      fact_key: action.factKey || null,
      timestamp: new Date().toISOString(),
    };

    // ── Allowed validations (P6) ──
    if (action.type === "health_check") {
      result.allowed = true;
      result.reason = "ALLOWED: health check is cheap validation, not rediscovery";
      this.allowedValidations.push(result);
      return result;
    }

    if (action.type === "exact_readback") {
      result.allowed = true;
      result.reason = "ALLOWED: exact readback (approval/campaign) is required validation";
      this.allowedValidations.push(result);
      return result;
    }

    if (action.type === "fingerprint_comparison") {
      result.allowed = true;
      result.reason = "ALLOWED: fingerprint comparison is required startup step";
      this.allowedValidations.push(result);
      return result;
    }

    // ── Context drift overrides everything ──
    if (contextDrift) {
      result.allowed = true;
      result.reason = "ALLOWED: context drift detected — targeted rediscovery permitted";
      this.allowedValidations.push(result);
      return result;
    }

    // ── Contradictory evidence overrides ──
    if (hasContradictoryEvidence) {
      result.allowed = true;
      result.reason = "ALLOWED: contradictory runtime evidence — targeted investigation permitted";
      this.allowedValidations.push(result);
      return result;
    }

    // ── Block rediscovery of certified facts ──
    if (action.type === "rediscover" && this.certifiedFactKeys.has(action.factKey)) {
      if (contextFingerprintUnchanged) {
        result.blocked = true;
        result.allowed = false;
        result.reason = `CERTIFIED_FACT_REDISCOVERY_BLOCKED: '${action.factKey}' is a certified fact, context fingerprint unchanged, no contradictory evidence. Use exact readback instead.`;
        this.blockedAttempts.push(result);
        return result;
      }
    }

    // ── Block broad archaeology ──
    if (action.type === "rediscover" && !action.factKey) {
      // Check if it matches known rediscovery patterns
      const blockedPatterns = [
        "find .env",
        "ps aux",
        "global grep",
        "Docker topology",
        "route guessing",
        "historical receipt search",
        "architecture inspection",
        "port probing",
        "config file search",
      ];
      const desc = (action.description || "").toLowerCase();
      for (const pattern of blockedPatterns) {
        if (desc.includes(pattern.toLowerCase())) {
          result.blocked = true;
          result.allowed = false;
          result.reason = `CERTIFIED_FACT_REDISCOVERY_BLOCKED: '${pattern}' is archaeology of certified architecture. Load context and verify fingerprint instead.`;
          this.blockedAttempts.push(result);
          return result;
        }
      }
    }

    // ── Default: allow non-certified-fact discovery ──
    result.allowed = true;
    result.reason = "ALLOWED: not a certified fact rediscovery or broad archaeology";
    this.allowedValidations.push(result);
    return result;
  }

  /**
   * Get metrics on blocked vs allowed attempts.
   */
  getMetrics() {
    return {
      blocked_attempts: this.blockedAttempts.length,
      allowed_validations: this.allowedValidations.length,
      total_evaluations: this.blockedAttempts.length + this.allowedValidations.length,
    };
  }

  reset() {
    this.blockedAttempts = [];
    this.allowedValidations = [];
  }
}

// ─── Startup Budget (P7) ────────────────────────────────────────────────

class StartupBudget {
  constructor() {
    this.maxActions = STARTUP_BUDGET.max_orientation_actions;
    this.allowedActions = STARTUP_BUDGET.allowed_actions;
    this.actionsTaken = [];
  }

  /**
   * Record an orientation action.
   * @param {string} action - The action type (must be in allowed_actions)
   * @returns {{ allowed: boolean, reason: string, remaining: number }}
   */
  record(action) {
    if (!this.allowedActions.includes(action)) {
      return {
        allowed: false,
        reason: `STARTUP_ARCHAEOLOGY_BLOCKED: '${action}' is not in the allowed startup actions list`,
        remaining: this.maxActions - this.actionsTaken.length,
      };
    }

    this.actionsTaken.push({
      action,
      timestamp: new Date().toISOString(),
    });

    const remaining = this.maxActions - this.actionsTaken.length;

    if (this.actionsTaken.length > this.maxActions) {
      return {
        allowed: false,
        reason: `STARTUP_ARCHAEOLOGY_BLOCKED: exceeded max ${this.maxActions} orientation actions. Load context, verify fingerprint, read exact object — then execute.`,
        remaining: 0,
      };
    }

    return {
      allowed: true,
      reason: `Orientation action recorded (${this.actionsTaken.length}/${this.maxActions}). ${remaining} remaining.`,
      remaining,
    };
  }

  /**
   * Check if budget is exhausted.
   */
  isExhausted() {
    return this.actionsTaken.length >= this.maxActions;
  }

  /**
   * Get remaining budget.
   */
  remaining() {
    return Math.max(0, this.maxActions - this.actionsTaken.length);
  }

  /**
   * Reset the budget (for a new mission).
   */
  reset() {
    this.actionsTaken = [];
  }

  /**
   * Get current state.
   */
  getState() {
    return {
      actions_taken: this.actionsTaken.length,
      max_actions: this.maxActions,
      remaining: this.remaining(),
      exhausted: this.isExhausted(),
      actions: this.actionsTaken.map((a) => a.action),
    };
  }
}

// ─── Context Update Policy (P9) ─────────────────────────────────────────

class ContextUpdatePolicy {
  constructor() {
    this.updateLog = [];
  }

  /**
   * After a successful repair, update ONLY impacted certified facts.
   * Do NOT invalidate the entire architecture because one authority changed.
   *
   * @param {string} impactedFactKey - The fact key that was repaired
   * @param {object} newFactValue - The new value for the fact
   * @param {string} repairDescription - What was repaired
   * @returns {{ updated: boolean, fact_key: string, new_fingerprint: string }}
   */
  updateFact(impactedFactKey, newFactValue, repairDescription) {
    if (!CERTIFIED_FACTS[impactedFactKey]) {
      return {
        updated: false,
        fact_key: impactedFactKey,
        reason: "Fact key not in certified registry",
      };
    }

    const oldFingerprint = CERTIFIED_FACTS[impactedFactKey].fingerprint;
    const newFingerprint = crypto
      .createHash("sha256")
      .update(JSON.stringify(newFactValue))
      .digest("hex");

    // Update the fact in place (immutable freeze is relaxed for updates)
    CERTIFIED_FACTS[impactedFactKey] = {
      ...CERTIFIED_FACTS[impactedFactKey],
      value: newFactValue,
      fingerprint: `${oldFingerprint}-updated-${newFingerprint.substring(0, 8)}`,
      certified_at: new Date().toISOString(),
      last_repaired: repairDescription,
    };

    const entry = {
      fact_key: impactedFactKey,
      old_fingerprint: oldFingerprint,
      new_fingerprint: CERTIFIED_FACTS[impactedFactKey].fingerprint,
      repair: repairDescription,
      timestamp: new Date().toISOString(),
    };
    this.updateLog.push(entry);

    return {
      updated: true,
      fact_key: impactedFactKey,
      old_fingerprint: oldFingerprint,
      new_fingerprint: CERTIFIED_FACTS[impactedFactKey].fingerprint,
    };
  }

  /**
   * Get the update log.
   */
  getUpdateLog() {
    return [...this.updateLog];
  }

  /**
   * Reset the update log.
   */
  reset() {
    this.updateLog = [];
  }
}

// ─── Contradiction Handler (P10) ────────────────────────────────────────

class ContradictionHandler {
  constructor() {
    this.contradictions = [];
  }

  /**
   * Handle a certified context contradiction.
   * Emit CERTIFIED_CONTEXT_CONTRADICTION and investigate ONLY that fact.
   *
   * @param {string} factKey - The certified fact that contradicts runtime evidence
   * @param {*} certifiedValue - The certified value
   * @param {*} runtimeValue - The runtime value that contradicts
   * @param {string} evidence - The runtime evidence that caused the contradiction
   * @returns {{ contradiction_detected: boolean, fact_key: string, investigation_scope: string }}
   */
  handle(factKey, certifiedValue, runtimeValue, evidence) {
    const contradiction = {
      type: "CERTIFIED_CONTEXT_CONTRADICTION",
      fact_key: factKey,
      certified_value: certifiedValue,
      runtime_value: runtimeValue,
      evidence: evidence,
      timestamp: new Date().toISOString(),
      investigation_scope: factKey, // ONLY investigate this fact
    };

    this.contradictions.push(contradiction);

    return {
      contradiction_detected: true,
      fact_key: factKey,
      investigation_scope: factKey,
      message: `CERTIFIED_CONTEXT_CONTRADICTION: '${factKey}' certified value differs from runtime evidence. Investigation scope limited to this fact only.`,
      contradiction,
    };
  }

  /**
   * Check if a fact has a known contradiction.
   */
  hasContradiction(factKey) {
    return this.contradictions.some((c) => c.fact_key === factKey);
  }

  /**
   * Get all contradictions.
   */
  getContradictions() {
    return [...this.contradictions];
  }

  /**
   * Clear contradictions after resolution.
   */
  clear() {
    this.contradictions = [];
  }
}

// ─── Metrics (P14) ──────────────────────────────────────────────────────

class ContextMetrics {
  constructor() {
    this.reset();
  }

  reset() {
    this.metrics = {
      startup_tool_calls: 0,
      certified_fact_rediscovery_attempts: 0,
      certified_fact_rediscovery_blocked: 0,
      architecture_context_load_ms: 0,
      fingerprint_check_ms: 0,
      resume_packet_load_ms: 0,
      targeted_drift_discovery_calls: 0,
      startup_orientation_actions: 0,
      startup_orientation_blocked: 0,
    };
  }

  increment(metricName, value = 1) {
    if (this.metrics[metricName] !== undefined) {
      this.metrics[metricName] += value;
    }
  }

  set(metricName, value) {
    if (this.metrics[metricName] !== undefined) {
      this.metrics[metricName] = value;
    }
  }

  getMetrics() {
    return { ...this.metrics };
  }

  /**
   * Get a compact summary for receipts.
   */
  getSummary() {
    const m = this.metrics;
    return {
      startup_tool_calls: m.startup_tool_calls,
      certified_rediscovery_attempts: m.certified_fact_rediscovery_attempts,
      certified_rediscovery_blocked: m.certified_fact_rediscovery_blocked,
      context_load_ms: m.architecture_context_load_ms,
      fingerprint_check_ms: m.fingerprint_check_ms,
      resume_packet_load_ms: m.resume_packet_load_ms,
      targeted_drift_calls: m.targeted_drift_discovery_calls,
      startup_actions: m.startup_orientation_actions,
      startup_actions_blocked: m.startup_orientation_blocked,
    };
  }
}

// ─── Main Context Manager ───────────────────────────────────────────────

class CertifiedArchitectureContext {
  constructor(options = {}) {
    this.contextPath = options.contextPath || CONTEXT_PATH;
    this.fingerprint = new ArchitectureFingerprint();
    this.gate = new NoRediscoveryGate();
    this.budget = new StartupBudget();
    this.updatePolicy = new ContextUpdatePolicy();
    this.contradictionHandler = new ContradictionHandler();
    this.metrics = new ContextMetrics();
    this.context = null;
    this.fingerprintValue = null;
    this.loadedAt = null;
  }

  /**
   * P1+P2: Load the certified architecture context.
   * Seeding from already-certified evidence only.
   * @returns {{ success: boolean, fact_count: number, load_ms: number }}
   */
  load() {
    const startTime = Date.now();

    // Load from persistent storage if available
    if (fs.existsSync(this.contextPath)) {
      try {
        this.context = JSON.parse(fs.readFileSync(this.contextPath, "utf8"));
        this.loadedAt = new Date().toISOString();
        this.metrics.set("architecture_context_load_ms", Date.now() - startTime);
        return {
          success: true,
          source: "persistent_storage",
          fact_count: Object.keys(this.context.facts || {}).length,
          load_ms: Date.now() - startTime,
        };
      } catch (_) {
        // Corrupt — regenerate from seed
      }
    }

    // Seed from certified facts (P2)
    this.context = {
      version: "1.0.0",
      generated_at: new Date().toISOString(),
      facts: {},
    };

    for (const [key, fact] of Object.entries(CERTIFIED_FACTS)) {
      this.context.facts[key] = {
        value: fact.value,
        source_authority: fact.source_authority,
        fingerprint: fact.fingerprint,
        certified_at: fact.certified_at,
        staleness_policy: fact.staleness_policy,
      };
    }

    this.loadedAt = new Date().toISOString();
    this.persist();

    this.metrics.set("architecture_context_load_ms", Date.now() - startTime);
    return {
      success: true,
      source: "seeded_from_certified_evidence",
      fact_count: Object.keys(this.context.facts).length,
      load_ms: Date.now() - startTime,
    };
  }

  /**
   * Persist the context to disk.
   */
  persist() {
    const dir = path.dirname(this.contextPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this.contextPath, JSON.stringify(this.context, null, 2));
  }

  /**
   * P3: Verify the architecture fingerprint.
   * @param {string} [storedFingerprint] - Optional stored fingerprint to compare against
   * @returns {{ status: string, unchanged: boolean, current: string, stored?: string }}
   */
  verifyFingerprint(storedFingerprint) {
    const startTime = Date.now();
    const current = this.fingerprint.generate();
    this.fingerprintValue = current;

    let result;
    if (storedFingerprint) {
      result = this.fingerprint.compare(storedFingerprint);
    } else {
      // Load stored fingerprint if available
      const stored = this.fingerprint.getCurrent();
      result = this.fingerprint.compare(stored);
    }

    this.metrics.set("fingerprint_check_ms", Date.now() - startTime);

    return {
      status: result.unchanged ? "CONTEXT_VALID" : "CONTEXT_DRIFT",
      unchanged: result.unchanged,
      current: result.current,
      stored: result.stored,
      check_ms: Date.now() - startTime,
    };
  }

  /**
   * Get a certified fact by key.
   * @param {string} factKey
   * @returns {object|null}
   */
  getFact(factKey) {
    if (this.context && this.context.facts && this.context.facts[factKey]) {
      return this.context.facts[factKey];
    }
    return CERTIFIED_FACTS[factKey] || null;
  }

  /**
   * Get all certified facts.
   */
  getAllFacts() {
    if (this.context && this.context.facts) {
      return this.context.facts;
    }
    return CERTIFIED_FACTS;
  }

  /**
   * P5+P6: Evaluate a discovery action against the no-rediscovery gate.
   */
  evaluateDiscovery(action, options = {}) {
    return this.gate.evaluate(action, options);
  }

  /**
   * P7: Record a startup orientation action.
   */
  recordStartupAction(action) {
    return this.budget.record(action);
  }

  /**
   * P9: Update a certified fact incrementally.
   */
  updateFact(factKey, newValue, description) {
    return this.updatePolicy.updateFact(factKey, newValue, description);
  }

  /**
   * P10: Handle a contradiction.
   */
  handleContradiction(factKey, certifiedValue, runtimeValue, evidence) {
    return this.contradictionHandler.handle(factKey, certifiedValue, runtimeValue, evidence);
  }

  /**
   * Get all metrics.
   */
  getMetrics() {
    return this.metrics.getMetrics();
  }

  /**
   * Get a compact receipt for the mission.
   */
  generateReceipt() {
    return {
      verdict: "HERMES_CERTIFIED_ARCHITECTURE_CONTEXT_ACTIVE",
      fact_count: Object.keys(CERTIFIED_FACTS).length,
      fingerprint: this.fingerprintValue || this.fingerprint.getCurrent(),
      fingerprint_status: this.fingerprintValue ? "verified" : "not_checked",
      gate_metrics: this.gate.getMetrics(),
      budget_state: this.budget.getState(),
      update_log: this.updatePolicy.getUpdateLog(),
      contradictions: this.contradictionHandler.getContradictions(),
      metrics: this.metrics.getSummary(),
      timestamp: new Date().toISOString(),
    };
  }
}

// ─── Export ──────────────────────────────────────────────────────────────

module.exports = {
  CertifiedArchitectureContext,
  ArchitectureFingerprint,
  NoRediscoveryGate,
  StartupBudget,
  ContextUpdatePolicy,
  ContradictionHandler,
  ContextMetrics,
  CERTIFIED_FACTS,
  STALENESS_POLICY,
  STARTUP_BUDGET,
};