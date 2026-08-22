#!/usr/bin/env node
/**
 * Hermes Convergence Governor v1
 *
 * Reusable component that prevents Hermes from continuing repository/runtime
 * discovery after sufficient evidence exists to answer the current engineering
 * question.
 *
 * Implements P1-P15 of the MISSION: HERMES_CONVERGENCE_GOVERNOR_AND_DISCOVERY_LOOP_GUARD_V1
 * specification.
 *
 * @module convergence-governor
 * @version 1.0.0
 */

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

// ─── Constants ───────────────────────────────────────────────────────────────

const STATE = Object.freeze({
  ORIENT: "ORIENT",
  DISCOVER: "DISCOVER",
  HYPOTHESIS_READY: "HYPOTHESIS_READY",
  IMPLEMENT: "IMPLEMENT",
  VERIFY: "VERIFY",
  REPORT: "REPORT",
});

const SUFFICIENCY = Object.freeze({
  INSUFFICIENT: "INSUFFICIENT",
  SUFFICIENT_TO_FORM_HYPOTHESIS: "SUFFICIENT_TO_FORM_HYPOTHESIS",
  SUFFICIENT_TO_IMPLEMENT: "SUFFICIENT_TO_IMPLEMENT",
  SUFFICIENT_TO_REPORT: "SUFFICIENT_TO_REPORT",
});

const STOP_REASON = Object.freeze({
  UNKNOWN_CANONICAL_STATE: "UNKNOWN_CANONICAL_STATE",
  CANONICAL_ROUTE_UNAVAILABLE: "CANONICAL_ROUTE_UNAVAILABLE",
  AUTHORIZATION_UNCLEAR: "AUTHORIZATION_UNCLEAR",
  INDEPENDENT_BLOCKER: "INDEPENDENT_BLOCKER",
  CERTIFICATION_FAILURE: "CERTIFICATION_FAILURE",
  NEW_DEFECT_DURING_EXECUTE: "NEW_DEFECT_DURING_EXECUTE",
  NEW_DEFECT_DURING_CERTIFY: "NEW_DEFECT_DURING_CERTIFY",
  REPAIR_REQUIRES_SCOPE_EXPANSION: "REPAIR_REQUIRES_SCOPE_EXPANSION",
  EXACT_REPEAT_DETECTED: "EXACT_REPEAT_DETECTED",
  SEMANTIC_REPEAT_DETECTED: "SEMANTIC_REPEAT_DETECTED",
  EVIDENCE_SUFFICIENT: "EVIDENCE_SUFFICIENT",
  FORCE_CONVERGENCE: "FORCE_CONVERGENCE",
  SIDE_QUEST_BLOCKED: "SIDE_QUEST_BLOCKED",
  UNAUTHORIZED_DISCOVERY_MUTATION_BLOCKED: "UNAUTHORIZED_DISCOVERY_MUTATION_BLOCKED",
  HARD_CONVERGENCE_TRIGGER: "HARD_CONVERGENCE_TRIGGER",
  STOP_DISCOVERY: "STOP_DISCOVERY",
  CODEGRAPH_BOOTSTRAP_REQUIRED: "CODEGRAPH_BOOTSTRAP_REQUIRED",
});

// ── Convergence Governor Enforcement States (P16) ──
// These distinguish between actual runtime enforcement and
// behavioral-only governance (LLM following instructions voluntarily).

const GOVERNOR_ENFORCEMENT_STATE = Object.freeze({
  RUNTIME_ENFORCED: "RUNTIME_ENFORCED",
  BEHAVIORAL_ONLY: "BEHAVIORAL_ONLY",
  NOT_ACTIVE: "NOT_ACTIVE",
});

const DISCOVERY_ACTION_TYPES = Object.freeze({
  GREP: "grep",
  FIND: "find",
  SEARCH_FILES: "search_files",
  READ_FILE: "read_file",
  TERMINAL: "terminal",
  WEB_SEARCH: "web_search",
  BROWSER_NAVIGATE: "browser_navigate",
  DOCKER_INSPECT: "docker_inspect",
  PROCESS_INSPECT: "process_inspect",
  ENV_INSPECT: "env_inspect",
  FILESYSTEM_SCAN: "filesystem_scan",
  REPO_SEARCH: "repo_search",
  SWAGGER_DISCOVERY: "swagger_discovery",
  WEB_RESEARCH: "web_research",
  AD_HOC_PAYLOAD: "ad_hoc_payload",
  AD_HOC_TRACKING_URL: "ad_hoc_tracking_url",
  MANUAL_PROVIDER_CONTRACT: "manual_provider_contract_resolution",
});

// Default per-phase discovery budgets (P2)
const DEFAULT_BUDGETS = Object.freeze({
  exact_same_command: { max: 1, label: "exact same command" },
  same_semantic_search: { max: 2, label: "same semantic search" },
  broad_repo_search: { max: 3, label: "broad repo search" },
  global_filesystem_search: { max: 2, label: "global filesystem search" },
  environment_archaeology: { max: 3, label: "environment archaeology" },
  docker_process_topology: { max: 3, label: "Docker/process topology" },
  swagger_api_discovery: { max: 2, label: "Swagger/API contract discovery" },
});

// Hard safety limit (P15)
const DEFAULT_HARD_CEILING = 50;

// Consecutive low-value threshold (P6)
const FORCE_CONVERGENCE_THRESHOLD = 3;

// ─── Utility ─────────────────────────────────────────────────────────────────

function sha256(str) {
  return crypto.createHash("sha256").update(str).digest("hex");
}

function normalizeCommand(cmd) {
  // Normalize a terminal command for semantic fingerprinting (P1).
  if (typeof cmd !== "string") return "";

  let normalized = cmd.trim();

  // Collapse multiple spaces
  normalized = normalized.replace(/\s+/g, " ");

  // Normalize grep variants: grep -rn → grep -r, grep -rN → grep -r
  normalized = normalized.replace(/grep -rn\b/g, "grep -r");
  normalized = normalized.replace(/grep -rN\b/g, "grep -r");
  normalized = normalized.replace(/grep -rn\b/g, "grep -r");

  // Normalize head variants: head -10 vs head -20 → head
  normalized = normalized.replace(/\bhead -\d+/g, "head");

  // Normalize tail variants
  normalized = normalized.replace(/\btail -\d+/g, "tail");

  // Normalize find | xargs grep → grep (same semantic family)
  const findXargsPattern = /find\s+[^|]+\|\s*xargs\s+grep\b/;
  if (findXargsPattern.test(normalized)) {
    // Extract the grep target and scope
    const grepMatch = normalized.match(/grep\s+(-[a-zA-Z]+\s+)*['"]?([^'"\s]+)['"]?\s+(\S+)/);
    if (grepMatch) {
      normalized = `grep -r ${grepMatch[2]} ${grepMatch[3]}`;
    }
  }

  // Normalize path aliases where determinable
  // e.g., /home/deploy/... → ~ (but only if we can detect the pattern)
  normalized = normalized.replace(/\/home\/deploy\/\.\.\./g, "~");

  // Normalize output formatting flags
  normalized = normalized.replace(/\|?\s*head\s+\d+/g, "");
  normalized = normalized.replace(/\|?\s*tail\s+\d+/g, "");
  normalized = normalized.replace(/\|?\s*sort\s+-u/g, "");
  normalized = normalized.replace(/\|?\s*uniq/g, "");

  // Collapse whitespace again after removals
  normalized = normalized.replace(/\s+/g, " ").trim();

  return normalized;
}

function extractSearchTarget(cmd) {
  // Extract the search target concept from a command for semantic hashing.
  const patterns = [
    /grep\s+(?:-[a-zA-Z]+\s+)*['"]?([^'"\s|]+)['"]?/,
    /find\s+[^|]+\s*-name\s+['"]?([^'"\s|]+)['"]?/,
    /search_files\s*\(\s*[^,]*,\s*['"]([^'"]+)['"]/,
    /grep\s+-r\s+['"]?([^'"\s|]+)['"]?\s+\S+/,
  ];

  for (const pat of patterns) {
    const m = cmd.match(pat);
    if (m) return m[1];
  }

  // Fallback: first quoted string or first word after command verb
  const quoteMatch = cmd.match(/['"]([^'"]+)['"]/);
  if (quoteMatch) return quoteMatch[1];

  const words = cmd.split(/\s+/);
  if (words.length > 1) return words[1];

  return cmd;
}

function extractScope(cmd) {
  // Extract the scope (directory/path) from a command.
  const scopePatterns = [
    /(?:grep|find|search_files)\s+\S+\s+(\/\S+)/,
    /grep\s+-r\s+\S+\s+(\/\S+)/,
    /find\s+(\/\S+)/,
  ];

  for (const pat of scopePatterns) {
    const m = cmd.match(pat);
    if (m) return m[1];
  }

  // Default: current working directory
  return ".";
}

function classifyActionType(cmd) {
  // Classify a command into a discovery action type.
  if (cmd.includes("grep") || cmd.includes("search_files")) return DISCOVERY_ACTION_TYPES.GREP;
  if (cmd.startsWith("find ")) return DISCOVERY_ACTION_TYPES.FIND;
  if (cmd.includes("docker") || cmd.includes("docker-compose")) return DISCOVERY_ACTION_TYPES.DOCKER_INSPECT;
  if (cmd.includes("ps ") || cmd.includes("top ") || cmd.includes("htop")) return DISCOVERY_ACTION_TYPES.PROCESS_INSPECT;
  if (cmd.includes("env ") || cmd.includes("printenv") || cmd.includes("echo $")) return DISCOVERY_ACTION_TYPES.ENV_INSPECT;
  if (cmd.includes("ls ") || cmd.includes("tree ") || cmd.includes("find .")) return DISCOVERY_ACTION_TYPES.FILESYSTEM_SCAN;
  if (cmd.includes("swagger") || cmd.includes("openapi")) return DISCOVERY_ACTION_TYPES.SWAGGER_DISCOVERY;
  if (cmd.includes("curl") || cmd.includes("wget") || cmd.includes("fetch")) return DISCOVERY_ACTION_TYPES.TERMINAL;
  if (cmd.includes("web_search") || cmd.includes("search_engine")) return DISCOVERY_ACTION_TYPES.WEB_SEARCH;
  if (cmd.includes("browser_navigate")) return DISCOVERY_ACTION_TYPES.BROWSER_NAVIGATE;
  if (cmd.includes("repo_search") || cmd.includes("grep -r")) return DISCOVERY_ACTION_TYPES.REPO_SEARCH;
  if (cmd.includes("POST /") || cmd.includes("PUT /") || cmd.includes("DELETE /")) return DISCOVERY_ACTION_TYPES.AD_HOC_PAYLOAD;
  if (cmd.includes("affiliate") || cmd.includes("aff_sub") || cmd.includes("tracking")) return DISCOVERY_ACTION_TYPES.AD_HOC_TRACKING_URL;
  if (cmd.includes("provider") && cmd.includes("contract")) return DISCOVERY_ACTION_TYPES.MANUAL_PROVIDER_CONTRACT;

  return DISCOVERY_ACTION_TYPES.TERMINAL;
}

function isReadOnly(cmd) {
  // Determine if a command is read-only (no mutation).
  const mutationPatterns = [
    /\bPOST\b/, /\bPUT\b/, /\bDELETE\b/, /\bmkdir\b/, /\brm\b/,
    /\bmv\b/, /\bcp\b/, /\bchmod\b/, /\bchown\b/, /\btouch\b/,
    /\bwrite_file\b/, /\bpatch\b/, /\bcreate\b/, /\binstall\b/,
    /\bexec\b/, /\bsubmitApproval\b/, /\bPLAY\b/,
  ];
  for (const pat of mutationPatterns) {
    if (pat.test(cmd)) return false;
  }
  return true;
}

// ─── Command Fingerprinter (P1) ─────────────────────────────────────────────

class CommandFingerprinter {
  constructor() {
    this.fingerprints = new Map(); // exact_hash → fingerprint
    this.semanticHashes = new Map(); // semantic_hash → Set of exact_hashes
  }

  /**
   * Fingerprint a command action.
   * @param {string} command - The exact command string
   * @param {object} [overrides] - Optional overrides for fingerprint fields
   * @returns {object} The fingerprint
   */
  fingerprint(command, overrides = {}) {
    const exactHash = sha256(command);
    const normalized = normalizeCommand(command);
    const semanticHash = sha256(normalized);
    const searchTarget = extractSearchTarget(normalized);
    const scope = extractScope(normalized);
    const actionType = classifyActionType(normalized);
    const readOnly = isReadOnly(normalized);

    const fp = {
      exact_command_hash: exactHash,
      semantic_command_hash: semanticHash,
      tool_type: overrides.tool_type || this._inferToolType(command),
      operation_type: actionType,
      search_target: searchTarget,
      search_concept: searchTarget,
      scope: scope,
      filters: this._extractFilters(normalized),
      mutation: !readOnly,
      read_only: readOnly,
      normalized_arguments: normalized,
      execution_count: 0,
      new_evidence_count: 0,
      last_execution_step: null,
      ...overrides,
    };

    // Update existing or store new
    if (this.fingerprints.has(exactHash)) {
      const existing = this.fingerprints.get(exactHash);
      existing.execution_count++;
      existing.last_execution_step = overrides.step || existing.last_execution_step;
      fp.execution_count = existing.execution_count;
      fp.new_evidence_count = existing.new_evidence_count;
    } else {
      fp.execution_count = 1;
      this.fingerprints.set(exactHash, fp);
    }

    // Track semantic hash mapping
    if (!this.semanticHashes.has(semanticHash)) {
      this.semanticHashes.set(semanticHash, new Set());
    }
    this.semanticHashes.get(semanticHash).add(exactHash);

    return fp;
  }

  _inferToolType(command) {
    if (command.startsWith("terminal(") || command.startsWith("terminal ")) return "terminal";
    if (command.startsWith("search_files(")) return "search_files";
    if (command.startsWith("read_file(")) return "read_file";
    if (command.startsWith("write_file(")) return "write_file";
    if (command.startsWith("patch(")) return "patch";
    if (command.startsWith("terminal(")) return "terminal";
    if (command.includes("web_search")) return "web_search";
    if (command.includes("browser_navigate")) return "browser";
    if (command.includes("lcm_")) return "lcm";
    if (command.includes("session_search")) return "session_search";
    return "unknown";
  }

  _extractFilters(normalized) {
    const filters = [];
    // Extract -e, --include, --exclude, path filters from grep
    const grepMatch = normalized.match(/grep\s+((?:-[a-zA-Z]+\s*[\s\S]*?))\s/);
    if (grepMatch) filters.push(grepMatch[1].trim());
    return filters;
  }

  getFingerprint(exactHash) {
    return this.fingerprints.get(exactHash) || null;
  }

  getSemanticFamily(semanticHash) {
    return this.semanticHashes.get(semanticHash) || new Set();
  }

  getStats() {
    let totalCommands = 0;
    let uniqueSemantic = 0;
    let repeats = 0;
    for (const [hash, fp] of this.fingerprints) {
      totalCommands++;
      if (fp.execution_count > 1) repeats++;
    }
    for (const [, set] of this.semanticHashes) {
      if (set.size > 0) uniqueSemantic++;
    }
    return { total_commands: totalCommands, unique_semantic_families: uniqueSemantic, repeated_families: repeats };
  }
}

// ─── Discovery Budget (P2) ──────────────────────────────────────────────────

class DiscoveryBudget {
  constructor(budgets = {}) {
    this.budgets = { ...DEFAULT_BUDGETS, ...budgets };
    this.counters = {};
    this.justifications = [];
    this._initCounters();
  }

  _initCounters() {
    for (const [key, cfg] of Object.entries(this.budgets)) {
      this.counters[key] = 0;
    }
    // Also track archaeology counters (anti-archaeology gate)
    this.counters.grep_count = 0;
    this.counters.find_count = 0;
    this.counters.repo_search_count = 0;
    this.counters.swagger_discovery_count = 0;
    this.counters.web_research_count = 0;
    this.counters.ad_hoc_payload_count = 0;
    this.counters.ad_hoc_tracking_url_count = 0;
    this.counters.manual_provider_contract_resolution_count = 0;
  }

  /**
   * Check if a discovery action is within budget.
   * @param {string} category - The budget category
   * @param {string} [justification] - Required when exceeding ceiling
   * @returns {{ allowed: boolean, reason: string }}
   */
  check(category, justification = "") {
    if (!this.budgets[category]) {
      return { allowed: true, reason: "unknown category, defaulting to allowed" };
    }

    const limit = this.budgets[category].max;
    this.counters[category] = (this.counters[category] || 0) + 1;

    if (this.counters[category] <= limit) {
      return { allowed: true, reason: `within budget (${this.counters[category]}/${limit})` };
    }

    // Over budget — check for NEW_EVIDENCE_JUSTIFICATION
    if (justification && justification.trim().length > 0) {
      this.justifications.push({ category, justification, step: this.counters[category] });
      return { allowed: true, reason: `over budget but justified: ${justification}` };
    }

    return {
      allowed: false,
      reason: `BUDGET_EXCEEDED: ${category} limit is ${limit}, already at ${this.counters[category]}. Provide NEW_EVIDENCE_JUSTIFICATION to continue.`,
    };
  }

  /**
   * Increment an archaeology counter (P2 anti-archaeology gate).
   */
  incrementArchaeologyCounter(counterName) {
    if (this.counters[counterName] !== undefined) {
      this.counters[counterName]++;
    }
  }

  /**
   * Check if all archaeology counters are zero (for BUILD_AND_CERTIFY).
   */
  allArchaeologyCountersZero() {
    const archaeologyCounters = [
      "grep_count", "find_count", "repo_search_count",
      "swagger_discovery_count", "web_research_count",
      "ad_hoc_payload_count", "ad_hoc_tracking_url_count",
      "manual_provider_contract_resolution_count",
    ];
    for (const c of archaeologyCounters) {
      if ((this.counters[c] || 0) !== 0) return false;
    }
    return true;
  }

  getCounters() {
    return { ...this.counters };
  }

  reset() {
    this._initCounters();
    this.justifications = [];
  }
}

// ─── Evidence Ledger (P3) ───────────────────────────────────────────────────

class EvidenceLedger {
  constructor() {
    this.hypotheses = new Map(); // hypothesis_id → hypothesis
    this.evidenceIndex = 0;
  }

  /**
   * Create or update a hypothesis in the ledger.
   * @param {string} hypothesisId
   * @param {object} data
   * @param {string} data.question - The question being investigated
   * @param {string[]} [data.evidence_for] - Evidence supporting the hypothesis
   * @param {string[]} [data.evidence_against] - Evidence contradicting the hypothesis
   * @param {string[]} [data.unknowns] - Known unknowns
   * @param {string[]} [data.blocking_unknowns] - Unknowns blocking progress
   * @param {number} [data.confidence] - 0.0 to 1.0
   * @param {string[]} [data.canonical_sources_found] - Canonical authorities located
   * @param {string} [data.next_information_needed] - What's still needed
   */
  recordHypothesis(hypothesisId, data) {
    if (!this.hypotheses.has(hypothesisId)) {
      this.hypotheses.set(hypothesisId, {
        hypothesis_id: hypothesisId,
        question: data.question || "",
        evidence_for: data.evidence_for || [],
        evidence_against: data.evidence_against || [],
        unknowns: data.unknowns || [],
        blocking_unknowns: data.blocking_unknowns || [],
        confidence: data.confidence || 0.0,
        canonical_sources_found: data.canonical_sources_found || [],
        next_information_needed: data.next_information_needed || "",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    } else {
      const existing = this.hypotheses.get(hypothesisId);
      if (data.evidence_for) existing.evidence_for.push(...data.evidence_for);
      if (data.evidence_against) existing.evidence_against.push(...data.evidence_against);
      if (data.unknowns) existing.unknowns.push(...data.unknowns);
      if (data.blocking_unknowns) existing.blocking_unknowns.push(...data.blocking_unknowns);
      if (data.canonical_sources_found) existing.canonical_sources_found.push(...data.canonical_sources_found);
      if (data.confidence !== undefined) existing.confidence = data.confidence;
      if (data.next_information_needed) existing.next_information_needed = data.next_information_needed;
      existing.updated_at = new Date().toISOString();
    }

    return this.hypotheses.get(hypothesisId);
  }

  /**
   * Record evidence for a hypothesis.
   */
  addEvidence(hypothesisId, direction, evidence) {
    const h = this.hypotheses.get(hypothesisId);
    if (!h) return null;

    if (direction === "for") {
      h.evidence_for.push(evidence);
      // New evidence may reduce unknowns
      this._tryResolveUnknowns(h, evidence);
    } else if (direction === "against") {
      h.evidence_against.push(evidence);
    }

    h.updated_at = new Date().toISOString();
    return h;
  }

  /**
   * Remove a blocking unknown if evidence addresses it.
   */
  _tryResolveUnknowns(hypothesis, evidence) {
    const evidenceLower = evidence.toLowerCase();
    hypothesis.blocking_unknowns = hypothesis.blocking_unknowns.filter((u) => {
      return !evidenceLower.includes(u.toLowerCase().substring(0, 20));
    });
  }

  /**
   * Check if a hypothesis has zero blocking unknowns.
   */
  isSufficient(hypothesisId) {
    const h = this.hypotheses.get(hypothesisId);
    if (!h) return false;
    return h.blocking_unknowns.length === 0;
  }

  getHypothesis(hypothesisId) {
    return this.hypotheses.get(hypothesisId) || null;
  }

  getAllHypotheses() {
    return Array.from(this.hypotheses.values());
  }

  /**
   * Restore hypotheses from persisted state (P9 context compaction continuity).
   * @param {object} persistedLedger - Serialized ledger data from _serializeLedger
   */
  restoreFromPersistedState(persistedLedger) {
    if (!persistedLedger || typeof persistedLedger !== "object") return;

    for (const [id, data] of Object.entries(persistedLedger)) {
      this.hypotheses.set(id, {
        hypothesis_id: id,
        question: data.question || "",
        evidence_for: data.evidence_for || [],
        evidence_against: data.evidence_against || [],
        unknowns: data.unknowns || [],
        blocking_unknowns: data.blocking_unknowns || [],
        confidence: data.confidence || 0.0,
        canonical_sources_found: data.canonical_sources_found || [],
        next_information_needed: data.next_information_needed || "",
        created_at: data.created_at || new Date().toISOString(),
        updated_at: data.updated_at || new Date().toISOString(),
      });
    }
  }

  /**
   * Get the overall sufficiency assessment across all hypotheses.
   */
  getOverallSufficiency() {
    const all = this.getAllHypotheses();
    if (all.length === 0) return SUFFICIENCY.INSUFFICIENT;

    const allSufficient = all.every((h) => h.blocking_unknowns.length === 0);
    const hasCanonical = all.some((h) => h.canonical_sources_found.length > 0);
    const highConfidence = all.every((h) => h.confidence >= 0.7);

    if (allSufficient && hasCanonical && highConfidence) return SUFFICIENCY.SUFFICIENT_TO_REPORT;
    if (allSufficient && hasCanonical) return SUFFICIENCY.SUFFICIENT_TO_IMPLEMENT;
    if (hasCanonical) return SUFFICIENCY.SUFFICIENT_TO_FORM_HYPOTHESIS;
    return SUFFICIENCY.INSUFFICIENT;
  }
}

// ─── Evidence Sufficiency Gate (P4) ─────────────────────────────────────────

class EvidenceSufficiencyGate {
  /**
   * Evaluate whether sufficient evidence exists.
   * @param {EvidenceLedger} ledger
   * @param {object} [options]
   * @param {string} [options.missionMode] - EXECUTE, DIAGNOSTIC, REPAIR, CERTIFY, BUILD_AND_CERTIFY
   * @param {string[]} [options.requiredCanonicalSources] - Required canonical sources
   * @returns {{ sufficiency: string, reason: string, can_proceed: boolean, recommended_transition: string }}
   */
  evaluateEvidenceSufficiency(ledger, options = {}) {
    const { missionMode = "EXECUTE", requiredCanonicalSources = [] } = options;
    const overall = ledger.getOverallSufficiency();
    const allHypotheses = ledger.getAllHypotheses();

    // Check for required canonical sources
    const missingSources = requiredCanonicalSources.filter((src) => {
      return !allHypotheses.some((h) =>
        h.canonical_sources_found.some((s) => s.includes(src))
      );
    });

    if (missingSources.length > 0) {
      return {
        sufficiency: SUFFICIENCY.INSUFFICIENT,
        reason: `Missing required canonical sources: ${missingSources.join(", ")}`,
        can_proceed: false,
        recommended_transition: "DISCOVER",
      };
    }

    // Mission-specific thresholds
    switch (missionMode) {
      case "EXECUTE":
      case "REPAIR":
        if (overall === SUFFICIENCY.SUFFICIENT_TO_IMPLEMENT || overall === SUFFICIENCY.SUFFICIENT_TO_REPORT) {
          // For REPAIR mode, SUFFICIENT_TO_REPORT maps to SUFFICIENT_TO_IMPLEMENT
          const sufficiencyForMode = (missionMode === "REPAIR" && overall === SUFFICIENCY.SUFFICIENT_TO_REPORT)
            ? SUFFICIENCY.SUFFICIENT_TO_IMPLEMENT
            : overall;
          const transitionForMode = (missionMode === "REPAIR" && overall === SUFFICIENCY.SUFFICIENT_TO_REPORT)
            ? "IMPLEMENT"
            : (overall === SUFFICIENCY.SUFFICIENT_TO_IMPLEMENT ? "IMPLEMENT" : "REPORT");
          return {
            sufficiency: sufficiencyForMode,
            reason: "Sufficient evidence for implementation/report",
            can_proceed: true,
            recommended_transition: transitionForMode,
          };
        }
        break;

      case "DIAGNOSTIC":
        if (overall === SUFFICIENCY.SUFFICIENT_TO_FORM_HYPOTHESIS || overall === SUFFICIENCY.SUFFICIENT_TO_REPORT) {
          return {
            sufficiency: overall,
            reason: "Sufficient evidence for diagnostic report",
            can_proceed: true,
            recommended_transition: "REPORT",
          };
        }
        break;

      case "CERTIFY":
      case "BUILD_AND_CERTIFY":
        if (overall === SUFFICIENCY.SUFFICIENT_TO_REPORT) {
          return {
            sufficiency: overall,
            reason: "Sufficient evidence for certification",
            can_proceed: true,
            recommended_transition: "REPORT",
          };
        }
        break;
    }

    // Default: not yet sufficient
    const nextStep = this._recommendNextStep(allHypotheses);
    return {
      sufficiency: overall,
      reason: `Evidence insufficient. ${nextStep}`,
      can_proceed: false,
      recommended_transition: "DISCOVER",
    };
  }

  _recommendNextStep(hypotheses) {
    const blocking = hypotheses.filter((h) => h.blocking_unknowns.length > 0);
    if (blocking.length === 0) return "All blocking unknowns resolved.";

    const next = blocking[0].next_information_needed;
    if (next) return `Next: ${next}`;

    return `Resolve blocking unknowns: ${blocking.map((h) => h.blocking_unknowns.join(", ")).join("; ")}`;
  }
}

// ─── Loop Detector (P5) ─────────────────────────────────────────────────────

class LoopDetector {
  constructor(fingerprinter) {
    this.fingerprinter = fingerprinter;
    this.exactRepeatThreshold = 2; // same exact command returns same result twice → third forbidden
    this.semanticRepeatThreshold = 2; // same semantic family produces no new evidence twice → further forbidden
    this.detectedLoops = [];
  }

  /**
   * Check for exact repeat detection.
   * @param {string} commandHash - SHA-256 of the exact command
   * @param {boolean} [sameResult=true] - Whether the result was materially identical
   * @returns {{ detected: boolean, type: string, details: object }}
   */
  checkExactRepeat(commandHash, sameResult = true) {
    const fp = this.fingerprinter.getFingerprint(commandHash);
    if (!fp) return { detected: false, type: null, details: {} };

    if (fp.execution_count >= this.exactRepeatThreshold && sameResult) {
      const loop = {
        type: "EXACT_REPEAT_DETECTED",
        command_hash: commandHash,
        execution_count: fp.execution_count,
        new_evidence_produced: fp.new_evidence_count,
        recommended_transition: "FORCE_CONVERGENCE",
        timestamp: new Date().toISOString(),
      };
      this.detectedLoops.push(loop);
      return { detected: true, type: "EXACT_REPEAT_DETECTED", details: loop };
    }

    return { detected: false, type: null, details: {} };
  }

  /**
   * Check for semantic repeat detection.
   * @param {string} semanticHash - SHA-256 of the normalized command
   * @param {number} newEvidenceCount - How many new evidence items this execution produced
   * @returns {{ detected: boolean, type: string, details: object }}
   */
  checkSemanticRepeat(semanticHash, newEvidenceCount = 0) {
    const family = this.fingerprinter.getSemanticFamily(semanticHash);
    if (family.size === 0) return { detected: false, type: null, details: {} };

    // Check if any member of this family has produced no new evidence on recent runs
    const totalExecutions = Array.from(family).reduce((sum, h) => {
      const fp = this.fingerprinter.getFingerprint(h);
      return sum + (fp ? fp.execution_count : 0);
    }, 0);

    const totalNewEvidence = Array.from(family).reduce((sum, h) => {
      const fp = this.fingerprinter.getFingerprint(h);
      return sum + (fp ? fp.new_evidence_count : 0);
    }, 0);

    if (totalExecutions >= this.semanticRepeatThreshold && totalNewEvidence === 0) {
      const loop = {
        type: "SEMANTIC_REPEAT_DETECTED",
        semantic_hash: semanticHash,
        family_size: family.size,
        total_executions: totalExecutions,
        total_new_evidence: totalNewEvidence,
        recommended_transition: "FORCE_CONVERGENCE",
        timestamp: new Date().toISOString(),
      };
      this.detectedLoops.push(loop);
      return { detected: true, type: "SEMANTIC_REPEAT_DETECTED", details: loop };
    }

    return { detected: false, type: null, details: {} };
  }

  getDetectedLoops() {
    return [...this.detectedLoops];
  }

  hasActiveLoop() {
    return this.detectedLoops.length > 0;
  }
}

// ─── Information Gain Scorer (P6) ───────────────────────────────────────────

class InformationGainScorer {
  constructor() {
    this.consecutiveLowValueCount = 0;
    this.totalScores = { HIGH: 0, MEDIUM: 0, LOW: 0, ZERO: 0 };
    this.history = [];
  }

  /**
   * Score a discovery action's information gain.
   * @param {object} evidence - The evidence produced by the action
   * @param {object} ledger - The current evidence ledger
   * @returns {{ score: string, reason: string }}
   */
  score(evidence, ledger) {
    let score;
    let reason;

    if (!evidence || Object.keys(evidence).length === 0) {
      score = "ZERO";
      reason = "No evidence produced";
    } else if (this._isDuplicate(evidence, ledger)) {
      score = "ZERO";
      reason = "Repeats already-recorded evidence";
    } else if (this._isLowValue(evidence, ledger)) {
      score = "LOW";
      reason = "Returns same files/processes/configuration without reducing blocking unknowns";
    } else if (this._isMediumValue(evidence, ledger)) {
      score = "MEDIUM";
      reason = "Partial reduction in unknowns; not yet sufficient";
    } else {
      score = "HIGH";
      reason = "Reduces a blocking unknown or provides canonical authority";
    }

    this.totalScores[score]++;
    this.history.push({ score, reason, timestamp: new Date().toISOString() });

    // Track consecutive low value
    if (score === "LOW" || score === "ZERO") {
      this.consecutiveLowValueCount++;
    } else {
      this.consecutiveLowValueCount = 0;
    }

    return { score, reason };
  }

  _isDuplicate(evidence, ledger) {
    const evidenceStr = JSON.stringify(evidence);
    for (const h of ledger.getAllHypotheses()) {
      for (const ef of h.evidence_for) {
        if (JSON.stringify(ef) === evidenceStr) return true;
      }
    }
    return false;
  }

  _isLowValue(evidence, ledger) {
    // Evidence that doesn't reduce any blocking unknown
    const hasBlockingUnknown = ledger.getAllHypotheses().some(
      (h) => h.blocking_unknowns.length > 0
    );
    if (!hasBlockingUnknown) return false;

    // Check if evidence addresses any blocking unknown
    const evidenceStr = JSON.stringify(evidence).toLowerCase();
    for (const h of ledger.getAllHypotheses()) {
      for (const bu of h.blocking_unknowns) {
        if (evidenceStr.includes(bu.toLowerCase().substring(0, 15))) return false;
      }
    }
    return true;
  }

  _isMediumValue(evidence, ledger) {
    // Evidence that partially addresses unknowns but doesn't resolve them
    return !this._isDuplicate(evidence, ledger) && !this._isLowValue(evidence, ledger);
  }

  /**
   * Check if force convergence review should trigger.
   */
  shouldForceConvergence() {
    return this.consecutiveLowValueCount >= FORCE_CONVERGENCE_THRESHOLD;
  }

  getConsecutiveLowValueCount() {
    return this.consecutiveLowValueCount;
  }

  getStats() {
    return {
      ...this.totalScores,
      consecutive_low_value: this.consecutiveLowValueCount,
      total_actions: Object.values(this.totalScores).reduce((a, b) => a + b, 0),
    };
  }
}

// ─── Side-Quest Detector (P7) ───────────────────────────────────────────────

class SideQuestDetector {
  constructor() {
    this.activeBlockingUnknowns = new Set();
    this.discoveryFamilies = new Map(); // family_name → blocking_unknown_id
    this.blockedSideQuests = [];
  }

  /**
   * Set the current blocking unknowns for the mission.
   * @param {string[]} unknownIds - List of blocking unknown IDs
   */
  setBlockingUnknowns(unknownIds) {
    this.activeBlockingUnknowns = new Set(unknownIds);
  }

  /**
   * Register a discovery family and map it to a blocking unknown.
   * @param {string} familyName - Name of the discovery family
   * @param {string} blockingUnknownId - The blocking unknown this maps to
   * @returns {{ allowed: boolean, reason: string }}
   */
  registerDiscoveryFamily(familyName, blockingUnknownId) {
    if (this.activeBlockingUnknowns.size === 0) {
      // No blocking unknowns — any new discovery is a side quest
      this.blockedSideQuests.push({
        family: familyName,
        reason: "No active blocking unknown — discovery is a side quest",
        timestamp: new Date().toISOString(),
      });
      return { allowed: false, reason: "SIDE_QUEST_BLOCKED: No active blocking unknown to map to", family: familyName };
    }

    if (!this.activeBlockingUnknowns.has(blockingUnknownId)) {
      this.blockedSideQuests.push({
        family: familyName,
        reason: `Blocking unknown ${blockingUnknownId} is not in the active set`,
        timestamp: new Date().toISOString(),
      });
      return {
        allowed: false,
        reason: `SIDE_QUEST_BLOCKED: ${blockingUnknownId} is not a current blocking unknown`,
        family: familyName,
      };
    }

    this.discoveryFamilies.set(familyName, blockingUnknownId);
    return { allowed: true, reason: `Mapped to blocking unknown ${blockingUnknownId}` };
  }

  getBlockedSideQuests() {
    return [...this.blockedSideQuests];
  }

  hasBlockedSideQuests() {
    return this.blockedSideQuests.length > 0;
  }
}

// ─── Mutation Escalation Guard (P8) ─────────────────────────────────────────

class MutationEscalationGuard {
  constructor() {
    this.prohibitedModes = new Set([
      "DIAGNOSTIC",
      "PREPARE",
      "READ_ONLY",
      "OFFLINE_CERTIFICATION",
    ]);
    this.blockedProbes = [];
    this.secretsDetected = [];
  }

  /**
   * Check if a mutation/probe is allowed given the current mission mode.
   * @param {string} mode - Current mission mode
   * @param {string} command - The command being considered
   * @param {boolean} isMutation - Whether the action mutates state
   * @returns {{ allowed: boolean, reason: string }}
   */
  check(mode, command, isMutation = false) {
    // Redact secrets before logging
    const redactedCommand = this._redactSecrets(command);

    if (!isMutation) {
      return { allowed: true, reason: "Read-only action is always allowed" };
    }

    // Mutation in prohibited mode
    if (this.prohibitedModes.has(mode)) {
      const block = {
        mode,
        command: redactedCommand,
        reason: `UNAUTHORIZED_DISCOVERY_MUTATION_BLOCKED: Mutations prohibited in ${mode} mode without explicit operator authorization`,
        timestamp: new Date().toISOString(),
      };
      this.blockedProbes.push(block);
      return { allowed: false, reason: block.reason };
    }

    // Check for specific forbidden patterns
    const forbiddenPatterns = [
      { pattern: /POST\s+\/execute/, name: "hand-built POST /execute" },
      { pattern: /POST\s+\/campaigns/, name: "direct provider campaign creation" },
      { pattern: /fake.?campaign/, name: "fake campaign payload" },
      { pattern: /manual.?payload/, name: "manual payload construction" },
      { pattern: /affiliate.*URL.*mutat/, name: "manual affiliate URL mutation" },
      { pattern: /direct.?provider.?write/, name: "direct provider write" },
    ];

    for (const { pattern, name } of forbiddenPatterns) {
      if (pattern.test(command)) {
        const block = {
          mode,
          command: redactedCommand,
          reason: `UNAUTHORIZED_DISCOVERY_MUTATION_BLOCKED: ${name} is forbidden in ${mode} mode`,
          timestamp: new Date().toISOString(),
        };
        this.blockedProbes.push(block);
        return { allowed: false, reason: block.reason };
      }
    }

    return { allowed: true, reason: "Mutation allowed in current mode" };
  }

  /**
   * Redact secrets from command strings before logging.
   */
  _redactSecrets(cmd) {
    let redacted = cmd;
    // Redact common secret patterns
    redacted = redacted.replace(/(API_KEY|api_key|apikey)\s*=\s*['"][^'"]+['"]/gi, "$1 = '[REDACTED]'");
    redacted = redacted.replace(/(TOKEN|token)\s*=\s*['"][^'"]+['"]/gi, "$1 = '[REDACTED]'");
    redacted = redacted.replace(/(SECRET|secret)\s*=\s*['"][^'"]+['"]/gi, "$SECRET = '[REDACTED]'");
    redacted = redacted.replace(/(password|PASSWD)\s*=\s*['"][^'"]+['"]/gi, "$1 = '[REDACTED]'");
    // Redact Bearer tokens
    redacted = redacted.replace(/Bearer\s+['"][^'"]+['"]/gi, "Bearer '[REDACTED]'");
    // Redact authorization headers
    redacted = redacted.replace(/Authorization:\s*['"][^'"]+['"]/gi, "Authorization: '[REDACTED]'");
    return redacted;
  }

  /**
   * Check if any secrets were detected in recent commands.
   */
  hasExposedSecrets() {
    return this.secretsDetected.length > 0;
  }

  getBlockedProbes() {
    return [...this.blockedProbes];
  }

  getSecretsDetected() {
    return [...this.secretsDetected];
  }
}

// ─── Context Compaction Continuity (P9) ─────────────────────────────────────

class ContextCompactionContinuity {
  constructor(persistPath = null) {
    this.persistPath = persistPath || path.join(
      process.env.HERMES_HOME || "/home/deploy/.hermes",
      "convergence-state.json"
    );
    this.state = {
      evidence_ledger: {},
      resolved_hypotheses: [],
      canonical_authorities_found: [],
      commands_already_executed: [],
      semantic_command_fingerprints: {},
      current_blocking_unknowns: [],
      phase_state: STATE.ORIENT,
      last_compaction_timestamp: null,
    };
  }

  /**
   * Persist state before context compaction.
   */
  persist(ledger, stateMachine) {
    this.state.evidence_ledger = this._serializeLedger(ledger);
    this.state.resolved_hypotheses = ledger.getAllHypotheses()
      .filter((h) => h.blocking_unknowns.length === 0)
      .map((h) => h.hypothesis_id);
    this.state.canonical_authorities_found = ledger.getAllHypotheses()
      .flatMap((h) => h.canonical_sources_found);
    this.state.current_blocking_unknowns = ledger.getAllHypotheses()
      .flatMap((h) => h.blocking_unknowns);
    this.state.phase_state = stateMachine ? stateMachine.getCurrentState() : STATE.ORIENT;
    this.state.last_compaction_timestamp = new Date().toISOString();

    try {
      const dir = path.dirname(this.persistPath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(this.persistPath, JSON.stringify(this.state, null, 2));
    } catch (e) {
      // Silently fail — compaction continuity is best-effort
    }
  }

  /**
   * Restore state after context compaction.
   * @returns {boolean} Whether state was successfully restored
   */
  restore() {
    try {
      if (!fs.existsSync(this.persistPath)) return false;
      const data = JSON.parse(fs.readFileSync(this.persistPath, "utf8"));
      this.state = { ...this.state, ...data };
      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Check if a fact should be re-discovered after compaction.
   * Only re-discover if source fingerprint changed or evidence became stale.
   */
  shouldRediscover(factKey, sourceFingerprint) {
    const existing = this.state.semantic_command_fingerprints[factKey];
    if (!existing) return true; // New fact, allow discovery
    if (existing.fingerprint !== sourceFingerprint) return true; // Fingerprint changed
    return false; // Same fingerprint, no rediscovery needed
  }

  _serializeLedger(ledger) {
    const all = ledger.getAllHypotheses();
    const result = {};
    for (const h of all) {
      result[h.hypothesis_id] = {
        question: h.question,
        evidence_for: h.evidence_for,
        evidence_against: h.evidence_against,
        unknowns: h.unknowns,
        blocking_unknowns: h.blocking_unknowns,
        confidence: h.confidence,
        canonical_sources_found: h.canonical_sources_found,
        next_information_needed: h.next_information_needed,
        created_at: h.created_at,
        updated_at: h.updated_at,
      };
    }
    return result;
  }

  getState() {
    return { ...this.state };
  }
}

// ─── Workflow State Machine (P10) ───────────────────────────────────────────

class WorkflowStateMachine {
  constructor() {
    this.states = [
      STATE.ORIENT,
      STATE.DISCOVER,
      STATE.HYPOTHESIS_READY,
      STATE.IMPLEMENT,
      STATE.VERIFY,
      STATE.REPORT,
    ];
    this.currentState = STATE.ORIENT;
    this.transitionHistory = [];
  }

  /**
   * Transition to a new state.
   * @param {string} newState
   * @returns {{ success: boolean, reason: string }}
   */
  transitionTo(newState) {
    const currentIdx = this.states.indexOf(this.currentState);
    const newIdx = this.states.indexOf(newState);

    if (newIdx === -1) {
      return { success: false, reason: `Unknown state: ${newState}` };
    }

    // Forward transitions only (no going back, except DISCOVER for blocker verification)
    if (newIdx < currentIdx && newState !== STATE.DISCOVER) {
      return {
        success: false,
        reason: `Cannot move backward from ${this.currentState} to ${newState} (except DISCOOVER for blocker verification)`,
      };
    }

    // DIAGNOSTIC shortcut: HYPOTHESIS_READY → REPORT
    // This is handled externally by setting the state directly

    const oldState = this.currentState;
    this.currentState = newState;
    this.transitionHistory.push({
      from: oldState,
      to: newState,
      timestamp: new Date().toISOString(),
    });

    return { success: true, reason: `Transitioned from ${oldState} to ${newState}` };
  }

  getCurrentState() {
    return this.currentState;
  }

  /**
   * Check if broad discovery is prohibited.
   */
  isBroadDiscoveryProhibited() {
    const prohibitedStates = [STATE.IMPLEMENT, STATE.VERIFY, STATE.REPORT];
    return prohibitedStates.includes(this.currentState);
  }

  /**
   * Get the transition history.
   */
  getHistory() {
    return [...this.transitionHistory];
  }
}

// ─── Force Convergence (P11) ─────────────────────────────────────────────────

class ForceConvergence {
  /**
   * Generate a convergence check output.
   * @param {object} context - { knownFacts, blockingUnknown, requiredNextAction, decisionContext }
   * @returns {object} Convergence check result
   */
  generateConvergenceCheck(context = {}) {
    const { knownFacts = [], blockingUnknown = null, requiredNextAction = null, decisionContext = null } = context;

    const check = {
      type: "CONVERGENCE_CHECK",
      timestamp: new Date().toISOString(),
      known_facts: knownFacts,
      blocking_unknown: blockingUnknown,
      required_next_action: requiredNextAction,
    };

    if (!blockingUnknown || blockingUnknown === "NONE") {
      check.decision = "TRANSITION_IMMEDIATELY";
      check.message = "No blocking unknown exists — transitioning to next phase without operator input.";
    } else {
      // Evaluate whether the blocking unknown is actually blocking a decision,
      // or whether the data supports a stronger verdict (e.g., TERMINATE).
      const negativeVerdict = this._evaluateNegativeEvidence(decisionContext, knownFacts);
      if (negativeVerdict) {
        check.decision = negativeVerdict.decision;
        check.message = negativeVerdict.message;
        check.negative_evidence = negativeVerdict.evidence;
      } else {
        check.decision = "PROCEED_WITH_CAUTION";
        check.message = `Blocking unknown '${blockingUnknown}' remains — proceed with the required next action.`;
      }
    }

    return check;
  }

  /**
   * Evaluate whether the snapshot data contains clear negative evidence
   * that supports a stronger decision than PROCEED_WITH_CAUTION.
   * @param {object|null} decisionContext - Snapshot data or decision inputs
   * @param {object[]} knownFacts - Known facts from the evidence ledger
   * @returns {object|null} Negative verdict object or null if no negative evidence is decisive
   */
  _evaluateNegativeEvidence(decisionContext, knownFacts) {
    if (!decisionContext) return null;

    const { data_quality, economics, funnel, decision_inputs, zones } = decisionContext;

    // CASE 6: data_quality FAIL → BLOCKED_CANONICAL_DATA
    if (data_quality && data_quality.status === "FAIL") {
      return {
        decision: "BLOCKED_CANONICAL_DATA",
        message: "Canonical data quality is FAIL — required authoritative facts are unavailable.",
        evidence: { data_quality_status: data_quality.status, blockers: data_quality.blockers || [] },
      };
    }

    // CASE 7: PARTIAL caused only by SPEND_WITHOUT_REVENUE while revenue=0 is an observed trustworthy value
    // → do not automatically suppress business decision; evaluate negative evidence below

    // Evaluate negative evidence from snapshot data
    const hasSpend = economics && economics.spend_usd > 0;
    const hasZeroRevenue = economics && economics.revenue_usd === 0;
    const hasZeroConversions = funnel && funnel.paid_conversions === 0;
    const hasNegativeZoneSignal = zones && zones.some && zones.some((z) => z.signal === "NEGATIVE");
    const noPositiveSignal = decision_inputs && decision_inputs.positive_signal_present === false;
    const hasNegativeSignal = decision_inputs && decision_inputs.negative_signal_present === true;
    const infoReady = decision_inputs && decision_inputs.information_readiness === "READY";

    // CASE 2: material spend/sample, 0 revenue, 0 conversions, no positive signal,
    // low additional information value → TERMINATE
    if (hasSpend && hasZeroRevenue && hasZeroConversions && noPositiveSignal && infoReady) {
      const evidence = {
        spend_usd: economics.spend_usd,
        revenue_usd: economics.revenue_usd,
        paid_conversions: funnel.paid_conversions,
        positive_signal_present: false,
        negative_signal_present: hasNegativeSignal,
        information_readiness: "READY",
        zone_signals: zones ? zones.map((z) => ({ zone_id: z.zone_id, signal: z.signal })) : [],
      };
      return {
        decision: "TERMINATE",
        message: "Negative evidence is sufficiently sampled: spend > 0, revenue = 0, conversions = 0, no positive signal, information readiness READY. Additional spend has negligible expected information/economic value.",
        evidence,
      };
    }

    // CASE 9: campaign already PAUSED + terminal negative evidence → TERMINATE recommendation remains valid
    // (Handled by the same logic as CASE 2 — PAUSED status doesn't change the negative evidence evaluation)

    // CASE 1: spend > 0, 0 revenue, 0 conversions, insufficient sample → WATCH, not TERMINATE
    // This case is handled by the evidence sufficiency gate, not here.
    // If we reach this point with insufficient sample, the caller should set a blocking unknown
    // that indicates insufficient sample, and PROCEED_WITH_CAUTION is appropriate.

    return null;
  }
}

// ─── Stop Conditions (P12) ──────────────────────────────────────────────────

class StopConditions {
  /**
   * Evaluate stop conditions for discovery.
   * @param {object} context
   * @param {EvidenceLedger} context.ledger
   * @param {LoopDetector} context.loopDetector
   * @param {InformationGainScorer} context.gainScorer
   * @param {SideQuestDetector} context.sideQuestDetector
   * @param {number} context.discoveryActionCount
   * @returns {{ should_stop: boolean, reason: string, stop_reason: string }}
   */
  evaluateStopConditions(context = {}) {
    const {
      ledger,
      loopDetector,
      gainScorer,
      sideQuestDetector,
      discoveryActionCount = 0,
      canonicalAuthorityIdentified = false,
      exactDefectLocalized = false,
      smallestRepairClear = false,
      remainingUnknownsAffectRepair = true,
      missionAcceptanceCanBeEvaluated = false,
    } = context;

    // P12: STOP DISCOVERY when ANY is true

    // Condition 1: canonical authority identified AND exact defect localized
    if (canonicalAuthorityIdentified && exactDefectLocalized) {
      return { should_stop: true, reason: "Canonical authority identified and exact defect localized", stop_reason: STOP_REASON.STOP_DISCOVERY };
    }

    // Condition 2: smallest repair is clear
    if (smallestRepairClear) {
      return { should_stop: true, reason: "Smallest repair is clear", stop_reason: STOP_REASON.STOP_DISCOVERY };
    }

    // Condition 3: remaining unknowns do not affect repair correctness
    if (!remainingUnknownsAffectRepair) {
      return { should_stop: true, reason: "Remaining unknowns do not affect repair correctness", stop_reason: STOP_REASON.STOP_DISCOVERY };
    }

    // Condition 4: mission acceptance criteria can already be evaluated
    if (missionAcceptanceCanBeEvaluated) {
      return { should_stop: true, reason: "Mission acceptance criteria can already be evaluated", stop_reason: STOP_REASON.STOP_DISCOVERY };
    }

    // Condition 5: repeated searches provide zero new evidence
    if (gainScorer && gainScorer.getConsecutiveLowValueCount() >= FORCE_CONVERGENCE_THRESHOLD) {
      return { should_stop: true, reason: "3 consecutive LOW/ZERO discovery actions — no new evidence being produced", stop_reason: STOP_REASON.FORCE_CONVERGENCE };
    }

    // Check loop detector
    if (loopDetector && loopDetector.hasActiveLoop()) {
      return { should_stop: true, reason: "Active discovery loop detected", stop_reason: STOP_REASON.SEMANTIC_REPEAT_DETECTED };
    }

    // Check side quests
    if (sideQuestDetector && sideQuestDetector.hasBlockedSideQuests()) {
      // Not a stop condition per se, but worth noting
    }

    // Check hard ceiling (P15)
    if (discoveryActionCount >= DEFAULT_HARD_CEILING) {
      return { should_stop: true, reason: `Hard safety limit of ${DEFAULT_HARD_CEILING} discovery actions reached`, stop_reason: STOP_REASON.HARD_CONVERGENCE_TRIGGER };
    }

    return { should_stop: false, reason: "Continue discovery", stop_reason: null };
  }
}

// ─── Metrics (P14) ──────────────────────────────────────────────────────────

class MetricsCollector {
  constructor() {
    this.reset();
  }

  reset() {
    this.metrics = {
      total_tool_calls: 0,
      discovery_tool_calls: 0,
      exact_repeat_count: 0,
      semantic_repeat_count: 0,
      zero_information_calls: 0,
      broad_search_count: 0,
      forced_convergence_count: 0,
      side_quest_block_count: 0,
      unauthorized_probe_block_count: 0,
      post_compaction_rediscovery_count: 0,
      hard_convergence_trigger_count: 0,
      stop_discovery_count: 0,
      evidence_sufficient_count: 0,
      codegraph_bootstrap_blocked_count: 0,
    };
  }

  increment(metricName) {
    if (this.metrics[metricName] !== undefined) {
      this.metrics[metricName]++;
    }
  }

  getMetrics() {
    return { ...this.metrics };
  }

  /**
   * Generate a compact metrics summary for receipts.
   */
  getSummary() {
    const m = this.metrics;
    return {
      total_tool_calls: m.total_tool_calls,
      discovery_tool_calls: m.discovery_tool_calls,
      exact_repeats: m.exact_repeat_count,
      semantic_repeats: m.semantic_repeat_count,
      zero_information_calls: m.zero_information_calls,
      broad_searches: m.broad_search_count,
      forced_convergence_triggers: m.forced_convergence_count,
      side_quests_blocked: m.side_quest_block_count,
      unauthorized_probes_blocked: m.unauthorized_probe_block_count,
      post_compaction_rediscovery: m.post_compaction_rediscovery_count,
      hard_convergence_triggers: m.hard_convergence_trigger_count,
      discovery_stops: m.stop_discovery_count + m.evidence_sufficient_count,
    };
  }
}

// ─── Main Convergence Governor ──────────────────────────────────────────────

class ConvergenceGovernor {
  constructor(options = {}) {
    this.fingerprinter = new CommandFingerprinter();
    this.budget = new DiscoveryBudget(options.budgets);
    this.ledger = new EvidenceLedger();
    this.sufficiencyGate = new EvidenceSufficiencyGate();
    this.loopDetector = new LoopDetector(this.fingerprinter);
    this.gainScorer = new InformationGainScorer();
    this.sideQuestDetector = new SideQuestDetector();
    this.mutationGuard = new MutationEscalationGuard();
    this.compactionContinuity = new ContextCompactionContinuity(options.persistPath);
    this.stateMachine = new WorkflowStateMachine();
    this.forceConvergence = new ForceConvergence();
    this.stopConditions = new StopConditions();
    this.metrics = new MetricsCollector();
    this.hardCeiling = options.hardCeiling || DEFAULT_HARD_CEILING;
    this.missionMode = options.missionMode || "EXECUTE";
    this.discoveryActionCount = 0;
    this.canonicalAuthorities = new Set();
    this.exactDefectLocalized = false;
    // CodeGraph bootstrap state
    this.codegraphBootstrapCompleted = false;
    this.codegraphBootstrapReceipt = null;
    this.codegraphBootstrapBlockedCount = 0;
    // Convergence governor runtime enforcement proof (P16)
    // RUNTIME_ENFORCED = actual tool-dispatch gate is active
    // BEHAVIORAL_ONLY = LLM following instructions voluntarily
    // NOT_ACTIVE = no governor is active
    this.governorEnforcementState = GOVERNOR_ENFORCEMENT_STATE.NOT_ACTIVE;
    this.governorRuntimeProof = {
      enforcement_module: null,
      dispatch_boundary: null,
      session_id: null,
      mission_id: null,
      bootstrap_state: null,
      blocked_action_count: 0,
      runtime_gate_version: null,
      runtime_gate_fingerprint: null,
      set_at: null,
    };
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /**
   * Record a discovery action and evaluate whether it should be allowed.
   * @param {object} action
   * @param {string} action.command - The command/action string
   * @param {string} [action.mode] - Current mission mode
   * @param {boolean} [action.isMutation] - Whether the action mutates state
   * @param {object} [action.evidence] - Evidence produced by this action
   * @param {string} [action.justification] - NEW_EVIDENCE_JUSTIFICATION for budget overruns
   * @param {string} [action.blockingUnknownId] - Blocking unknown this maps to (P7)
   * @param {string} [action.discoveryFamily] - Discovery family name (P7)
   * @param {boolean} [action.sameResult] - Whether the result is materially identical to prior runs (for loop detection)
   * @returns {{ allowed: boolean, reason: string, convergence_check?: object }}
   */
  recordAction(action) {
    const {
      command,
      mode = this.missionMode,
      isMutation = false,
      evidence = null,
      justification = "",
      blockingUnknownId = null,
      discoveryFamily = null,
      sameResult = true,
    } = action;

    // ── CodeGraph Bootstrap Gate ──
    // Block discovery actions until CodeGraph bootstrap is complete.
    // This is the mandatory enforcement boundary: raw repository/runtime
    // discovery cannot proceed before CodeGraph has resolved the repo.
    if (!this.codegraphBootstrapCompleted) {
      this.codegraphBootstrapBlockedCount++;
      this.metrics.increment("codegraph_bootstrap_blocked_count");
      return {
        allowed: false,
        reason: "CODEGRAPH_BOOTSTRAP_REQUIRED: CodeGraph mandatory bootstrap has not been completed. Mission startup must run lah_context_resolve() before any discovery action.",
        stop_reason: STOP_REASON.CODEGRAPH_BOOTSTRAP_REQUIRED,
        convergence_check: this.forceConvergence.generateConvergenceCheck({
          knownFacts: this._getKnownFacts(),
          blockingUnknown: this._getBlockingUnknown(),
          requiredNextAction: "Complete CodeGraph bootstrap before discovery",
        }),
      };
    }

    this.metrics.increment("total_tool_calls");

    // P8: Mutation escalation guard check
    if (isMutation) {
      const mutationCheck = this.mutationGuard.check(mode, command, isMutation);
      if (!mutationCheck.allowed) {
        this.metrics.increment("unauthorized_probe_block_count");
        return {
          allowed: false,
          reason: mutationCheck.reason,
          stop_reason: STOP_REASON.UNAUTHORIZED_DISCOVERY_MUTATION_BLOCKED,
        };
      }
    }

    // P1: Fingerprint the command
    const fingerprint = this.fingerprinter.fingerprint(command, { step: this.discoveryActionCount });

    // Track new evidence count for semantic repeat detection
    if (evidence && fingerprint.new_evidence_count !== undefined) {
      fingerprint.new_evidence_count++;
    }

    // P5: Check for exact repeat
    const exactRepeat = this.loopDetector.checkExactRepeat(fingerprint.exact_command_hash, sameResult);
    if (exactRepeat.detected) {
      this.metrics.increment("exact_repeat_count");
      return {
        allowed: false,
        reason: `EXACT_REPEAT_DETECTED: Command executed ${fingerprint.execution_count} times with identical results`,
        stop_reason: STOP_REASON.EXACT_REPEAT_DETECTED,
        convergence_check: this.forceConvergence.generateConvergenceCheck({
          knownFacts: this._getKnownFacts(),
          blockingUnknown: this._getBlockingUnknown(),
          requiredNextAction: "Transition to IMPLEMENT or REPORT phase",
        }),
      };
    }

    // P5: Check for semantic repeat
    const semanticRepeat = this.loopDetector.checkSemanticRepeat(fingerprint.semantic_command_hash, evidence ? 1 : 0);
    if (semanticRepeat.detected) {
      this.metrics.increment("semantic_repeat_count");
      return {
        allowed: false,
        reason: `SEMANTIC_REPEAT_DETECTED: Semantic family ${fingerprint.semantic_command_hash.substring(0, 12)}... produced no new evidence`,
        stop_reason: STOP_REASON.SEMANTIC_REPEAT_DETECTED,
        convergence_check: this.forceConvergence.generateConvergenceCheck({
          knownFacts: this._getKnownFacts(),
          blockingUnknown: this._getBlockingUnknown(),
          requiredNextAction: "Force convergence — transition to next phase",
        }),
      };
    }

    // P2: Check discovery budget
    const budgetCategory = this._categorizeDiscoveryAction(command);
    const budgetCheck = this.budget.check(budgetCategory, justification);
    if (!budgetCheck.allowed) {
      return {
        allowed: false,
        reason: budgetCheck.reason,
        stop_reason: STOP_REASON.STOP_DISCOVERY,
      };
    }

    // P7: Side-quest check
    if (discoveryFamily && blockingUnknownId) {
      const sideQuestCheck = this.sideQuestDetector.registerDiscoveryFamily(discoveryFamily, blockingUnknownId);
      if (!sideQuestCheck.allowed) {
        this.metrics.increment("side_quest_block_count");
        return {
          allowed: false,
          reason: sideQuestCheck.reason,
          stop_reason: STOP_REASON.SIDE_QUEST_BLOCKED,
        };
      }
    }

    // P6: Score information gain
    const gainScore = this.gainScorer.score(evidence, this.ledger);
    if (gainScore.score === "ZERO") {
      this.metrics.increment("zero_information_calls");
    } else if (gainScore.score === "LOW") {
      this.metrics.increment("zero_information_calls"); // Count low as near-zero
    }

    // P15: Hard safety limit
    this.discoveryActionCount++;
    this.metrics.increment("discovery_tool_calls");

    if (this.discoveryActionCount >= this.hardCeiling) {
      this.metrics.increment("hard_convergence_trigger_count");
      return {
        allowed: false,
        reason: `HARD_CONVERGENCE_TRIGGER: ${this.hardCeiling} discovery actions exceeded without state transition`,
        stop_reason: STOP_REASON.HARD_CONVERGENCE_TRIGGER,
        convergence_check: this.forceConvergence.generateConvergenceCheck({
          knownFacts: this._getKnownFacts(),
          blockingUnknown: this._getBlockingUnknown(),
          requiredNextAction: "STOP DISCOVERY — hard ceiling reached",
        }),
      };
    }

    // P6: Force convergence check
    if (this.gainScorer.shouldForceConvergence()) {
      this.metrics.increment("forced_convergence_count");
      return {
        allowed: false,
        reason: "FORCE_CONVERGENCE_REVIEW: 3 consecutive LOW/ZERO discovery actions",
        stop_reason: STOP_REASON.FORCE_CONVERGENCE,
        convergence_check: this.forceConvergence.generateConvergenceCheck({
          knownFacts: this._getKnownFacts(),
          blockingUnknown: this._getBlockingUnknown(),
          requiredNextAction: "Review: what do I know? What unknown blocks progress? Is it required? What single narrow action resolves it?",
        }),
      };
    }

    // P4: Check evidence sufficiency
    const sufficiency = this.sufficiencyGate.evaluateEvidenceSufficiency(this.ledger, { missionMode: mode });
    if (sufficiency.can_proceed && sufficiency.sufficiency === SUFFICIENCY.SUFFICIENT_TO_REPORT) {
      this.metrics.increment("evidence_sufficient_count");
      return {
        allowed: true,
        reason: sufficiency.reason,
        sufficiency: sufficiency.sufficiency,
        recommended_transition: sufficiency.recommended_transition,
      };
    }

    // Record evidence in ledger if provided
    if (evidence) {
      // Auto-record evidence to the most recent hypothesis if any exist
      const hypotheses = this.ledger.getAllHypotheses();
      if (hypotheses.length > 0) {
        const latest = hypotheses[hypotheses.length - 1];
        this.ledger.addEvidence(latest.hypothesis_id, "for", JSON.stringify(evidence));
      }
    }

    return {
      allowed: true,
      reason: `Discovery action recorded (budget: ${budgetCategory} ${this.budget.getCounters()[budgetCategory] || 0})`,
      sufficiency: sufficiency.sufficiency,
    };
  }

  /**
   * Record a hypothesis in the evidence ledger.
   */
  recordHypothesis(hypothesisId, data) {
    return this.ledger.recordHypothesis(hypothesisId, data);
  }

  /**
   * Transition the workflow state machine.
   */
  transitionState(newState) {
    const result = this.stateMachine.transitionTo(newState);
    if (result.success && newState === STATE.IMPLEMENT) {
      // Transitioning to IMPLEMENT — prohibit broad discovery
      this.discoveryActionCount = 0; // Reset discovery counter for implementation phase
    }
    return result;
  }

  /**
   * Mark CodeGraph bootstrap as completed.
   * Called by the startup orchestrator after successful lah_context_resolve().
   * @param {object} receipt - The CodeGraph bootstrap receipt
   */
  setCodeGraphBootstrap(receipt) {
    this.codegraphBootstrapCompleted = true;
    this.codegraphBootstrapReceipt = receipt;
    this.canonicalAuthorities.add("codegraph");
  }

  /**
   * Check whether CodeGraph bootstrap has been completed.
   * @returns {boolean}
   */
  isCodeGraphBootstrapCompleted() {
    return this.codegraphBootstrapCompleted;
  }

  /**
   * Get CodeGraph bootstrap metrics.
   * @returns {object}
   */
  getCodeGraphMetrics() {
    return {
      codegraph_bootstrap_completed: this.codegraphBootstrapCompleted,
      codegraph_bootstrap_blocked_count: this.codegraphBootstrapBlockedCount,
      codegraph_bootstrap_receipt: this.codegraphBootstrapReceipt,
    };
  }

  /**
   * Set the convergence governor runtime enforcement proof.
   * This must be called with machine-verifiable evidence that the
   * governor is actively enforcing at the tool-dispatch boundary.
   *
   * @param {object} proof
   * @param {string} proof.state - RUNTIME_ENFORCED | BEHAVIORAL_ONLY | NOT_ACTIVE
   * @param {string} [proof.enforcement_module] - Name of the enforcement module
   * @param {string} [proof.dispatch_boundary] - The dispatch boundary (e.g. handle_function_call)
   * @param {string} [proof.session_id] - The session ID
   * @param {string} [proof.mission_id] - The mission ID
   * @param {string} [proof.bootstrap_state] - The bootstrap state
   * @param {number} [proof.blocked_action_count] - Count of blocked actions
   * @param {string} [proof.runtime_gate_version] - Version of the runtime gate
   * @param {string} [proof.runtime_gate_fingerprint] - SHA-256 fingerprint of the gate
   */
  setRuntimeEnforcementProof(proof) {
    const state = proof.state || GOVERNOR_ENFORCEMENT_STATE.NOT_ACTIVE;
    if (
      state !== GOVERNOR_ENFORCEMENT_STATE.RUNTIME_ENFORCED &&
      state !== GOVERNOR_ENFORCEMENT_STATE.BEHAVIORAL_ONLY &&
      state !== GOVERNOR_ENFORCEMENT_STATE.NOT_ACTIVE
    ) {
      throw new Error(
        `Invalid governor enforcement state '${state}'. ` +
          `Valid states: ${Object.values(GOVERNOR_ENFORCEMENT_STATE).join(", ")}`
      );
    }
    this.governorEnforcementState = state;
    this.governorRuntimeProof = {
      enforcement_module: proof.enforcement_module || null,
      dispatch_boundary: proof.dispatch_boundary || null,
      session_id: proof.session_id || null,
      mission_id: proof.mission_id || null,
      bootstrap_state: proof.bootstrap_state || null,
      blocked_action_count: proof.blocked_action_count || 0,
      runtime_gate_version: proof.runtime_gate_version || null,
      runtime_gate_fingerprint: proof.runtime_gate_fingerprint || null,
      set_at: new Date().toISOString(),
    };
  }

  /**
   * Get the convergence governor runtime enforcement proof.
   * @returns {object}
   */
  getRuntimeEnforcementProof() {
    return {
      state: this.governorEnforcementState,
      proof: this.governorRuntimeProof,
    };
  }

  /**
   * Check whether the governor has RUNTIME_ENFORCED proof.
   * BEHAVIORAL_ONLY and NOT_ACTIVE do NOT satisfy runtime enforcement requirements.
   * @returns {boolean}
   */
  isRuntimeEnforced() {
    return this.governorEnforcementState === GOVERNOR_ENFORCEMENT_STATE.RUNTIME_ENFORCED;
  }

  /**
   * Get the current workflow state.
   */
  getCurrentState() {
    return this.stateMachine.getCurrentState();
  }

  /**
   * Persist state for context compaction continuity (P9).
   */
  persistState() {
    this.compactionContinuity.persist(this.ledger, this.stateMachine);
  }

  /**
   * Restore state after context compaction (P9).
   * Restores both the compaction continuity state and the evidence ledger.
   */
  restoreState() {
    const restored = this.compactionContinuity.restore();
    if (restored) {
      // Also restore the evidence ledger from persisted state
      const persistedLedger = this.compactionContinuity.state.evidence_ledger;
      if (persistedLedger && typeof persistedLedger === "object") {
        this.ledger.restoreFromPersistedState(persistedLedger);
      }
    }
    return restored;
  }

  /**
   * Get the full convergence status.
   */
  getStatus() {
    return {
      state: this.stateMachine.getCurrentState(),
      mission_mode: this.missionMode,
      discovery_actions: this.discoveryActionCount,
      hard_ceiling: this.hardCeiling,
      budget_counters: this.budget.getCounters(),
      evidence_sufficiency: this.sufficiencyGate.evaluateEvidenceSufficiency(this.ledger),
      loops_detected: this.loopDetector.getDetectedLoops().length,
      side_quests_blocked: this.sideQuestDetector.getBlockedSideQuests().length,
      unauthorized_probes_blocked: this.mutationGuard.getBlockedProbes().length,
      consecutive_low_value: this.gainScorer.getConsecutiveLowValueCount(),
      metrics: this.metrics.getSummary(),
      fingerprint_stats: this.fingerprinter.getStats(),
    };
  }

  /**
   * Generate a compact receipt for the mission.
   */
  generateReceipt() {
    return {
      verdict: this._determineVerdict(),
      state: this.stateMachine.getCurrentState(),
      discovery_actions: this.discoveryActionCount,
      metrics: this.metrics.getSummary(),
      budget_counters: this.budget.getCounters(),
      evidence_sufficiency: this.sufficiencyGate.evaluateEvidenceSufficiency(this.ledger),
      loops_detected: this.loopDetector.getDetectedLoops(),
      side_quests_blocked: this.sideQuestDetector.getBlockedSideQuests(),
      unauthorized_probes_blocked: this.mutationGuard.getBlockedProbes(),
      fingerprint_stats: this.fingerprinter.getStats(),
      canonical_authorities: Array.from(this.canonicalAuthorities),
      exact_defect_localized: this.exactDefectLocalized,
      codegraph_bootstrap: this.getCodeGraphMetrics(),
      convergence_governor_enforcement: {
        state: this.governorEnforcementState,
        is_runtime_enforced: this.isRuntimeEnforced(),
        proof: this.governorRuntimeProof,
      },
      timestamp: new Date().toISOString(),
    };
  }

  // ── Internal helpers ────────────────────────────────────────────────────

  _categorizeDiscoveryAction(command) {
    const cmd = command.toLowerCase();
    if (cmd.includes("grep") && !cmd.includes("grep -r")) return "exact_same_command";
    if (cmd.includes("grep -r") || cmd.includes("search_files")) return "same_semantic_search";
    if (cmd.includes("find ") && cmd.includes("repo")) return "broad_repo_search";
    if (cmd.includes("find ") && cmd.includes("/")) return "global_filesystem_search";
    if (cmd.includes("env ") || cmd.includes("printenv")) return "environment_archaeology";
    if (cmd.includes("docker") || cmd.includes("ps ")) return "docker_process_topology";
    if (cmd.includes("swagger") || cmd.includes("openapi")) return "swagger_api_discovery";
    return "same_semantic_search";
  }

  _getKnownFacts() {
    return this.ledger.getAllHypotheses().map((h) => ({
      id: h.hypothesis_id,
      question: h.question,
      confidence: h.confidence,
      blocking_unknowns: h.blocking_unknowns,
    }));
  }

  _getBlockingUnknown() {
    for (const h of this.ledger.getAllHypotheses()) {
      if (h.blocking_unknowns.length > 0) return h.blocking_unknowns[0];
    }
    return "NONE";
  }

  _determineVerdict() {
    const sufficiency = this.sufficiencyGate.evaluateEvidenceSufficiency(this.ledger);
    const metrics = this.metrics.getMetrics();

    if (metrics.exact_repeat_count > 0 || metrics.semantic_repeat_count > 0) {
      if (metrics.unauthorized_probe_block_count > 0) return "BLOCKED_WORKFLOW_INTEGRATION";
      return "PARTIALLY_CERTIFIED";
    }

    if (sufficiency.sufficiency === SUFFICIENCY.SUFFICIENT_TO_REPORT) {
      if (metrics.exact_repeat_count === 0 && metrics.unauthorized_probe_block_count === 0) {
        return "HERMES_CONVERGENCE_GOVERNOR_CERTIFIED";
      }
      return "PARTIALLY_CERTIFIED";
    }

    if (sufficiency.sufficiency === SUFFICIENCY.INSUFFICIENT) {
      return "BLOCKED_HARNESS_ARCHITECTURE";
    }

    return "PARTIALLY_CERTIFIED";
  }
}

// ─── Export ─────────────────────────────────────────────────────────────────

module.exports = {
  ConvergenceGovernor,
  CommandFingerprinter,
  DiscoveryBudget,
  EvidenceLedger,
  EvidenceSufficiencyGate,
  LoopDetector,
  InformationGainScorer,
  SideQuestDetector,
  MutationEscalationGuard,
  ContextCompactionContinuity,
  WorkflowStateMachine,
  ForceConvergence,
  StopConditions,
  MetricsCollector,
  STATE,
  SUFFICIENCY,
  STOP_REASON,
  GOVERNOR_ENFORCEMENT_STATE,
  DEFAULT_BUDGETS,
  DEFAULT_HARD_CEILING,
  FORCE_CONVERGENCE_THRESHOLD,
  DISCOVERY_ACTION_TYPES,
};