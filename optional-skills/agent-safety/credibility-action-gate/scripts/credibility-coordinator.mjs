#!/usr/bin/env node
/*
 * credibility-coordinator
 *
 * Deterministic coordinator for claim-quality review lanes.
 * Reads lane JSON plus operator policy, writes an analysis-only action disposition.
 */

import fs from "node:fs";
import path from "node:path";

const SEVERITY_RANK = {
  info: 0,
  low: 1,
  medium: 2,
  high: 3,
  critical: 4,
};

const VALID_LANE_STATUSES = new Set(["completed", "missing", "not_applicable", "error"]);

const DEFAULT_POLICY = {
  policy_name: "default-policy",
  policy_version: "0.1.0",
  requested_action: {
    type: "bounded_action",
    size: 1,
    unit: "configured_unit",
    is_repeat_action: false,
  },
  authority: {
    max_autonomous_action_size: 1,
    small_test_action_size: 1,
    serious_action_threshold: 1,
  },
  required_lanes: ["evidence", "external_context"],
  required_lanes_for_serious_action: ["evidence", "external_context"],
  hard_blocker_codes: [
    "POLICY_AUTHORITY_EXCEEDED",
    "DOMAIN_HARD_BLOCKER",
    "CONTRADICTION_IN_RECORD",
  ],
  disposition_preferences: {
    reject_on_any_critical: true,
    reject_on_any_high: true,
    missing_required_lane: "monitor_until_new_evidence",
    missing_required_lane_for_serious_action: "monitor_until_new_evidence",
    multiple_medium_concerns: "eligible_for_small_test_action",
    repeat_action_without_fresh_reason: "monitor_until_new_evidence",
  },
};

function usage() {
  return `Usage:
  node scripts/credibility-coordinator.mjs --policy policy.json --lane evidence=evidence.json [--lane external_context=external.json] [--out disposition.json]

Options:
  --policy FILE       Operator policy JSON. If omitted, conservative defaults are used.
  --lane NAME=FILE    Review lane JSON. Repeat for multiple lanes.
  --out FILE          Write output JSON to FILE instead of stdout.
`;
}

function parseArgs(argv) {
  const args = { lanes: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      process.stdout.write(usage());
      process.exit(0);
    }
    if (arg === "--policy") args.policyFile = argv[++index];
    else if (arg === "--lane") args.lanes.push(argv[++index]);
    else if (arg === "--out") args.out = argv[++index];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function writeJson(file, data) {
  fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`);
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asNumber(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function normalizeSeverity(severity) {
  const normalized = String(severity ?? "info").toLowerCase();
  return Object.prototype.hasOwnProperty.call(SEVERITY_RANK, normalized) ? normalized : "info";
}

function severityRank(severity) {
  return SEVERITY_RANK[normalizeSeverity(severity)] ?? 0;
}

function highestSeverity(findings) {
  return asArray(findings).reduce((highest, finding) => {
    return severityRank(finding.severity) > severityRank(highest) ? normalizeSeverity(finding.severity) : highest;
  }, "info");
}

function mergePolicy(input) {
  const policy = input && typeof input === "object" ? input : {};
  return {
    ...DEFAULT_POLICY,
    ...policy,
    requested_action: {
      ...DEFAULT_POLICY.requested_action,
      ...(policy.requested_action ?? {}),
    },
    authority: {
      ...DEFAULT_POLICY.authority,
      ...(policy.authority ?? {}),
    },
    disposition_preferences: {
      ...DEFAULT_POLICY.disposition_preferences,
      ...(policy.disposition_preferences ?? {}),
    },
    required_lanes: Array.isArray(policy.required_lanes) ? policy.required_lanes : DEFAULT_POLICY.required_lanes,
    required_lanes_for_serious_action: Array.isArray(policy.required_lanes_for_serious_action)
      ? policy.required_lanes_for_serious_action
      : DEFAULT_POLICY.required_lanes_for_serious_action,
    hard_blocker_codes: Array.isArray(policy.hard_blocker_codes)
      ? policy.hard_blocker_codes
      : DEFAULT_POLICY.hard_blocker_codes,
  };
}

function parseLaneSpec(spec) {
  const splitAt = spec.indexOf("=");
  if (splitAt <= 0 || splitAt === spec.length - 1) throw new Error(`Lane must be NAME=FILE, got: ${spec}`);
  return {
    laneType: spec.slice(0, splitAt),
    file: spec.slice(splitAt + 1),
  };
}

function normalizeFinding(finding) {
  return {
    code: String(finding?.code ?? "UNSPECIFIED_FINDING"),
    severity: normalizeSeverity(finding?.severity),
    summary: String(finding?.summary ?? finding?.description ?? ""),
    evidence_refs: asArray(finding?.evidence_refs),
  };
}

function normalizeLane(laneType, raw) {
  if (raw?.lane_type !== undefined && String(raw.lane_type) !== laneType) {
    throw new Error(`Lane spec "${laneType}" does not match lane_type "${raw.lane_type}" in lane JSON.`);
  }
  const status = String(raw?.status ?? "completed");
  if (!VALID_LANE_STATUSES.has(status)) {
    throw new Error(`Invalid status "${status}" for lane "${laneType}".`);
  }
  const lane = {
    ...(raw && typeof raw === "object" ? raw : {}),
    lane_type: String(raw?.lane_type ?? laneType),
    status,
    confidence: raw?.confidence ?? null,
    findings: asArray(raw?.findings).map(normalizeFinding),
    positives: asArray(raw?.positives),
    decision: raw?.decision && typeof raw.decision === "object" ? raw.decision : {},
    limitations: asArray(raw?.limitations),
  };
  if (["missing", "error"].includes(status)) {
    lane.findings.push({
      code: status === "missing" ? "LANE_MISSING" : "LANE_ERROR",
      severity: "medium",
      summary: `${lane.lane_type} lane status is ${status}.`,
      evidence_refs: [],
    });
  }
  lane.highest_severity = highestSeverity(lane.findings);
  return lane;
}

function loadLanes(laneSpecs) {
  return laneSpecs.map((spec) => {
    const parsed = parseLaneSpec(spec);
    return normalizeLane(parsed.laneType, readJson(parsed.file));
  });
}

function isSeriousAction(policy) {
  const size = asNumber(policy.requested_action?.size, 0);
  const threshold = asNumber(policy.authority?.serious_action_threshold, Number.POSITIVE_INFINITY);
  return size >= threshold;
}

function laneMap(lanes) {
  const map = new Map();
  for (const lane of lanes) map.set(lane.lane_type, lane);
  return map;
}

function missingLanes(policy, lanes) {
  const map = laneMap(lanes);
  const required = new Set(asArray(policy.required_lanes));
  if (isSeriousAction(policy)) {
    for (const lane of asArray(policy.required_lanes_for_serious_action)) required.add(lane);
  }
  return [...required].filter((lane) => !map.has(lane) || map.get(lane).status !== "completed");
}

function collectSignals(policy, lanes) {
  const signals = [];
  const hardCodes = new Set(asArray(policy.hard_blocker_codes));
  const allFindings = lanes.flatMap((lane) =>
    lane.findings.map((finding) => ({ ...finding, lane_type: lane.lane_type })));
  const highFindings = allFindings.filter((finding) => severityRank(finding.severity) >= SEVERITY_RANK.high);
  const criticalFindings = allFindings.filter((finding) => severityRank(finding.severity) >= SEVERITY_RANK.critical);
  const mediumPlusFindings = allFindings.filter((finding) => severityRank(finding.severity) >= SEVERITY_RANK.medium);
  const concernLanes = new Set(mediumPlusFindings.map((finding) => finding.lane_type));
  const hardBlockerFindings = allFindings.filter((finding) => hardCodes.has(finding.code));
  const missing = missingLanes(policy, lanes);
  const map = laneMap(lanes);
  const evidence = map.get("evidence")?.decision ?? {};
  const external = map.get("external_context")?.decision ?? {};
  const repeatSupport = String(evidence.repeat_or_top_up_support ?? external.repeat_or_top_up_support ?? "not_applicable");
  const coreSupport = String(evidence.supports_core_claim ?? "unknown");
  const actionSizeSupport = lanes
    .filter((lane) => Object.prototype.hasOwnProperty.call(lane.decision, "supports_requested_action_size"))
    .map((lane) => ({
      lane_type: lane.lane_type,
      value: String(lane.decision.supports_requested_action_size ?? "unknown"),
    }));
  const publicContext = String(external.public_context_plausibility ?? "not_applicable");
  const sourceIndependence = String(external.source_independence ?? "not_applicable");
  const linkage = String(evidence.claimant_linkage ?? external.claimant_linkage ?? "unknown");

  if (hardBlockerFindings.length > 0) {
    signals.push({
      code: "HARD_BLOCKER_CODE_PRESENT",
      severity: "critical",
      summary: "At least one finding matches the operator policy hard-blocker list.",
      refs: hardBlockerFindings.map((finding) => `${finding.lane_type}:${finding.code}`),
    });
  }
  if (criticalFindings.length > 0) {
    signals.push({
      code: "ANY_CRITICAL_FINDING",
      severity: "critical",
      summary: "At least one lane emitted a critical finding.",
      refs: criticalFindings.map((finding) => `${finding.lane_type}:${finding.code}`),
    });
  }
  if (highFindings.length > 0) {
    signals.push({
      code: "ANY_HIGH_FINDING",
      severity: "high",
      summary: "At least one lane emitted a high-or-higher finding.",
      refs: highFindings.map((finding) => `${finding.lane_type}:${finding.code}`),
    });
  }
  if (mediumPlusFindings.length >= 2 || concernLanes.size >= 2) {
    signals.push({
      code: "MULTIPLE_MEDIUM_OR_CROSS_LANE_CONCERNS",
      severity: "medium",
      summary: "Multiple material concerns appear across the record.",
      refs: mediumPlusFindings.map((finding) => `${finding.lane_type}:${finding.code}`),
    });
  }
  if (missing.length > 0) {
    signals.push({
      code: "REQUIRED_LANE_UNAVAILABLE",
      severity: "medium",
      summary: "One or more required lanes are missing, errored, or not applicable on the current record.",
      refs: missing,
    });
  }
  if (["no", "contradiction", "unsupported"].includes(coreSupport) &&
      !["some_found", "all_found"].includes(publicContext)) {
    signals.push({
      code: "CORE_CLAIM_UNSUPPORTED_AND_CONTEXT_WEAK",
      severity: "high",
      summary: "The core claim is unsupported or contradicted and public context is weak or absent.",
      refs: ["evidence:supports_core_claim", "external_context:public_context_plausibility"],
    });
  }
  const unsupportedActionSize = actionSizeSupport.filter((entry) => ["no", "unsupported"].includes(entry.value));
  if (unsupportedActionSize.length > 0) {
    signals.push({
      code: "REQUESTED_ACTION_SIZE_UNSUPPORTED",
      severity: "high",
      summary: "At least one lane says the current record does not support the requested action size.",
      refs: unsupportedActionSize.map((entry) => `${entry.lane_type}:supports_requested_action_size`),
    });
  } else {
    const weakActionSize = actionSizeSupport.filter((entry) => ["partial", "unknown"].includes(entry.value));
    if (weakActionSize.length > 0) {
      signals.push({
        code: "REQUESTED_ACTION_SIZE_NOT_FULLY_SUPPORTED",
        severity: "medium",
        summary: "At least one lane only partially supports, or cannot confirm, the requested action size.",
        refs: weakActionSize.map((entry) => `${entry.lane_type}:supports_requested_action_size`),
      });
    }
  }
  if (["weak", "none"].includes(sourceIndependence)) {
    signals.push({
      code: "SOURCE_INDEPENDENCE_WEAK",
      severity: "medium",
      summary: "External context does not provide strong independent support.",
      refs: ["external_context:source_independence"],
    });
  }
  if (["unverified", "contradicted"].includes(linkage)) {
    signals.push({
      code: "IDENTITY_OR_LINKAGE_UNVERIFIED",
      severity: linkage === "contradicted" ? "high" : "medium",
      summary: "The record does not sufficiently connect the claimant to the claimed need, asset, or beneficiary.",
      refs: ["claimant_linkage"],
    });
  }
  if (policy.requested_action?.is_repeat_action && ["none", "weak"].includes(repeatSupport)) {
    signals.push({
      code: "REPEAT_ACTION_WITHOUT_FRESH_REASON",
      severity: "medium",
      summary: "Repeat action lacks a fresh reason on the current record.",
      refs: ["repeat_or_top_up_support"],
    });
  }

  const requestedSize = asNumber(policy.requested_action?.size, 0);
  const maxSize = asNumber(policy.authority?.max_autonomous_action_size, 0);
  if (requestedSize > maxSize) {
    signals.push({
      code: "POLICY_AUTHORITY_EXCEEDED",
      severity: "critical",
      summary: "Requested action size exceeds configured autonomous authority.",
      refs: ["policy.authority.max_autonomous_action_size"],
    });
  }

  return {
    all_findings: allFindings,
    high_findings: highFindings,
    medium_plus_findings: mediumPlusFindings,
    concern_lane_count: concernLanes.size,
    missing_lanes: missing,
    signals,
  };
}

function chooseDisposition(policy, signals) {
  const prefs = policy.disposition_preferences ?? {};
  const hasCode = (code) => signals.some((signal) => signal.code === code);
  const anyCritical = signals.some((signal) => severityRank(signal.severity) >= SEVERITY_RANK.critical);
  const anyHigh = signals.some((signal) => severityRank(signal.severity) >= SEVERITY_RANK.high);
  const anyMedium = signals.some((signal) => severityRank(signal.severity) >= SEVERITY_RANK.medium);
  const requestedSize = asNumber(policy.requested_action?.size, 0);
  const maxSize = asNumber(policy.authority?.max_autonomous_action_size, 0);
  const smallSize = asNumber(policy.authority?.small_test_action_size, Math.min(requestedSize, maxSize));

  if (hasCode("POLICY_AUTHORITY_EXCEEDED") || hasCode("HARD_BLOCKER_CODE_PRESENT")) {
    return {
      disposition: "blocked_by_operator_or_legal_policy",
      action_size_guidance: "do_not_act_without_policy_change",
      maximum_recommended_size: 0,
    };
  }
  if (anyCritical && prefs.reject_on_any_critical !== false) {
    return {
      disposition: "reject_current_record",
      action_size_guidance: "do_not_act_on_current_record",
      maximum_recommended_size: 0,
    };
  }
  if (anyHigh && prefs.reject_on_any_high !== false) {
    return {
      disposition: "reject_current_record",
      action_size_guidance: "do_not_act_on_current_record",
      maximum_recommended_size: 0,
    };
  }
  if (hasCode("REQUIRED_LANE_UNAVAILABLE")) {
    return {
      disposition: prefs.missing_required_lane ??
        prefs.missing_required_lane_for_serious_action ??
        "monitor_until_new_evidence",
      action_size_guidance: "wait_for_required_lanes",
      maximum_recommended_size: 0,
    };
  }
  if (hasCode("REPEAT_ACTION_WITHOUT_FRESH_REASON")) {
    return {
      disposition: prefs.repeat_action_without_fresh_reason ?? "monitor_until_new_evidence",
      action_size_guidance: "do_not_repeat_without_new_record_support",
      maximum_recommended_size: 0,
    };
  }
  if (hasCode("MULTIPLE_MEDIUM_OR_CROSS_LANE_CONCERNS")) {
    return {
      disposition: prefs.multiple_medium_concerns ?? "eligible_for_small_test_action",
      action_size_guidance: "smallest_configured_test_action",
      maximum_recommended_size: Math.min(smallSize, maxSize, requestedSize),
    };
  }
  if (anyMedium) {
    return {
      disposition: "eligible_for_small_test_action",
      action_size_guidance: "smallest_configured_test_action",
      maximum_recommended_size: Math.min(smallSize, maxSize, requestedSize),
    };
  }
  if (requestedSize <= maxSize) {
    return {
      disposition: "eligible_for_full_policy_action",
      action_size_guidance: "requested_action_within_policy",
      maximum_recommended_size: requestedSize,
    };
  }
  return {
    disposition: "eligible_for_bounded_action",
    action_size_guidance: "cap_to_operator_policy",
    maximum_recommended_size: maxSize,
  };
}

function confidenceFor(signals, lanes, collected) {
  if (signals.some((signal) => severityRank(signal.severity) >= SEVERITY_RANK.high)) return "low";
  if (collected.missing_lanes.length > 0) return "low_due_to_missing_lanes";
  if (signals.some((signal) => severityRank(signal.severity) >= SEVERITY_RANK.medium)) return "low_to_medium";
  const laneConfidences = lanes.map((lane) => lane.confidence).filter(Boolean);
  if (laneConfidences.includes("low")) return "low_to_medium";
  if (laneConfidences.includes("medium")) return "medium";
  return "medium_to_high";
}

function coordinate(policyInput, lanes) {
  const policy = mergePolicy(policyInput);
  const collected = collectSignals(policy, lanes);
  const selected = chooseDisposition(policy, collected.signals);
  return {
    review_type: "credibility-action-gate",
    review_version: "0.1.0",
    reviewed_at: new Date().toISOString(),
    status: "completed",
    analysis_only: true,
    policy: {
      policy_name: policy.policy_name,
      policy_version: policy.policy_version,
      action_domain: policy.action_domain ?? "general",
      requested_action: policy.requested_action,
      authority: policy.authority,
    },
    disposition: selected.disposition,
    confidence: confidenceFor(collected.signals, lanes, collected),
    action_size_guidance: selected.action_size_guidance,
    maximum_recommended_size: selected.maximum_recommended_size,
    reasons: collected.signals.map((signal) => ({
      code: signal.code,
      severity: signal.severity,
      summary: signal.summary,
      refs: signal.refs,
    })),
    missing_lanes: collected.missing_lanes,
    lane_summary: lanes.map((lane) => ({
      lane_type: lane.lane_type,
      status: lane.status,
      confidence: lane.confidence,
      highest_severity: lane.highest_severity,
      finding_count: lane.findings.length,
    })),
    lane_details: lanes,
    integration_guidance: {
      does_not_execute_action: true,
      use_as_gate_not_mission_selector: true,
      avoid_public_accusation: "Use record-scoped language unless independent evidence supports a stronger public claim.",
    },
  };
}

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    const policy = args.policyFile ? readJson(args.policyFile) : DEFAULT_POLICY;
    const lanes = loadLanes(args.lanes);
    const result = coordinate(policy, lanes);
    if (args.out) writeJson(args.out, result);
    else process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`credibility-coordinator error: ${error.message}\n`);
    process.exit(1);
  }
}

main();
