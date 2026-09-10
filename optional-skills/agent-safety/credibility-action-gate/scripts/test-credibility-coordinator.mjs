#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT = fileURLToPath(new URL("./credibility-coordinator.mjs", import.meta.url));

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "credibility-gate-"));
}

function writeJson(dir, name, value) {
  const file = path.join(dir, name);
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
  return file;
}

function runCase({ policy, lanes }) {
  const dir = tempDir();
  const policyFile = writeJson(dir, "policy.json", policy);
  const args = [SCRIPT, "--policy", policyFile];
  for (const [name, lane] of Object.entries(lanes)) {
    args.push("--lane", `${name}=${writeJson(dir, `${name}.json`, lane)}`);
  }
  const output = execFileSync("node", args, { encoding: "utf8" });
  return JSON.parse(output);
}

function runCaseExpectError({ policy, lanes }) {
  const dir = tempDir();
  const policyFile = writeJson(dir, "policy.json", policy);
  const args = [SCRIPT, "--policy", policyFile];
  for (const [name, lane] of Object.entries(lanes)) {
    args.push("--lane", `${name}=${writeJson(dir, `${name}.json`, lane)}`);
  }
  assert.throws(() => execFileSync("node", args, { encoding: "utf8", stdio: "pipe" }));
}

function basePolicy(overrides = {}) {
  return {
    policy_name: "test",
    requested_action: { type: "bounded_action", size: 5, unit: "points", is_repeat_action: false },
    authority: { max_autonomous_action_size: 10, small_test_action_size: 1, serious_action_threshold: 5 },
    required_lanes: ["evidence", "external_context"],
    required_lanes_for_serious_action: ["evidence", "external_context", "graph_history"],
    hard_blocker_codes: ["DOMAIN_HARD_BLOCKER", "CONTRADICTION_IN_RECORD"],
    disposition_preferences: {
      reject_on_any_critical: true,
      reject_on_any_high: true,
      missing_required_lane_for_serious_action: "monitor_until_new_evidence",
      multiple_medium_concerns: "eligible_for_small_test_action",
      repeat_action_without_fresh_reason: "monitor_until_new_evidence",
    },
    ...overrides,
  };
}

const cleanEvidence = {
  lane_type: "evidence",
  status: "completed",
  confidence: "high",
  findings: [],
  decision: {
    supports_core_claim: "yes",
    supports_requested_action_size: "yes",
    claimant_linkage: "verified",
    repeat_or_top_up_support: "not_applicable",
  },
};

const cleanExternal = {
  lane_type: "external_context",
  status: "completed",
  confidence: "high",
  findings: [],
  decision: {
    public_context_plausibility: "all_found",
    source_independence: "strong",
  },
};

const cleanGraph = {
  lane_type: "graph_history",
  status: "completed",
  confidence: "medium",
  findings: [],
  decision: {},
};

function test(name, fn) {
  try {
    fn();
    process.stdout.write(`ok - ${name}\n`);
  } catch (error) {
    process.stderr.write(`not ok - ${name}\n${error.stack}\n`);
    process.exitCode = 1;
  }
}

test("clean required lanes allow full policy action", () => {
  const result = runCase({ policy: basePolicy(), lanes: { evidence: cleanEvidence, external_context: cleanExternal, graph_history: cleanGraph } });
  assert.equal(result.disposition, "eligible_for_full_policy_action");
});

test("critical hard blocker blocks by policy", () => {
  const evidence = {
    ...cleanEvidence,
    findings: [{ code: "DOMAIN_HARD_BLOCKER", severity: "critical", summary: "Blocked by policy." }],
  };
  const result = runCase({ policy: basePolicy(), lanes: { evidence, external_context: cleanExternal, graph_history: cleanGraph } });
  assert.equal(result.disposition, "blocked_by_operator_or_legal_policy");
});

test("high finding rejects current record", () => {
  const external = {
    ...cleanExternal,
    findings: [{ code: "COPIED_TEXT_OR_TITLE_RISK", severity: "high", summary: "Highly similar copied record." }],
  };
  const result = runCase({ policy: basePolicy(), lanes: { evidence: cleanEvidence, external_context: external, graph_history: cleanGraph } });
  assert.equal(result.disposition, "reject_current_record");
});

test("multiple medium concerns shrink to small test action", () => {
  const evidence = {
    ...cleanEvidence,
    findings: [{ code: "IDENTITY_OR_LINKAGE_UNVERIFIED", severity: "medium", summary: "Linkage partial." }],
    decision: { ...cleanEvidence.decision, claimant_linkage: "partial" },
  };
  const external = {
    ...cleanExternal,
    findings: [{ code: "SOURCE_INDEPENDENCE_WEAK", severity: "medium", summary: "Independent sources weak." }],
    decision: { ...cleanExternal.decision, source_independence: "weak" },
  };
  const result = runCase({ policy: basePolicy(), lanes: { evidence, external_context: external, graph_history: cleanGraph } });
  assert.equal(result.disposition, "eligible_for_small_test_action");
  assert.equal(result.maximum_recommended_size, 1);
});

test("missing graph lane on serious action monitors", () => {
  const result = runCase({ policy: basePolicy(), lanes: { evidence: cleanEvidence, external_context: cleanExternal } });
  assert.equal(result.disposition, "monitor_until_new_evidence");
  assert(result.missing_lanes.includes("graph_history"));
});

test("missing required lane on non-serious action monitors", () => {
  const policy = basePolicy({
    requested_action: { type: "bounded_action", size: 1, unit: "points", is_repeat_action: false },
    authority: { max_autonomous_action_size: 10, small_test_action_size: 1, serious_action_threshold: 5 },
    required_lanes_for_serious_action: ["graph_history"],
  });
  const result = runCase({ policy, lanes: { evidence: cleanEvidence } });
  assert.equal(result.disposition, "monitor_until_new_evidence");
  assert(result.missing_lanes.includes("external_context"));
});

test("not applicable required lane does not satisfy required lane", () => {
  const graph = { ...cleanGraph, status: "not_applicable" };
  const result = runCase({ policy: basePolicy(), lanes: { evidence: cleanEvidence, external_context: cleanExternal, graph_history: graph } });
  assert.equal(result.disposition, "monitor_until_new_evidence");
  assert(result.missing_lanes.includes("graph_history"));
});

test("unsupported requested action size rejects current record", () => {
  const evidence = {
    ...cleanEvidence,
    decision: { ...cleanEvidence.decision, supports_requested_action_size: "unsupported" },
  };
  const result = runCase({ policy: basePolicy(), lanes: { evidence, external_context: cleanExternal, graph_history: cleanGraph } });
  assert.equal(result.disposition, "reject_current_record");
  assert(result.reasons.some((reason) => reason.code === "REQUESTED_ACTION_SIZE_UNSUPPORTED"));
});

test("partially supported requested action size shrinks to small test action", () => {
  const evidence = {
    ...cleanEvidence,
    decision: { ...cleanEvidence.decision, supports_requested_action_size: "partial" },
  };
  const result = runCase({ policy: basePolicy(), lanes: { evidence, external_context: cleanExternal, graph_history: cleanGraph } });
  assert.equal(result.disposition, "eligible_for_small_test_action");
  assert.equal(result.maximum_recommended_size, 1);
});

test("unsupported core claim and weak context rejects", () => {
  const evidence = {
    ...cleanEvidence,
    decision: { ...cleanEvidence.decision, supports_core_claim: "unsupported" },
  };
  const external = {
    ...cleanExternal,
    decision: { ...cleanExternal.decision, public_context_plausibility: "not_found", source_independence: "none" },
  };
  const result = runCase({ policy: basePolicy(), lanes: { evidence, external_context: external, graph_history: cleanGraph } });
  assert.equal(result.disposition, "reject_current_record");
});

test("repeat action without fresh reason monitors", () => {
  const policy = basePolicy({ requested_action: { type: "bounded_action", size: 2, unit: "points", is_repeat_action: true } });
  const evidence = {
    ...cleanEvidence,
    decision: { ...cleanEvidence.decision, repeat_or_top_up_support: "none" },
  };
  const result = runCase({ policy, lanes: { evidence, external_context: cleanExternal, graph_history: cleanGraph } });
  assert.equal(result.disposition, "monitor_until_new_evidence");
});

test("requested action above authority blocks", () => {
  const policy = basePolicy({ requested_action: { type: "bounded_action", size: 50, unit: "points", is_repeat_action: false } });
  const result = runCase({ policy, lanes: { evidence: cleanEvidence, external_context: cleanExternal, graph_history: cleanGraph } });
  assert.equal(result.disposition, "blocked_by_operator_or_legal_policy");
});

test("lane type mismatch is rejected", () => {
  runCaseExpectError({ policy: basePolicy(), lanes: { evidence: { ...cleanEvidence, lane_type: "external_context" } } });
});

test("invalid lane status is rejected", () => {
  runCaseExpectError({ policy: basePolicy(), lanes: { evidence: { ...cleanEvidence, status: "done" } } });
});

test("lane details preserve unknown fields", () => {
  const evidence = { ...cleanEvidence, reviewer_note: "kept" };
  const result = runCase({ policy: basePolicy(), lanes: { evidence, external_context: cleanExternal, graph_history: cleanGraph } });
  const detail = result.lane_details.find((lane) => lane.lane_type === "evidence");
  assert.equal(detail.reviewer_note, "kept");
});

if (process.exitCode) process.exit(process.exitCode);
