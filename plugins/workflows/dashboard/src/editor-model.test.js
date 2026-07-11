import { describe, expect, it } from "vitest";
import {
  supportedEditorCoverage,
  editorSections,
  changeNodeType,
  conditionFromForm,
  resultContractFromRows,
  inputRowsFromTrigger,
  inputSchemaFromRows,
  triggerIntakeFromForm,
  readyPathFromTrigger,
  workflowIdFromText,
  RESULT_CONTRACT_PRIMITIVES,
  CONDITION_OPS,
} from "./editor-model.js";

describe("supportedEditorCoverage", () => {
  it("returns exact expected coverage", () => {
    expect(supportedEditorCoverage()).toEqual({
      triggers: ["manual", "schedule"],
      nodes: ["agent_task", "fail", "join", "parallel", "pass", "switch", "wait"],
      intakeModes: ["continuous", "single"],
    });
  });
});

describe("editorSections", () => {
  it("returns metadata, trigger, and node sections for a spec", () => {
    const spec = {
      id: "test",
      name: "Test",
      version: 1,
      triggers: [{ id: "manual", type: "manual" }],
      nodes: {
        work: { type: "agent_task", profile: "worker", prompt: "Do it" },
        done: { type: "pass", output: "ok" },
      },
      edges: [],
    };
    const sections = editorSections(spec);
    expect(sections.some((s) => s.kind === "metadata")).toBe(true);
    expect(sections.some((s) => s.kind === "trigger" && s.id === "manual")).toBe(true);
    expect(sections.some((s) => s.kind === "node" && s.id === "work" && s.type === "agent_task")).toBe(true);
    expect(sections.some((s) => s.kind === "node" && s.id === "done" && s.type === "pass")).toBe(true);
  });

  it("returns empty array for null spec", () => {
    expect(editorSections(null)).toEqual([]);
  });

  it("lists type-specific fields for each node section", () => {
    const spec = { nodes: { w: { type: "wait", seconds: 10 } } };
    const sections = editorSections(spec);
    const waitSection = sections.find((s) => s.kind === "node" && s.id === "w");
    expect(waitSection.fields).toContain("seconds");
  });
});

describe("changeNodeType", () => {
  it("previews and removes fields incompatible with the next node type", () => {
    const spec = {
      nodes: {
        work: {
          type: "agent_task",
          profile: "worker",
          prompt: "Do it",
          result_contract: { ok: "boolean" },
        },
      },
    };
    const result = changeNodeType(spec, "work", "pass");
    expect(result.removedFields.sort()).toEqual(["profile", "prompt", "result_contract"]);
    expect(result.spec.nodes.work).toEqual({ type: "pass", output: {} });
  });

  it("adds defaults when converting to agent_task", () => {
    const spec = { nodes: { step: { type: "pass", output: "hello" } } };
    const result = changeNodeType(spec, "step", "agent_task");
    expect(result.removedFields).toEqual(["output"]);
    expect(result.spec.nodes.step.type).toBe("agent_task");
    expect(result.spec.nodes.step.profile).toBe("default");
    expect(result.spec.nodes.step.prompt).toBeTruthy();
    expect(result.spec.nodes.step.result_contract).toBeTruthy();
  });

  it("adds defaults when converting to wait", () => {
    const spec = { nodes: { step: { type: "pass" } } };
    const result = changeNodeType(spec, "step", "wait");
    expect(result.spec.nodes.step.type).toBe("wait");
    expect(result.spec.nodes.step.seconds).toBe(60);
  });

  it("adds defaults when converting to switch", () => {
    const spec = { nodes: { step: { type: "pass" } } };
    const result = changeNodeType(spec, "step", "switch");
    expect(result.spec.nodes.step.type).toBe("switch");
    expect(result.spec.nodes.step.cases).toEqual([]);
  });

  it("adds defaults when converting to fail", () => {
    const spec = { nodes: { step: { type: "pass" } } };
    const result = changeNodeType(spec, "step", "fail");
    expect(result.spec.nodes.step.type).toBe("fail");
    expect(result.spec.nodes.step.output).toBe("Workflow failed.");
  });

  it("preserves title across type change", () => {
    const spec = { nodes: { step: { type: "pass", title: "My Step" } } };
    const result = changeNodeType(spec, "step", "wait");
    expect(result.spec.nodes.step.title).toBe("My Step");
    expect(result.spec.nodes.step.type).toBe("wait");
  });

  it("returns empty removedFields for missing node", () => {
    const spec = { nodes: {} };
    const result = changeNodeType(spec, "missing", "pass");
    expect(result.removedFields).toEqual([]);
  });

  it("does not mutate the original spec", () => {
    const spec = { nodes: { work: { type: "agent_task", profile: "w", prompt: "p" } } };
    changeNodeType(spec, "work", "pass");
    expect(spec.nodes.work.type).toBe("agent_task");
    expect(spec.nodes.work.profile).toBe("w");
  });
});

describe("node type round-trips", () => {
  it("agent_task round-trips result_contract through rows", () => {
    const contract = { summary: "string", status: "string", verdict: "approved|rejected" };
    const rows = Object.keys(contract).map((key) => ({ key, type: contract[key] }));
    const restored = resultContractFromRows(rows);
    expect(restored).toEqual(contract);
  });

  it("switch round-trips cases through condition forms", () => {
    const form = { op: "eq", leftPath: "$.input.status", rightValue: "approved" };
    const condition = conditionFromForm(form);
    expect(condition).toEqual({ op: "eq", left: { path: "$.input.status" }, right: "approved" });
  });

  it("pass preserves output through changeNodeType", () => {
    const spec = { nodes: { step: { type: "pass", output: "hello", title: "Step" } } };
    const result = changeNodeType(spec, "step", "pass");
    expect(result.spec.nodes.step.output).toBe("hello");
    expect(result.spec.nodes.step.title).toBe("Step");
  });

  it("fail preserves output through changeNodeType", () => {
    const spec = { nodes: { step: { type: "fail", output: "Bad", title: "Err" } } };
    const result = changeNodeType(spec, "step", "fail");
    expect(result.spec.nodes.step.output).toBe("Bad");
    expect(result.spec.nodes.step.title).toBe("Err");
  });

  it("wait preserves seconds through changeNodeType", () => {
    const spec = { nodes: { step: { type: "wait", seconds: 120, title: "Pause" } } };
    const result = changeNodeType(spec, "step", "wait");
    expect(result.spec.nodes.step.seconds).toBe(120);
    expect(result.spec.nodes.step.title).toBe("Pause");
  });

  it("switch preserves cases and default through changeNodeType", () => {
    const cases = [{ name: "ok", when: { op: "eq", left: { path: "$.input.status" }, right: "ok" } }];
    const spec = { nodes: { decide: { type: "switch", cases, default: "fallback", title: "Decide" } } };
    const result = changeNodeType(spec, "decide", "switch");
    expect(result.spec.nodes.decide.cases).toEqual(cases);
    expect(result.spec.nodes.decide.default).toBe("fallback");
  });

  it("parallel and join preserve title through changeNodeType", () => {
    const spec = { nodes: { fan: { type: "parallel", title: "Fan Out" } } };
    const result = changeNodeType(spec, "fan", "parallel");
    expect(result.spec.nodes.fan.title).toBe("Fan Out");
    expect(result.spec.nodes.fan.type).toBe("parallel");
  });
});

describe("conditionFromForm", () => {
  it("returns null for missing op", () => {
    expect(conditionFromForm(null)).toBeNull();
    expect(conditionFromForm({})).toBeNull();
  });

  it("returns null for unsupported op", () => {
    expect(conditionFromForm({ op: "invalid_op", leftPath: "$.x", rightValue: "y" })).toBeNull();
  });

  it("builds exists condition from path", () => {
    expect(conditionFromForm({ op: "exists", path: "$.input.repo" })).toEqual({
      op: "exists",
      path: "$.input.repo",
    });
  });

  it("builds missing condition from path", () => {
    expect(conditionFromForm({ op: "missing", path: "$.input.field" })).toEqual({
      op: "missing",
      path: "$.input.field",
    });
  });

  it("builds eq condition from form", () => {
    expect(
      conditionFromForm({ op: "eq", leftPath: "$.input.status", rightValue: "approved" })
    ).toEqual({ op: "eq", left: { path: "$.input.status" }, right: "approved" });
  });

  it("builds comparison conditions", () => {
    expect(conditionFromForm({ op: "gt", leftPath: "$.input.count", rightValue: 5 })).toEqual({
      op: "gt",
      left: { path: "$.input.count" },
      right: 5,
    });
  });

  it("builds string op conditions", () => {
    expect(
      conditionFromForm({ op: "contains", leftPath: "$.input.text", rightValue: "error" })
    ).toEqual({ op: "contains", left: { path: "$.input.text" }, right: "error" });
  });

  it("returns null for exists without path", () => {
    expect(conditionFromForm({ op: "exists" })).toBeNull();
  });

  it("returns null for comparison without leftPath", () => {
    expect(conditionFromForm({ op: "eq", rightValue: "x" })).toBeNull();
  });

  it("returns null for comparison without rightValue", () => {
    expect(conditionFromForm({ op: "eq", leftPath: "$.x" })).toBeNull();
  });
});

describe("resultContractFromRows", () => {
  it("converts primitive rows to contract object", () => {
    expect(
      resultContractFromRows([
        { key: "summary", type: "string" },
        { key: "count", type: "number" },
        { key: "ok", type: "boolean" },
      ])
    ).toEqual({ summary: "string", count: "number", ok: "boolean" });
  });

  it("supports all primitive tokens", () => {
    const rows = RESULT_CONTRACT_PRIMITIVES.map((t) => ({ key: t, type: t }));
    expect(resultContractFromRows(rows)).toEqual({
      string: "string",
      number: "number",
      boolean: "boolean",
      array: "array",
      object: "object",
    });
  });

  it("supports enum tokens with pipe separator", () => {
    expect(resultContractFromRows([{ key: "verdict", type: "approved|rejected" }])).toEqual({
      verdict: "approved|rejected",
    });
  });

  it("skips empty or invalid rows", () => {
    expect(
      resultContractFromRows([
        { key: "", type: "string" },
        { key: "ok", type: "" },
        { key: "bad", type: "unknown_type" },
        { key: "single", type: "just_one" },
      ])
    ).toEqual({});
  });

  it("returns empty object for null input", () => {
    expect(resultContractFromRows(null)).toEqual({});
  });
});

describe("trigger intake and input schema conversions", () => {
  it("round-trips input schema through rows", () => {
    const schema = {
      repo_path: { kind: "repo_path", required: true },
      message: { kind: "text", default: "hello" },
    };
    const trigger = { input_schema: schema };
    const rows = inputRowsFromTrigger(trigger);
    expect(rows.length).toBe(2);
    const restored = inputSchemaFromRows(rows);
    expect(restored.repo_path.kind).toBe("repo_path");
    expect(restored.repo_path.required).toBe(true);
    expect(restored.message.kind).toBe("text");
    expect(restored.message.default).toBe("hello");
  });

  it("triggerIntakeFromForm builds intake with mode and optional fields", () => {
    expect(triggerIntakeFromForm("continuous", "$.input.key", "$.input.ready")).toEqual({
      mode: "continuous",
      dedupe_key: "$.input.key",
      ready_when: { op: "exists", path: "$.input.ready" },
    });
  });

  it("triggerIntakeFromForm defaults to single mode", () => {
    expect(triggerIntakeFromForm(null, "", "")).toEqual({ mode: "single" });
  });

  it("readyPathFromTrigger extracts path from intake.ready_when", () => {
    expect(readyPathFromTrigger({ intake: { ready_when: { op: "exists", path: "$.input.x" } } })).toBe("$.input.x");
  });

  it("readyPathFromTrigger returns empty for missing intake", () => {
    expect(readyPathFromTrigger(null)).toBe("");
    expect(readyPathFromTrigger({})).toBe("");
  });
});

describe("workflowIdFromText", () => {
  it("converts text to slug", () => {
    expect(workflowIdFromText("My Workflow")).toBe("my_workflow");
  });

  it("falls back to workflow_draft for empty input", () => {
    expect(workflowIdFromText("")).toBe("workflow_draft");
  });
});

describe("unsupported scope", () => {
  it("editor must NOT offer batch, document upload, split strategy, item source, webhook, kanban_event, send_message, subworkflow, or loop controls", () => {
    const coverage = supportedEditorCoverage();
    const unsupported = [
      "batch",
      "document_upload",
      "split_strategy",
      "item_source",
      "webhook",
      "kanban_event",
      "send_message",
      "subworkflow",
      "loop",
    ];
    const allSupported = [...coverage.triggers, ...coverage.nodes, ...coverage.intakeModes];
    unsupported.forEach((u) => {
      expect(allSupported).not.toContain(u);
    });
  });

  it("node type fields do not include unsupported fields", () => {
    const spec = {
      nodes: {
        work: {
          type: "agent_task",
          profile: "w",
          prompt: "p",
          result_contract: {},
        },
      },
    };
    const result = changeNodeType(spec, "work", "pass");
    const node = result.spec.nodes.work;
    expect(node.batch).toBeUndefined();
    expect(node.loop).toBeUndefined();
    expect(node.subworkflow).toBeUndefined();
    expect(node.split_strategy).toBeUndefined();
    expect(node.webhook).toBeUndefined();
    expect(node.kanban_event).toBeUndefined();
    expect(node.send_message).toBeUndefined();
  });

  it("condition ops are leaf ops only, no unsupported composite controls", () => {
    expect(CONDITION_OPS).not.toContain("batch");
    expect(CONDITION_OPS).not.toContain("loop");
    CONDITION_OPS.forEach((op) => {
      expect(typeof op).toBe("string");
      expect(op.length).toBeGreaterThan(0);
    });
  });
});
