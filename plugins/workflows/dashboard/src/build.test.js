import { describe, expect, it } from "vitest";
import {
  acceptCandidate,
  applyServerDraft,
  buildApiHelpers,
  candidateReducer,
  createCandidateState,
  diffForCandidate,
  isDraftDirty,
  pushUndo,
  recordPublishResult,
  rejectCandidate,
  semanticWorkflowDiff,
} from "./build.js";

describe("semanticWorkflowDiff", () => {
  it("summarizes trigger, node, and routing changes from the plan example", () => {
    const before = {
      triggers: [{ id: "manual", type: "manual" }],
      nodes: { review: { type: "pass" } },
      edges: [],
    };
    const after = {
      triggers: [
        { id: "manual", type: "manual", input_schema: { repo: { kind: "repo_path" } } },
      ],
      nodes: {
        review: { type: "agent_task", profile: "reviewer", prompt: "Review" },
      },
      edges: [{ from: "review", to: "done" }],
    };

    expect(semanticWorkflowDiff(before, after).map((s) => s.kind)).toEqual([
      "triggers",
      "nodes",
      "routing",
    ]);
  });

  it("suppresses unchanged sections and never emits an empty section", () => {
    const before = {
      id: "x",
      name: "X",
      version: 1,
      triggers: [{ id: "manual", type: "manual" }],
      nodes: { a: { type: "pass" } },
      edges: [{ from: "a", to: "a" }],
    };
    const after = { ...before, name: "X Renamed" };

    const sections = semanticWorkflowDiff(before, after);
    expect(sections.map((s) => s.kind)).toEqual(["metadata"]);
    expect(sections.every((s) => Array.isArray(s.items) && s.items.length > 0)).toBe(true);
  });

  it("reports runtime changes when retry/catch blocks change", () => {
    const before = {
      triggers: [{ id: "manual", type: "manual" }],
      nodes: { a: { type: "agent_task", profile: "p", prompt: "p" } },
      edges: [{ from: "a", to: "a" }],
    };
    const after = {
      triggers: [{ id: "manual", type: "manual" }],
      nodes: {
        a: {
          type: "agent_task",
          profile: "p",
          prompt: "p",
          retry: { max_attempts: 3 },
          catch: "b",
        },
        b: { type: "fail", output: { message: "x" } },
      },
      edges: [{ from: "a", to: "a" }],
    };

    const kinds = semanticWorkflowDiff(before, after).map((s) => s.kind);
    expect(kinds).toContain("runtime");
    expect(kinds).toContain("nodes");
  });

  it("reports routing changes for switch and parallel branches", () => {
    const before = {
      triggers: [{ id: "manual", type: "manual" }],
      nodes: { decide: { type: "switch", cases: [{ name: "ok" }] } },
      edges: [{ from: "decide.ok", to: "ok_node" }],
    };
    const after = {
      triggers: [{ id: "manual", type: "manual" }],
      nodes: { decide: { type: "switch", cases: [{ name: "ok" }, { name: "bad" }] } },
      edges: [
        { from: "decide.ok", to: "ok_node" },
        { from: "decide.bad", to: "fail_node" },
      ],
    };

    const kinds = semanticWorkflowDiff(before, after).map((s) => s.kind);
    expect(kinds).toContain("routing");
  });

  it("keeps a stable order: metadata, triggers, nodes, routing, runtime", () => {
    const before = {
      triggers: [{ id: "manual", type: "manual" }],
      nodes: { a: { type: "agent_task", profile: "p", prompt: "p" } },
      edges: [{ from: "a", to: "a" }],
    };
    const after = {
      id: "renamed",
      triggers: [{ id: "manual", type: "manual", input_schema: { x: { kind: "text" } } }],
      nodes: {
        a: { type: "agent_task", profile: "p", prompt: "p2", catch: "b" },
        b: { type: "fail", output: { message: "x" } },
      },
      edges: [{ from: "a", to: "a" }, { from: "b", to: "b" }],
    };

    const kinds = semanticWorkflowDiff(before, after).map((s) => s.kind);
    expect(kinds).toEqual(["metadata", "triggers", "nodes", "routing", "runtime"]);
  });
});

describe("candidate reducer", () => {
  it("starts with no candidate and an empty undo stack", () => {
    const state = createCandidateState();
    expect(state.candidateDraft).toBeNull();
    expect(state.candidateSource).toBeNull();
    expect(state.savedDraft).toBeNull();
    expect(state.workingDraft).toBeNull();
    expect(state.undoStack).toEqual([]);
  });

  it("accepts a generate candidate and saves it through the draft API", async () => {
    const calls = [];
    const api = {
      putDraft: (id, body) => { calls.push(["put", id, body]); return Promise.resolve({ draft: body.spec }); },
      getDraft: () => Promise.resolve(null),
      deleteDraft: () => Promise.resolve({ deleted: true }),
      publish: () => Promise.resolve({ definition: { version: 1 } }),
    };
    let state = createCandidateState();
    state = candidateReducer(state, { type: "set-working", spec: { id: "wf", name: "WF" } });
    state = await candidateReducer(state, {
      type: "candidate",
      source: "generate",
      draft: {
        spec: { id: "wf", name: "WF v2" },
        summary: "Generates the workflow.",
        assumptions: ["Used default trigger"],
        warnings: [],
      },
    }, api);

    expect(state.candidateSource).toBe("generate");
    expect(state.candidateDraft.name).toBe("WF v2");
    expect(state.workingDraft.name).toBe("WF");
    expect(state.savedDraft).toBeNull();

    const before = JSON.parse(JSON.stringify(state));
    state = await candidateReducer(state, { type: "accept" }, api);
    expect(state.candidateDraft).toBeNull();
    expect(state.candidateSource).toBeNull();
    expect(state.workingDraft.name).toBe("WF v2");
    expect(state.savedDraft && state.savedDraft.name).toBe("WF v2");
    expect(calls[0][0]).toBe("put");
    expect(state.undoStack[0]).toEqual(before.workingDraft);
  });

  it("rejects a candidate and leaves the draft unchanged", async () => {
    const calls = [];
    const api = {
      putDraft: (id, body) => { calls.push(["put", id, body]); return Promise.resolve({}); },
      getDraft: () => Promise.resolve(null),
      deleteDraft: () => Promise.resolve({ deleted: true }),
      publish: () => Promise.resolve({}),
    };
    let state = createCandidateState();
    state = candidateReducer(state, { type: "set-working", spec: { id: "wf", name: "WF" } });
    state = await candidateReducer(state, {
      type: "candidate",
      source: "refine",
      draft: {
        spec: { id: "wf", name: "WF Refined" },
        summary: "Refined.",
        assumptions: [],
        warnings: [],
      },
    }, api);

    const beforeWorking = JSON.parse(JSON.stringify(state.workingDraft));
    const beforeSaved = state.savedDraft;
    state = await candidateReducer(state, { type: "reject" }, api);

    expect(state.candidateDraft).toBeNull();
    expect(state.candidateSource).toBeNull();
    expect(state.workingDraft).toEqual(beforeWorking);
    expect(state.savedDraft).toBe(beforeSaved);
    expect(calls).toEqual([]);
  });

  it("publishes with expected_latest_version and clears the draft only on success", async () => {
    const calls = [];
    const api = {
      putDraft: (id, body) => { calls.push(["put", id, body]); return Promise.resolve({}); },
      getDraft: () => Promise.resolve(null),
      deleteDraft: (id) => { calls.push(["del", id]); return Promise.resolve({ deleted: true }); },
      publish: (id, body) => {
        calls.push(["publish", id, body]);
        return Promise.resolve({ definition: { version: body.expected_latest_version + 1 } });
      },
    };
    let state = createCandidateState();
    state = candidateReducer(state, { type: "set-working", spec: { id: "wf", name: "WF" } });
    state = await candidateReducer(state, {
      type: "publish",
      expected_latest_version: 3,
    }, api);

    expect(calls[0]).toEqual(["publish", "wf", { expected_latest_version: 3 }]);
    expect(state.savedDraft && state.savedDraft.name).toBe("WF");
    expect(state.candidateDraft).toBeNull();
  });

  it("preserves the working draft on a 409 publish conflict", async () => {
    const calls = [];
    const api = {
      putDraft: () => { calls.push("put"); return Promise.resolve({}); },
      getDraft: () => Promise.resolve(null),
      deleteDraft: () => Promise.resolve({}),
      publish: () => {
        const err = new Error("409: conflict");
        err.status = 409;
        err.code = "workflow_version_conflict";
        err.hint = "Reload the latest version and review the draft again.";
        throw err;
      },
    };
    let state = createCandidateState();
    state = candidateReducer(state, { type: "set-working", spec: { id: "wf", name: "WF" } });

    await expect(
      candidateReducer(state, { type: "publish", expected_latest_version: 1 }, api)
    ).rejects.toMatchObject({ status: 409, code: "workflow_version_conflict" });

    const failed = await candidateReducer(state, { type: "publish", expected_latest_version: 1 }, {
      ...api,
      publish: () => { throw Object.assign(new Error("409: conflict"), { status: 409 }); },
    });
    expect(failed.publishConflict).toBeTruthy();
    expect(failed.publishConflict.code).toBe("workflow_version_conflict");
    expect(failed.publishConflict.hint).toMatch(/Reload|Review/);
    expect(failed.workingDraft).toEqual({ id: "wf", name: "WF" });
    expect(failed.savedDraft).toBeNull();
    expect(calls).toEqual([]);
  });

  it("does not overwrite a dirty working draft when refreshing from the server", async () => {
    let state = createCandidateState();
    state = candidateReducer(state, { type: "set-working", spec: { id: "wf", name: "Editor Edits" } });
    state = candidateReducer(state, {
      type: "set-saved",
      draft: { id: "wf", name: "Server Version" },
    });
    expect(isDraftDirty(state)).toBe(true);

    state = await candidateReducer(state, {
      type: "refresh",
      serverDraft: { id: "wf", name: "Even Newer Server" },
    }, {
      putDraft: () => Promise.resolve({}),
      getDraft: () => Promise.resolve({ id: "wf", name: "Even Newer Server" }),
      deleteDraft: () => Promise.resolve({}),
      publish: () => Promise.resolve({}),
    });

    expect(state.workingDraft.name).toBe("Editor Edits");
  });

  it("overwrites a clean working draft when refreshing from the server", async () => {
    let state = createCandidateState();
    state = candidateReducer(state, {
      type: "set-saved",
      draft: { id: "wf", name: "Server Version" },
    });
    state = candidateReducer(state, { type: "set-working", spec: { id: "wf", name: "Server Version" } });
    expect(isDraftDirty(state)).toBe(false);

    state = await candidateReducer(state, {
      type: "refresh",
      serverDraft: { id: "wf", name: "Newer Server" },
    }, {
      putDraft: () => Promise.resolve({}),
      getDraft: () => Promise.resolve({ id: "wf", name: "Newer Server" }),
      deleteDraft: () => Promise.resolve({}),
      publish: () => Promise.resolve({}),
    });

    expect(state.workingDraft.name).toBe("Newer Server");
  });
});

describe("local undo", () => {
  it("keeps the last 20 accepted working drafts", () => {
    let state = createCandidateState();
    for (let i = 0; i < 25; i += 1) {
      state = pushUndo(state, { id: "wf", name: "v" + i });
    }
    expect(state.undoStack).toHaveLength(20);
    expect(state.undoStack[0].name).toBe("v5");
    expect(state.undoStack[19].name).toBe("v24");
  });
});

describe("diffForCandidate", () => {
  it("diffs the candidate against the working draft", () => {
    const state = {
      savedDraft: null,
      workingDraft: { id: "wf", name: "Before", version: 1, triggers: [], nodes: {}, edges: [] },
      candidateDraft: { id: "wf", name: "After", version: 1, triggers: [], nodes: {}, edges: [] },
      candidateSource: "refine",
    };
    const sections = diffForCandidate(state);
    expect(sections.map((s) => s.kind)).toEqual(["metadata"]);
  });

  it("returns no sections when there is no candidate", () => {
    expect(diffForCandidate(createCandidateState())).toEqual([]);
  });
});

describe("buildApiHelpers", () => {
  it("builds draft and publish helpers that route through the host api", () => {
    const calls = [];
    const api = (path, options) => { calls.push([path, options]); return Promise.resolve({ ok: true }); };
    const helpers = buildApiHelpers(api);

    return Promise.all([
      helpers.putDraft("wf", { spec: { id: "wf" }, base_version: 1 }),
      helpers.getDraft("wf"),
      helpers.deleteDraft("wf"),
      helpers.publish("wf", { expected_latest_version: 2 }),
    ]).then(() => {
      expect(calls[0][0]).toBe("/definitions/wf/draft");
      expect(calls[0][1]).toMatchObject({ method: "PUT" });
      expect(calls[1][0]).toBe("/definitions/wf/draft");
      expect(calls[1][1]).toMatchObject({ method: "GET" });
      expect(calls[2][0]).toBe("/definitions/wf/draft");
      expect(calls[2][1]).toMatchObject({ method: "DELETE" });
      expect(calls[3][0]).toBe("/definitions/wf/publish");
      expect(calls[3][1]).toMatchObject({ method: "POST" });
    });
  });
});

describe("acceptCandidate + recordPublishResult direct helpers", () => {
  it("promotes the candidate and pushes the previous working draft onto undo", () => {
    const state = {
      savedDraft: null,
      workingDraft: { id: "wf", name: "Old" },
      candidateDraft: { id: "wf", name: "New" },
      candidateSource: "refine",
      undoStack: [],
    };
    const next = acceptCandidate(state);
    expect(next.workingDraft.name).toBe("New");
    expect(next.candidateDraft).toBeNull();
    expect(next.undoStack[0]).toEqual({ id: "wf", name: "Old" });
  });

  it("clears the candidate on reject without touching the working draft", () => {
    const state = {
      savedDraft: { id: "wf", name: "Saved" },
      workingDraft: { id: "wf", name: "Editor" },
      candidateDraft: { id: "wf", name: "AI Proposal" },
      candidateSource: "generate",
      undoStack: [],
    };
    const next = rejectCandidate(state);
    expect(next.candidateDraft).toBeNull();
    expect(next.candidateSource).toBeNull();
    expect(next.workingDraft).toBe(state.workingDraft);
    expect(next.savedDraft).toBe(state.savedDraft);
  });

  it("applies the server draft on success and preserves on conflict", () => {
    const state = {
      savedDraft: null,
      workingDraft: { id: "wf", name: "Editor" },
      candidateDraft: null,
      candidateSource: null,
      undoStack: [],
      publishConflict: null,
    };
    const ok = recordPublishResult(state, { ok: true, savedDraft: { id: "wf", name: "Server" } });
    expect(ok.savedDraft.name).toBe("Server");
    expect(ok.publishConflict).toBeNull();

    const conflict = applyServerDraft(state, {
      ok: false,
      status: 409,
      code: "workflow_version_conflict",
      hint: "Reload the latest version and review the draft again.",
    });
    expect(conflict.publishConflict.code).toBe("workflow_version_conflict");
    expect(conflict.savedDraft).toBe(state.savedDraft);
    expect(conflict.workingDraft).toBe(state.workingDraft);
  });
});