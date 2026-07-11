import { describe, expect, it } from "vitest";
import {
  serializeFilters,
  historyListPath,
  detailUrl,
  canCancel,
  buildRerunBody,
  buildDetailGraphParams,
} from "./history.js";

describe("serializeFilters", () => {
  it("includes workflow_id when provided", () => {
    expect(serializeFilters({ workflowId: "demo" })).toContain("workflow_id=demo");
  });

  it("serializes status filter", () => {
    expect(serializeFilters({ status: "succeeded" })).toContain("status=succeeded");
  });

  it("serializes version filter", () => {
    expect(serializeFilters({ version: 2 })).toContain("version=2");
  });

  it("serializes trigger_id filter", () => {
    expect(serializeFilters({ triggerId: "manual" })).toContain("trigger_id=manual");
  });

  it("serializes before cursor as two params", () => {
    const qs = serializeFilters({ before: { createdAt: 100, executionId: "wfexec_a" } });
    expect(qs).toContain("before_created_at=100");
    expect(qs).toContain("before_execution_id=wfexec_a");
  });

  it("serializes limit", () => {
    expect(serializeFilters({ limit: 50 })).toContain("limit=50");
  });

  it("omits empty/null/undefined values", () => {
    const qs = serializeFilters({});
    expect(qs).toBe("");
  });

  it("combines multiple filters", () => {
    const qs = serializeFilters({ workflowId: "demo", status: "queued", limit: 10 });
    expect(qs).toContain("workflow_id=demo");
    expect(qs).toContain("status=queued");
    expect(qs).toContain("limit=10");
  });
});

describe("historyListPath", () => {
  it("builds /executions with filters", () => {
    expect(historyListPath({ workflowId: "demo" })).toBe("/executions?workflow_id=demo");
  });

  it("builds bare /executions for all mode", () => {
    expect(historyListPath({})).toBe("/executions");
  });
});

describe("detailUrl", () => {
  it("builds detail URL from execution id", () => {
    expect(detailUrl("wfexec_abc")).toBe("/executions/wfexec_abc/detail");
  });

  it("returns empty string for falsy id", () => {
    expect(detailUrl("")).toBe("");
    expect(detailUrl(null)).toBe("");
  });
});

describe("canCancel", () => {
  it("returns true for queued", () => {
    expect(canCancel("queued")).toBe(true);
  });

  it("returns true for running", () => {
    expect(canCancel("running")).toBe(true);
  });

  it("returns true for waiting", () => {
    expect(canCancel("waiting")).toBe(true);
  });

  it("returns false for succeeded", () => {
    expect(canCancel("succeeded")).toBe(false);
  });

  it("returns false for cancelled", () => {
    expect(canCancel("cancelled")).toBe(false);
  });

  it("returns false for failed", () => {
    expect(canCancel("failed")).toBe(false);
  });
});

describe("buildRerunBody", () => {
  it("wraps input in { input }", () => {
    expect(buildRerunBody({ x: 1 })).toEqual({ input: { x: 1 } });
  });

  it("returns empty input for falsy", () => {
    expect(buildRerunBody(null)).toEqual({ input: {} });
    expect(buildRerunBody(undefined)).toEqual({ input: {} });
  });
});

describe("buildDetailGraphParams", () => {
  it("includes version when present", () => {
    expect(buildDetailGraphParams({ version: 3 })).toEqual({ version: 3 });
  });

  it("returns empty object when no version", () => {
    expect(buildDetailGraphParams({})).toEqual({});
    expect(buildDetailGraphParams(null)).toEqual({});
  });
});
