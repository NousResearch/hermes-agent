import { describe, expect, it } from "vitest";
import { decorateGraphItems, graphItems } from "./graph.js";

const spec = {
  triggers: [{ id: "manual", type: "manual" }],
  nodes: { pass: { type: "pass" } },
};

describe("graphItems", () => {
  it("keeps renderer identity separate from trigger subtype", () => {
    expect(graphItems(spec)[0]).toMatchObject({
      id: "manual",
      rendererType: "trigger",
      specKind: "trigger",
      triggerType: "manual",
    });
  });

  it("preserves membership when statuses change", () => {
    const before = graphItems(spec);
    const after = decorateGraphItems(before, { pass: "succeeded" });
    expect(after.map((item) => item.id)).toEqual(["manual", "pass"]);
    expect(after.find((item) => item.id === "pass").status).toBe("succeeded");
  });
});
