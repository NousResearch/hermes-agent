// Node assertions for the Team chart layout (compiled from src/lib/orgchart.ts).
import assert from "node:assert/strict";
import { layoutTeam, channelKey } from "./.build/orgchart.js";

const col = (layout, id) => layout.nodes.find((n) => n.id === id)?.column;
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log("ok -", name); };

test("channels first, then the agents they route to, then who those hand to", () => {
  const l = layoutTeam(
    [{ id: "support", may_assign_to: ["ops"] }, { id: "ops", may_assign_to: ["finance"] }, { id: "finance" }],
    [{ id: "tg", routes: [{ agent: "support" }] }],
  );
  assert.equal(col(l, channelKey("tg")), 0);
  assert.deepEqual(["support", "ops", "finance"].map((id) => col(l, id)), [1, 2, 3]);
  assert.equal(l.edges.filter((e) => e.kind === "route").length, 1);
});

test("a two-way hand-off is marked mutual, and the return leg is drawn as an arc", () => {
  const l = layoutTeam(
    [{ id: "support", may_assign_to: ["ops"] }, { id: "ops", may_assign_to: ["support"] }],
    [{ id: "tg", routes: [{ agent: "support" }] }],
  );
  const fwd = l.edges.find((e) => e.from === "support" && e.to === "ops");
  const ret = l.edges.find((e) => e.from === "ops" && e.to === "support");
  assert.ok(fwd.mutual && ret.mutual);
  assert.equal(fwd.back, false);
  assert.equal(ret.back, true);
});

test("without channels, agents nobody hands work to start the chart", () => {
  const l = layoutTeam([{ id: "a", may_assign_to: ["b"] }, { id: "b" }], []);
  assert.deepEqual([col(l, "a"), col(l, "b")], [0, 1]);
});

test("an agent reachable only through a cycle still gets a place", () => {
  const l = layoutTeam(
    [{ id: "x", may_assign_to: ["y"] }, { id: "y", may_assign_to: ["x"] }, { id: "solo" }], [],
  );
  assert.equal(l.nodes.length, 3);
  assert.ok(l.nodes.every((n) => Number.isFinite(n.x) && Number.isFinite(n.y)));
});

test("a hand-off to an agent that does not exist is reported, not drawn", () => {
  const l = layoutTeam([{ id: "a", may_assign_to: ["ghost", "a"] }], []);
  assert.deepEqual(l.dangling, [{ from: "a", to: "ghost" }]);
  assert.equal(l.edges.length, 0, "and a self-hand-off is not an edge");
});

test("a channel with no routes falls back to its allowed agents", () => {
  const l = layoutTeam([{ id: "a" }], [{ id: "slack", allowed_agents: ["a"] }]);
  assert.equal(l.edges.length, 1);
});

test("cards never overlap", () => {
  const agents = Array.from({ length: 7 }, (_, i) => ({ id: `a${i}`, may_assign_to: i ? [] : ["a1", "a2", "a3"] }));
  const l = layoutTeam(agents, [{ id: "c", routes: [{ agent: "a0" }] }]);
  const seen = new Set(l.nodes.map((n) => `${n.x},${n.y}`));
  assert.equal(seen.size, l.nodes.length);
  assert.ok(l.width > 0 && l.height > 0);
});

console.log(`${passed} passed`);
