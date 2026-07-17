const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadLayoutApi() {
  const bundlePath = path.join(__dirname, "..", "dashboard", "dist", "index.js");
  const bundle = fs.readFileSync(bundlePath, "utf8");
  const start = bundle.indexOf("  const GRAPH_NODE_W");
  const end = bundle.indexOf("\n  function GraphTaskNode", start);
  assert.ok(start >= 0 && end > start, "layout source region must be discoverable");
  const source = bundle.slice(start, end);
  return vm.runInNewContext(
    `(function () { ${source}; return { buildTaskGraphLayout }; })()`,
    { Map, Set, Math, Object, Array },
  );
}

function task(id, priority = 0) {
  return { id, title: id, status: "todo", priority };
}

function board(tasks, links) {
  return {
    columns: [{ name: "todo", tasks }],
    links,
  };
}

function center(node, axis) {
  return axis === "y"
    ? node.y + node.height / 2
    : node.x + node.width / 2;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function assertNoOverlaps(nodes) {
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      const overlaps = a.x < b.x + b.width && a.x + a.width > b.x
        && a.y < b.y + b.height && a.y + a.height > b.y;
      assert.equal(overlaps, false, `${a.id} overlaps ${b.id}`);
    }
  }
}

function edgeEndpoints(edge) {
  const values = edge.d.match(/-?\d+(?:\.\d+)?/g).map(Number);
  return { start: values.slice(0, 2), end: values.slice(-2) };
}

const api = loadLayoutApi();

const PRESETS = ["balanced-horizontal", "balanced-vertical", "compact"];
const LAYOUT_FIXTURES = [
  {
    name: "fork",
    ids: ["root", "branch-a", "branch-b", "branch-c"],
    links: [
      { parent_id: "root", child_id: "branch-a" },
      { parent_id: "root", child_id: "branch-b" },
      { parent_id: "root", child_id: "branch-c" },
    ],
  },
  {
    name: "join",
    ids: ["source-a", "source-b", "source-c", "merge"],
    links: [
      { parent_id: "source-a", child_id: "merge" },
      { parent_id: "source-b", child_id: "merge" },
      { parent_id: "source-c", child_id: "merge" },
    ],
  },
  {
    name: "chain",
    ids: ["step-a", "step-b", "step-c", "step-d"],
    links: [
      { parent_id: "step-a", child_id: "step-b" },
      { parent_id: "step-b", child_id: "step-c" },
      { parent_id: "step-c", child_id: "step-d" },
    ],
  },
  {
    name: "disconnected",
    ids: ["left-a", "left-b", "right-a", "right-b"],
    links: [
      { parent_id: "left-a", child_id: "left-b" },
      { parent_id: "right-a", child_id: "right-b" },
    ],
  },
  {
    name: "cyclic",
    ids: ["cycle-a", "cycle-b", "cycle-c"],
    links: [
      { parent_id: "cycle-a", child_id: "cycle-b" },
      { parent_id: "cycle-b", child_id: "cycle-c" },
      { parent_id: "cycle-c", child_id: "cycle-a" },
    ],
  },
  {
    name: "unlinked",
    ids: ["island-a", "island-b", "island-c"],
    links: [],
  },
];

for (const preset of PRESETS) {
  for (const fixture of LAYOUT_FIXTURES) {
    test(`${preset} satisfies the full ${fixture.name} fixture contract`, () => {
      const tasks = fixture.ids.map((id) => Object.assign(task(id), { title: `Task ${id}` }));
      const data = board(tasks, fixture.links);
      const first = api.buildTaskGraphLayout(data, data, preset);
      const second = api.buildTaskGraphLayout(data, data, preset);

      assert.deepEqual(first, second, "layout must be deterministic");
      assert.deepEqual(
        Array.from(first.nodes, (node) => node.id).sort(),
        fixture.ids.slice().sort(),
        "layout must include every expected node exactly once",
      );
      for (const node of first.nodes) {
        const expectedTask = tasks.find((candidate) => candidate.id === node.id);
        assert.equal(node.task.title, expectedTask.title, `${node.id} title must be preserved`);
        assert.ok(node.task.title.trim().length > 0, `${node.id} title must remain non-empty`);
        assert.ok(node.x >= 0 && node.x + node.width <= first.width, `${node.id} exceeds x bounds`);
        assert.ok(node.y >= 0 && node.y + node.height <= first.height, `${node.id} exceeds y bounds`);
      }
      for (const island of first.islands) {
        assert.ok(island.x >= 0 && island.x + island.width <= first.width, `${island.id} exceeds x bounds`);
        assert.ok(island.y >= 0 && island.y + island.height <= first.height, `${island.id} exceeds y bounds`);
      }
      assertNoOverlaps(first.nodes);
    });
  }
}

test("balanced horizontal centers a three-to-one fan-in", () => {
  const data = board(
    [task("research-a"), task("research-b"), task("research-c"), task("synthesis")],
    [
      { parent_id: "research-a", child_id: "synthesis" },
      { parent_id: "research-b", child_id: "synthesis" },
      { parent_id: "research-c", child_id: "synthesis" },
    ],
  );
  const layout = api.buildTaskGraphLayout(data, data, "balanced-horizontal");
  const nodes = new Map(layout.nodes.map((node) => [node.id, node]));
  const sourceMean = ["research-a", "research-b", "research-c"]
    .map((id) => center(nodes.get(id), "y"))
    .reduce((sum, value) => sum + value, 0) / 3;
  assert.ok(Math.abs(center(nodes.get("synthesis"), "y") - sourceMean) <= 8);
});

test("balanced vertical centers a one-to-three fan-out", () => {
  const data = board(
    [task("brief"), task("a"), task("b"), task("c")],
    [
      { parent_id: "brief", child_id: "a" },
      { parent_id: "brief", child_id: "b" },
      { parent_id: "brief", child_id: "c" },
    ],
  );
  const layout = api.buildTaskGraphLayout(data, data, "balanced-vertical");
  const nodes = new Map(layout.nodes.map((node) => [node.id, node]));
  const childMean = ["a", "b", "c"]
    .map((id) => center(nodes.get(id), "x"))
    .reduce((sum, value) => sum + value, 0) / 3;
  assert.ok(Math.abs(center(nodes.get("brief"), "x") - childMean) <= 8);
  for (const edge of layout.edges) {
    const source = nodes.get(edge.source.id);
    const target = nodes.get(edge.target.id);
    assert.deepEqual(edgeEndpoints(edge), {
      start: [source.x + source.width / 2, source.y + source.height],
      end: [target.x + target.width / 2, target.y],
    });
  }
});

for (const [preset, axis, crossSize, laneGap, rankAxis, rankSize, rankGap] of [
  ["balanced-horizontal", "y", 116, 34, "x", 260, 88],
  ["balanced-vertical", "x", 260, 34, "y", 116, 72],
  ["compact", "y", 116, 18, "x", 260, 52],
]) {
  test(`${preset} reconciles simultaneous fan-in and fan-out`, () => {
    const data = board(
      [task("a"), task("b"), task("c"), task("d")],
      [
        { parent_id: "a", child_id: "c" },
        { parent_id: "a", child_id: "d" },
        { parent_id: "b", child_id: "c" },
      ],
    );
    const layout = api.buildTaskGraphLayout(data, data, preset);
    const nodes = new Map(layout.nodes.map((node) => [node.id, node]));
    const fanOutDeviation = Math.abs(
      center(nodes.get("a"), axis)
        - mean([center(nodes.get("c"), axis), center(nodes.get("d"), axis)]),
    );
    const fanInDeviation = Math.abs(
      center(nodes.get("c"), axis)
        - mean([center(nodes.get("a"), axis), center(nodes.get("b"), axis)]),
    );
    assert.ok(fanOutDeviation <= 8, `fan-out deviation ${fanOutDeviation}px exceeds 8px`);
    assert.ok(fanInDeviation <= 8, `fan-in deviation ${fanInDeviation}px exceeds 8px`);
    assert.equal(
      center(nodes.get("c"), rankAxis) - center(nodes.get("a"), rankAxis),
      rankSize + rankGap,
      "rank spacing must match the selected preset",
    );
    assert.ok(
      Math.abs(center(nodes.get("a"), axis) - center(nodes.get("b"), axis))
        >= crossSize + laneGap,
      "source lane must preserve the configured collision gap",
    );
    assert.ok(
      Math.abs(center(nodes.get("c"), axis) - center(nodes.get("d"), axis))
        >= crossSize + laneGap,
      "target lane must preserve the configured collision gap",
    );
  });

  test(`${preset} preserves spacing for an infeasible complete fan`, () => {
    const sourceIds = ["source-a", "source-b", "source-c"];
    const targetIds = ["target-a", "target-b"];
    const data = board(
      [...sourceIds, ...targetIds].map((id) => task(id)),
      sourceIds.flatMap((parent_id) => targetIds.map((child_id) => ({ parent_id, child_id }))),
    );
    const first = api.buildTaskGraphLayout(data, data, preset);
    const second = api.buildTaskGraphLayout(data, data, preset);
    assert.deepEqual(first, second, "complete-fan layout must be deterministic");
    const nodes = new Map(first.nodes.map((node) => [node.id, node]));
    for (const rankIds of [sourceIds, targetIds]) {
      const centers = rankIds.map((id) => center(nodes.get(id), axis)).sort((a, b) => a - b);
      for (let index = 1; index < centers.length; index += 1) {
        assert.ok(
          centers[index] - centers[index - 1] >= crossSize + laneGap,
          `adjacent ${preset} centers must preserve configured lane spacing`,
        );
      }
    }
    assertNoOverlaps(first.nodes);
  });
}

test("layout rejects null link entries without crashing", () => {
  const data = board([task("a")], [null]);
  assert.doesNotThrow(() => api.buildTaskGraphLayout(data, data, "compact"));
  assert.equal(api.buildTaskGraphLayout(data, data, "compact").edges.length, 0);
});

test("layout treats a non-array links value as empty", () => {
  const data = board([task("a"), task("b")], { parent_id: "a", child_id: "b" });
  assert.doesNotThrow(() => api.buildTaskGraphLayout(data, data, "compact"));
  assert.equal(api.buildTaskGraphLayout(data, data, "compact").edges.length, 0);
});

test("layout rejects links with missing endpoints without crashing", () => {
  const data = board(
    [task("a"), task("b")],
    [{ parent_id: "a" }, { child_id: "b" }, {}, "a->b"],
  );
  assert.doesNotThrow(() => api.buildTaskGraphLayout(data, data, "compact"));
  assert.equal(api.buildTaskGraphLayout(data, data, "compact").edges.length, 0);
});

test("layout tolerates a dependency cycle", () => {
  const data = board(
    [task("a"), task("b")],
    [
      { parent_id: "a", child_id: "b" },
      { parent_id: "b", child_id: "a" },
    ],
  );
  assert.doesNotThrow(() => api.buildTaskGraphLayout(data, data, "balanced-vertical"));
  assert.equal(api.buildTaskGraphLayout(data, data, "balanced-vertical").edges.length, 2);
});

test("full layout contract is deterministic, bounded, anchored, and read-only", () => {
  const links = [
    Object.freeze({ parent_id: "root", child_id: "a" }),
    Object.freeze({ parent_id: "root", child_id: "b" }),
    Object.freeze({ parent_id: "a", child_id: "merge" }),
    Object.freeze({ parent_id: "b", child_id: "merge" }),
  ];
  const data = board(
    [task("root"), task("b", 2), task("a", 2), task("merge"), task("island")],
    Object.freeze(links),
  );
  const linksBefore = JSON.stringify(data.links);
  const first = api.buildTaskGraphLayout(data, data, "compact");
  const second = api.buildTaskGraphLayout(data, data, "compact");
  assert.deepEqual(first, second);
  assert.equal(JSON.stringify(data.links), linksBefore);
  assert.deepEqual(first.edges.map((edge) => edge.id), [
    "root->a", "root->b", "a->merge", "b->merge",
  ]);
  assert.equal(first.componentCount, 2);
  assert.equal(first.nodes.length, 5);
  assert.equal(first.islands.length, 2);
  assert.ok(first.width >= 720);
  assert.ok(first.height >= 480);
  assertNoOverlaps(first.nodes);

  for (const node of first.nodes) {
    assert.equal(node.width, 260);
    assert.equal(node.height, 116);
    assert.ok(Number.isInteger(node.componentIndex));
    assert.ok(node.x >= 0 && node.x + node.width <= first.width);
    assert.ok(node.y >= 0 && node.y + node.height <= first.height);
  }
  for (const island of first.islands) {
    assert.equal(typeof island.isUnlinked, "boolean");
    assert.ok(Array.isArray(island.taskIds));
    assert.equal(island.seedTaskId, island.taskIds[0]);
    assert.ok(island.x >= 0 && island.x + island.width <= first.width);
    assert.ok(island.y >= 0 && island.y + island.height <= first.height);
  }

  const nodes = new Map(first.nodes.map((node) => [node.id, node]));
  assert.equal(nodes.get("a").x - nodes.get("root").x, 260 + 52);
  assert.equal(nodes.get("merge").x - nodes.get("a").x, 260 + 52);
  assert.ok(Math.abs(center(nodes.get("a"), "y") - center(nodes.get("b"), "y")) >= 116 + 18);
  for (const edge of first.edges) {
    const source = nodes.get(edge.source.id);
    const target = nodes.get(edge.target.id);
    assert.deepEqual(edgeEndpoints(edge), {
      start: [source.x + source.width, source.y + source.height / 2],
      end: [target.x, target.y + target.height / 2],
    });
  }
});
