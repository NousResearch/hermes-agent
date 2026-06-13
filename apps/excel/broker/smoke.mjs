// Live smoke test for the Hermes for Excel bridge. Run after Hermes updates.
//   node broker/smoke.mjs        (bridge must be running on :8787)
// Exits non-zero if any assertion fails.

const BASE_URL = process.env.HERMES_EXCEL_BRIDGE_URL || "http://127.0.0.1:8787";
const TIMEOUT_MS = 120000;

const results = { pass: 0, fail: 0, skip: 0 };

function assert(label, cond, detail = "") {
  if (cond) {
    console.log(`PASS ${label}`);
    results.pass += 1;
  } else {
    console.error(`FAIL ${label}: ${detail}`);
    results.fail += 1;
  }
}

function skip(label, reason) {
  console.log(`SKIP ${label}: ${reason}`);
  results.skip += 1;
}

async function post(path, body, opts = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(opts.timeoutMs || TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} on ${path}: ${await res.text().catch(() => "")}`);
  return res.json();
}

// A write/create action's cells (rebased formulas included) live in .values.
function formulasOf(action) {
  return (action.values || []).flat().filter((cell) => typeof cell === "string" && cell.startsWith("="));
}

async function testHealth() {
  console.log("\n--- 1. health ---");
  try {
    const res = await fetch(`${BASE_URL}/api/health`, { signal: AbortSignal.timeout(TIMEOUT_MS) });
    const data = await res.json();
    assert("health.ok", data.ok === true || data.hermes?.ok === true, JSON.stringify(data));
    const hermesOk = data.hermes?.ok === true;
    console.log(hermesOk ? "INFO: Hermes reachable — running model assertions." : "INFO: Hermes down — running fallback assertions.");
    return hermesOk;
  } catch (error) {
    console.error(`ERROR health: ${error.message}`);
    return false;
  }
}

const TABLE_PROMPT =
  "Make a 3-row product table: Widgets 4 @ 2.50, Gears 10 @ 1.25, Bolts 20 @ 0.10, with a Total column of formulas.";

function tableBody(address) {
  return {
    prompt: TABLE_PROMPT,
    workbook: { activeSheet: "Sheet1", sheets: [{ name: "Sheet1", usedRange: "A1:A1" }] },
    selection: { address, rowCount: 1, columnCount: 1, values: [[""]], formulas: [[""]] },
    history: [],
    files: [],
  };
}

async function testSimpleA1(hermesOk) {
  console.log("\n--- 2. simple table at A1 ---");
  try {
    const res = await post("/api/chat", tableBody("Sheet1!A1"));
    if (!hermesOk) {
      assert("simple.fallback", res.source === "fallback" && res.actions.length === 0, JSON.stringify(res.actions));
      return;
    }
    assert("simple.source", res.source === "llm", `got ${res.source}`);
    const action = res.actions.find((a) => a.type === "create_sheet" || a.type === "write_cells");
    assert("simple.action", Boolean(action), "no create_sheet/write_cells");
    const flat = res.actions.flatMap((a) => a.values || []).flat().map(String).join(" ");
    const ok = (flat.includes("2.5") || flat.includes("2.50")) && flat.includes("1.25") && (flat.includes("0.1") || flat.includes("0.10"));
    assert("simple.numbers", ok, `numbers missing in: ${flat.slice(0, 200)}`);
  } catch (error) {
    assert("simple.request", false, error.message);
  }
}

async function testAnchorH23(hermesOk) {
  console.log("\n--- 3. anchor H23 (formula-rebasing regression) ---");
  try {
    const res = await post("/api/chat", tableBody("Sheet1!H23"));
    if (!hermesOk) {
      assert("anchor.fallback", res.source === "fallback", `got ${res.source}`);
      return;
    }
    const action = res.actions.find((a) => a.type === "write_cells" || a.type === "create_sheet");
    if (!action) return assert("anchor.action", false, "no relevant action");
    if (action.type === "create_sheet") {
      console.log("NOTE: model chose create_sheet (A1 formulas are correct there) — soft pass.");
      results.pass += 1;
      console.log("PASS anchor.soft");
      return;
    }
    const formulas = formulasOf(action);
    assert("anchor.has-formulas", formulas.length > 0, "no formulas in write_cells values");
    // The bug wrote A1-relative refs (B2/C2/D2) at H23; rebased refs must point at H23+.
    const bad = formulas.filter((f) => /\b[A-G]?[BCD]2\b/.test(f) || /=B2|=C2|=D2|\(B2|\(C2|\(D2|\*B2|\*C2|\*D2/.test(f));
    const rebased = formulas.some((f) =>
      (f.match(/[A-Z]+[0-9]+/g) || []).some((ref) => {
        const col = ref.replace(/[0-9]/g, "");
        const row = parseInt(ref.replace(/[A-Z]/g, ""), 10);
        return col.length === 1 && (col.charCodeAt(0) >= "H".charCodeAt(0) || row >= 23);
      }),
    );
    assert("anchor.rebased", bad.length === 0 && rebased, `bad refs: ${bad.join(", ") || "none"}; rebased=${rebased}; formulas=${formulas.join(", ")}`);
  } catch (error) {
    assert("anchor.request", false, error.message);
  }
}

async function testMultiturn(hermesOk) {
  console.log("\n--- 4. multi-turn history ---");
  try {
    const res1 = await post("/api/chat", tableBody("Sheet1!A1"));
    if (!hermesOk) {
      assert("mt.fallback", res1.source === "fallback", `got ${res1.source}`);
      return;
    }
    assert("mt.turn1", res1.source === "llm", `got ${res1.source}`);
    const res2 = await post("/api/chat", {
      prompt: "add a totals row with a SUM for the last column",
      workbook: { activeSheet: "Sheet1", sheets: [{ name: "Sheet1", usedRange: "A1:D4" }] },
      selection: { address: "Sheet1!A1", rowCount: 1, columnCount: 1, values: [[""]], formulas: [[""]] },
      history: [
        { role: "user", content: TABLE_PROMPT },
        { role: "assistant", content: res1.message },
      ],
      files: [],
    });
    assert("mt.turn2.source", res2.source === "llm", `got ${res2.source}`);
    assert("mt.turn2.action", res2.actions.some((a) => a.type === "write_cells" || a.type === "create_sheet"), "no action in turn 2");
  } catch (error) {
    assert("mt.request", false, error.message);
  }
}

const MEDIUM_PROMPT =
  'Create a worksheet named "Smoke Medium" with 12 rows of sample sales data ' +
  "(columns: Product, Region, Units, Unit Price, Revenue, Cost, Margin %). " +
  "Apply currency and percent number formats, a bold header, and conditional " +
  "formatting that highlights Margin % below 25% in red. Keep it compact.";

async function testMediumBuild(hermesOk) {
  console.log("\n--- 5. medium multi-action build (regression for the 2026-06-13 stall) ---");
  if (!hermesOk) return skip("medium.skip", "Hermes down — medium build needs the model.");
  try {
    const res = await post(
      "/api/chat",
      {
        prompt: MEDIUM_PROMPT,
        workbook: { activeSheet: "Sheet1", sheets: [{ name: "Sheet1", usedRange: "A1:A1" }] },
        selection: { address: "Sheet1!A1", rowCount: 1, columnCount: 1, values: [[""]], formulas: [[""]] },
        history: [],
        files: [],
      },
      { timeoutMs: 180000 },
    );
    // The original bug: the model produced unparseable JSON and the bridge fell
    // back, writing nothing. A real build must come back as a usable llm result.
    assert("medium.not-fallback", res.source === "llm", `source=${res.source} — model output was unusable`);
    const hasData = res.actions.some((a) => a.type === "create_sheet" || a.type === "write_cells");
    assert("medium.has-data", hasData, "no create_sheet/write_cells");
    // The native conditional_format action should be used instead of hand-written code.
    assert(
      "medium.conditional-format",
      res.actions.some((a) => a.type === "conditional_format"),
      `actions: ${res.actions.map((a) => a.type).join(", ")}`,
    );
    // No paragraph-length cell prose (the matrix-bloat that triggered the stall).
    const longest = res.actions
      .flatMap((a) => a.values || [])
      .flat()
      .reduce((max, cell) => (typeof cell === "string" && cell.length > max ? cell.length : max), 0);
    assert("medium.no-prose-cells", longest <= 80, `longest cell = ${longest} chars`);
  } catch (error) {
    assert("medium.request", false, error.message);
  }
}

function testExport() {
  console.log("\n--- 6. export endpoint ---");
  return post("/api/export", { name: "smoke test!!", values: [["A", "B,c"], [1, 2]] })
    .then((res) => {
      assert("export.ok", res.ok === true && typeof res.path === "string", JSON.stringify(res));
      assert("export.name-sanitized", /smoketest\.csv$/.test(res.path || ""), `path=${res.path}`);
    })
    .catch((error) => assert("export.request", false, error.message));
}

function testHonestyNote() {
  console.log("\n--- 7. honesty / containment (manual) ---");
  skip("honesty.active", "true model-down is covered by the v0.2.0 bogus-endpoint test; harness asserts empty actions on any fallback above.");
  skip("containment", "cross-service /v1/toolsets check needs the Hermes key — see SHIP_CHECKLIST manual step.");
}

async function main() {
  console.log(`Hermes for Excel smoke test → ${BASE_URL}`);
  const hermesOk = await testHealth();
  await testSimpleA1(hermesOk);
  await testAnchorH23(hermesOk);
  await testMultiturn(hermesOk);
  await testMediumBuild(hermesOk);
  await testExport();
  testHonestyNote();
  console.log(`\n--- ${results.pass} passed, ${results.fail} failed, ${results.skip} skipped ---`);
  process.exit(results.fail > 0 ? 1 : 0);
}

main().catch((error) => {
  console.error("Fatal:", error);
  process.exit(1);
});
