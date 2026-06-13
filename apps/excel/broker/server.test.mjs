import { test } from "node:test";
import { strict as assert } from "node:assert";
import {
  parseDelimitedText,
  markdownTableFromRows,
  extractJsonObject,
  normalizeMatrix,
  normalizeAction,
  normalizeActions,
  legacyWriteToActions,
  parseMoney,
  extractedField,
  followingValue,
  moneyValueForLabel,
  statementSummaryRows,
  statementTransactionRows,
  truncateText,
  windowsPathToWsl,
  extensionOf,
  compactSelection,
  normalizeWorkbook,
  buildChatMessages,
  buildSystemPrompt,
  fallbackResponse,
  scanWrittenCells,
  matrixToCsv,
  safeExportName,
  expectsWorkbookActions,
  claimsWorkbookChange,
  promptWantsWorkbookOutput,
} from "./server.mjs";
import {
  columnLettersToNumber,
  numberToColumnLetters,
  anchorFromAddress,
  translateFormula,
  translateMatrixFormulas,
} from "./formula-rebase.mjs";

test("parseDelimitedText: quoted cells, escaped quotes, CRLF, blank rows, TSV", () => {
  const csv = 'a "b", c\r\n"d ""e"", f"\r\n';
  assert.deepStrictEqual(parseDelimitedText(csv, ","), [
    ["a b", " c"],
    ['d "e", f'],
  ]);

  assert.deepStrictEqual(parseDelimitedText("a,b\r\n\r\n\r\n", ","), [["a", "b"]]);

  assert.deepStrictEqual(parseDelimitedText("x\ty\n1\t2", "\t"), [
    ["x", "y"],
    ["1", "2"],
  ]);
});

test("markdownTableFromRows: header, separator, pipe escaping, omitted note", () => {
  const md = markdownTableFromRows([["A", "B"], ["1", "2"], ["3", "4"]], 2);
  assert.ok(md.includes("| A | B |"));
  assert.ok(md.includes("| --- | --- |"));
  assert.ok(md.includes("| 1 | 2 |"));
  assert.ok(!md.includes("| 3 | 4 |"));
  assert.ok(md.includes("[1 more row(s) omitted]"));

  const pipeMd = markdownTableFromRows([["A|B"], ["C"]], 10);
  assert.ok(pipeMd.includes("A\\|B"));
});

test("extractJsonObject: plain, fenced, embedded in prose, garbage", () => {
  assert.deepStrictEqual(extractJsonObject('{"a":1}'), { a: 1 });
  assert.deepStrictEqual(extractJsonObject('```json\n{"b":2}\n```'), { b: 2 });
  assert.deepStrictEqual(extractJsonObject('Here is the result {"c":3} as requested'), { c: 3 });
  assert.strictEqual(extractJsonObject("no json here"), null);
});

test("extractJsonObject: repairs single stray or missing brackets from local models", () => {
  // The exact malformed shape observed from hermes-agent on 2026-06-11:
  // a stray } after the values matrix's last row.
  const observed =
    '{"message":"Creating Verify sheet.","actions":[{"type":"create_sheet","name":"Verify",' +
    '"values":[["Item","Qty"],["Apples",4]}],"auto_fit":true},{"type":"format_cells","range":"Verify!A1:B1","style":["header"]}]}';
  const repaired = extractJsonObject(observed);
  assert.ok(repaired);
  assert.strictEqual(repaired.actions.length, 2);
  assert.strictEqual(repaired.actions[0].type, "create_sheet");
  assert.deepStrictEqual(repaired.actions[0].values, [
    ["Item", "Qty"],
    ["Apples", 4],
  ]);
  assert.strictEqual(repaired.actions[1].type, "format_cells");

  // Truncated output: closers appended.
  const truncated = '{"message":"hi","actions":[{"type":"create_sheet","values":[["A","B"]';
  const closed = extractJsonObject(truncated);
  assert.ok(closed);
  assert.strictEqual(closed.message, "hi");

  // Trailing prose after the root object is ignored.
  assert.deepStrictEqual(extractJsonObject('{"d":4} Hope that helps!'), { d: 4 });

  // Trailing commas removed.
  assert.deepStrictEqual(extractJsonObject('{"e":[1,2,],}'), { e: [1, 2] });

  // Brackets inside strings are not treated as structure.
  assert.deepStrictEqual(extractJsonObject('{"f":"a } weird ] string"}'), { f: "a } weird ] string" });
});

test("normalizeMatrix: pads ragged rows, coerces objects, passes scalars, caps size", () => {
  assert.deepStrictEqual(normalizeMatrix([[1], [2, 3]]), [
    [1, ""],
    [2, 3],
  ]);

  assert.strictEqual(normalizeMatrix([[{ x: 1 }]])[0][0], "[object Object]");

  const pass = [[null, "s", 1, true]];
  assert.deepStrictEqual(normalizeMatrix(pass), pass);

  assert.strictEqual(normalizeMatrix(null), null);
  assert.strictEqual(normalizeMatrix("string"), null);

  const big = Array.from({ length: 201 }, (_, r) => Array.from({ length: 31 }, (_, c) => `${r}-${c}`));
  const capped = normalizeMatrix(big);
  assert.strictEqual(capped.length, 200);
  assert.strictEqual(capped[0].length, 30);
});

test("normalizeAction: all action types plus unknown", () => {
  assert.strictEqual(normalizeAction({ type: "write_cells" }), null);
  const wc = normalizeAction({ type: "write_cells", values: [[1]] });
  // start_cell is empty when omitted; normalizeActions resolves it later.
  assert.strictEqual(wc.start_cell, "");
  assert.strictEqual(wc.allow_overwrite, true);
  assert.strictEqual(normalizeAction({ type: "write_cells", start_cell: "Sheet1!C5", values: [[1]] }).start_cell, "Sheet1!C5");

  const cs = normalizeAction({ type: "create_sheet", values: [[1]] });
  assert.strictEqual(cs.name, "Hermes Output");
  const csLong = normalizeAction({ type: "create_sheet", name: "A".repeat(50), values: [[1]] });
  assert.strictEqual(csLong.name.length, 31);

  const fc = normalizeAction({ type: "format_cells", style: ["header"] });
  assert.deepStrictEqual(fc.style, ["header"]);

  assert.strictEqual(normalizeAction({ type: "execute_office_js" }), null);
  assert.ok(normalizeAction({ type: "execute_office_js", code: "context.workbook.load();" }));

  const rr = normalizeAction({ type: "read_range", range: "Sheet2!A1:B10", reason: "check totals" });
  assert.strictEqual(rr.type, "read_range");
  assert.strictEqual(rr.range, "Sheet2!A1:B10");
  assert.strictEqual(rr.reason, "check totals");
  const rrLong = normalizeAction({ type: "read_range", range: "X".repeat(120), reason: "Y".repeat(300) });
  assert.strictEqual(rrLong.range.length, 80);
  assert.strictEqual(rrLong.reason.length, 200);
  assert.strictEqual(normalizeAction({ type: "read_range" }), null);

  assert.strictEqual(normalizeAction({ type: "unknown" }), null);
});

test("normalizeAction: conditional_format normalizes operator synonyms, defaults colors, requires range", () => {
  assert.strictEqual(normalizeAction({ type: "conditional_format" }), null);

  // The Margin % < 25% case: percent cells are decimals, so value is 0.25.
  const margin = normalizeAction({ type: "conditional_format", range: "Hermes Medium Test!G2:G31", operator: "lessThan", value: 0.25 });
  assert.strictEqual(margin.type, "conditional_format");
  assert.strictEqual(margin.range, "Hermes Medium Test!G2:G31");
  assert.strictEqual(margin.operator, "lessThan");
  assert.strictEqual(margin.value, 0.25);
  assert.strictEqual(margin.fill_color, "#FFC7CE");
  assert.strictEqual(margin.font_color, "#9C0006");
  assert.strictEqual(margin.value2, undefined);

  // Operator synonyms a local model is likely to emit all map onto the enum names.
  assert.strictEqual(normalizeAction({ type: "conditional_format", range: "A1", operator: "below", value: 0 }).operator, "lessThan");
  assert.strictEqual(normalizeAction({ type: "conditional_format", range: "A1", operator: ">=", value: 0 }).operator, "greaterThanOrEqual");
  assert.strictEqual(normalizeAction({ type: "conditional_format", range: "A1", operator: "greater than", value: 0 }).operator, "greaterThan");
  assert.strictEqual(normalizeAction({ type: "conditional_format", range: "A1", operator: "garbage", value: 0 }).operator, "lessThan");

  // `threshold` is accepted as an alias for `value`.
  assert.strictEqual(normalizeAction({ type: "conditional_format", range: "A1", operator: "greaterThan", threshold: 100 }).value, 100);

  // between carries value2; custom colors pass through.
  const band = normalizeAction({ type: "conditional_format", range: "A1:A9", operator: "between", value: 10, value2: 20, fill_color: "#FFEB9C", font_color: "#9C6500" });
  assert.strictEqual(band.operator, "between");
  assert.strictEqual(band.value2, 20);
  assert.strictEqual(band.fill_color, "#FFEB9C");
  assert.strictEqual(band.font_color, "#9C6500");

  // A non-between operator drops a stray value2.
  assert.strictEqual(normalizeAction({ type: "conditional_format", range: "A1", operator: "lessThan", value: 1, value2: 2 }).value2, undefined);
});

test("normalizeActions: prefers actions array, falls back to legacy write", () => {
  const fromActions = normalizeActions({ actions: [{ type: "write_cells", values: [[1]] }] }, {});
  assert.strictEqual(fromActions.length, 1);
  assert.strictEqual(fromActions[0].type, "write_cells");

  const newSheet = normalizeActions({ write: { mode: "new_sheet", values: [[1, 2]] } }, {});
  assert.strictEqual(newSheet.length, 1);
  assert.strictEqual(newSheet[0].type, "create_sheet");

  const selection = normalizeActions(
    { write: { mode: "selection", values: [[1]] } },
    { selection: { address: "B2" } },
  );
  assert.strictEqual(selection.length, 1);
  assert.strictEqual(selection[0].type, "write_cells");
  assert.strictEqual(selection[0].start_cell, "B2");

  assert.deepStrictEqual(normalizeActions({ write: { mode: "none", values: [[1]] } }, {}), []);
  assert.deepStrictEqual(legacyWriteToActions(null, {}), []);
});

test("parseMoney: plain, currency, paren negatives, blanks, unparseable", () => {
  assert.strictEqual(parseMoney("1,234.56"), 1234.56);
  assert.strictEqual(parseMoney("$2,000.00"), 2000);
  assert.strictEqual(parseMoney("(1,234.56)"), -1234.56);
  assert.strictEqual(parseMoney("$(15.00)"), -15);
  assert.strictEqual(parseMoney(""), "");
  assert.strictEqual(parseMoney("n/a"), "");
  assert.strictEqual(parseMoney("1.2.3"), "1.2.3");
});

test("extractedField, followingValue, moneyValueForLabel", () => {
  assert.strictEqual(extractedField("Account Holder: Lance Electric\n", "Account Holder"), "Lance Electric");

  assert.strictEqual(followingValue("Ending Balance\n$12,345.67\n", "Ending Balance"), "$12,345.67");

  assert.strictEqual(moneyValueForLabel("Ending Balance .... $12,345.67", "Ending Balance"), "$12,345.67");
  assert.strictEqual(moneyValueForLabel("Ending Balance\n$12,345.67", "Ending Balance"), "$12,345.67");
});

const statementFixture = {
  name: "stmt.pdf",
  type: "application/pdf",
  size: 1000,
  extraction_status: "parsed",
  extraction_method: "docling",
  extracted_text: [
    "First Example Bank",
    "Account Holder: Lance Electric",
    "Ending Balance",
    "$1,400.00",
    "Transactions",
    "01/01/2023",
    "Deposit",
    "$500.00",
    "$1,500.00",
    "01/02/2023",
    "Withdrawal",
    "$100.00",
    "$1,400.00",
  ].join("\n"),
};

test("statementSummaryRows: pulls real fields from extracted text", () => {
  const summary = statementSummaryRows(statementFixture);
  assert.ok(summary);
  assert.deepStrictEqual(summary[0], ["Field", "Value"]);
  const ending = summary.find((row) => row[0] === "Ending balance");
  assert.ok(ending);
  assert.strictEqual(ending[1], 1400);
  const holder = summary.find((row) => row[0] === "Account holder");
  assert.strictEqual(holder[1], "Lance Electric");
});

test("statementTransactionRows: dates, debit/credit columns, balances", () => {
  const rows = statementTransactionRows(statementFixture);
  assert.ok(rows);
  assert.deepStrictEqual(rows[0], ["Date", "Description", "Debit", "Credit", "Balance"]);

  const deposit = rows.find((row) => row[1] === "Deposit");
  assert.ok(deposit);
  assert.strictEqual(deposit[2], "");
  assert.strictEqual(deposit[3], 500);
  assert.strictEqual(deposit[4], 1500);

  const withdrawal = rows.find((row) => row[1] === "Withdrawal");
  assert.ok(withdrawal);
  assert.strictEqual(withdrawal[2], 100);
  assert.strictEqual(withdrawal[3], "");
  assert.strictEqual(withdrawal[4], 1400);
});

test("truncateText: cuts at maxChars with note, normalizes CRLF", () => {
  assert.strictEqual(truncateText("A".repeat(100), 10), `${"A".repeat(10)}\n\n[truncated 90 characters]`);
  assert.strictEqual(truncateText("A\r\nB", 100), "A\nB");
});

test("windowsPathToWsl and extensionOf", () => {
  assert.strictEqual(windowsPathToWsl("C:\\Users\\User\\file.pdf"), "/mnt/c/Users/User/file.pdf");
  assert.strictEqual(extensionOf("A.PDF"), ".pdf");
  assert.strictEqual(extensionOf("noext"), "");
});

test("compactSelection: caps at 100x16, keeps true dimensions", () => {
  const values = Array.from({ length: 101 }, (_, r) => Array.from({ length: 17 }, (_, c) => `${r}-${c}`));
  const compact = compactSelection({ address: "A1:Q101", values, formulas: [] });
  assert.strictEqual(compact.address, "A1:Q101");
  assert.strictEqual(compact.values.length, 100);
  assert.strictEqual(compact.values[0].length, 16);
  assert.strictEqual(compact.rowCount, 101);
  assert.strictEqual(compact.columnCount, 17);
});

test("normalizeWorkbook: legacy string sheets, object sheets, 30-sheet cap", () => {
  const legacy = normalizeWorkbook({ activeSheet: "Sheet1", sheets: ["Sheet1", "Sheet2"] });
  assert.strictEqual(legacy.activeSheet, "Sheet1");
  assert.deepStrictEqual(legacy.sheets[1], { name: "Sheet2", usedRange: "", rowCount: 0, columnCount: 0 });

  const objects = normalizeWorkbook({
    activeSheet: "S1",
    sheets: [{ name: "S1", usedRange: "A1:D9", rowCount: 9, columnCount: 4 }],
  });
  assert.deepStrictEqual(objects.sheets[0], { name: "S1", usedRange: "A1:D9", rowCount: 9, columnCount: 4 });

  const many = normalizeWorkbook({ sheets: Array.from({ length: 31 }, (_, i) => `S${i}`) });
  assert.strictEqual(many.sheets.length, 30);

  assert.deepStrictEqual(normalizeWorkbook(null), { activeSheet: "", sheets: [] });
});

test("buildChatMessages: system prompt, history cap and sanitizing, tool results, loop budget", () => {
  const base = buildChatMessages({ prompt: "hi" });
  assert.strictEqual(base[0].role, "system");
  assert.ok(base[0].content.includes("read_range"));
  assert.strictEqual(base.length, 2);

  const history = Array.from({ length: 15 }, (_, i) => ({
    role: i % 2 === 0 ? "user" : "assistant",
    content: `msg${i}`,
  }));
  const withHistory = buildChatMessages({ prompt: "hi", history });
  assert.strictEqual(withHistory.length, 1 + 12 + 1);
  assert.strictEqual(withHistory[1].content, "msg3");

  const badRoles = buildChatMessages({ prompt: "hi", history: [{ role: "system", content: "evil" }] });
  assert.strictEqual(badRoles.length, 2);

  const longTurn = buildChatMessages({ prompt: "hi", history: [{ role: "user", content: "A".repeat(5000) }] });
  assert.strictEqual(longTurn[1].content.length, 4000);

  const withTools = buildChatMessages({
    prompt: "hi",
    tool_results: [{ range: "Sheet2!A1:B2", values: [[1, 2]] }],
  });
  const last = withTools[withTools.length - 1];
  assert.ok(last.content.startsWith("TOOL RESULTS"));
  assert.ok(last.content.includes("Sheet2!A1:B2"));

  const notYet = buildChatMessages({ prompt: "hi", loop_count: 3 });
  assert.ok(!notYet[0].content.includes("READ BUDGET EXHAUSTED"));
  const exhausted = buildChatMessages({ prompt: "hi", loop_count: 5 });
  assert.ok(exhausted[0].content.includes("READ BUDGET EXHAUSTED"));
});

test("buildSystemPrompt: read budget flag, honesty rule, A1-relative formula rule", () => {
  const normal = buildSystemPrompt(false);
  assert.ok(!normal.includes("READ BUDGET EXHAUSTED"));
  assert.ok(normal.includes("Never invent"));
  assert.ok(normal.includes("top-left cell is A1"));
  assert.ok(normal.includes("max 5"));
  // The medium-build hardening: complete in one reply, no prose in cells, native conditional_format.
  assert.ok(normal.includes("COMPLETE THE ENTIRE TASK IN THIS ONE REPLY"));
  assert.ok(normal.includes("conditional_format"));
  assert.ok(/never put a sentence/i.test(normal));
  assert.ok(normal.includes("0.25"));

  const exhausted = buildSystemPrompt(true);
  assert.ok(exhausted.includes("READ BUDGET EXHAUSTED"));
  assert.ok(exhausted.includes("Never invent"));
});

test("columnLettersToNumber / numberToColumnLetters: round-trip and bounds", () => {
  assert.strictEqual(columnLettersToNumber("A"), 1);
  assert.strictEqual(columnLettersToNumber("Z"), 26);
  assert.strictEqual(columnLettersToNumber("AA"), 27);
  assert.strictEqual(columnLettersToNumber("XFD"), 16384);
  assert.strictEqual(columnLettersToNumber("a"), 1);
  assert.strictEqual(numberToColumnLetters(1), "A");
  assert.strictEqual(numberToColumnLetters(26), "Z");
  assert.strictEqual(numberToColumnLetters(27), "AA");
  assert.strictEqual(numberToColumnLetters(16384), "XFD");
  for (const n of [1, 2, 26, 27, 52, 703, 16384]) assert.strictEqual(columnLettersToNumber(numberToColumnLetters(n)), n);
});

test("anchorFromAddress: sheet qualifier and range tail", () => {
  assert.deepStrictEqual(anchorFromAddress("Sheet1!H23"), { rowOffset: 22, colOffset: 7 });
  assert.deepStrictEqual(anchorFromAddress("H23"), { rowOffset: 22, colOffset: 7 });
  assert.deepStrictEqual(anchorFromAddress("H23:K30"), { rowOffset: 22, colOffset: 7 });
  assert.deepStrictEqual(anchorFromAddress("'My Sheet'!B2"), { rowOffset: 1, colOffset: 1 });
  assert.deepStrictEqual(anchorFromAddress(""), { rowOffset: 0, colOffset: 0 });
  assert.deepStrictEqual(anchorFromAddress("garbage"), { rowOffset: 0, colOffset: 0 });
});

// The crown-jewel correctness function. Cases pinned by the peer-session handoff,
// including the three bugs the first inline implementation had.
test("translateFormula: anchor H23 (rowOffset 22, colOffset 7) canonical cases", () => {
  assert.strictEqual(translateFormula("=B2*C2", 22, 7), "=I24*J24");
  assert.strictEqual(translateFormula("=SUM(D2:D4)", 22, 7), "=SUM(K24:K26)");
  assert.strictEqual(translateFormula("=B2*C2", 0, 0), "=B2*C2");
  assert.strictEqual(translateFormula("=$B$2", 22, 7), "=$B$2");
  assert.strictEqual(translateFormula("=$B2", 22, 7), "=$B24");
  assert.strictEqual(translateFormula("=B$2", 22, 7), "=I$2");
  assert.strictEqual(translateFormula("=LOG10(B2)", 22, 7), "=LOG10(I24)");
  assert.strictEqual(translateFormula("Widgets", 22, 7), "Widgets");
  assert.strictEqual(translateFormula(4, 22, 7), 4);
});

test("translateFormula: bug #1 — quoted string literals are never shifted", () => {
  assert.strictEqual(translateFormula('=IF(A1>0,"See B2",C1)', 22, 7), '=IF(H23>0,"See B2",J23)');
  assert.strictEqual(translateFormula('="Total of B2:B9"&A1', 22, 7), '="Total of B2:B9"&H23');
  assert.strictEqual(translateFormula('=A1&""""&B2', 22, 7), '=H23&""""&I24');
});

test("translateFormula: bug #2 — sheet-qualified refs and ranges are skipped whole", () => {
  assert.strictEqual(translateFormula("=Sheet1!B5+'Bank Rec'!A1", 22, 7), "=Sheet1!B5+'Bank Rec'!A1");
  assert.strictEqual(translateFormula("='Bank Rec'!A1:B5", 22, 7), "='Bank Rec'!A1:B5");
  // A bare ref next to a qualified one still shifts.
  assert.strictEqual(translateFormula("=Sheet2!B2+A1", 22, 7), "=Sheet2!B2+H23");
});

test("translateFormula: bug #3 — out-of-grid refs are left unchanged, not clamped", () => {
  // XFD is the last column; shifting it right must NOT clamp to XFD — leave it.
  assert.strictEqual(translateFormula("=XFD1", 0, 7), "=XFD1");
  assert.strictEqual(translateFormula("=A1048576", 7, 0), "=A1048576");
});

test("translateMatrixFormulas: shifts formula cells, leaves literals", () => {
  const table = [
    ["Item", "Qty", "Price", "Total"],
    ["Widgets", 4, 2.5, "=B2*C2"],
    ["Gears", 10, 1.25, "=B3*C3"],
    ["", "", "", "=SUM(D2:D3)"],
  ];
  const atA1 = translateMatrixFormulas(table, 0, 0);
  assert.strictEqual(atA1[1][3], "=B2*C2");

  const atH23 = translateMatrixFormulas(table, 22, 7);
  assert.strictEqual(atH23[1][3], "=I24*J24");
  assert.strictEqual(atH23[2][3], "=I25*J25");
  assert.strictEqual(atH23[3][3], "=SUM(K24:K25)");
  assert.strictEqual(atH23[1][0], "Widgets");
  assert.strictEqual(atH23[1][1], 4);
});

test("normalizeActions: resolves anchor from selection and rebases formulas (the H23 regression)", () => {
  const parsed = {
    actions: [
      {
        type: "write_cells",
        values: [
          ["Item", "Total"],
          ["Widgets", "=A2*B2"],
        ],
      },
    ],
  };
  // Model omitted start_cell; selection is H23 -> must resolve and rebase.
  const resolved = normalizeActions(parsed, { selection: { address: "Sheet1!H23" } });
  assert.strictEqual(resolved[0].start_cell, "Sheet1!H23");
  assert.strictEqual(resolved[0].values[1][1], "=H24*I24");

  // create_sheet is always A1: formulas must NOT be shifted even if a selection exists.
  const sheet = normalizeActions(
    { actions: [{ type: "create_sheet", name: "X", values: [["h"], ["=A1"]] }] },
    { selection: { address: "Sheet1!H23" } },
  );
  assert.strictEqual(sheet[0].values[1][0], "=A1");
});

test("scanWrittenCells: error cells and all-zero formula columns", () => {
  const clean = [
    ["Item", "Total"],
    ["A", 10],
    ["B", 20],
  ];
  assert.strictEqual(scanWrittenCells(clean).ok, true);

  const errored = [
    ["Item", "Total"],
    ["A", "#REF!"],
    ["B", "#DIV/0!"],
  ];
  const e = scanWrittenCells(errored);
  assert.strictEqual(e.ok, false);
  assert.strictEqual(e.errors.length, 2);

  // the $0.00 symptom: a Total column that came out all zeros
  const zeros = [
    ["Item", "Total"],
    ["A", 0],
    ["B", 0],
    ["C", 0],
  ];
  const z = scanWrittenCells(zeros);
  assert.strictEqual(z.ok, false);
  assert.deepStrictEqual(z.zeroFormulaColumns, [1]);

  // a single zero is not enough to flag (need >=2 and column all-zero)
  const oneZero = [
    ["Item", "Total"],
    ["A", 0],
    ["B", 5],
  ];
  assert.strictEqual(scanWrittenCells(oneZero).ok, true);
  assert.strictEqual(scanWrittenCells("nope").ok, true);
});

test("expectsWorkbookActions / claimsWorkbookChange: the claims-success-without-actions guard", () => {
  // Jeff's live transcript, 2026-06-11: "can you put this into a spreadsheet?"
  assert.ok(expectsWorkbookActions({ prompt: "can you put this into a spreadsheet?", files: [] }));
  assert.ok(expectsWorkbookActions({ prompt: "ok can you organize it into a the spread sheet for me", files: [] }));
  assert.ok(expectsWorkbookActions({ prompt: "anything", files: [{ extracted_text: "statement text" }] }));
  assert.ok(!expectsWorkbookActions({ prompt: "what is the ending balance?", files: [] }));
  assert.ok(!expectsWorkbookActions({ prompt: "thanks!", files: [{ extraction_status: "failed" }] }));

  assert.ok(claimsWorkbookChange("Done."));
  assert.ok(claimsWorkbookChange("The spreadsheet has been populated with the statement data"));
  assert.ok(claimsWorkbookChange("Created a summary sheet."));
  assert.ok(!claimsWorkbookChange("The ending balance is $6,450.00."));
  assert.ok(!claimsWorkbookChange("I need the statement file to do that."));
});

test("fallbackResponse: write-intent prompt with parsed data builds a sheet, not a preview dump", () => {
  // Jeff's live transcript, 2026-06-11: model call timed out and the fallback
  // dumped raw markdown instead of building from the already-parsed statement.
  const body = {
    prompt: "Parse thhis PDF and make a workbook with its financial data",
    workbook: { activeSheet: "Sheet1" },
    selection: { address: "Sheet1!A1" },
    files: [statementFixture],
  };
  assert.ok(promptWantsWorkbookOutput(body.prompt));
  const out = fallbackResponse(body, "This operation was aborted");
  assert.strictEqual(out.source, "file-parser");
  assert.strictEqual(out.actions.length, 1);
  assert.strictEqual(out.actions[0].type, "create_sheet");
  assert.ok(out.actions[0].values.length > 3, "sheet should carry the parsed statement rows");
  assert.ok(!/##/.test(out.message), "no raw markdown dump in the message");

  // Pure questions still get the Q&A answer, not a sheet.
  const qa = fallbackResponse(
    { prompt: "can you read this pdf?", workbook: {}, selection: {}, files: [statementFixture] },
    "This operation was aborted",
  );
  assert.strictEqual((qa.actions || []).length, 0);
});

test("matrixToCsv and safeExportName", () => {
  assert.strictEqual(
    matrixToCsv([
      ["a", "b,c"],
      ['quote"d', "line\nbreak"],
    ]),
    'a,"b,c"\r\n"quote""d","line\nbreak"',
  );
  assert.strictEqual(safeExportName("My Report"), "MyReport.csv");
  assert.strictEqual(safeExportName("already.csv"), "already.csv");
  assert.strictEqual(safeExportName(""), "hermes-export.csv");
  assert.strictEqual(safeExportName("../../etc/passwd"), "etcpasswd.csv");
});
