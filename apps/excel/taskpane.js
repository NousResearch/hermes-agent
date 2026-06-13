const brokerUrls = [
  window.location.origin,
  "http://127.0.0.1:8787",
  "http://localhost:8787",
].filter((url, index, list) => url && list.indexOf(url) === index);

const state = {
  files: [],
  selection: null,
  workbook: null,
  sending: false,
  workStartedAt: 0,
  workTimer: null,
  workStages: [],
  lastSentFiles: [],
  bridgeOk: true,
  bridgeTimer: null,
  history: [],
  undoStack: [],
  workbookKey: "",
  controller: null,
  reviewMode: false,
};

const els = {
  status: document.getElementById("status"),
  dropzone: document.getElementById("dropzone"),
  dropLabel: document.getElementById("dropLabel"),
  fileInput: document.getElementById("fileInput"),
  fileList: document.getElementById("fileList"),
  messages: document.getElementById("messages"),
  prompt: document.getElementById("prompt"),
  chatForm: document.getElementById("chatForm"),
  workIndicator: document.getElementById("workIndicator"),
  workStage: document.getElementById("workStage"),
  workElapsed: document.getElementById("workElapsed"),
  sendButton: document.getElementById("sendButton"),
  undoButton: document.getElementById("undoButton"),
  clearButton: document.getElementById("clearButton"),
  cancelButton: document.getElementById("cancelButton"),
  reviewToggle: document.getElementById("reviewToggle"),
};

function setStatus(text) {
  els.status.textContent = text;
}

const renderedMessages = [];

function addMessage(role, text, persist = true) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  els.messages.appendChild(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  if (persist) renderedMessages.push({ role, text });
}

function historyStorageKey() {
  return `hermes-chat:${state.workbookKey || "default"}`;
}

function loadChatHistory() {
  try {
    const raw = localStorage.getItem(historyStorageKey());
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data && Array.isArray(data.history)) state.history = data.history.slice(-40);
    if (data && Array.isArray(data.messages)) {
      const restored = data.messages.slice(-80);
      for (const message of restored) addMessage(message.role, message.text, false);
      renderedMessages.push(...restored);
    }
  } catch {
    // Office's webview may deny storage; the chat just starts empty.
  }
}

function saveChatHistory() {
  try {
    localStorage.setItem(
      historyStorageKey(),
      JSON.stringify({ history: state.history.slice(-40), messages: renderedMessages.slice(-80) }),
    );
  } catch {}
}

function clearChat() {
  state.history = [];
  state.undoStack = [];
  renderedMessages.length = 0;
  try {
    localStorage.removeItem(historyStorageKey());
  } catch {}
  els.messages.replaceChildren();
  addMessage("hermes", "Chat cleared. Workbook changes were not reverted; use Undo for that.", false);
  setStatus("Ready");
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function stageForElapsed(seconds) {
  let current = state.workStages[0] || "Working...";
  for (const stage of state.workStages) {
    if (seconds >= stage.after) current = stage.text;
  }
  return typeof current === "string" ? current : current.text;
}

function setWorkStage(text) {
  if (!els.workIndicator.hidden) {
    els.workStage.textContent = text;
    els.workElapsed.textContent = `Elapsed ${formatElapsed(Date.now() - state.workStartedAt)}`;
  }
  setStatus(text);
}

function startWorkIndicator(filesToSend) {
  const hasLargeFile = filesToSend.some((file) => file.size > 1024 * 1024);
  const hasFiles = filesToSend.length > 0;
  state.workStartedAt = Date.now();
  state.workStages = hasFiles
    ? [
        { after: 0, text: `Preparing ${filesToSend.length} file(s)...` },
        { after: 3, text: "Uploading files to the local bridge..." },
        { after: 8, text: hasLargeFile ? "Docling is parsing the document. Large PDFs can take a few minutes..." : "Reading attached file text..." },
        { after: 45, text: "Still parsing. This is normal for large bank statements..." },
        { after: 120, text: "Still working. Docling is extracting tables and statement text..." },
        { after: 210, text: "Long parse still running. Keep this pane open..." },
      ]
    : [
        { after: 0, text: "Reading workbook context..." },
        { after: 3, text: "Asking Hermes..." },
        { after: 20, text: "Waiting on the local model..." },
        { after: 60, text: "Still waiting on the local model..." },
      ];
  els.workIndicator.hidden = false;
  els.sendButton.disabled = true;
  if (els.cancelButton) els.cancelButton.hidden = false;
  setWorkStage(stageForElapsed(0));
  clearInterval(state.workTimer);
  state.workTimer = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - state.workStartedAt) / 1000);
    setWorkStage(stageForElapsed(elapsedSeconds));
  }, 1000);
}

function stopWorkIndicator(finalStatus = "Ready") {
  clearInterval(state.workTimer);
  state.workTimer = null;
  state.workStartedAt = 0;
  state.workStages = [];
  els.workIndicator.hidden = true;
  if (els.cancelButton) els.cancelButton.hidden = true;
  els.sendButton.disabled = false;
  setStatus(finalStatus);
}

async function checkBridgeHealth(showStatus = false) {
  for (const brokerUrl of brokerUrls) {
    try {
      const response = await fetch(`${brokerUrl}/api/health`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const health = await response.json();
      state.bridgeOk = true;
      if (showStatus && !state.sending) {
        setStatus(health.ok ? "Ready" : "Ready, but a local service is degraded");
      }
      return health;
    } catch {}
  }
  state.bridgeOk = false;
  if (!state.sending) setStatus("Hermes bridge offline. Restart the Hermes bridge.");
  return null;
}

function startBridgeMonitor() {
  clearInterval(state.bridgeTimer);
  checkBridgeHealth(true);
  state.bridgeTimer = setInterval(() => checkBridgeHealth(false), 15000);
}

function updateFileList() {
  els.fileList.replaceChildren();
  for (const file of state.files) {
    const li = document.createElement("li");
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    li.innerHTML = `<span>${file.name}</span><span>${sizeKb} KB</span>`;
    els.fileList.appendChild(li);
  }
  els.dropLabel.textContent = state.files.length
    ? `${state.files.length} file(s) attached. Drop more here.`
    : "Drop files here, into the message box, or click to attach";
}

function clearFiles() {
  state.files = [];
  els.fileInput.value = "";
  updateFileList();
}

function describeFiles(files) {
  if (!files.length) return "";
  return files.map((file) => `${file.name} (${Math.max(1, Math.round(file.size / 1024))} KB)`).join(", ");
}

function wantsPriorFiles(prompt) {
  return /\b(this|that|the|previous|last|attached|same)\s+(pdf|file|files|statement|bank statement|document|doc|attachment|upload|scan)\b/i.test(
    prompt,
  );
}

async function addFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  state.files.push(...files);
  updateFileList();
  setStatus(`${state.files.length} file(s) attached`);
}

async function fileToPayload(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return {
    name: file.name,
    type: file.type || "application/octet-stream",
    size: file.size,
    base64: btoa(binary),
  };
}

async function readWorkbookContext() {
  return Excel.run(async (context) => {
    const workbook = context.workbook;
    const sheets = workbook.worksheets;
    const active = sheets.getActiveWorksheet();
    const selected = workbook.getSelectedRange();
    sheets.load("items/name");
    active.load("name");
    selected.load(["address", "values", "formulas", "rowCount", "columnCount"]);
    await context.sync();

    const usedRanges = sheets.items.map((sheet) => {
      const usedRange = sheet.getUsedRangeOrNullObject();
      usedRange.load(["address", "rowCount", "columnCount"]);
      return { sheet, usedRange };
    });
    await context.sync();

    state.workbook = {
      activeSheet: active.name,
      sheets: usedRanges.map(({ sheet, usedRange }) =>
        usedRange.isNullObject
          ? { name: sheet.name, usedRange: "", rowCount: 0, columnCount: 0 }
          : {
              name: sheet.name,
              usedRange: usedRange.address,
              rowCount: usedRange.rowCount,
              columnCount: usedRange.columnCount,
            },
      ),
    };
    state.selection = {
      address: selected.address,
      values: selected.values,
      formulas: selected.formulas,
      rowCount: selected.rowCount,
      columnCount: selected.columnCount,
    };

    setStatus("Ready");
    return { workbook: state.workbook, selection: state.selection };
  });
}

async function executeReadRange(rangeRef) {
  try {
    return await Excel.run(async (context) => {
      const range = rangeFromRef(context, rangeRef);
      range.load(["address", "rowCount", "columnCount"]);
      await context.sync();

      const truncated = range.rowCount > 300 || range.columnCount > 30;
      const target = truncated
        ? range
            .getCell(0, 0)
            .getResizedRange(Math.min(range.rowCount, 300) - 1, Math.min(range.columnCount, 30) - 1)
        : range;
      target.load(["values", "formulas"]);
      await context.sync();

      const result = { range: range.address, values: target.values, formulas: target.formulas };
      if (truncated) result.truncated = true;
      return result;
    });
  } catch (error) {
    return { range: rangeRef, error: error.message };
  }
}

async function postChat(payload, options = {}) {
  const body = JSON.stringify(payload);
  const errors = [];
  for (const brokerUrl of brokerUrls) {
    try {
      const response = await fetch(`${brokerUrl}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        signal: options.signal,
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    } catch (error) {
      if (error.name === "AbortError") throw error;
      errors.push(`${brokerUrl}: ${error.message}`);
    }
  }
  throw new Error(`Could not reach the local Hermes bridge. Tried ${errors.join(" | ")}`);
}

async function askHermes(prompt, filesToSend) {
  state.controller = new AbortController();
  const context = await readWorkbookContext();
  const files = await Promise.all(filesToSend.map(fileToPayload));

  let toolResults = [];
  let parsedFiles = null;
  let response = null;

  // The model may ask to read ranges before answering; loop up to 5 read rounds.
  for (let loopCount = 0; loopCount < 6; loopCount += 1) {
    try {
      response = await postChat(
        {
          prompt,
          history: state.history.slice(-12),
          ...context,
          files: loopCount === 0 ? files : [],
          parsed_files: loopCount > 0 && parsedFiles ? parsedFiles : undefined,
          tool_results: toolResults.length ? toolResults : undefined,
          loop_count: loopCount,
        },
        { signal: state.controller.signal },
      );
    } catch (error) {
      if (error.name === "AbortError") throw new Error("Canceled.");
      throw error;
    }

    if (response.parsed_files) parsedFiles = response.parsed_files;
    const readActions = (response.actions || []).filter((action) => action && action.type === "read_range");
    if (!readActions.length || loopCount >= 5) {
      response.actions = (response.actions || []).filter((action) => action && action.type !== "read_range");
      return response;
    }

    setWorkStage(`Hermes is reading ${readActions.map((action) => action.range).join(", ")}...`);
    const reads = await Promise.all(readActions.map((action) => executeReadRange(action.range)));
    toolResults = [...toolResults, ...reads];
  }
  return response;
}

async function verifyWrittenRange(address) {
  try {
    return await Excel.run(async (context) => {
      const range = rangeFromRef(context, address);
      range.load(["rowCount", "columnCount"]);
      await context.sync();
      const truncated = range.rowCount > 300 || range.columnCount > 30;
      const target = truncated
        ? range.getCell(0, 0).getResizedRange(Math.min(range.rowCount, 300) - 1, Math.min(range.columnCount, 30) - 1)
        : range;
      target.load("values");
      await context.sync();
      return target.values;
    });
  } catch {
    return null;
  }
}

// Mirror of the broker's scanWrittenCells (kept in sync; tested in server.test.mjs).
function scanWrittenCells(values) {
  if (!Array.isArray(values)) return { errors: [], zeroFormulaColumns: [], ok: true };
  const errorRegex = /^#(REF|DIV\/0|VALUE|NAME\?|N\/A|NULL|NUM)!?/i;
  const errors = [];
  const nonEmpty = {};
  const zeros = {};
  for (let r = 0; r < values.length; r += 1) {
    const row = values[r];
    if (!Array.isArray(row)) continue;
    for (let c = 0; c < row.length; c += 1) {
      const cell = row[c];
      const str = cell === null || cell === undefined ? "" : String(cell);
      if (str && errorRegex.test(str)) errors.push({ r, c, value: str });
      if (r === 0 || str === "") continue;
      nonEmpty[c] = (nonEmpty[c] || 0) + 1;
      if (cell === 0 || str === "0") zeros[c] = (zeros[c] || 0) + 1;
    }
  }
  const zeroFormulaColumns = [];
  for (const key of Object.keys(nonEmpty)) {
    const c = Number(key);
    if (nonEmpty[c] >= 2 && nonEmpty[c] === (zeros[c] || 0)) zeroFormulaColumns.push(c);
  }
  zeroFormulaColumns.sort((a, b) => a - b);
  return { errors, zeroFormulaColumns, ok: errors.length === 0 && zeroFormulaColumns.length === 0 };
}

function verificationWarning(address, scan) {
  const parts = [];
  if (scan.errors.length) parts.push(`${scan.errors.length} error cell(s)`);
  if (scan.zeroFormulaColumns.length) parts.push(`${scan.zeroFormulaColumns.length} all-zero column(s)`);
  return `Warning: ${address} shows ${parts.join(" and ")} — the formulas may not have landed correctly.`;
}

function cancelWork() {
  if (state.controller) {
    state.controller.abort();
    state.controller = null;
  }
  setStatus("Canceling...");
}

function describeActions(actions) {
  return (actions || [])
    .map((action) => {
      if (!action) return null;
      if (action.type === "write_cells") {
        const rows = (action.values || []).length;
        const cols = (action.values || [])[0]?.length || 0;
        return `Write ${rows}×${cols} to ${action.start_cell}`;
      }
      if (action.type === "create_sheet") return `Create sheet "${action.name}" (${(action.values || []).length} rows)`;
      if (action.type === "format_cells") return `Format ${action.range}`;
      if (action.type === "conditional_format") return `Highlight ${action.range} where value ${action.operator} ${action.value}`;
      if (action.type === "execute_office_js") return `Run workbook script: ${action.explanation}`;
      return null;
    })
    .filter(Boolean)
    .join("\n");
}

function normalizeMatrix(values) {
  if (!Array.isArray(values)) return [];
  const rows = values
    .filter((row) => Array.isArray(row))
    .map((row) =>
      row.map((cell) => {
        if (cell === null || ["string", "number", "boolean"].includes(typeof cell)) return cell;
        return String(cell);
      }),
    );
  const width = Math.max(0, ...rows.map((row) => row.length));
  return rows.map((row) => row.concat(Array(Math.max(0, width - row.length)).fill("")));
}

function parseStartCell(startCell) {
  const ref = String(startCell || state.selection?.address || "A1").trim();
  const bang = ref.lastIndexOf("!");
  const sheetName = bang >= 0 ? ref.slice(0, bang).replace(/^'|'$/g, "") : state.workbook?.activeSheet;
  const address = bang >= 0 ? ref.slice(bang + 1) : ref;
  return { sheetName, address: address || "A1" };
}

function rangeFromRef(context, ref) {
  const { sheetName, address } = parseStartCell(ref);
  const sheet = sheetName
    ? context.workbook.worksheets.getItem(sheetName)
    : context.workbook.worksheets.getActiveWorksheet();
  return sheet.getRange(address);
}

function safeSheetName(name, fallback = "Hermes Output") {
  return String(name || fallback)
    .replace(/[\[\]:*?/\\]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 31) || fallback;
}

function excelNumberFormat(style, action) {
  const format = String(action.number_format || "").trim();
  if (format && !["number", "integer", "currency", "percent", "ratio", "text"].includes(format)) return format;
  const named = format || style;
  const symbol = action.currency_symbol || "$";
  const dp = Number.isFinite(Number(action.number_format_dp)) ? Number(action.number_format_dp) : 2;
  const decimals = dp > 0 ? `.${"0".repeat(dp)}` : "";
  if (named === "currency") return `${symbol}#,##0${decimals};[Red](${symbol}#,##0${decimals})`;
  if (named === "number") return `#,##0${decimals};[Red](#,##0${decimals})`;
  if (named === "integer") return "#,##0;[Red](#,##0)";
  if (named === "percent") return `0${decimals}%`;
  if (named === "ratio") return "0.0x";
  if (named === "text") return "@";
  return null;
}

function borderWeight(weight) {
  const normalized = String(weight || "thin").toLowerCase();
  if (normalized === "medium") return Excel.BorderWeight.medium;
  if (normalized === "thick") return Excel.BorderWeight.thick;
  return Excel.BorderWeight.thin;
}

function setBorder(format, edge, weight, color = "#9CA3AF") {
  const border = format.borders.getItem(edge);
  if (weight === "none") {
    border.style = Excel.BorderLineStyle.none;
    return;
  }
  border.style = Excel.BorderLineStyle.continuous;
  border.weight = borderWeight(weight);
  border.color = color;
}

function fillNumberFormat(range, format, rows, columns) {
  range.numberFormat = Array.from({ length: rows }, () => Array.from({ length: columns }, () => format));
}

function styleRange(range, rows, columns, action = {}) {
  const styles = Array.isArray(action.style) ? action.style : action.style ? [action.style] : [];
  const hasStyle = (name) => styles.includes(name);

  range.format.font.name = action.font_name || "Arial";
  range.format.font.size = action.font_size || 10;

  if (hasStyle("header")) {
    range.format.font.bold = true;
    range.format.font.color = "#FFFFFF";
    range.format.fill.color = "#1F4E78";
    range.format.horizontalAlignment = Excel.HorizontalAlignment.center;
    range.format.verticalAlignment = Excel.VerticalAlignment.center;
    range.format.wrapText = true;
  }
  if (hasStyle("total-row")) {
    range.format.font.bold = true;
    setBorder(range.format, Excel.BorderIndex.edgeTop, "thin", "#374151");
  }
  if (hasStyle("subtotal")) {
    range.format.font.bold = true;
    setBorder(range.format, Excel.BorderIndex.edgeTop, "thin", "#9CA3AF");
  }
  if (hasStyle("input")) {
    range.format.fill.color = "#FFF2CC";
    range.format.font.color = "#0000FF";
  }
  if (hasStyle("blank-section")) {
    range.format.fill.color = "#F3F4F6";
  }

  const numberStyle = styles.find((name) => ["number", "integer", "currency", "percent", "ratio", "text"].includes(name));
  const numberFormat = excelNumberFormat(numberStyle, action);
  if (numberFormat) fillNumberFormat(range, numberFormat, rows, columns);

  if (typeof action.bold === "boolean") range.format.font.bold = action.bold;
  if (typeof action.italic === "boolean") range.format.font.italic = action.italic;
  if (typeof action.underline === "boolean") range.format.font.underline = action.underline ? "Single" : "None";
  if (action.font_color) range.format.font.color = String(action.font_color);
  if (action.fill_color) range.format.fill.color = String(action.fill_color);
  if (action.horizontal_alignment) range.format.horizontalAlignment = action.horizontal_alignment;
  if (action.vertical_alignment) range.format.verticalAlignment = action.vertical_alignment;
  if (typeof action.wrap_text === "boolean") range.format.wrapText = action.wrap_text;
  if (Number.isFinite(Number(action.column_width))) range.format.columnWidth = Number(action.column_width) * 7.2;
  if (Number.isFinite(Number(action.row_height))) range.format.rowHeight = Number(action.row_height);

  const borderColor = action.border_color || "#9CA3AF";
  if (action.borders) {
    for (const edge of [
      Excel.BorderIndex.edgeTop,
      Excel.BorderIndex.edgeBottom,
      Excel.BorderIndex.edgeLeft,
      Excel.BorderIndex.edgeRight,
      Excel.BorderIndex.insideHorizontal,
      Excel.BorderIndex.insideVertical,
    ]) {
      setBorder(range.format, edge, action.borders, borderColor);
    }
  }
  if (action.border_top) setBorder(range.format, Excel.BorderIndex.edgeTop, action.border_top, borderColor);
  if (action.border_bottom) setBorder(range.format, Excel.BorderIndex.edgeBottom, action.border_bottom, borderColor);
  if (action.border_left) setBorder(range.format, Excel.BorderIndex.edgeLeft, action.border_left, borderColor);
  if (action.border_right) setBorder(range.format, Excel.BorderIndex.edgeRight, action.border_right, borderColor);

  if (action.auto_fit !== false) {
    range.format.autofitColumns();
    range.format.autofitRows();
  }
}

function applyProfessionalTableFormat(sheet, target, values) {
  const rowCount = values.length;
  const columnCount = Math.max(...values.map((row) => row.length));
  styleRange(target, rowCount, columnCount, { borders: "thin", auto_fit: true });

  const header = target.getCell(0, 0).getResizedRange(0, columnCount - 1);
  styleRange(header, 1, columnCount, { style: "header", auto_fit: true });

  values.forEach((row, index) => {
    const labelText = row.map((cell) => String(cell ?? "").toLowerCase()).join(" ");
    if (index > 0 && /\b(total|net income|gross profit|ebitda|ending balance)\b/.test(labelText)) {
      const totalRow = target.getCell(index, 0).getResizedRange(0, columnCount - 1);
      styleRange(totalRow, 1, columnCount, { style: "total-row", auto_fit: true });
    }
  });

  if (rowCount > 1 && columnCount > 1) {
    const body = target.getCell(1, 1).getResizedRange(rowCount - 2, columnCount - 2);
    fillNumberFormat(body, '#,##0.00;[Red](#,##0.00);"-"', rowCount - 1, columnCount - 1);
  }

  sheet.freezePanes.freezeRows(1);
}

async function beforeWriteCellsSnapshot(action, values) {
  try {
    await Excel.run(async (context) => {
      const start = rangeFromRef(context, action.start_cell || action.startCell);
      const target = start.getResizedRange(values.length - 1, Math.max(...values.map((row) => row.length)) - 1);
      target.load(["address", "formulas", "numberFormat"]);
      await context.sync();
      state.undoStack.push({
        kind: "write_cells",
        address: target.address,
        formulas: target.formulas,
        numberFormat: target.numberFormat,
      });
      if (state.undoStack.length > 10) state.undoStack.shift();
    });
  } catch {
    // A failed snapshot only means this particular write cannot be undone.
  }
}

function createSheetUndoRecord(name) {
  state.undoStack.push({ kind: "create_sheet", name });
  if (state.undoStack.length > 10) state.undoStack.shift();
}

async function undoLast() {
  if (!state.undoStack.length) {
    addMessage("hermes", "Nothing to undo.", false);
    return;
  }
  const record = state.undoStack.pop();
  try {
    if (record.kind === "write_cells") {
      await Excel.run(async (context) => {
        const range = rangeFromRef(context, record.address);
        range.formulas = record.formulas;
        range.numberFormat = record.numberFormat;
        await context.sync();
      });
      addMessage("hermes", `Undid the last write: restored ${record.address}.`);
    } else if (record.kind === "create_sheet") {
      await Excel.run(async (context) => {
        const sheet = context.workbook.worksheets.getItemOrNullObject(record.name);
        await context.sync();
        if (!sheet.isNullObject) sheet.delete();
        await context.sync();
      });
      addMessage("hermes", `Removed the sheet Hermes created: ${record.name}.`);
    }
    saveChatHistory();
  } catch (error) {
    addMessage("hermes", `Undo failed: ${error.message}`);
  }
}

async function writeCellsAction(action) {
  const values = normalizeMatrix(action.values || action.table);
  if (!values.length) return null;

  await beforeWriteCellsSnapshot(action, values);
  return Excel.run(async (context) => {
    const start = rangeFromRef(context, action.start_cell || action.startCell);
    const target = start.getResizedRange(values.length - 1, Math.max(...values.map((row) => row.length)) - 1);
    target.values = values;
    if (action.auto_format) styleRange(target, values.length, values[0].length, { borders: "thin", auto_fit: true });
    else {
      target.format.autofitColumns();
      target.format.autofitRows();
    }
    target.load("address");
    await context.sync();
    return { status: `Wrote ${values.length} row(s) to ${target.address}.`, address: target.address };
  });
}

async function createSheetAction(action) {
  const values = normalizeMatrix(action.values || action.table);
  if (!values.length) return null;

  return Excel.run(async (context) => {
    const sheets = context.workbook.worksheets;
    sheets.load("items/name");
    await context.sync();

    const baseName = safeSheetName(action.name || action.sheet_name || action.sheetName);
    const existing = new Set(sheets.items.map((sheet) => sheet.name.toLowerCase()));
    let name = baseName;
    for (let index = 2; existing.has(name.toLowerCase()); index += 1) {
      name = safeSheetName(`${baseName.slice(0, 27)} ${index}`);
    }

    const sheet = sheets.add(name);
    const target = sheet.getRange("A1").getResizedRange(
      values.length - 1,
      Math.max(...values.map((row) => row.length)) - 1,
    );
    target.values = values;
    applyProfessionalTableFormat(sheet, target, values);
    sheet.activate();
    target.load("address");
    await context.sync();
    createSheetUndoRecord(name);
    return { status: `Created ${name} and wrote ${values.length} row(s) to ${target.address}.`, address: target.address };
  });
}

async function formatCellsAction(action) {
  return Excel.run(async (context) => {
    const range = rangeFromRef(context, action.range);
    range.load(["address", "rowCount", "columnCount"]);
    await context.sync();
    styleRange(range, range.rowCount, range.columnCount, action);
    await context.sync();
    return `Formatted ${range.address}.`;
  });
}

function conditionalOperator(name) {
  const map = {
    lessThan: Excel.ConditionalCellValueOperator.lessThan,
    lessThanOrEqual: Excel.ConditionalCellValueOperator.lessThanOrEqual,
    greaterThan: Excel.ConditionalCellValueOperator.greaterThan,
    greaterThanOrEqual: Excel.ConditionalCellValueOperator.greaterThanOrEqual,
    equalTo: Excel.ConditionalCellValueOperator.equalTo,
    notEqualTo: Excel.ConditionalCellValueOperator.notEqualTo,
    between: Excel.ConditionalCellValueOperator.between,
    notBetween: Excel.ConditionalCellValueOperator.notBetween,
  };
  return map[name] || Excel.ConditionalCellValueOperator.lessThan;
}

// A cell-value rule formula is an Excel expression; a bare number string works.
function conditionalFormula(value) {
  if (value === null || value === undefined || value === "") return "0";
  return String(value);
}

async function conditionalFormatAction(action) {
  return Excel.run(async (context) => {
    const range = rangeFromRef(context, action.range);
    const cf = range.conditionalFormats.add(Excel.ConditionalFormatType.cellValue);
    if (action.fill_color) cf.cellValue.format.fill.color = String(action.fill_color);
    if (action.font_color) cf.cellValue.format.font.color = String(action.font_color);
    const operator = conditionalOperator(action.operator);
    const rule = { operator, formula1: conditionalFormula(action.value) };
    if (
      operator === Excel.ConditionalCellValueOperator.between ||
      operator === Excel.ConditionalCellValueOperator.notBetween
    ) {
      rule.formula2 = conditionalFormula(action.value2 ?? action.value);
    }
    cf.cellValue.rule = rule;
    await context.sync();
    return `Applied conditional formatting to ${action.range}.`;
  });
}

async function executeOfficeJsAction(action) {
  const code = String(action.code || "").trim();
  if (!code) return null;
  if (/\bExcel\.run\s*\(/.test(code)) {
    throw new Error("Hermes action used nested Excel.run; ask again with direct Office.js code.");
  }

  return Excel.run(async (context) => {
    const fn = new Function("context", `"use strict"; return (async () => {\n${code}\n})();`);
    await fn(context);
    await context.sync();
    return action.explanation ? `Ran workbook edit: ${action.explanation}` : "Ran workbook edit.";
  });
}

function actionsFromLegacyWrite(write) {
  if (!write || write.mode === "none") return [];
  const values = write.table || write.values;
  if (write.mode === "new_sheet") {
    return [{ type: "create_sheet", name: write.name || "Hermes Output", values }];
  }
  if (write.mode === "selection") {
    return [{ type: "write_cells", start_cell: state.selection?.address || "A1", values }];
  }
  return [];
}

async function runWorkbookActions(result) {
  const actions = Array.isArray(result.actions) ? result.actions : actionsFromLegacyWrite(result.write);
  const statusLines = [];
  for (const action of actions) {
    if (!action || typeof action !== "object") continue;
    // One bad action (usually model-authored execute_office_js code) must not
    // abort the rest or mask the statuses of writes that already succeeded.
    try {
      if (action.type === "write_cells" || action.type === "create_sheet") {
        const res = action.type === "write_cells" ? await writeCellsAction(action) : await createSheetAction(action);
        if (!res) continue;
        statusLines.push(res.status);
        // Re-read what we just wrote and warn if formulas didn't land (error cells / all-zero columns).
        const written = await verifyWrittenRange(res.address);
        if (written) {
          const scan = scanWrittenCells(written);
          if (!scan.ok) statusLines.push(verificationWarning(res.address, scan));
        }
      }
      if (action.type === "format_cells") statusLines.push(await formatCellsAction(action));
      if (action.type === "conditional_format") statusLines.push(await conditionalFormatAction(action));
      if (action.type === "execute_office_js") statusLines.push(await executeOfficeJsAction(action));
      if (action.type === "export") statusLines.push(await exportAction(action));
    } catch (error) {
      statusLines.push(`One step failed (${action.type}): ${error.message} — other changes were still applied.`);
    }
  }
  return statusLines.filter(Boolean);
}

async function exportAction(action) {
  try {
    for (const brokerUrl of brokerUrls) {
      try {
        const response = await fetch(`${brokerUrl}/api/export`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ name: action.name, values: action.values || action.table }),
        });
        if (!response.ok) continue;
        const data = await response.json();
        return `Exported a CSV to ${data.path || "the add-in exports folder"}.`;
      } catch {}
    }
  } catch {}
  return "Could not write the CSV export.";
}

function fileStatusLines(result) {
  const files = Array.isArray(result.files) ? result.files : [];
  return files
    .filter((file) => file?.name)
    .map((file) => {
      if (file.extraction_status === "parsed") {
        return `${file.name}: read with ${file.extraction_method || "parser"}.`;
      }
      if (file.extraction_status === "failed") {
        return `${file.name}: not readable (${file.extraction_error || "parser failed"}).`;
      }
      return `${file.name}: attached.`;
    });
}

function wireDropzone() {
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
  const preventFileOpen = (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  };
  const claimFileDrop = (event) => {
    preventFileOpen(event);
    if (hasFiles(event)) event.stopPropagation();
  };
  const setDragging = (isDragging) => {
    els.dropzone.classList.toggle("dragover", isDragging);
    els.chatForm.classList.toggle("dragover", isDragging);
  };
  const addDropTarget = (target) => {
    for (const eventName of ["dragenter", "dragover"]) {
      target.addEventListener(eventName, (event) => {
        claimFileDrop(event);
        if (hasFiles(event)) setDragging(true);
      });
    }
    for (const eventName of ["dragleave", "dragend"]) {
      target.addEventListener(eventName, (event) => {
        if (hasFiles(event)) setDragging(false);
      });
    }
    target.addEventListener("drop", (event) => {
      claimFileDrop(event);
      setDragging(false);
      addFiles(event.dataTransfer?.files);
    });
  };

  // Office's webview may otherwise navigate/open the dropped file.
  for (const eventName of ["dragenter", "dragover", "drop"]) {
    window.addEventListener(eventName, preventFileOpen, true);
    document.addEventListener(eventName, preventFileOpen, true);
  }

  addDropTarget(document.body);
  addDropTarget(els.chatForm);
  addDropTarget(els.prompt);
  addDropTarget(els.messages);
  addDropTarget(els.dropzone);
  addDropTarget(els.fileInput);
  els.fileInput.addEventListener("change", (event) => addFiles(event.target.files));
}

async function applyResultAndRecord(result, filesToSend, fileLines) {
  const statusLines = await runWorkbookActions(result);
  if (statusLines.length) addMessage("hermes", statusLines.join("\n"));
  state.history.push({
    role: "assistant",
    content: [result.message || "Done.", ...fileLines, ...statusLines].filter(Boolean).join("\n"),
  });
  if (state.history.length > 40) state.history = state.history.slice(-40);
  saveChatHistory();
  if (filesToSend.length) state.lastSentFiles = filesToSend;
  if (filesToSend.length) clearFiles();
}

function renderReviewButtons(result, filesToSend, fileLines) {
  const bar = document.createElement("div");
  bar.className = "pending-actions";

  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "primary";
  applyBtn.textContent = "Apply";
  applyBtn.addEventListener("click", async () => {
    bar.remove();
    state.sending = true;
    startWorkIndicator([]);
    setWorkStage("Applying workbook output...");
    try {
      await applyResultAndRecord(result, filesToSend, fileLines);
      stopWorkIndicator("Ready");
    } catch (error) {
      stopWorkIndicator("Error");
      addMessage("hermes", `Error: ${error.message}`);
    } finally {
      state.sending = false;
      state.controller = null;
      els.sendButton.disabled = false;
    }
  });

  const discardBtn = document.createElement("button");
  discardBtn.type = "button";
  discardBtn.className = "ghost";
  discardBtn.textContent = "Discard";
  discardBtn.addEventListener("click", () => {
    bar.remove();
    addMessage("hermes", "Discarded — no changes made.");
  });

  bar.appendChild(applyBtn);
  bar.appendChild(discardBtn);
  els.messages.appendChild(bar);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function wireActions() {
  els.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });

  els.undoButton.addEventListener("click", undoLast);
  els.clearButton.addEventListener("click", clearChat);
  if (els.cancelButton) els.cancelButton.addEventListener("click", cancelWork);
  if (els.reviewToggle) {
    els.reviewToggle.addEventListener("change", (event) => {
      state.reviewMode = event.target.checked;
    });
  }

  els.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.sending) return;
    const prompt = els.prompt.value.trim();
    if (!prompt) return;
    const health = await checkBridgeHealth(false);
    if (!health) {
      addMessage("hermes", "I cannot reach the local Hermes bridge right now. Restart the Hermes bridge window, then reopen this pane.");
      return;
    }
    const filesToSend = state.files.length ? [...state.files] : wantsPriorFiles(prompt) ? [...state.lastSentFiles] : [];
    const fileDescription = describeFiles(filesToSend);
    const reusedFiles = !state.files.length && filesToSend.length;
    const userText = fileDescription
      ? `${prompt}\n\n${reusedFiles ? "Using previous attachment" : "Attached"}: ${fileDescription}`
      : prompt;
    addMessage("user", userText);
    state.history.push({ role: "user", content: userText });
    els.prompt.value = "";
    state.sending = true;
    startWorkIndicator(filesToSend);
    try {
      const result = await askHermes(prompt, filesToSend);
      addMessage("hermes", result.message || "Done.");
      const fileLines = fileStatusLines(result);
      if (fileLines.length) addMessage("hermes", fileLines.join("\n"));

      const actions = Array.isArray(result.actions) ? result.actions : actionsFromLegacyWrite(result.write);
      const applyable = actions.filter((action) => action && action.type !== "read_range");

      if (state.reviewMode && applyable.length) {
        // Hold the changes; let the user approve them first.
        addMessage("hermes", `Review these changes:\n${describeActions(applyable)}`);
        renderReviewButtons(result, filesToSend, fileLines);
        stopWorkIndicator("Waiting for review");
        return;
      }

      setWorkStage("Applying workbook output...");
      await applyResultAndRecord(result, filesToSend, fileLines);
      stopWorkIndicator("Ready");
    } catch (error) {
      stopWorkIndicator(error.message === "Canceled." ? "Canceled" : "Error");
      addMessage("hermes", error.message === "Canceled." ? "Canceled." : `Error: ${error.message}`);
      state.history.push({ role: "assistant", content: error.message === "Canceled." ? "Canceled." : `Error: ${error.message}` });
      saveChatHistory();
    } finally {
      state.sending = false;
      state.controller = null;
      els.sendButton.disabled = false;
    }
  });
}

Office.onReady((info) => {
  if (info.host !== Office.HostType.Excel) {
    setStatus("Open this add-in inside Excel.");
    return;
  }
  try {
    state.workbookKey = Office.context?.document?.url || "default";
  } catch {
    state.workbookKey = "default";
  }
  loadChatHistory();
  wireDropzone();
  wireActions();
  startBridgeMonitor();
  setStatus("Ready");
});
