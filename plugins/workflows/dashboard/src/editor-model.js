// Pure form conversions for the workflow editor. No DOM, no React, no side-effects.
// Shared by app.js inspector renderers and editor-model.test.js.

export var SUPPORTED_TRIGGERS = ["manual", "schedule"];
export var SUPPORTED_NODES = ["agent_task", "fail", "join", "parallel", "pass", "switch", "wait"];
export var SUPPORTED_INTAKE_MODES = ["continuous", "single"];
export var RESULT_CONTRACT_PRIMITIVES = ["string", "number", "boolean", "array", "object"];
export var CONDITION_OPS = ["eq", "ne", "gt", "gte", "lt", "lte", "exists", "missing", "contains", "starts_with", "ends_with"];
export var SCALAR_INPUT_KINDS = ["text", "long_text", "prompt", "criteria", "url", "repo_path", "boolean", "number", "integer"];

// ponytail: fields specific to each node type, excluding common (type, title).
var NODE_TYPE_FIELDS = {
  agent_task: ["profile", "prompt", "result_contract", "provider", "model", "skills", "workspace_kind", "workspace_path", "goal_mode", "goal_max_turns", "max_retries"],
  switch: ["cases", "default"],
  wait: ["seconds"],
  pass: ["output"],
  fail: ["output"],
  parallel: [],
  join: [],
};

export function supportedEditorCoverage() {
  return {
    triggers: SUPPORTED_TRIGGERS.slice(),
    nodes: SUPPORTED_NODES.slice(),
    intakeModes: SUPPORTED_INTAKE_MODES.slice(),
  };
}

export function editorSections(spec) {
  if (!spec) return [];
  var sections = [];
  sections.push({ kind: "metadata", fields: ["id", "name", "version", "enabled"] });
  var triggers = Array.isArray(spec.triggers) ? spec.triggers : [];
  triggers.forEach(function (t) {
    var type = t.type || t.trigger_type || "manual";
    var fields = ["type", "title", "input_schema", "intake"];
    if (type === "schedule") fields.push("schedule");
    sections.push({ kind: "trigger", id: t.id || t.name || "", type: type, fields: fields });
  });
  var nodes = spec.nodes || {};
  Object.keys(nodes).forEach(function (id) {
    var node = nodes[id] || {};
    var type = node.type || "pass";
    var typeFields = NODE_TYPE_FIELDS[type] || [];
    sections.push({ kind: "node", id: id, type: type, fields: ["type", "title"].concat(typeFields) });
  });
  return sections;
}

export function changeNodeType(spec, nodeId, nextType) {
  var next = JSON.parse(JSON.stringify(spec || {}));
  var nodes = next.nodes || {};
  var node = nodes[nodeId];
  if (!node) return { spec: next, removedFields: [] };

  var oldType = node.type || "pass";
  var oldFields = NODE_TYPE_FIELDS[oldType] || [];
  var newFieldSet = {};
  (NODE_TYPE_FIELDS[nextType] || []).forEach(function (f) { newFieldSet[f] = true; });

  var removedFields = [];
  oldFields.forEach(function (f) {
    if (!newFieldSet[f] && node[f] !== undefined) {
      removedFields.push(f);
      delete node[f];
    }
  });

  node.type = nextType;

  // Add defaults for the new type when the field is missing.
  if (nextType === "agent_task") {
    if (!node.profile) node.profile = "default";
    if (!node.prompt) node.prompt = "Return JSON only matching the result contract.";
    if (!node.result_contract) node.result_contract = { summary: "string", status: "string" };
  } else if (nextType === "wait") {
    if (node.seconds === undefined) node.seconds = 60;
  } else if (nextType === "fail") {
    if (!node.output) node.output = "Workflow failed.";
  } else if (nextType === "switch") {
    if (!node.cases) node.cases = [];
  } else if (nextType === "pass") {
    if (!node.output) node.output = {};
  }

  return { spec: next, removedFields: removedFields.sort() };
}

export function conditionFromForm(form) {
  if (!form || !form.op) return null;
  var op = form.op;
  if (CONDITION_OPS.indexOf(op) === -1) return null;

  if (op === "exists" || op === "missing") {
    var path = form.path || "";
    if (!path) return null;
    return { op: op, path: path };
  }

  var leftPath = form.leftPath || "";
  var rightValue = form.rightValue;
  if (!leftPath) return null;
  if (rightValue === undefined || rightValue === null) return null;

  return { op: op, left: { path: leftPath }, right: rightValue };
}

export function resultContractFromRows(rows) {
  var contract = {};
  (rows || []).forEach(function (row) {
    var key = (row && row.key || "").trim();
    var type = (row && row.type || "").trim();
    if (!key || !type) return;
    if (RESULT_CONTRACT_PRIMITIVES.indexOf(type) !== -1) {
      contract[key] = type;
    } else if (type.indexOf("|") !== -1) {
      var parts = type.split("|").map(function (s) { return s.trim(); }).filter(Boolean);
      if (parts.length >= 2) contract[key] = type;
    }
  });
  return contract;
}

export function workflowIdFromText(value) {
  var text = String(value || "workflow draft").trim().toLowerCase();
  var slug = text.replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 64);
  if (slug && /^[a-z]/.test(slug)) return slug;
  return "workflow_draft";
}

export function inputRowsFromTrigger(trigger) {
  var schema = trigger && trigger.input_schema && typeof trigger.input_schema === "object" && !Array.isArray(trigger.input_schema) ? trigger.input_schema : {};
  return Object.keys(schema).sort().map(function (name) {
    var field = schema[name] && typeof schema[name] === "object" && !Array.isArray(schema[name]) ? schema[name] : {};
    var kind = String(field.kind || "text");
    if (kind === "object" || kind === "json" || kind === "document") kind = "text";
    if (SCALAR_INPUT_KINDS.indexOf(kind) === -1) kind = "text";
    return {
      name: name,
      kind: kind,
      required: !!field.required,
      defaultValue: field.default === undefined || field.default === null ? "" : String(field.default),
      minLength: field.min_length === undefined || field.min_length === null ? "" : String(field.min_length),
      maxLength: field.max_length === undefined || field.max_length === null ? "" : String(field.max_length),
    };
  });
}

export function inputSchemaFromRows(rows) {
  var schema = {};
  (rows || []).forEach(function (row) {
    var name = workflowIdFromText(row && row.name ? row.name : "").replace(/-/g, "_");
    if (!name) return;
    var kind = SCALAR_INPUT_KINDS.indexOf(row.kind) !== -1 ? row.kind : "text";
    var field = { kind: kind };
    if (row.required) field.required = true;
    if (row.defaultValue !== undefined && String(row.defaultValue).trim() !== "") field.default = String(row.defaultValue);
    var minLength = String(row.minLength || "").trim();
    var maxLength = String(row.maxLength || "").trim();
    if (minLength) field.min_length = Math.max(0, parseInt(minLength, 10) || 0);
    if (maxLength) field.max_length = Math.max(0, parseInt(maxLength, 10) || 0);
    schema[name] = field;
  });
  return schema;
}

export function triggerIntakeFromForm(mode, dedupeKey, readyPath) {
  var safeMode = mode === "continuous" ? "continuous" : "single";
  var intake = { mode: safeMode };
  var dedupe = String(dedupeKey || "").trim();
  var ready = String(readyPath || "").trim();
  if (dedupe) intake.dedupe_key = dedupe;
  if (ready) intake.ready_when = { op: "exists", path: ready };
  return intake;
}

export function readyPathFromTrigger(trigger) {
  var cond = trigger && trigger.intake && trigger.intake.ready_when;
  if (!cond || typeof cond !== "object" || Array.isArray(cond)) return "";
  if (String(cond.op || "") === "exists" && typeof cond.path === "string") return cond.path;
  return "";
}
