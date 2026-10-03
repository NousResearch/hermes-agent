// Behavioral probe for the blocked-task action section in the shipped dashboard bundle.
// Extracts and runs the real components with a minimal Preact-compatible h() stub.
const fs = require("fs");

const bundlePath = process.argv[2];
const src = fs.readFileSync(bundlePath, "utf8");

function extractFunction(name) {
  const start = src.indexOf(`function ${name}`);
  if (start === -1) {
    console.error(`${name} not found in bundle`);
    process.exit(1);
  }
  const bodyStart = src.indexOf("{", start);
  let depth = 0;
  let end = bodyStart;
  for (; end < src.length; end++) {
    if (src[end] === "{") depth++;
    else if (src[end] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  return src.slice(start, end + 1);
}

function h(type, props, ...children) {
  if (typeof type === "function") return type(props || {});
  return { type, props: props || {}, children };
}
global.h = h;

// Safe in this test probe: evaluate only named functions extracted from the checked-in bundle.
eval(extractFunction("BlockActionSection"));
eval(extractFunction("TaskBlockActionSection"));

function textOf(node) {
  if (node == null || node === false) return [];
  if (Array.isArray(node)) return node.flatMap(textOf);
  if (typeof node === "string" || typeof node === "number") return [String(node)];
  return textOf(node.children);
}

const internalContract = {
  disposition: "Internal owner action",
  owner: "systems",
  action: "Repair the worker lease",
  action_required: false,
  reply_format: "No reply required.",
  consequence_if_no_action: "Recovery remains blocked.",
  next_action: "Retry after repair.",
  retry_condition: "Worker lease is healthy.",
  auto_resume: true,
};
const internalTree = TaskBlockActionSection({
  task: { status: "blocked", block_action: internalContract },
});
const internalText = textOf(internalTree);
const requiredInternalText = [
  "Required action",
  "No action needed from Matt",
  "Disposition", "Internal owner action",
  "Owner", "systems",
  "Required action", "Repair the worker lease",
  "Reply format", "No reply required.",
  "If no action", "Recovery remains blocked.",
  "Next action", "Retry after repair.",
  "Retry condition", "Worker lease is healthy.",
  "Auto-resume", "yes",
];
for (const text of requiredInternalText) {
  if (!internalText.includes(text)) {
    console.error(`FAIL: internal block drawer omitted ${JSON.stringify(text)}: ${JSON.stringify(internalText)}`);
    process.exit(1);
  }
}

const mattTree = TaskBlockActionSection({
  task: {
    status: "blocked",
    block_action: {
      ...internalContract,
      disposition: "Matt action required",
      owner: "Matt",
      action: "Choose A or B",
      action_required: true,
      reply_format: "Reply with A or B.",
      auto_resume: false,
    },
  },
});
const mattText = textOf(mattTree);
if (mattText.includes("No action needed from Matt")) {
  console.error("FAIL: Matt-action block rendered the no-action banner");
  process.exit(1);
}
if (!mattText.includes("Reply with A or B.")) {
  console.error("FAIL: Matt-action block omitted its reply format");
  process.exit(1);
}

if (TaskBlockActionSection({ task: { status: "running", block_action: internalContract } }) !== null) {
  console.error("FAIL: non-blocked task rendered a blocked-action section");
  process.exit(1);
}

console.log("PASS");
