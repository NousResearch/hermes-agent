import fs from "fs";
import path from "path";

const targetPath = "node_modules/@spectrum-ts/imessage/dist/index.js";
let source = fs.readFileSync(targetPath, "utf8");

// Handle both CRLF and LF
const isCRLF = source.includes("\r\n");
const normalized = isCRLF ? source.split("\r\n").join("\n") : source;

const oldCode = `const toCachedPoll = (input) => {
\tconst poll = asPoll({
\t\ttitle: input.title,
\t\toptions: input.options.map((optionInfo) => ({ title: optionInfo.text }))
\t});
\tconst optionsByIdentifier = /* @__PURE__ */ new Map();
\tfor (const [index, optionInfo] of input.options.entries()) {
\t\tconst option = poll.options[index];
\t\tif (option && optionInfo.optionIdentifier) optionsByIdentifier.set(optionInfo.optionIdentifier, option);
\t}
\treturn {
\t\tpoll,
\t\toptionsByIdentifier
\t};
};`;

const newCode = `const toCachedPoll = (input) => {
\tconst pollTitle = (input && input.title && String(input.title).trim()) ? String(input.title).trim() : "投票";
\tconst rawOpts = (input && Array.isArray(input.options)) ? input.options : [];
\tconst mappedOpts = rawOpts.map((optionInfo) => ({
\t\ttitle: (optionInfo && (optionInfo.text || optionInfo.title) && String(optionInfo.text || optionInfo.title).trim()) ? String(optionInfo.text || optionInfo.title).trim() : "选项"
\t}));
\twhile (mappedOpts.length < 2) {
\t\tmappedOpts.push({ title: \`选项 \${mappedOpts.length + 1}\` });
\t}
\tconst poll = asPoll({
\t\ttitle: pollTitle,
\t\toptions: mappedOpts
\t});
\tconst optionsByIdentifier = /* @__PURE__ */ new Map();
\tfor (const [index, optionInfo] of rawOpts.entries()) {
\t\tconst option = poll.options[index];
\t\tif (option && optionInfo && optionInfo.optionIdentifier) optionsByIdentifier.set(optionInfo.optionIdentifier, option);
\t}
\treturn {
\t\tpoll,
\t\toptionsByIdentifier
\t};
};`;

if (!normalized.includes(oldCode)) {
  console.log("oldCode not found in file (already patched or different version)");
} else {
  let patched = normalized.replace(oldCode, newCode);
  if (isCRLF) {
    patched = patched.split("\n").join("\r\n");
  }
  fs.writeFileSync(targetPath, patched, "utf8");
  console.log("Successfully patched toCachedPoll in", targetPath);
}
