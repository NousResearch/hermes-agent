#!/usr/bin/env node
// Patch spectrum-ts' iMessage inbound mapper until upstream preserves mixed
// text + attachment Apple events. The current spectrum-ts mapper returns only
// buildAttachmentMessage(...) whenever attachments are present, which drops
// event.message.content.text before Hermes can see it.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MARKER = "Hermes patch: Preserve mixed text + attachment iMessage payloads";

function scriptDir() {
  return path.dirname(fileURLToPath(import.meta.url));
}

function replaceOnce(source, from, to, label) {
  const count = source.split(from).length - 1;
  if (count !== 1) {
    throw new Error(`expected exactly one ${label} match, found ${count}`);
  }
  return source.replace(from, to);
}

function replaceFirst(source, from, to, label) {
  if (!source.includes(from)) {
    throw new Error(`expected at least one ${label} match, found 0`);
  }
  return source.replace(from, to);
}

function addTextChildSnippet(messageExpr) {
  return `if (text2) {\n      items.unshift({\n        ...base,\n        id: formatChildId(0, messageGuidStr),\n        content: asText(text2),\n        partIndex: 0,\n        parentId: messageGuidStr\n      });\n    }`;
}

function patchRebuild(source) {
  source = replaceOnce(
    source,
    `  const attachments = messageAttachments(message);\n  if (attachments.length === 1) {`,
    `  const attachments = messageAttachments(message);\n  const text2 = message.content.text;\n  if (attachments.length === 1) {`,
    "rebuild text capture"
  );
  source = replaceOnce(
    source,
    `    return buildAttachmentMessage(client, base, info, messageGuidStr, 0);`,
    `    const msg2 = await buildAttachmentMessage(\n      client,\n      base,\n      info,\n      text2 ? formatChildId(1, messageGuidStr) : messageGuidStr,\n      text2 ? 1 : 0,\n      text2 ? messageGuidStr : void 0\n    );\n    if (text2) {\n      const textMsg = {\n        ...base,\n        id: formatChildId(0, messageGuidStr),\n        content: asText(text2),\n        partIndex: 0,\n        parentId: messageGuidStr\n      };\n      return {\n        ...base,\n        id: messageGuidStr,\n        content: asProviderGroup([textMsg, msg2])\n      };\n    }\n    return msg2;`,
    "rebuild single attachment"
  );
  source = replaceFirst(
    source,
    `          formatChildId(i, messageGuidStr),\n          i,\n          messageGuidStr`,
    `          formatChildId(text2 ? i + 1 : i, messageGuidStr),\n          text2 ? i + 1 : i,\n          messageGuidStr`,
    "rebuild multi attachment child index"
  );
  source = replaceFirst(
    source,
    `    return {\n      ...base,\n      id: messageGuidStr,\n      content: asProviderGroup(items)\n    };\n  }\n  if (getBalloonBundleId(message) === URL_BALLOON_BUNDLE_ID) {`,
    `    ${addTextChildSnippet("message")}\n    return {\n      ...base,\n      id: messageGuidStr,\n      content: asProviderGroup(items)\n    };\n  }\n  if (getBalloonBundleId(message) === URL_BALLOON_BUNDLE_ID) {`,
    "rebuild multi attachment text child"
  );
  source = replaceFirst(
    source,
    `  const text2 = message.content.text;\n  return {\n    ...base,`,
    `  return {\n    ...base,`,
    "rebuild duplicate text declaration"
  );
  return source;
}

function patchInbound(source) {
  source = replaceOnce(
    source,
    `  const attachments = messageAttachments(event.message);\n  if (attachments.length === 1) {`,
    `  const attachments = messageAttachments(event.message);\n  const text2 = event.message.content.text;\n  if (attachments.length === 1) {`,
    "inbound text capture"
  );
  source = replaceOnce(
    source,
    `      messageGuidStr,\n      0\n    );\n    cacheMessage(cache, msg2);\n    return [msg2];`,
    `      text2 ? formatChildId(1, messageGuidStr) : messageGuidStr,\n      text2 ? 1 : 0,\n      text2 ? messageGuidStr : void 0\n    );\n    if (text2) {\n      const textMsg = {\n        ...base,\n        id: formatChildId(0, messageGuidStr),\n        content: asText(text2),\n        partIndex: 0,\n        parentId: messageGuidStr\n      };\n      const parent = {\n        ...base,\n        id: messageGuidStr,\n        content: asProviderGroup([textMsg, msg2])\n      };\n      cacheMessage(cache, parent);\n      return [parent];\n    }\n    cacheMessage(cache, msg2);\n    return [msg2];`,
    "inbound single attachment"
  );
  source = replaceOnce(
    source,
    `          formatChildId(i, messageGuidStr),\n          i,\n          messageGuidStr`,
    `          formatChildId(text2 ? i + 1 : i, messageGuidStr),\n          text2 ? i + 1 : i,\n          messageGuidStr`,
    "inbound multi attachment child index"
  );
  source = replaceOnce(
    source,
    `    const parent = {\n      ...base,\n      id: messageGuidStr,\n      content: asProviderGroup(items)\n    };`,
    `    ${addTextChildSnippet("event.message")}\n    const parent = {\n      ...base,\n      id: messageGuidStr,\n      content: asProviderGroup(items)\n    };`,
    "inbound multi attachment text child"
  );
  source = replaceOnce(
    source,
    `  const text2 = event.message.content.text;\n  const msg = {`,
    `  const msg = {`,
    "inbound duplicate text declaration"
  );
  return source;
}

export function patchSpectrumTs(root = scriptDir()) {
  const dist = path.join(root, "node_modules", "spectrum-ts", "dist");
  if (!fs.existsSync(dist)) {
    throw new Error(`spectrum-ts dist not found: ${dist}`);
  }
  const files = fs.readdirSync(dist)
    .filter((name) => name.endsWith(".js"))
    .map((name) => path.join(dist, name));

  for (const file of files) {
    const raw = fs.readFileSync(file, "utf8");
    if (raw.includes(MARKER)) {
      return { patched: false, file, reason: "already patched" };
    }
    // Normalize to LF for matching so the patch works regardless of the
    // checkout's line-ending style (Windows git autocrlf produces CRLF,
    // which would otherwise defeat the \n-based search strings). The
    // original EOL style is restored on write.
    const CR = String.fromCharCode(13);
    const CRLF = CR + "\n";
    const usedCRLF = raw.includes(CRLF);
    const original = usedCRLF ? raw.split(CRLF).join("\n") : raw;
    if (!original.includes("var toInboundMessages = async") ||
        !original.includes("var rebuildFromAppleMessage = async")) {
      continue;
    }
    let patched = original;
    patched = patchRebuild(patched);
    patched = patchInbound(patched);
    patched = `// ${MARKER}\n${patched}`;
    if (usedCRLF) {
      patched = patched.split("\n").join(CRLF);
    }
    fs.writeFileSync(file, patched, "utf8");
    return { patched: true, file };
  }
  throw new Error("could not find spectrum-ts iMessage inbound chunk to patch");
}

const _invokedDirectly =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (_invokedDirectly) {
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumTs(root);
    const action = result.patched ? "patched" : "ok";
    console.error(`photon-sidecar: spectrum mixed attachment patch ${action}: ${result.file}`);
  } catch (err) {
    console.error(`photon-sidecar: spectrum mixed attachment patch failed: ${err?.stack || err}`);
    process.exit(1);
  }
}
