#!/usr/bin/env node
// Apply narrow compatibility fixes to spectrum-ts' bundled iMessage mapper.
// Older releases return only
// buildAttachmentMessage(...) whenever attachments are present, which drops
// `message.content.text` before Hermes can see it. We rewrite the two inbound
// mappers — `rebuildFromAppleMessage` (used by `space.getMessage`) and
// `toInboundMessages` (used by the live stream) — so a bubble carrying both
// text and attachment(s) surfaces as a group whose first child is the typed
// text. Paths with no text are rewritten to byte-identical behavior, so only
// mixed text+attachment messages change shape.
//
// Since spectrum-ts 5.x split the SDK into scoped packages, the iMessage mapper
// lives in `@spectrum-ts/imessage/dist/index.js` (it used to be a chunk under
// `spectrum-ts/dist`). The published output is tab-indented and uses
// `const ... = async` declarations; the anchors below match that exactly and
// fail loudly if a future spectrum-ts reshapes a path that still needs repair.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MARKER = "Hermes patch: Preserve mixed text + attachment iMessage payloads";
const POLL_TITLE_MARKER = "Hermes patch: Accept empty inbound iMessage poll titles";
const POLL_CACHE_MARKER = "Hermes patch: Preserve outbound iMessage poll metadata";
const POLL_RECOVERY_MARKER = "Hermes patch: Merge and persist iMessage poll option identifiers";
// Patch names reported by patchSpectrumTs() and the sidecar's /healthz.
const PATCH_MARKERS = {
  mixedAttachments: MARKER,
  pollEmptyTitle: POLL_TITLE_MARKER,
  pollMetadataCache: POLL_CACHE_MARKER,
  pollVoteRecovery: POLL_RECOVERY_MARKER,
};
const POLL_PATCHES = ["pollEmptyTitle", "pollMetadataCache", "pollVoteRecovery"];

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

function replaceExactly(source, from, to, expected, label) {
  const count = source.split(from).length - 1;
  if (count !== expected) {
    throw new Error(
      `expected exactly ${expected} ${label} matches, found ${count}`
    );
  }
  return source.split(from).join(to);
}

// The text-first child of a mixed text+attachment group, indented `tabs` deep
// (the object's closing brace sits at `tabs`; its properties one level in).
function textChild(tabs) {
  const t = "\t".repeat(tabs);
  return (
    `{\n${t}\t...base,\n${t}\tid: formatChildId(0, messageGuidStr),` +
    `\n${t}\tcontent: asText(text2),\n${t}\tpartIndex: 0,` +
    `\n${t}\tparentId: messageGuidStr\n${t}}`
  );
}

function patchRebuild(source) {
  // Capture the bubble text before the attachment branches consume it. The
  // existing no-attachment branch keeps its own `const text` declaration, so a
  // distinct name avoids a redeclaration.
  source = replaceOnce(
    source,
    `\tconst attachments = messageAttachments(message);\n\tif (attachments.length === 1) {`,
    `\tconst attachments = messageAttachments(message);\n\tconst text2 = message.content.text;\n\tif (attachments.length === 1) {`,
    "rebuild text capture"
  );
  // Single attachment: when text is present, push it to slot 0 and the
  // attachment to slot 1, then wrap both in a group.
  source = replaceOnce(
    source,
    `\t\treturn buildAttachmentMessage(client, base, info, messageGuidStr, 0);`,
    `\t\tconst msg2 = await buildAttachmentMessage(client, base, info, text2 ? formatChildId(1, messageGuidStr) : messageGuidStr, text2 ? 1 : 0, text2 ? messageGuidStr : void 0);\n\t\tif (text2) {\n\t\t\tconst textMsg = ${textChild(3)};\n\t\t\treturn {\n\t\t\t\t...base,\n\t\t\t\tid: messageGuidStr,\n\t\t\t\tcontent: asProviderGroup([textMsg, msg2])\n\t\t\t};\n\t\t}\n\t\treturn msg2;`,
    "rebuild single attachment"
  );
  // Multi attachment: prepend the text child to the group's items.
  source = replaceOnce(
    source,
    `\t\treturn {\n\t\t\t...base,\n\t\t\tid: messageGuidStr,\n\t\t\tcontent: asProviderGroup(items)\n\t\t};`,
    `\t\tif (text2) {\n\t\t\titems.unshift(${textChild(3)});\n\t\t}\n\t\treturn {\n\t\t\t...base,\n\t\t\tid: messageGuidStr,\n\t\t\tcontent: asProviderGroup(items)\n\t\t};`,
    "rebuild multi attachment text child"
  );
  return source;
}

function patchInbound(source) {
  source = replaceOnce(
    source,
    `\tconst attachments = messageAttachments(event.message);\n\tif (attachments.length === 1) {`,
    `\tconst attachments = messageAttachments(event.message);\n\tconst text2 = event.message.content.text;\n\tif (attachments.length === 1) {`,
    "inbound text capture"
  );
  source = replaceOnce(
    source,
    `\t\tconst msg = await buildAttachmentMessage(client, base, info, messageGuidStr, 0);\n\t\tcacheMessage(cache, msg);\n\t\treturn [msg];`,
    `\t\tconst msg = await buildAttachmentMessage(client, base, info, text2 ? formatChildId(1, messageGuidStr) : messageGuidStr, text2 ? 1 : 0, text2 ? messageGuidStr : void 0);\n\t\tif (text2) {\n\t\t\tconst textMsg = ${textChild(3)};\n\t\t\tconst parent = {\n\t\t\t\t...base,\n\t\t\t\tid: messageGuidStr,\n\t\t\t\tcontent: asProviderGroup([textMsg, msg])\n\t\t\t};\n\t\t\tcacheMessage(cache, parent);\n\t\t\treturn [parent];\n\t\t}\n\t\tcacheMessage(cache, msg);\n\t\treturn [msg];`,
    "inbound single attachment"
  );
  source = replaceOnce(
    source,
    `\t\tconst parent = {\n\t\t\t...base,\n\t\t\tid: messageGuidStr,\n\t\t\tcontent: asProviderGroup(items)\n\t\t};`,
    `\t\tif (text2) {\n\t\t\titems.unshift(${textChild(3)});\n\t\t}\n\t\tconst parent = {\n\t\t\t...base,\n\t\t\tid: messageGuidStr,\n\t\t\tcontent: asProviderGroup(items)\n\t\t};`,
    "inbound multi attachment text child"
  );
  return source;
}

// Shift attachment part indices by one when a text child occupies slot 0. The
// push line is byte-identical in both mappers, so patch both occurrences.
function patchChildIndices(source) {
  return replaceExactly(
    source,
    `items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));`,
    `items.push(await buildAttachmentMessage(client, base, info, formatChildId(text2 ? i + 1 : i, messageGuidStr), text2 ? i + 1 : i, messageGuidStr));`,
    2,
    "multi attachment child index"
  );
}

function patchEmptyPollTitles(source) {
  if (source.includes(POLL_TITLE_MARKER)) {
    return source;
  }
  // Photon currently returns an empty poll title in both the live `created`
  // delta and `client.polls.get()`, even when the outbound poll had a title.
  // spectrum-ts validates the reconstructed Poll before it can map a vote's
  // option identifier, so the empty title drops every selection. The title is
  // internal metadata on this inbound path; Hermes only consumes the selected
  // option title. Supply a non-empty placeholder and preserve the option map.
  const anchor = `\tconst poll = asPoll({\n\t\ttitle: input.title,`;
  return replaceOnce(
    source,
    anchor,
    `\t// ${POLL_TITLE_MARKER}\n\tconst poll = asPoll({\n\t\ttitle: input.title || "Poll",`,
    "empty inbound poll title"
  );
}

function patchPollMetadataCache(source) {
  if (source.includes(POLL_CACHE_MARKER)) {
    return source;
  }
  if (!source.includes("const toCachedPoll =")) {
    return source;
  }

  // Spectrum 12.7 only populates its poll cache from inbound events and
  // polls.get(). Photon can omit option identifiers from both, while the
  // polls.create() response still has the identifiers needed to turn a later
  // vote id into its choice title. Seed the same per-client cache on send and
  // do not let a less-complete created/refresh payload overwrite it.
  source = replaceOnce(
    source,
    `const cachePollInfo = (cache, info) => {\n\tconst cached = toCachedPoll(info);\n\tcache.set(info.pollMessageGuid, cached);\n\treturn cached;\n};`,
    `// ${POLL_CACHE_MARKER}\nconst cachePoll = (cache, id, info) => {\n\tconst cached = toCachedPoll(info);\n\tconst existing = cache.get(id);\n\tif (existing && existing.optionsByIdentifier.size > cached.optionsByIdentifier.size) return existing;\n\tcache.set(id, cached);\n\treturn cached;\n};\nconst cachePollInfo = (cache, info) => cachePoll(cache, info.pollMessageGuid, info);`,
    "poll metadata cache helper"
  );
  source = replaceOnce(
    source,
    `\t\tconst cached = toCachedPoll({\n\t\t\ttitle: event.delta.title,\n\t\t\toptions: event.delta.options\n\t\t});\n\t\tcache.set(event.pollMessageGuid, cached);\n\t\treturn cached;`,
    `\t\treturn cachePoll(cache, event.pollMessageGuid, {\n\t\t\ttitle: event.delta.title,\n\t\t\toptions: event.delta.options\n\t\t});`,
    "inbound poll metadata cache"
  );
  source = replaceOnce(
    source,
    `\t\tcase "poll":\n\t\t\tif (replyTo) throw unsupportedRemoteContent("poll", "polls cannot be sent as replies");\n\t\t\treturn outboundPoll(spaceId, await remote.polls.create(chat, content.title, content.options.map((option) => option.title)), content);`,
    `\t\tcase "poll": {\n\t\t\tif (replyTo) throw unsupportedRemoteContent("poll", "polls cannot be sent as replies");\n\t\t\tconst created = await remote.polls.create(chat, content.title, content.options.map((option) => option.title));\n\t\t\tcachePollInfo(getPollCache(remote), { ...created, title: created.title || content.title });\n\t\t\treturn outboundPoll(spaceId, created, content);\n\t\t}`,
    "outbound poll metadata cache"
  );
  source = replaceOnce(
    source,
    `\tconst pollCache = getPollCache(clients);`,
    `\t// ${POLL_CACHE_MARKER}: share outbound metadata with this line's stream.`,
    "shared poll cache declaration"
  );
  return replaceOnce(
    source,
    `\t\treturn clientStream(entry.client, pollCache, entry.phone, includeGroupEvents, tracker ? contactShareHandler(tracker, profileSyncGate) : void 0, recover);`,
    `\t\treturn clientStream(entry.client, getPollCache(entry.client), entry.phone, includeGroupEvents, tracker ? contactShareHandler(tracker, profileSyncGate) : void 0, recover);`,
    "per-client poll stream cache"
  );
}

// Runs after the metadata-cache patch, so it applies to pristine files and to
// files an older Hermes already patched. Photon can omit identifiers from a
// later created/optionAdded/get payload; merge instead of replacing so known
// identifiers survive. The sidecar installs `globalThis.__hermesPhotonPollSeeds`
// (poll-seeds.mjs) to keep send-time identifiers across restarts.
function patchPollVoteRecovery(source) {
  if (source.includes(POLL_RECOVERY_MARKER)) {
    return source;
  }
  source = replaceOnce(
    source,
    `const cachePoll = (cache, id, info) => {\n\tconst cached = toCachedPoll(info);\n\tconst existing = cache.get(id);\n\tif (existing && existing.optionsByIdentifier.size > cached.optionsByIdentifier.size) return existing;\n\tcache.set(id, cached);\n\treturn cached;\n};`,
    `const cachePoll = (cache, id, info) => {\n\t// ${POLL_RECOVERY_MARKER}\n\tconst existing = cache.get(id) ?? globalThis.__hermesPhotonPollSeeds?.get(id);\n\tconst cached = toCachedPoll(existing && !info.title ? { ...info, title: existing.poll.title } : info);\n\tfor (const [identifier, option] of existing?.optionsByIdentifier ?? []) {\n\t\tif (cached.optionsByIdentifier.has(identifier)) continue;\n\t\tcached.optionsByIdentifier.set(identifier, cached.poll.options.find((choice) => choice.title === option.title) ?? option);\n\t}\n\tcache.set(id, cached);\n\treturn cached;\n};`,
    "poll identifier merge"
  );
  source = replaceOnce(
    source,
    `\t\t\tcachePollInfo(getPollCache(remote), { ...created, title: created.title || content.title });\n`,
    `\t\t\tcachePollInfo(getPollCache(remote), { ...created, title: created.title || content.title });\n\t\t\tglobalThis.__hermesPhotonPollSeeds?.remember(created.pollMessageGuid, created.title || content.title, created.options);\n`,
    "outbound poll seed persistence"
  );
  return replaceOnce(
    source,
    `\t\tif (refreshed) cached = refreshed;\n\t}\n\tconst message = buildPollOptionMessage({`,
    `\t\tif (refreshed) cached = refreshed;\n\t}\n\tif (!cached.optionsByIdentifier.has(optionId)) console.error(\`photon-sidecar: WARNING poll vote dropped: unknown option identifier for poll \${event.pollMessageGuid}\`);\n\tconst message = buildPollOptionMessage({`,
    "unknown poll option warning"
  );
}

function patchStates(sources) {
  const states = {};
  for (const [name, marker] of Object.entries(PATCH_MARKERS)) {
    states[name] = sources.some((source) => source.includes(marker)) ? "applied" : "missing";
  }
  if (states.mixedAttachments === "missing" && sources.some(upstreamPreservesMixed)) {
    states.mixedAttachments = "not-needed";
  }
  if (!sources.some((source) => source.includes("asPoll("))) {
    for (const name of POLL_PATCHES) states[name] = "not-needed";
  }
  return states;
}

// spectrum-ts 12.x replaced the attachment-only branches with
// `buildUnwrappedContentMessage` + `toOrderedParts`, which already emits a
// group containing both text and attachments. There is nothing left for
// Hermes to patch; keep the legacy v8 path for older pinned installs.
function upstreamPreservesMixed(source) {
  return source.includes("const buildUnwrappedContentMessage = async") &&
    source.includes("const parts = toOrderedParts(message.content.text, attachments);");
}

// Write every changed chunk to a temp file first, then rename each into place,
// so a crash or full disk never leaves a truncated SDK module.
function writeAtomically(writes) {
  const temps = [];
  try {
    for (const { file, patched } of writes) {
      const temp = `${file}.hermes-${process.pid}.tmp`;
      fs.writeFileSync(temp, patched, "utf8");
      temps.push([temp, file]);
    }
    for (const [temp, file] of temps) fs.renameSync(temp, file);
  } finally {
    for (const [temp] of temps) fs.rmSync(temp, { force: true });
  }
}

export function patchSpectrumTs(root = scriptDir()) {
  const dist = path.join(
    root,
    "node_modules",
    "@spectrum-ts",
    "imessage",
    "dist"
  );
  if (!fs.existsSync(dist)) {
    throw new Error(`@spectrum-ts/imessage dist not found: ${dist}`);
  }
  const files = fs.readdirSync(dist)
    .filter((name) => name.endsWith(".js"))
    .sort()
    .map((name) => path.join(dist, name));

  const writes = [];
  const originals = [];
  const finals = [];
  let firstCandidate;
  try {
    for (const file of files) {
      const raw = fs.readFileSync(file, "utf8");
      // Normalize to LF for matching so the patch works regardless of the
      // checkout's line-ending style (Windows git autocrlf produces CRLF,
      // which would otherwise defeat the \n-based search strings). The
      // original EOL style is restored on write. Indentation in the published
      // tarball is tabs; the anchors match that directly.
      const CR = String.fromCharCode(13);
      const CRLF = CR + "\n";
      const usedCRLF = raw.includes(CRLF);
      const original = usedCRLF ? raw.split(CRLF).join("\n") : raw;
      const hasMixedMapper =
        original.includes("const toInboundMessages = async") &&
        original.includes("const rebuildFromAppleMessage = async");
      const hasPollMapper = original.includes("const toCachedPoll =");
      const hasPolls = original.includes("asPoll(");
      if (!hasMixedMapper && !hasPollMapper && !hasPolls) {
        continue;
      }
      firstCandidate ??= file;
      originals.push(original);

      let patched = original;
      if (hasPollMapper) {
        patched = patchEmptyPollTitles(patched);
        patched = patchPollMetadataCache(patched);
        if (patched.includes(POLL_CACHE_MARKER)) {
          patched = patchPollVoteRecovery(patched);
        }
      }
      if (hasMixedMapper && !original.includes(MARKER) && !upstreamPreservesMixed(original)) {
        patched = patchRebuild(patched);
        patched = patchInbound(patched);
        patched = patchChildIndices(patched);
        patched = `// ${MARKER}\n${patched}`;
      }
      finals.push(patched);
      if (patched !== original) {
        writes.push({ file, patched: usedCRLF ? patched.split("\n").join(CRLF) : patched });
      }
    }
    if (!firstCandidate) {
      throw new Error("could not find @spectrum-ts/imessage iMessage inbound chunk to patch");
    }
    // An SDK that still builds polls but renamed the mapper would otherwise
    // pass silently and keep dropping votes from empty-title polls.
    const states = patchStates(finals);
    const missing = POLL_PATCHES.filter((name) => states[name] === "missing");
    if (missing.length) {
      throw new Error(
        `poll vote patch missing (${missing.join(", ")}): @spectrum-ts/imessage builds polls ` +
        "but no known poll mapper was found; inbound votes may be dropped"
      );
    }
    writeAtomically(writes);
    if (writes.length > 0) {
      return { patched: true, file: writes[0].file, patches: states };
    }
    return {
      patched: false,
      file: firstCandidate,
      reason: "already patched or not needed",
      patches: states,
    };
  } catch (err) {
    // Nothing was written: report what the files on disk already carry.
    err.patches = patchStates(originals);
    throw err;
  }
}

const _invokedDirectly =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (_invokedDirectly) {
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumTs(root);
    const action = result.patched ? "patched" : "ok";
    console.error(`photon-sidecar: spectrum patch ${action}: ${result.file} ${JSON.stringify(result.patches)}`);
  } catch (err) {
    console.error(`photon-sidecar: WARNING spectrum patch failed ${JSON.stringify(err?.patches ?? {})}: ${err?.stack || err}`);
    process.exit(1);
  }
}
