"""Regression tests for Hermes' Spectrum mixed text+attachment workaround."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import textwrap
import time
import urllib.request
from pathlib import Path


_PATCHER = Path("plugins/platforms/photon/sidecar/patch-spectrum-mixed-attachments.mjs")


def _sidecar_env(port: int) -> dict[str, str]:
    return {
        **os.environ,
        "PHOTON_PROJECT_ID": "test-project",
        "PHOTON_PROJECT_SECRET": "test-secret",
        "PHOTON_SIDECAR_PORT": str(port),
        "PHOTON_SIDECAR_TOKEN": "test-token",
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_sidecar_fixture(tmp_path: Path, *, sdk_available: bool) -> Path:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    shutil.copyfile("plugins/platforms/photon/sidecar/index.mjs", sidecar / "index.mjs")
    # index.mjs imports sibling helper modules — copy every non-patch .mjs so
    # the fixture keeps working as helpers are extracted from index.mjs.
    for helper in Path("plugins/platforms/photon/sidecar").glob("*.mjs"):
        if helper.name in ("index.mjs", "patch-spectrum-mixed-attachments.mjs"):
            continue
        shutil.copyfile(helper, sidecar / helper.name)
    (sidecar / "patch-spectrum-mixed-attachments.mjs").write_text(
        "export function patchSpectrumTs() { throw new Error('forced patch failure'); }\n",
        encoding="utf-8",
    )

    if not sdk_available:
        return sidecar

    package = sidecar / "node_modules" / "spectrum-ts"
    (package / "providers").mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "spectrum-ts",
                "type": "module",
                "exports": {
                    ".": "./index.js",
                    "./providers/imessage": "./providers/imessage.js",
                },
            }
        ),
        encoding="utf-8",
    )
    (package / "index.js").write_text(
        textwrap.dedent(
            """
            export async function Spectrum() {
              return {
                messages: { [Symbol.asyncIterator]() { return { next: () => new Promise(() => {}) }; } },
                stop: async () => undefined,
              };
            }
            export const attachment = value => value;
            export const voice = value => value;
            export const text = value => value;
            export const markdown = value => value;
            export const typing = value => value;
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package / "providers" / "imessage.js").write_text(
        "export function imessage() { return {}; }\nimessage.config = () => ({});\n",
        encoding="utf-8",
    )
    return sidecar


def test_sidecar_patch_failure_still_reaches_health_endpoint(tmp_path: Path) -> None:
    """The compatibility patch is optional when the SDK itself remains usable."""
    sidecar = _write_sidecar_fixture(tmp_path, sdk_available=True)
    port = _free_port()
    proc = subprocess.Popen(
        ["node", "index.mjs"],
        cwd=sidecar,
        env=_sidecar_env(port),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/healthz",
        data=b"{}",
        headers={"X-Hermes-Sidecar-Token": "test-token"},
        method="POST",
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with urllib.request.urlopen(request, timeout=0.5) as response:
                    payload = json.load(response)
                break
            except OSError:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

        assert payload["ok"] is True
        assert proc.poll() is None
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=5)

    assert "forced patch failure" in stderr


def _tabify(src: str) -> str:
    """Convert the fixture's two-space indentation to the tab indentation that
    spectrum-ts ships in `@spectrum-ts/imessage/dist`, so the patch anchors
    (which match tabs) apply exactly as they do against a real install."""
    out = []
    for line in src.split("\n"):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        out.append("\t" * (indent // 2) + " " * (indent % 2) + stripped)
    return "\n".join(out)


# An executable slice combining spectrum-ts 8.x's attachment mappers with
# 12.7's poll send/cache/resolve path, plus stubs of their closed-over helpers.
# It mirrors each published shape (including tab indentation) so the anchors
# exercise the dependency patch and exports the functions needed to assert the
# resulting behavior.
_SPECTRUM_IMESSAGE_FIXTURE = """
const asPoll = (input) => {
  if (!input.title) throw new Error("poll title must not be empty");
  return { type: "poll", ...input };
};
const asPollOption = (input) => ({ type: "poll_option", ...input });
class PollCache {
  map = new Map();
  get(id) { return this.map.get(id); }
  set(id, poll) { this.map.set(id, poll); }
}
const pollCaches = new WeakMap();
const getPollCache = (owner) => {
  let cache = pollCaches.get(owner);
  if (!cache) {
    cache = new PollCache();
    pollCaches.set(owner, cache);
  }
  return cache;
};
const outboundPoll = (spaceId, poll, content) => ({ id: poll.pollMessageGuid, content, space: { id: spaceId } });
const unsupportedRemoteContent = () => new Error("unsupported");
const send = async (remote, spaceId, content, replyTo) => {
  const chat = spaceId;
  switch (content.type) {
    case "poll":
      if (replyTo) throw unsupportedRemoteContent("poll", "polls cannot be sent as replies");
      return outboundPoll(spaceId, await remote.polls.create(chat, content.title, content.options.map((option) => option.title)), content);
    default: throw unsupportedRemoteContent(content.type);
  }
};
const toCachedPoll = (input) => {
  const poll = asPoll({
    title: input.title,
    options: input.options.map((optionInfo) => ({ title: optionInfo.text }))
  });
  const optionsByIdentifier = new Map();
  for (const [index, optionInfo] of input.options.entries()) {
    const option = poll.options[index];
    if (option && optionInfo.optionIdentifier) optionsByIdentifier.set(optionInfo.optionIdentifier, option);
  }
  return { poll, optionsByIdentifier };
};
const cachePollInfo = (cache, info) => {
  const cached = toCachedPoll(info);
  cache.set(info.pollMessageGuid, cached);
  return cached;
};
const cachePollEvent = (cache, event) => {
  if (event.delta.type === "created" || event.delta.type === "optionAdded") try {
    const cached = toCachedPoll({
      title: event.delta.title,
      options: event.delta.options
    });
    cache.set(event.pollMessageGuid, cached);
    return cached;
  } catch (e) {}
};
const resolvePoll = async (client, cache, event) => {
  const cached = cache.get(event.pollMessageGuid);
  if (cached) return cached;
  return cachePollInfo(cache, await client.polls.get(event.pollMessageGuid));
};
const buildPollOptionMessage = (input) => {
  const option = input.cached.optionsByIdentifier.get(input.optionId);
  if (!option) return;
  return {
    id: `${input.event.pollMessageGuid}:${input.optionId}`,
    content: asPollOption({ option, poll: input.cached.poll, selected: input.selected })
  };
};
const refreshPollMetadata = async (client, pollCache, event) => {
  const info = await client.polls.get(event.pollMessageGuid);
  if (!info) return;
  cachePollInfo(pollCache, info);
  return pollCache.get(info.pollMessageGuid);
};
const toPollOptionMessage = async (client, pollCache, event) => {
  const optionId = event.delta.optionIdentifier;
  if (!optionId) return [];
  let cached = await resolvePoll(client, pollCache, event);
  if (!cached) return [];
  if (!cached.optionsByIdentifier.has(optionId)) {
    const refreshed = await refreshPollMetadata(client, pollCache, event);
    if (refreshed) cached = refreshed;
  }
  const message = buildPollOptionMessage({
    cached, event, optionId, selected: event.delta.type === "voted"
  });
  return message ? [message] : [];
};
const clientStream = (client, pollCache) => ({ client, pollCache });
const contactShareHandler = () => undefined;
const createStreamGroup = () => ({
  builds: [],
  add(key, build) { this.builds.push(build); }
});
const lineKey = () => "line";
const getCloudRecover = () => undefined;
const isSharedMode = () => false;
const getContactShareTracker = () => undefined;
const messages$1 = (clients, projectConfig, profileSyncGate) => {
  const pollCache = getPollCache(clients);
  const staticShareEnabled = projectConfig?.profile?.imessageSynced === true;
  const recover = getCloudRecover(clients);
  const shared = isSharedMode(clients);
  const includeGroupEvents = !shared;
  const build = (entry) => () => {
    const tracker = staticShareEnabled || profileSyncGate ? getContactShareTracker(entry.client) : void 0;
    return clientStream(entry.client, pollCache, entry.phone, includeGroupEvents, tracker ? contactShareHandler(tracker, profileSyncGate) : void 0, recover);
  };
  const group = createStreamGroup({ label: "imessage.messages" });
  for (const entry of clients) group.add(lineKey(entry), build(entry));
  return group;
};
const formatChildId = (partIndex, parentGuid) => `p:${partIndex}/${parentGuid}`;
const asText = (text) => ({ type: "text", text });
const asCustom = (message) => ({ type: "custom" });
const asProviderGroup = (items) => ({ type: "group", items });
const messageAttachments = (message) => message.content.attachments ?? [];
const buildMessageBase = (message, chatGuidHint, timestamp, phone) => ({ direction: "inbound", sender: { id: "s" }, space: { id: "sp", type: "dm", phone }, timestamp });
const buildAttachmentMessage = async (client, base, info, id, partIndex, parentId) => {
  const msg = { ...base, id, content: { type: "attachment", id: info.guid }, partIndex };
  if (parentId !== void 0) msg.parentId = parentId;
  return msg;
};
const cacheMessage = (cache, message) => { cache.set(message.id, message); };
const rebuildFromAppleMessage = async (client, message, phone, chatGuidHint) => {
  const messageGuidStr = message.guid;
  const base = buildMessageBase(message, chatGuidHint, message.dateCreated ?? /* @__PURE__ */ new Date(), phone);
  const attachments = messageAttachments(message);
  if (attachments.length === 1) {
    const info = attachments[0];
    if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");
    return buildAttachmentMessage(client, base, info, messageGuidStr, 0);
  }
  if (attachments.length > 1) {
    const items = [];
    for (let i = 0; i < attachments.length; i++) {
      const info = attachments[i];
      if (!info) continue;
      items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));
    }
    return {
      ...base,
      id: messageGuidStr,
      content: asProviderGroup(items)
    };
  }
  const text = message.content.text;
  return {
    ...base,
    id: messageGuidStr,
    content: text ? asText(text) : asCustom(message)
  };
};
const toInboundMessages = async (client, cache, event, phone) => {
  const base = buildMessageBase(event.message, event.chatGuid, event.occurredAt, phone);
  const messageGuidStr = event.message.guid;
  const attachments = messageAttachments(event.message);
  if (attachments.length === 1) {
    const info = attachments[0];
    if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");
    const msg = await buildAttachmentMessage(client, base, info, messageGuidStr, 0);
    cacheMessage(cache, msg);
    return [msg];
  }
  if (attachments.length > 1) {
    const items = [];
    for (let i = 0; i < attachments.length; i++) {
      const info = attachments[i];
      if (!info) continue;
      items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));
    }
    const parent = {
      ...base,
      id: messageGuidStr,
      content: asProviderGroup(items)
    };
    cacheMessage(cache, parent);
    return [parent];
  }
  const text = event.message.content.text;
  const msg = {
    ...base,
    id: messageGuidStr,
    content: text ? asText(text) : asCustom(event.message)
  };
  cacheMessage(cache, msg);
  return [msg];
};
export { cachePollEvent, getPollCache, messages$1, rebuildFromAppleMessage, send, toCachedPoll, toInboundMessages, toPollOptionMessage };
"""


def _write_fixture(tmp_path: Path) -> Path:
    dist = tmp_path / "node_modules" / "@spectrum-ts" / "imessage" / "dist"
    dist.mkdir(parents=True)
    chunk = dist / "index.js"
    chunk.write_text(_tabify(_SPECTRUM_IMESSAGE_FIXTURE), encoding="utf-8")
    return chunk


def test_spectrum_patch_rewrites_the_imessage_mapper(tmp_path: Path) -> None:
    """The dependency patch must apply to the 8.x `@spectrum-ts/imessage` chunk
    and rewrite both inbound mappers to thread text through attachment bubbles."""
    chunk = _write_fixture(tmp_path)

    result = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    patched = chunk.read_text(encoding="utf-8")
    assert "Preserve mixed text + attachment iMessage payloads" in patched
    # Single-attachment bubbles wrap the text + attachment in a group...
    assert "content: asProviderGroup([textMsg, msg2])" in patched  # rebuild
    assert "content: asProviderGroup([textMsg, msg])" in patched  # inbound
    # ...multi-attachment bubbles keep the group and shift attachment indices.
    assert "content: asProviderGroup(items)" in patched
    assert "formatChildId(text2 ? i + 1 : i, messageGuidStr)" in patched
    # The text is captured in both mappers before the attachment branches run.
    assert "const text2 = message.content.text;" in patched
    assert "const text2 = event.message.content.text;" in patched

    # Photon can return an empty title for a poll that was sent with one. The
    # patched mapper must still reconstruct its option-id map so a vote reaches
    # the clarify response path.
    probe = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            (
                f"import {{cachePollEvent,messages$1,send,toCachedPoll,toPollOptionMessage}} from {json.dumps(chunk.as_uri())};"
                "const options=["
                "{text:'Route',optionIdentifier:'choice-1'},"
                "{text:'Calendar',optionIdentifier:'choice-2'}];"
                "const empty=toCachedPoll({title:'',options});"
                "const missing=toCachedPoll({options});"
                "const named=toCachedPoll({title:'Question?',options});"
                "const remote={polls:{"
                "create:async()=>({pollMessageGuid:'poll-1',title:'',options}),"
                "get:async()=>({pollMessageGuid:'poll-1',title:'',options:options.map(o=>({text:o.text,optionIdentifier:''}))})}};"
                "await send(remote,'space-1',{type:'poll',title:'Question?',options:named.poll.options});"
                "const streams=messages$1([{client:remote,phone:'line-1'}],{});"
                "const cache=streams.builds[0]().pollCache;"
                "cachePollEvent(cache,{pollMessageGuid:'poll-1',delta:{type:'created',title:'',"
                "options:options.map(o=>({text:o.text,optionIdentifier:''}))}});"
                "const [vote]=await toPollOptionMessage(remote,cache,{pollMessageGuid:'poll-1',"
                "delta:{type:'voted',optionIdentifier:'choice-1'}});"
                "console.log(JSON.stringify({emptyTitle:empty.poll.title,"
                "missingTitle:missing.poll.title,namedTitle:named.poll.title,"
                "choice:empty.optionsByIdentifier.get('choice-1').title,"
                "voteTitle:vote.content.option.title,votePollTitle:vote.content.poll.title}));"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout) == {
        "emptyTitle": "Poll",
        "missingTitle": "Poll",
        "namedTitle": "Question?",
        "choice": "Route",
        "voteTitle": "Route",
        "votePollTitle": "Question?",
    }

    # Re-running is a no-op (idempotent self-heal on every sidecar start).
    again = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert chunk.read_text(encoding="utf-8") == patched


def test_spectrum_patch_rejects_unknown_poll_mapper_without_partial_writes(
    tmp_path: Path,
) -> None:
    """A valid earlier chunk must not hide or be written before an unknown poll shape."""
    dist = tmp_path / "node_modules" / "@spectrum-ts" / "imessage" / "dist"
    dist.mkdir(parents=True)
    valid = dist / "a-valid.js"
    valid_source = _tabify(_SPECTRUM_IMESSAGE_FIXTURE)
    valid.write_text(valid_source, encoding="utf-8")
    poll = dist / "b-unknown-poll.js"
    unknown_source = _tabify(
        """
const toCachedPoll = (input) => {
  const poll = asPoll({
    title: normalizeTitle(input.title),
    options: input.options.map((optionInfo) => ({ title: optionInfo.text }))
  });
  return { poll };
};
"""
    )
    poll.write_text(unknown_source, encoding="utf-8")

    result = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "expected exactly one empty inbound poll title match, found 0" in result.stderr
    assert valid.read_text(encoding="utf-8") == valid_source
    assert poll.read_text(encoding="utf-8") == unknown_source
