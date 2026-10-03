import assert from "node:assert/strict";
import test from "node:test";

import { localIMessage } from "@spectrum-ts/imessage-local";
import { imessage } from "spectrum-ts/providers/imessage";

const appleMessage = {
  guid: "message-1",
  isFromMe: false,
  isAudioMessage: false,
  sender: { address: "+15555550100", service: "iMessage" },
  chatGuids: ["any;-;+15555550100"],
  content: {
    text: "before\uFFFCafter",
    attachments: [{
      guid: "attachment-1",
      fileName: "photo.jpg",
      mimeType: "image/jpeg",
      totalBytes: 123,
      uti: "public.jpeg",
    }],
  },
};

function contentSummary(messages) {
  return messages.map((message) => ({
    type: message.content.type,
    partIndex: message.partIndex,
    text: message.content.text,
  }));
}

function queueStream(items = []) {
  let waiting;
  let closed = false;
  return {
    [Symbol.asyncIterator]() {
      return this;
    },
    async next() {
      if (items.length > 0) return { done: false, value: items.shift() };
      if (closed) return { done: true, value: undefined };
      return new Promise((resolve) => {
        waiting = resolve;
      });
    },
    async return() {
      closed = true;
      waiting?.({ done: true, value: undefined });
      return { done: true, value: undefined };
    },
    async close() {
      await this.return();
    },
  };
}

test("local provider preserves text around an attachment", { timeout: 5000 }, async () => {
  let watcherCallbacks;
  const client = {
    async startWatching(callbacks) {
      watcherCallbacks = callbacks;
    },
    async stopWatching() {},
  };
  const messages = localIMessage.config().__definition.messages({ client });
  const iterator = messages[Symbol.asyncIterator]();
  const firstPending = iterator.next();

  for (let attempt = 0; attempt < 20 && !watcherCallbacks; attempt += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert(watcherCallbacks, "local provider did not start its Messages watcher");

  watcherCallbacks.onIncomingMessage({
    id: appleMessage.guid,
    chatId: appleMessage.chatGuids[0],
    chatKind: "dm",
    participant: appleMessage.sender.address,
    createdAt: new Date("2026-07-27T12:00:00Z"),
    kind: "text",
    reaction: null,
    retractedAt: null,
    hasAttachments: true,
    isAudioMessage: false,
    threadRootMessageId: null,
    text: appleMessage.content.text,
    attachments: [{
      id: "attachment-1",
      fileName: "photo.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 123,
      localPath: null,
      uti: "public.jpeg",
    }],
  });

  const emitted = [];
  for (let index = 0; index < 3; index += 1) {
    const result = index === 0 ? await firstPending : await iterator.next();
    assert.equal(result.done, false);
    emitted.push(result.value);
  }
  assert.deepEqual(contentSummary(emitted), [
    { type: "text", partIndex: 0, text: "before" },
    { type: "attachment", partIndex: 1, text: undefined },
    { type: "text", partIndex: 2, text: "after" },
  ]);
  await iterator.return();
});

test("cloud provider preserves the same mixed-message order", { timeout: 5000 }, async () => {
  const messageEvents = queueStream([{
    type: "message.received",
    sequence: 1,
    occurredAt: new Date("2026-07-27T12:00:00Z"),
    chatGuid: appleMessage.chatGuids[0],
    message: appleMessage,
  }]);
  const pollEvents = queueStream();
  const client = {
    messages: { subscribeEvents: () => messageEvents },
    polls: { subscribeEvents: () => pollEvents },
    events: { async *catchUp() {} },
  };
  const messages = imessage.config().__definition.messages({
    client: [{ phone: "shared", client }],
    projectConfig: undefined,
  });
  const iterator = messages[Symbol.asyncIterator]();
  const result = await iterator.next();

  assert.equal(result.done, false);
  assert.equal(result.value.content.type, "group");
  assert.deepEqual(contentSummary(result.value.content.items), [
    { type: "text", partIndex: 0, text: "before" },
    { type: "attachment", partIndex: 1, text: undefined },
    { type: "text", partIndex: 2, text: "after" },
  ]);
  await iterator.return();
});
