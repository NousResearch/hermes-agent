import assert from "node:assert/strict";
import test from "node:test";

import { normalizeReplyContent } from "../reply-content.mjs";

test("normalizes reply metadata and inner content", async () => {
  const normalized = await normalizeReplyContent(
    {
      content: { type: "text", text: "answer" },
      target: {
        id: "target-1",
        direction: "outbound",
        content: { type: "text", text: "question" },
      },
    },
    async (content) => ({ ...content, normalized: true }),
    (target) => target.content.text
  );

  assert.deepEqual(normalized, {
    type: "reply",
    content: { type: "text", text: "answer", normalized: true },
    targetMessageId: "target-1",
    targetDirection: "outbound",
    targetText: "question",
  });
});

test("keeps unresolved reply targets nullable", async () => {
  const normalized = await normalizeReplyContent(
    { content: { type: "attachment" } },
    async (content) => content,
    () => null
  );

  assert.equal(normalized.targetMessageId, null);
  assert.equal(normalized.targetDirection, null);
  assert.equal(normalized.targetText, null);
});
