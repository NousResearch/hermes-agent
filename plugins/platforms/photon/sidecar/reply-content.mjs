export async function normalizeReplyContent(
  content,
  normalizeContent,
  targetText
) {
  const target = content.target;
  return {
    type: "reply",
    content: await normalizeContent(content.content),
    targetMessageId: target?.id ?? null,
    targetDirection: target?.direction ?? null,
    targetText: targetText(target),
  };
}
