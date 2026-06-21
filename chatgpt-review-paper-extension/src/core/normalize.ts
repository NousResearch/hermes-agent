import type { Conversation, Message, MessageRole } from './types';

type RawNode = { message?: any; parent?: string | null; children?: string[] };

function roleOf(raw: string | undefined): MessageRole {
  if (raw === 'assistant' || raw === 'system' || raw === 'tool') return raw;
  return 'user';
}

function textFromPart(part: unknown): string {
  if (typeof part === 'string') return part;
  if (part && typeof part === 'object') {
    const obj = part as Record<string, unknown>;
    if (typeof obj.text === 'string') return obj.text;
    if (typeof obj.content === 'string') return obj.content;
    if (typeof obj.url === 'string') return `[non-text: ${obj.url}]`;
  }
  return '';
}

export function flattenContent(content: any): string {
  if (!content) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content.parts)) return content.parts.map(textFromPart).map((part: string) => part.trim()).filter(Boolean).join('\n').trim();
  if (Array.isArray(content.content_type)) return content.content_type.map(textFromPart).map((part: string) => part.trim()).filter(Boolean).join('\n').trim();
  if (typeof content.text === 'string') return content.text;
  return '';
}

export function normalizeConversation(raw: any): Conversation {
  const mapping = (raw.mapping ?? {}) as Record<string, RawNode>;
  const messages: Message[] = Object.values(mapping)
    .map((node) => node.message)
    .filter(Boolean)
    .map((message: any) => ({
      id: String(message.id),
      role: roleOf(message.author?.role),
      create_time: Number(message.create_time ?? 0),
      text: flattenContent(message.content),
    }))
    .filter((message) => message.id && message.create_time && message.text)
    .sort((a, b) => a.create_time - b.create_time);

  return dedupeMessages({
    id: String(raw.conversation_id ?? raw.id),
    title: String(raw.title ?? 'Untitled conversation'),
    create_time: Number(raw.create_time ?? messages.at(0)?.create_time ?? 0),
    update_time: Number(raw.update_time ?? messages.at(-1)?.create_time ?? 0),
    messages,
  });
}

export function dedupeMessages(conversation: Conversation): Conversation {
  const seen = new Set<string>();
  return {
    ...conversation,
    messages: conversation.messages.filter((message) => {
      if (seen.has(message.id)) return false;
      seen.add(message.id);
      return true;
    }),
  };
}

export function sliceConversationWindow(conversation: Conversation, startEpoch: number, endEpoch: number, contextRadius = 1): Conversation | null {
  const inWindowIndexes = conversation.messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => message.create_time >= startEpoch && message.create_time <= endEpoch)
    .map(({ index }) => index);
  if (inWindowIndexes.length === 0) return null;

  const keep = new Set<number>();
  for (const index of inWindowIndexes) {
    for (let i = Math.max(0, index - contextRadius); i <= Math.min(conversation.messages.length - 1, index + contextRadius); i += 1) {
      keep.add(i);
    }
  }

  return {
    ...conversation,
    messages: conversation.messages
      .filter((_message, index) => keep.has(index))
      .map((message) => ({ ...message, inWindow: message.create_time >= startEpoch && message.create_time <= endEpoch })),
  };
}

export function conversationsToPromptText(conversations: Conversation[]): string {
  return conversations.map((conversation) => {
    const body = conversation.messages.map((message) => {
      const marker = message.inWindow === false ? 'context' : 'in-window';
      return `- [${marker}] ${message.role} @ ${new Date(message.create_time * 1000).toISOString()}: ${message.text}`;
    }).join('\n');
    return `# ${conversation.title} (${conversation.id})\n${body}`;
  }).join('\n\n---\n\n');
}
