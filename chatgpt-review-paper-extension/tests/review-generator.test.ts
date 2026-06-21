import { describe, expect, it, vi } from 'vitest';
import { ReviewGenerator } from '../src/core/review-generator';
import type { ConversationSource, Summarizer } from '../src/core/interfaces';
import type { Conversation } from '../src/core/types';

function source(conversations: Conversation[]): ConversationSource {
  return {
    listConversations: vi.fn(async () => conversations.map(({ id, title, create_time, update_time }) => ({ id, title, create_time, update_time }))),
    getConversation: vi.fn(async (id: string) => conversations.find((c) => c.id === id)!),
  };
}

describe('ReviewGenerator', () => {
  it('fetches every updated conversation but summarizes only in-window messages', async () => {
    const now = 100_000;
    const conversations: Conversation[] = [
      { id: 'old-thread', title: 'Old thread updated today', create_time: 100, update_time: 990, messages: [
        { id: 'old', role: 'user', create_time: 95_000, text: 'outside' },
        { id: 'new', role: 'user', create_time: 99_500, text: 'inside' },
      ] },
      { id: 'only-old', title: 'No in-window messages', create_time: 100, update_time: 980, messages: [
        { id: 'old2', role: 'assistant', create_time: 95_000, text: 'outside' },
      ] },
    ];
    const summarizer: Summarizer = { summarize: vi.fn(async () => '## Review\nConcrete output') };
    const generator = new ReviewGenerator(source(conversations), summarizer, { now: () => now });

    const review = await generator.generate({ windowDays: 1 / 24, tier: 'take', language: 'ja' });

    expect(summarizer.summarize).toHaveBeenCalledWith(
      expect.objectContaining({ conversations: [expect.objectContaining({ id: 'old-thread' })] }),
      'take',
      'ja',
      undefined,
    );
    expect(review.conversationCount).toBe(1);
    expect(review.markdown).toContain('Concrete output');
  });

  it('reuses cached conversations when regenerating at another depth', async () => {
    const conversations: Conversation[] = [{ id: 'c1', title: 'Cache me', create_time: 0, update_time: 1000, messages: [{ id: 'm1', role: 'user', create_time: 999, text: 'inside' }] }];
    const src = source(conversations);
    const summarizer: Summarizer = { summarize: vi.fn(async (_content, tier) => `tier:${tier}`) };
    const generator = new ReviewGenerator(src, summarizer, { now: () => 1000 });

    await generator.generate({ windowDays: 1, tier: 'ume' });
    await generator.generate({ windowDays: 1, tier: 'matsu' });

    expect(src.getConversation).toHaveBeenCalledTimes(1);
    expect(summarizer.summarize).toHaveBeenCalledTimes(2);
  });
});
