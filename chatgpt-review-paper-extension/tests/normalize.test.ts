import { describe, expect, it } from 'vitest';
import { normalizeConversation, sliceConversationWindow, dedupeMessages } from '../src/core/normalize';
import type { Conversation } from '../src/core/types';

describe('normalization and window slicing', () => {
  it('orders mapped backend messages and flattens text parts', () => {
    const raw = {
      id: 'c1',
      title: 'Project notes',
      create_time: 10,
      update_time: 30,
      mapping: {
        b: { message: { id: 'b', author: { role: 'assistant' }, create_time: 30, content: { parts: ['Done'] } }, parent: 'a', children: [] },
        a: { message: { id: 'a', author: { role: 'user' }, create_time: 20, content: { parts: ['Please ', { text: 'summarize' }] } }, parent: null, children: ['b'] },
      },
    };

    expect(normalizeConversation(raw)).toEqual<Conversation>({
      id: 'c1',
      title: 'Project notes',
      create_time: 10,
      update_time: 30,
      messages: [
        { id: 'a', role: 'user', create_time: 20, text: 'Please\nsummarize' },
        { id: 'b', role: 'assistant', create_time: 30, text: 'Done' },
      ],
    });
  });

  it('keeps only in-window messages plus minimal surrounding context', () => {
    const conversation: Conversation = {
      id: 'c1', title: 'Long thread', create_time: 0, update_time: 400,
      messages: [
        { id: 'm1', role: 'user', create_time: 100, text: 'old setup' },
        { id: 'm2', role: 'assistant', create_time: 200, text: 'setup answer' },
        { id: 'm3', role: 'user', create_time: 300, text: 'today question' },
        { id: 'm4', role: 'assistant', create_time: 400, text: 'today answer' },
      ],
    };

    const sliced = sliceConversationWindow(conversation, 250, 450);
    expect(sliced?.messages.map((m) => [m.id, m.inWindow])).toEqual([
      ['m2', false],
      ['m3', true],
      ['m4', true],
    ]);
  });

  it('drops conversations with no in-window messages', () => {
    const conversation: Conversation = { id: 'c1', title: 'Old', create_time: 0, update_time: 10, messages: [{ id: 'm1', role: 'user', create_time: 10, text: 'old' }] };
    expect(sliceConversationWindow(conversation, 100, 200)).toBeNull();
  });

  it('deduplicates repeated message ids', () => {
    const conversation: Conversation = { id: 'c1', title: 'Dupes', create_time: 0, update_time: 2, messages: [
      { id: 'x', role: 'user', create_time: 1, text: 'first' },
      { id: 'x', role: 'user', create_time: 1, text: 'repeat' },
      { id: 'y', role: 'assistant', create_time: 2, text: 'second' },
    ] };
    expect(dedupeMessages(conversation).messages.map((m) => m.text)).toEqual(['first', 'second']);
  });
});
