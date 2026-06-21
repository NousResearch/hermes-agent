import { openDB, type DBSchema } from 'idb';
import type { Conversation, ReviewOutput, ReviewRequest } from '../core/types';
import type { ReviewStore } from '../core/interfaces';

interface ReviewDb extends DBSchema {
  conversations: { key: string; value: Conversation & { cacheKey: string } };
  reviews: { key: string; value: ReviewOutput & { cacheKey: string } };
}

const dbPromise = openDB<ReviewDb>('chatgpt-review-paper', 1, {
  upgrade(db) {
    db.createObjectStore('conversations', { keyPath: 'cacheKey' });
    db.createObjectStore('reviews', { keyPath: 'cacheKey' });
  },
});

export class IndexedDbReviewStore implements ReviewStore {
  async getConversation(id: string, updateTime: number): Promise<Conversation | undefined> {
    const hit = await (await dbPromise).get('conversations', `${id}:${updateTime}`);
    if (!hit) return undefined;
    const { cacheKey: _cacheKey, ...conversation } = hit;
    return conversation;
  }
  async putConversation(conversation: Conversation): Promise<void> {
    await (await dbPromise).put('conversations', { ...conversation, cacheKey: `${conversation.id}:${conversation.update_time}` });
  }
  async putReview(request: ReviewRequest, output: ReviewOutput): Promise<void> {
    await (await dbPromise).put('reviews', { ...output, cacheKey: `${request.windowDays}:${request.tier}:${request.language ?? 'auto'}:${output.window.endISO}` });
  }
}
