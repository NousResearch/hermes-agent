import type { ConversationSource, ReviewStore, Summarizer } from './interfaces';
import type { Conversation, NormalizedContent, ReviewOutput, ReviewRequest } from './types';
import { sliceConversationWindow } from './normalize';

export class MemoryReviewStore implements ReviewStore {
  private conversations = new Map<string, Conversation>();
  async getConversation(id: string, updateTime: number): Promise<Conversation | undefined> { return this.conversations.get(`${id}:${updateTime}`); }
  async putConversation(conversation: Conversation): Promise<void> { this.conversations.set(`${conversation.id}:${conversation.update_time}`, conversation); }
  async putReview(): Promise<void> { /* in-memory tests do not persist reviews */ }
}

export class ReviewGenerator {
  private readonly store: ReviewStore;
  private readonly now: () => number;

  constructor(private source: ConversationSource, private summarizer: Summarizer, options: { store?: ReviewStore; now?: () => number } = {}) {
    this.store = options.store ?? new MemoryReviewStore();
    this.now = options.now ?? (() => Date.now() / 1000);
  }

  async generate(request: ReviewRequest, onProgress?: (message: string) => void): Promise<ReviewOutput> {
    const endEpoch = this.now();
    const startEpoch = endEpoch - request.windowDays * 24 * 60 * 60;
    const window = { startISO: new Date(startEpoch * 1000).toISOString(), endISO: new Date(endEpoch * 1000).toISOString(), days: request.windowDays };
    onProgress?.('Listing ChatGPT conversations…');
    const summaries = await this.source.listConversations(window.startISO, onProgress);
    const sliced: Conversation[] = [];

    for (const summary of summaries) {
      let full = await this.store.getConversation(summary.id, summary.update_time);
      if (!full) {
        onProgress?.(`Fetching ${summary.title || summary.id}…`);
        full = await this.source.getConversation(summary.id, summary, onProgress);
        await this.store.putConversation(full);
        await delay(150);
      }
      const inWindow = sliceConversationWindow(full, startEpoch, endEpoch);
      if (inWindow) sliced.push(inWindow);
    }

    if (sliced.length === 0) {
      return { generatedAt: endEpoch, window, tier: request.tier, conversationCount: 0, markdown: '## No conversations in range\n\nNo ChatGPT messages were found in the selected window.' };
    }

    const content: NormalizedContent = { window, conversations: sliced };
    onProgress?.('Summarizing review paper…');
    const markdown = await this.summarizer.summarize(content, request.tier, request.language, onProgress);
    const output: ReviewOutput = { generatedAt: endEpoch, window, tier: request.tier, conversationCount: sliced.length, markdown };
    await this.store.putReview(request, output);
    return output;
  }
}

function delay(ms: number): Promise<void> { return new Promise((resolve) => setTimeout(resolve, ms)); }
