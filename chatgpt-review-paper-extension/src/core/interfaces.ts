import type { Conversation, ConversationSummary, NormalizedContent, ReviewOutput, ReviewRequest, ReviewTier } from './types';

export interface ConversationSource {
  listConversations(sinceISO: string, onProgress?: (message: string) => void): Promise<ConversationSummary[]>;
  getConversation(id: string, summary?: ConversationSummary, onProgress?: (message: string) => void): Promise<Conversation>;
}

export interface Summarizer {
  summarize(content: NormalizedContent, tier: ReviewTier, language: string | undefined, onProgress?: (message: string) => void): Promise<string>;
}

export interface ReviewStore {
  getConversation(id: string, updateTime: number): Promise<Conversation | undefined>;
  putConversation(conversation: Conversation): Promise<void>;
  putReview(request: ReviewRequest, output: ReviewOutput): Promise<void>;
}
