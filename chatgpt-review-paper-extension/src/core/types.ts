export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';

export type Message = {
  id: string;
  role: MessageRole;
  create_time: number;
  text: string;
  inWindow?: boolean;
};

export type Conversation = {
  id: string;
  title: string;
  create_time: number;
  update_time: number;
  messages: Message[];
};

export type ConversationSummary = Omit<Conversation, 'messages'>;

export type ReviewTier = 'matsu' | 'take' | 'ume';

export type ReviewRequest = {
  windowDays: number;
  tier: ReviewTier;
  language?: string;
};

export type ReviewOutput = {
  generatedAt: number;
  window: { startISO: string; endISO: string; days: number };
  tier: ReviewRequest['tier'];
  conversationCount: number;
  markdown: string;
};

export type NormalizedContent = {
  window: ReviewOutput['window'];
  conversations: Conversation[];
};

export type Settings = {
  provider: 'openai' | 'anthropic';
  apiKey?: string;
  defaultTier: ReviewTier;
  language?: string;
  disclaimerAccepted?: boolean;
};
