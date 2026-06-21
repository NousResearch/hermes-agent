import type { ConversationSource } from '../core/interfaces';
import type { Conversation, ConversationSummary } from '../core/types';
import { normalizeConversation } from '../core/normalize';

export type ChatGptRpc = <T = any>(message: Record<string, unknown>) => Promise<T>;

/**
 * All undocumented ChatGPT backend-api details are confined to this adapter and
 * the chatgpt.com content script that executes the actual fetches in-origin.
 *
 * LIVE VERIFICATION STATUS (2026-06-21): browser automation reached Cloudflare
 * challenge, so this environment could not confirm signed-in DevTools network
 * shapes. The adapter therefore centralizes endpoint assumptions and surfaces
 * clear failures; re-verify this file against live chatgpt.com Network before
 * depending on it in production.
 */
export class ChatGptBackendApiSource implements ConversationSource {
  constructor(private rpc: ChatGptRpc) {}

  async listConversations(sinceISO: string, onProgress?: (message: string) => void): Promise<ConversationSummary[]> {
    const since = Date.parse(sinceISO) / 1000;
    const summaries: ConversationSummary[] = [];
    let offset = 0;
    const limit = 28;
    while (true) {
      onProgress?.(`Listing conversations ${offset + 1}–${offset + limit}…`);
      const page = await this.rpc<any>({ type: 'CHATGPT_LIST_CONVERSATIONS', offset, limit });
      const items = page.items ?? page.conversations ?? [];
      for (const item of items) {
        summaries.push({
          id: String(item.id),
          title: String(item.title ?? 'Untitled conversation'),
          create_time: Number(item.create_time ?? 0),
          update_time: Number(item.update_time ?? item.create_time ?? 0),
        });
      }
      const oldest = Math.min(...items.map((item: any) => Number(item.update_time ?? item.create_time ?? 0)).filter(Boolean));
      if (!items.length || oldest < since || items.length < limit) break;
      offset += limit;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    return summaries.filter((summary) => summary.update_time >= since);
  }

  async getConversation(id: string): Promise<Conversation> {
    const raw = await this.rpc<any>({ type: 'CHATGPT_GET_CONVERSATION', id });
    return normalizeConversation(raw);
  }
}
