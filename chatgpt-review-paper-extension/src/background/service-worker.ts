import { ReviewGenerator } from '../core/review-generator';
import { OpenAiSummarizer, AnthropicSummarizer } from '../core/summarizers';
import { ChatGptBackendApiSource } from '../source/chatgpt-backend-api-source';
import { getSettings } from '../settings';
import { IndexedDbReviewStore } from '../storage/db';
import type { ReviewRequest } from '../core/types';

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});

async function getChatGptTab(): Promise<chrome.tabs.Tab | undefined> {
  const tabs = await chrome.tabs.query({ url: 'https://chatgpt.com/*' });
  return tabs[0];
}

async function ensureChatGptTab(): Promise<chrome.tabs.Tab> {
  const existing = await getChatGptTab();
  if (existing?.id) return existing;
  return chrome.tabs.create({ url: 'https://chatgpt.com/' });
}

async function rpc(tabId: number, message: Record<string, unknown>): Promise<any> {
  const response = await chrome.tabs.sendMessage(tabId, { target: 'CHATGPT_REVIEW_CONTENT', ...message });
  if (!response?.ok) throw new Error(response?.error ?? 'No response from ChatGPT content script. Open/reload chatgpt.com and try again.');
  return response.data ?? response;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message?.type === 'OPEN_CHATGPT') {
      await ensureChatGptTab();
      return { ok: true };
    }
    if (message?.type === 'CHECK_SESSION') {
      const tab = await getChatGptTab();
      if (!tab?.id) return { ok: true, loggedIn: false };
      return rpc(tab.id, { type: 'CHATGPT_SESSION_STATUS' });
    }
    if (message?.type === 'GENERATE_REVIEW') {
      const tab = await ensureChatGptTab();
      if (!tab.id) throw new Error('Could not open ChatGPT tab.');
      const settings = await getSettings();
      if (!settings.apiKey) throw new Error('Set an OpenAI or Anthropic API key in Settings first.');
      const summarizer = settings.provider === 'anthropic' ? new AnthropicSummarizer(settings.apiKey) : new OpenAiSummarizer(settings.apiKey);
      const source = new ChatGptBackendApiSource((payload) => rpc(tab.id!, payload));
      const generator = new ReviewGenerator(source, summarizer, { store: new IndexedDbReviewStore() });
      return { ok: true, output: await generator.generate(message.request as ReviewRequest) };
    }
    return { ok: false, error: `Unknown message: ${message?.type}` };
  })().then(sendResponse).catch((error) => sendResponse({ ok: false, error: error instanceof Error ? error.message : String(error) }));
  return true;
});
