import type { Summarizer } from './interfaces';
import type { NormalizedContent, ReviewTier } from './types';
import { conversationsToPromptText } from './normalize';
import { systemPrompt, tierInstruction } from './prompts';

const MAX_CHARS_PER_CALL = 28_000;

export abstract class BaseLLMSummarizer implements Summarizer {
  constructor(protected apiKey: string) {}
  protected abstract complete(messages: { role: 'system' | 'user'; content: string }[]): Promise<string>;

  async summarize(content: NormalizedContent, tier: ReviewTier, language?: string, onProgress?: (message: string) => void): Promise<string> {
    const source = conversationsToPromptText(content.conversations);
    if (source.length <= MAX_CHARS_PER_CALL) {
      return this.complete([{ role: 'system', content: systemPrompt(tier, language) }, { role: 'user', content: `Window: ${JSON.stringify(content.window)}\n\nSource conversations:\n${source}` }]);
    }

    onProgress?.('Large window detected; summarizing per conversation first…');
    const partials: string[] = [];
    for (const conversation of content.conversations) {
      const text = conversationsToPromptText([conversation]);
      partials.push(await this.complete([
        { role: 'system', content: `${systemPrompt('take', language)}\nCreate an evidence-preserving intermediate summary for map-reduce. Include concrete topics, decisions, unresolved points.` },
        { role: 'user', content: text.slice(0, MAX_CHARS_PER_CALL) },
      ]));
    }
    return this.complete([
      { role: 'system', content: systemPrompt(tier, language) },
      { role: 'user', content: `Synthesize these per-conversation summaries for ${JSON.stringify(content.window)}. ${tierInstruction(tier)}\n\n${partials.join('\n\n---\n\n')}` },
    ]);
  }
}

export class OpenAiSummarizer extends BaseLLMSummarizer {
  async complete(messages: { role: 'system' | 'user'; content: string }[]): Promise<string> {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: 'gpt-4.1-mini', temperature: 0.2, messages }),
    });
    if (!res.ok) throw new Error(`OpenAI request failed: ${res.status} ${await res.text()}`);
    const json = await res.json();
    return json.choices?.[0]?.message?.content ?? '';
  }
}

export class AnthropicSummarizer extends BaseLLMSummarizer {
  async complete(messages: { role: 'system' | 'user'; content: string }[]): Promise<string> {
    const system = messages.find((m) => m.role === 'system')?.content ?? '';
    const user = messages.filter((m) => m.role === 'user').map((m) => m.content).join('\n\n');
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST', headers: { 'content-type': 'application/json', 'x-api-key': this.apiKey, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      body: JSON.stringify({ model: 'claude-3-5-haiku-latest', max_tokens: 4096, temperature: 0.2, system, messages: [{ role: 'user', content: user }] }),
    });
    if (!res.ok) throw new Error(`Anthropic request failed: ${res.status} ${await res.text()}`);
    const json = await res.json();
    return json.content?.map((p: any) => p.text ?? '').join('\n') ?? '';
  }
}
