import { marked } from 'marked';
import { getSettings, saveSettings } from '../settings';
import type { ReviewOutput, ReviewTier, Settings } from '../core/types';
import './styles.css';

const app = document.querySelector<HTMLDivElement>('#app')!;
let lastReview: ReviewOutput | undefined;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, any> = {}, children: Array<Node | string> = []): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === 'class') node.className = value;
    else if (key === 'for') node.setAttribute('for', value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value !== undefined) (node as any)[key] = value;
  });
  children.forEach((child) => node.append(child));
  return node;
}

async function send<T>(message: any): Promise<T> {
  const response = await chrome.runtime.sendMessage(message);
  if (!response?.ok) throw new Error(response?.error ?? 'Extension request failed');
  return response as T;
}

async function render(): Promise<void> {
  const settings = await getSettings();
  app.innerHTML = '';
  app.append(el('h1', {}, ['ChatGPT Review Paper']));
  if (!settings.disclaimerAccepted) app.append(disclaimer(settings));
  app.append(await sessionPanel());
  app.append(settingsPanel(settings));
  app.append(generatePanel(settings));
  app.append(el('section', { id: 'status', class: 'status' }, ['Ready.']));
  app.append(el('article', { id: 'review', class: 'review' }));
}

function disclaimer(settings: Settings): HTMLElement {
  return el('section', { class: 'disclaimer' }, [
    'Personal-use tool: reads your own ChatGPT data via your signed-in browser session, stores data locally, and may break if ChatGPT changes internals. ',
    el('button', { onclick: async () => { await saveSettings({ ...settings, disclaimerAccepted: true }); await render(); } }, ['OK']),
  ]);
}

async function sessionPanel(): Promise<HTMLElement> {
  const section = el('section', { class: 'card' }, [el('h2', {}, ['Session'])]);
  try {
    const status = await send<{ loggedIn: boolean }>({ type: 'CHECK_SESSION' });
    section.append(el('p', { class: status.loggedIn ? 'ok' : 'warn' }, [status.loggedIn ? 'Logged in to chatgpt.com.' : 'No active ChatGPT session detected.']));
  } catch (error) {
    section.append(el('p', { class: 'warn' }, [(error as Error).message]));
  }
  section.append(el('button', { onclick: () => send({ type: 'OPEN_CHATGPT' }) }, ['Open ChatGPT and log in']));
  return section;
}

function settingsPanel(settings: Settings): HTMLElement {
  const provider = el('select', { id: 'provider' }, [
    el('option', { value: 'openai', selected: settings.provider === 'openai' }, ['OpenAI']),
    el('option', { value: 'anthropic', selected: settings.provider === 'anthropic' }, ['Anthropic']),
  ]);
  const apiKey = el('input', { id: 'apiKey', type: 'password', placeholder: 'LLM API key', value: settings.apiKey ?? '' });
  const language = el('input', { id: 'language', placeholder: 'Output language, e.g. Japanese (blank = auto)', value: settings.language ?? '' });
  return el('section', { class: 'card' }, [
    el('h2', {}, ['Settings']),
    el('label', {}, ['Provider', provider]),
    el('label', {}, ['API key', apiKey]),
    el('label', {}, ['Output language', language]),
    el('button', { onclick: async () => {
      await saveSettings({ ...settings, provider: (provider as HTMLSelectElement).value as Settings['provider'], apiKey: (apiKey as HTMLInputElement).value, language: (language as HTMLInputElement).value });
      setStatus('Settings saved.');
    } }, ['Save settings']),
  ]);
}

function generatePanel(settings: Settings): HTMLElement {
  const days = el('input', { id: 'days', type: 'number', min: '1', value: '7' });
  const tier = el('select', { id: 'tier' }, [
    el('option', { value: 'matsu', selected: settings.defaultTier === 'matsu' }, ['松 — 5-min read']),
    el('option', { value: 'take', selected: settings.defaultTier === 'take' }, ['竹 — 1-min read']),
    el('option', { value: 'ume', selected: settings.defaultTier === 'ume' }, ['梅 — 20-sec read']),
  ]);
  return el('section', { class: 'card' }, [
    el('h2', {}, ['Generate']),
    el('label', {}, ['Time window (days)', days]),
    el('label', {}, ['Coverage tier', tier]),
    el('button', { class: 'primary', onclick: async () => generate(Number((days as HTMLInputElement).value), (tier as HTMLSelectElement).value as ReviewTier) }, ['Generate review']),
    el('div', { class: 'toolbar' }, [
      el('button', { onclick: copyReview }, ['Copy']),
      el('button', { onclick: exportMarkdown }, ['Export Markdown']),
      el('button', { onclick: exportPdf }, ['Export PDF']),
    ]),
  ]);
}

async function generate(windowDays: number, tier: ReviewTier): Promise<void> {
  try {
    const settings = await getSettings();
    setStatus('Working: listing → fetching → summarizing…');
    const response = await send<{ output: ReviewOutput }>({ type: 'GENERATE_REVIEW', request: { windowDays, tier, language: settings.language || undefined } });
    lastReview = response.output;
    const review = document.querySelector<HTMLElement>('#review')!;
    review.innerHTML = await marked.parse(response.output.markdown);
    setStatus(`Done. ${response.output.conversationCount} conversations covered.`);
  } catch (error) { setStatus((error as Error).message, true); }
}

function setStatus(message: string, error = false): void {
  const status = document.querySelector<HTMLElement>('#status');
  if (status) { status.textContent = message; status.className = `status ${error ? 'error' : ''}`; }
}

async function copyReview(): Promise<void> { if (lastReview) await navigator.clipboard.writeText(lastReview.markdown); }
function exportMarkdown(): void { if (lastReview) download(`chatgpt-review-${lastReview.window.days}d-${lastReview.tier}.md`, lastReview.markdown, 'text/markdown'); }
function exportPdf(): void { window.print(); }
function download(filename: string, content: string, type: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

render().catch((error) => { app.textContent = (error as Error).message; });
