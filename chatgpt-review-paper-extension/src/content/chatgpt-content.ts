const API_BASE = 'https://chatgpt.com/backend-api';

async function chatgptFetch(path: string, init?: RequestInit): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, { credentials: 'include', ...init, headers: { accept: 'application/json', ...(init?.headers ?? {}) } });
  if (!res.ok) throw new Error(`ChatGPT backend-api request failed: ${res.status} ${await res.text()}`);
  return res.json();
}

async function hasSession(): Promise<boolean> {
  // [VERIFY LIVE] Historically this endpoint returns { accessToken } when logged in.
  const res = await fetch('https://chatgpt.com/api/auth/session', { credentials: 'include', headers: { accept: 'application/json' } });
  if (!res.ok) return false;
  const json = await res.json().catch(() => ({}));
  return Boolean(json?.accessToken || json?.user);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message?.target !== 'CHATGPT_REVIEW_CONTENT') return;
    if (message.type === 'CHATGPT_SESSION_STATUS') {
      sendResponse({ ok: true, loggedIn: await hasSession() });
      return;
    }
    if (!(await hasSession())) throw new Error('No active chatgpt.com session. Open ChatGPT and log in.');
    if (message.type === 'CHATGPT_LIST_CONVERSATIONS') {
      // [VERIFY LIVE] Endpoint/params/shape must be checked against DevTools Network.
      sendResponse({ ok: true, data: await chatgptFetch(`/conversations?offset=${encodeURIComponent(message.offset ?? 0)}&limit=${encodeURIComponent(message.limit ?? 28)}&order=updated`) });
      return;
    }
    if (message.type === 'CHATGPT_GET_CONVERSATION') {
      // [VERIFY LIVE] Endpoint/shape must be checked against DevTools Network.
      sendResponse({ ok: true, data: await chatgptFetch(`/conversation/${encodeURIComponent(message.id)}`) });
      return;
    }
    throw new Error(`Unknown ChatGPT content message: ${message.type}`);
  })().catch((error) => sendResponse({ ok: false, error: error instanceof Error ? error.message : String(error) }));
  return true;
});
