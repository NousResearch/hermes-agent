// Read-later helpers shared by the News widget and the Reading List widget.

import { store } from "./store.js";
import { toast } from "./utils.js";

const MAX_SAVED = 100;
const MAX_READ_URLS = 500;

export function isRead(url) {
  return Boolean(store.state.newsRead?.[url]);
}

export function markRead(url) {
  if (!url || isRead(url)) return;
  store.update((state) => {
    if (!state.newsRead) state.newsRead = {};
    state.newsRead[url] = Date.now();
    const urls = Object.keys(state.newsRead);
    if (urls.length > MAX_READ_URLS) {
      // drop the oldest entries
      urls.sort((a, b) => state.newsRead[a] - state.newsRead[b]);
      for (const old of urls.slice(0, urls.length - MAX_READ_URLS)) {
        delete state.newsRead[old];
      }
    }
  }, "reading");
}

export function isSaved(url) {
  return store.state.reading.items.some((i) => i.url === url);
}

export function saveForLater(item) {
  if (isSaved(item.url)) {
    toast("Already on your reading list");
    return false;
  }
  store.update((state) => {
    state.reading.items.unshift({
      title: item.title,
      url: item.url,
      summary: item.summary || "",
      source: item.source || "",
      saved: new Date().toISOString(),
    });
    state.reading.items = state.reading.items.slice(0, MAX_SAVED);
  }, "reading");
  toast("Saved to reading list");
  return true;
}

export function removeSaved(url) {
  store.update((state) => {
    state.reading.items = state.reading.items.filter((i) => i.url !== url);
  }, "reading");
}
