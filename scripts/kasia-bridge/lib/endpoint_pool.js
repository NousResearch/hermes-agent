const DEFAULT_BASE_BACKOFF_MS = 5_000;
const DEFAULT_MAX_BACKOFF_MS = 120_000;

function uniqueUrls(urls = []) {
  return [...new Set(urls.map((value) => String(value || "").trim()).filter(Boolean))];
}

export class EndpointPool {
  constructor({
    urls = [],
    name = "endpoint",
    nowFn = () => Date.now(),
    baseBackoffMs = DEFAULT_BASE_BACKOFF_MS,
    maxBackoffMs = DEFAULT_MAX_BACKOFF_MS,
  } = {}) {
    this.name = name;
    this.nowFn = nowFn;
    this.baseBackoffMs = baseBackoffMs;
    this.maxBackoffMs = maxBackoffMs;
    this._cursor = 0;
    this._state = new Map();
    this.setUrls(urls);
  }

  setUrls(urls = []) {
    const nextUrls = uniqueUrls(urls);
    const nextState = new Map();
    for (const url of nextUrls) {
      nextState.set(url, this._state.get(url) || {
        url,
        failures: 0,
        backoff_until_ms: 0,
        last_success_ms: 0,
        last_error: null,
      });
    }
    this._state = nextState;
    if (this._cursor >= nextUrls.length) {
      this._cursor = 0;
    }
  }

  urls() {
    return [...this._state.keys()];
  }

  get activeUrl() {
    const urls = this.urls();
    if (urls.length === 0) {
      return null;
    }
    const candidates = this.getCandidates();
    return candidates[0] || urls[0] || null;
  }

  getCandidates() {
    const urls = this.urls();
    if (urls.length === 0) {
      return [];
    }
    const nowMs = this.nowFn();
    const ordered = [];
    for (let offset = 0; offset < urls.length; offset += 1) {
      ordered.push(urls[(this._cursor + offset) % urls.length]);
    }
    const available = ordered.filter((url) => {
      const state = this._state.get(url);
      return !state || state.backoff_until_ms <= nowMs;
    });
    return available.length > 0 ? available : ordered;
  }

  markSuccess(url) {
    const normalized = String(url || "").trim();
    const state = this._state.get(normalized);
    if (!state) {
      return;
    }
    state.failures = 0;
    state.backoff_until_ms = 0;
    state.last_success_ms = this.nowFn();
    state.last_error = null;
    const urls = this.urls();
    const index = urls.indexOf(normalized);
    if (index >= 0) {
      this._cursor = index;
    }
  }

  markFailure(url, error) {
    const normalized = String(url || "").trim();
    const state = this._state.get(normalized);
    if (!state) {
      return;
    }
    state.failures += 1;
    state.last_error = error == null ? null : String(error);
    const backoffMs = Math.min(
      this.maxBackoffMs,
      this.baseBackoffMs * Math.max(1, state.failures)
    );
    state.backoff_until_ms = this.nowFn() + backoffMs;
    const urls = this.urls();
    const index = urls.indexOf(normalized);
    if (index >= 0) {
      this._cursor = (index + 1) % Math.max(1, urls.length);
    }
  }

  snapshot() {
    const urls = this.urls();
    const nowMs = this.nowFn();
    const unavailable = urls.filter((url) => {
      const state = this._state.get(url);
      return state && state.backoff_until_ms > nowMs;
    }).length;
    return {
      activeUrl: this.activeUrl,
      totalCount: urls.length,
      degraded: unavailable > 0 && urls.length > 1,
      endpoints: urls.map((url) => {
        const state = this._state.get(url);
        return {
          url,
          failures: state?.failures || 0,
          backoffUntilMs: state?.backoff_until_ms || 0,
          lastSuccessMs: state?.last_success_ms || 0,
          lastError: state?.last_error || null,
        };
      }),
    };
  }
}
