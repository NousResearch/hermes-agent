const DEFAULT_MAINNET_KNS_URL = "https://api.knsdomains.org/mainnet/api/v1";
const DEFAULT_TESTNET_KNS_URL = "https://api.knsdomains.org/tn10/api/v1";
const DEFAULT_CACHE_TTL_MS = 15 * 60 * 1000;
const DEFAULT_NEGATIVE_CACHE_TTL_MS = 5 * 60 * 1000;

function joinUrl(baseUrl, path) {
  const normalizedBase = String(baseUrl || "").replace(/\/+$/, "");
  const normalizedPath = String(path || "").replace(/^\/+/, "");
  return new URL(`${normalizedBase}/${normalizedPath}`);
}

export function defaultKnsUrlForNetwork(network = "mainnet") {
  return String(network || "").trim().toLowerCase().startsWith("mainnet")
    ? DEFAULT_MAINNET_KNS_URL
    : DEFAULT_TESTNET_KNS_URL;
}

export function normalizeKnsName(value) {
  const trimmed = String(value || "").trim().toLowerCase();
  if (!trimmed) {
    return null;
  }
  const fullName = trimmed.endsWith(".kas") ? trimmed : `${trimmed}.kas`;
  const label = fullName.slice(0, -4);
  if (!label || !/^[a-z0-9-]+$/i.test(label)) {
    return null;
  }
  return fullName;
}

function isCacheEntryFresh(entry, nowMs) {
  return Boolean(entry) && Number(entry.expires_at_ms || 0) > nowMs;
}

export class KnsClient {
  constructor({
    baseUrl,
    network = "mainnet",
    fetchImpl,
    nowFn = () => Date.now(),
    cacheTtlMs = DEFAULT_CACHE_TTL_MS,
    negativeCacheTtlMs = DEFAULT_NEGATIVE_CACHE_TTL_MS,
  } = {}) {
    this.fetchImpl = fetchImpl || fetch;
    this.nowFn = nowFn;
    this.cacheTtlMs = cacheTtlMs;
    this.negativeCacheTtlMs = negativeCacheTtlMs;
    this.baseUrl = String(baseUrl || defaultKnsUrlForNetwork(network)).trim();
  }

  isEnabled() {
    return Boolean(this.baseUrl);
  }

  async resolveTarget(target, cache = {}) {
    const normalizedName = normalizeKnsName(target);
    if (!normalizedName || !this.isEnabled()) {
      return null;
    }

    const nowMs = this.nowFn();
    const cacheKey = normalizedName.toLowerCase();
    const cached = cache?.by_name?.[cacheKey];
    if (isCacheEntryFresh(cached, nowMs)) {
      return cached.address || null;
    }

    const encodedName = encodeURIComponent(normalizedName);
    const response = await this._fetchJson(`/${encodedName}/owner`);
    const ownerAddress = String(response?.data?.owner || "").trim();
    const assetName = String(response?.data?.asset || "").trim().toLowerCase();
    const success = Boolean(response?.success) && ownerAddress && assetName === cacheKey;
    const expiresAtMs = nowMs + (success ? this.cacheTtlMs : this.negativeCacheTtlMs);

    cache.by_name[cacheKey] = {
      key: cacheKey,
      address: success ? ownerAddress : null,
      resolved_at_ms: nowMs,
      expires_at_ms: expiresAtMs,
      error: success ? null : "KNS domain did not resolve",
    };

    if (success) {
      cache.by_address[ownerAddress.toLowerCase()] = {
        key: ownerAddress.toLowerCase(),
        name: cacheKey,
        resolved_at_ms: nowMs,
        expires_at_ms: nowMs + this.cacheTtlMs,
        error: null,
      };
      return ownerAddress;
    }

    return null;
  }

  async lookupPrimaryName(address, cache = {}) {
    const normalizedAddress = String(address || "").trim();
    if (!normalizedAddress || !this.isEnabled()) {
      return null;
    }

    const nowMs = this.nowFn();
    const cacheKey = normalizedAddress.toLowerCase();
    const cached = cache?.by_address?.[cacheKey];
    if (isCacheEntryFresh(cached, nowMs)) {
      return cached.name || null;
    }

    const encodedAddress = encodeURIComponent(normalizedAddress);
    const response = await this._fetchJson(`/primary-name/${encodedAddress}`);
    const domain = String(response?.data?.domain?.name || "").trim().toLowerCase();
    const ownerAddress = String(response?.data?.ownerAddress || normalizedAddress).trim();
    const normalizedName = normalizeKnsName(domain);
    const success = Boolean(response?.success) && normalizedName;
    const expiresAtMs = nowMs + (success ? this.cacheTtlMs : this.negativeCacheTtlMs);

    cache.by_address[cacheKey] = {
      key: cacheKey,
      name: success ? normalizedName : null,
      resolved_at_ms: nowMs,
      expires_at_ms: expiresAtMs,
      error: success ? null : "No KNS primary name found",
    };

    if (success) {
      cache.by_name[normalizedName] = {
        key: normalizedName,
        address: ownerAddress,
        resolved_at_ms: nowMs,
        expires_at_ms: nowMs + this.cacheTtlMs,
        error: null,
      };
      return normalizedName;
    }

    return null;
  }

  async _fetchJson(path) {
    const response = await this.fetchImpl(joinUrl(this.baseUrl, path), {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`KNS request failed (${response.status}): ${body}`);
    }
    return await response.json();
  }
}
