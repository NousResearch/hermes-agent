function isBlockedUrl(urlStr) {
  try {
    const u = new URL(urlStr);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false;
    const h = u.hostname;
    if (h === '127.0.0.1' || h === 'localhost' || h === '::1' || h === '0.0.0.0') return true;
    if (h.startsWith('127.') || h.startsWith('10.') || h.startsWith('169.254.')) return true;
    if (h.startsWith('192.168.')) return true;
    const p = h.split('.');
    if (p.length === 4 && p[0] === '172') {
      const n = parseInt(p[1], 10);
      if (n >= 16 && n <= 31) return true;
    }
    return false;
  } catch (e) { return false; }
}

module.exports = { isBlockedUrl, MAX_DOWNLOAD_BYTES: 50 * 1024 * 1024, MAX_FETCH_BYTES: 16 * 1024 * 1024 };
