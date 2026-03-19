/**
 * DEXAI - Cosmos Collective AI-Powered DEX Screener
 * Complete Frontend Application
 */

/* ============================================
   API BASE & CONSTANTS
   ============================================ */

const API = window.location.pathname.startsWith('/dex') ? '/dex/api' : (window.location.pathname.startsWith('/DEXAI') ? '/DEXAI/api' : '/api');

const PLACEHOLDER = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Ccircle cx='20' cy='20' r='20' fill='%231a1a28'/%3E%3Ctext x='20' y='25' text-anchor='middle' fill='%234a5568' font-size='16'%3E%3F%3C/text%3E%3C/svg%3E";

/* ============================================
   STATE
   ============================================ */

let currentSort = 'trending';
let currentOffset = 0;
let currentToken = null;
let chartInstance = null;
let allTokens = [];
let ws = null;
let refreshInterval = null;
let lastFetchTime = null;
let chartResizeObserver = null;
let liveInterval = null;
let liveSeries = null;
let liveCandleSeries = null;
let liveVolumeSeries = null;
let liveDataPoints = [];
let liveCandles = [];
let currentCandle = null;
let liveLastPrice = null;
let livePriceDirection = 0;
let chartType = 'candle'; // 'candle', 'line', 'bar'
let pumpHistory = [];     // recent price changes for velocity detection
let pumpOverlayActive = false;
let tradesInterval = null;

/* Wallet & Boost */
const ECOSYSTEM_WALLET = '3fSS5RVErbgcVgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw';
const FARNS_MINT = '9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS';
const BOOST_PRICES = { 1: 25, 2: 50, 3: 100 };
let walletProvider = null;
let walletAddress = null;
let solPrice = null;
let selectedBoostLevel = 1;

/* X Connection */
let xConnected = false;
let xUsername = null;

/* ============================================
   UTILITY FUNCTIONS
   ============================================ */

function formatPrice(price) {
    if (price === null || price === undefined || isNaN(price)) return '--';
    const p = Number(price);
    if (p === 0) return '$0.00';
    if (p < 0.0000001) return '$' + p.toExponential(2);
    if (p < 0.00001) return '$' + p.toFixed(9);
    if (p < 0.001) return '$' + p.toFixed(7);
    if (p < 0.01) return '$' + p.toFixed(6);
    if (p < 1) return '$' + p.toFixed(4);
    if (p < 1000) return '$' + p.toFixed(2);
    return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    const n = Number(num);
    if (n === 0) return '$0';
    if (n >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + n.toFixed(2);
}

function formatPercent(pct) {
    if (pct === null || pct === undefined || isNaN(pct)) return '--';
    const p = Number(pct);
    const sign = p >= 0 ? '+' : '';
    return sign + p.toFixed(2) + '%';
}

function formatAddr(addr) {
    if (!addr || addr.length < 10) return addr || '--';
    return addr.slice(0, 6) + '...' + addr.slice(-4);
}

function timeAgo(timestamp) {
    if (!timestamp) return 'never';
    const now = Date.now();
    const ts = typeof timestamp === 'string' ? new Date(timestamp).getTime() : timestamp;
    const diff = Math.max(0, now - ts);
    const seconds = Math.floor(diff / 1000);
    if (seconds < 5) return 'just now';
    if (seconds < 60) return seconds + 's ago';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + 'm ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + 'h ago';
    const days = Math.floor(hours / 24);
    return days + 'd ago';
}

function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

function showToast(msg, type) {
    type = type || 'info';
    const container = document.getElementById('toasts');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = msg;
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });
    setTimeout(() => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => { toast.remove(); }, 400);
    }, 4000);
}

function chgClass(val) {
    if (val === null || val === undefined || isNaN(val)) return '';
    return Number(val) >= 0 ? 'pos' : 'neg';
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/* ============================================
   PARTICLES
   ============================================ */

function initParticles() {
    const canvas = document.getElementById('ambientCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let particles = [];
    const PARTICLE_COUNT = 100;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener('resize', resize);

    // Create particles
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.5,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            opacity: Math.random() * 0.15 + 0.05
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            // Wrap around edges
            if (p.x < -10) p.x = canvas.width + 10;
            if (p.x > canvas.width + 10) p.x = -10;
            if (p.y < -10) p.y = canvas.height + 10;
            if (p.y > canvas.height + 10) p.y = -10;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 255, 136, ' + p.opacity + ')';
            ctx.fill();
        }
        requestAnimationFrame(draw);
    }

    draw();
}

/* ============================================
   TICKER TAPE
   ============================================ */

function buildTicker(tokens) {
    const ticker = document.getElementById('ticker');
    if (!ticker || !tokens || tokens.length === 0) return;

    const items = tokens.slice(0, 30).map(t => {
        const chg = (t.priceChange && t.priceChange.h24 !== undefined) ? t.priceChange.h24 : null;
        const cls = chgClass(chg);
        const sym = escapeHtml(t.symbol || '???');
        const price = formatPrice(t.price);
        const chgStr = chg !== null ? formatPercent(chg) : '--';
        return '<span class="ticker-item">' +
            '<span class="ti-sym">' + sym + '</span>' +
            '<span class="ti-price">' + price + '</span>' +
            '<span class="ti-chg ' + cls + '">' + chgStr + '</span>' +
            '</span>';
    }).join('');

    // Duplicate content for seamless scrolling loop
    ticker.innerHTML = '<div class="ticker-track">' + items + items + '</div>';
}

/* ============================================
   SKELETON LOADING
   ============================================ */

function showSkeletons(count) {
    count = count || 12;
    const tbody = document.getElementById('tblBody');
    if (!tbody) return;
    let html = '';
    for (let i = 0; i < count; i++) {
        html += '<tr class="skel-row">';
        for (let j = 0; j < 12; j++) {
            html += '<td><div class="skel"></div></td>';
        }
        html += '</tr>';
    }
    tbody.innerHTML = html;
}

/* ============================================
   LOAD TOKENS (LIST VIEW)
   ============================================ */

async function loadTokens(sort, limit, offset) {
    sort = sort || currentSort;
    limit = limit || 100;
    offset = offset || 0;
    currentSort = sort;
    currentOffset = offset;

    // Update nav active state
    const navBtns = document.querySelectorAll('#mainNav .nav-btn');
    navBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-sort') === sort);
    });

    // Update view title
    const titles = {
        trending: 'Trending',
        collective: 'Collective Picks',
        whales: 'Whale Heat',
        volume: 'Top Volume',
        velocity: 'High Velocity',
        'new': 'New Pairs',
        gainers: 'Top Gainers',
        losers: 'Top Losers'
    };
    const viewTitle = document.getElementById('viewTitle');
    if (viewTitle) viewTitle.textContent = titles[sort] || 'Tokens';

    // Show skeletons while loading
    if (offset === 0) {
        showSkeletons(12);
    }

    try {
        const url = API + '/tokens?sort=' + encodeURIComponent(sort) +
            '&limit=' + limit + '&offset=' + offset;
        const res = await fetch(url);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        const tokens = data.tokens || [];
        allTokens = offset === 0 ? tokens : allTokens.concat(tokens);

        renderTable(tokens, offset);

        // Update live count
        const liveCount = document.getElementById('liveCount');
        if (liveCount) liveCount.textContent = (data.total || allTokens.length).toLocaleString();

        // Update ticker on first load
        if (offset === 0 && tokens.length > 0) {
            buildTicker(tokens);
        }

        // Update "updated X ago"
        lastFetchTime = Date.now();
        updateMetaRefresh();

        // Show/hide load more
        const loadMoreEl = document.getElementById('loadMore');
        if (loadMoreEl) {
            const total = data.total || 0;
            loadMoreEl.style.display = (allTokens.length < total) ? 'flex' : 'none';
        }
    } catch (err) {
        console.error('Failed to load tokens:', err);
        const tbody = document.getElementById('tblBody');
        if (tbody && offset === 0) {
            tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;padding:40px;color:#666680;">Failed to load tokens. Retrying...</td></tr>';
        }
        showToast('Failed to load tokens', 'error');
    }
}

function updateMetaRefresh() {
    const el = document.getElementById('metaRefresh');
    if (el && lastFetchTime) {
        el.textContent = 'Updated ' + timeAgo(lastFetchTime);
    }
}

/* ============================================
   RENDER TABLE
   ============================================ */

function renderTable(tokens, offset) {
    offset = offset || 0;
    const tbody = document.getElementById('tblBody');
    if (!tbody) return;

    if ((!tokens || tokens.length === 0) && offset === 0) {
        tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;padding:40px;color:#666680;">No tokens found</td></tr>';
        return;
    }

    const html = tokens.map((t, i) => {
        const rank = offset + i + 1;
        const sym = escapeHtml(t.symbol || '???');
        const name = escapeHtml(t.name || '');
        const addr = t.address || '';
        const imgSrc = t.imageUrl || PLACEHOLDER;

        // Price changes
        const chg5m = t.priceChange ? t.priceChange.m5 : null;
        const chg1h = t.priceChange ? t.priceChange.h1 : null;
        const chg24h = t.priceChange ? t.priceChange.h24 : null;

        // Volume
        const vol24 = t.volume ? t.volume.h24 : null;

        // Txns for mini bar
        const txnH1 = t.txns && t.txns.h1 ? t.txns.h1 : { buys: 0, sells: 0 };
        const totalTxn = (txnH1.buys || 0) + (txnH1.sells || 0);
        const buyPct = totalTxn > 0 ? Math.round(((txnH1.buys || 0) / totalTxn) * 100) : 50;

        // AI badge
        let aiBadge = '<span class="ai-badge">--</span>';
        if (t.aiScore !== null && t.aiScore !== undefined) {
            const sc = Number(t.aiScore);
            const aCls = sc >= 70 ? 'high' : (sc >= 40 ? 'mid' : 'low');
            aiBadge = '<span class="ai-badge ' + aCls + '">' + sc + '</span>';
        }

        // Platform badges
        let platformBadge = '';
        if (t.isBags || (t.platform === 'bags')) {
            platformBadge = '<span class="platform-badge bags-badge">BAGS</span>';
        } else if (t.platform === 'bonk') {
            platformBadge = '<span class="platform-badge bonk-badge">BONK</span>';
        }
        // Collective pick indicator
        let pickBadge = '';
        if (t.collectivePick) pickBadge = '<span class="pick-badge" title="Collective Pick">&#9733;</span>';
        // Whale heat indicator
        let whaleBadge = '';
        if (t.whaleHeat > 5) whaleBadge = '<span class="whale-badge" title="Whale Activity: ' + Math.round(t.whaleHeat) + '">&#x1F40B;</span>';

        return '<tr onclick="viewToken(\'' + addr + '\')">' +
            '<td class="td-rank">' + rank + '</td>' +
            '<td class="td-token"><div class="tok-cell">' +
                '<img src="' + imgSrc + '" alt="' + sym + '" onerror="this.src=\'' + PLACEHOLDER + '\'" class="tok-img">' +
                '<div class="tok-info"><span class="tok-sym">' + sym + platformBadge + pickBadge + whaleBadge + '</span><span class="tok-name">' + name + '</span></div>' +
            '</div></td>' +
            '<td class="td-price mono">' + formatPrice(t.price) + '</td>' +
            '<td class="td-chg mono ' + chgClass(chg5m) + '">' + formatPercent(chg5m) + '</td>' +
            '<td class="td-chg mono ' + chgClass(chg1h) + '">' + formatPercent(chg1h) + '</td>' +
            '<td class="td-chg mono ' + chgClass(chg24h) + '">' + formatPercent(chg24h) + '</td>' +
            '<td class="td-vol mono">' + formatNumber(vol24) + '</td>' +
            '<td class="td-liq mono">' + formatNumber(t.liquidity) + '</td>' +
            '<td class="td-mcap mono">' + formatNumber(t.marketCap || t.fdv) + '</td>' +
            '<td class="td-txn"><div class="txn-mini"><div class="txn-mini-buy" style="width:' + buyPct + '%"></div></div></td>' +
            '<td class="td-ai">' + aiBadge + '</td>' +
            '<td class="td-act"><button class="view-btn" onclick="event.stopPropagation();viewToken(\'' + addr + '\')">View</button></td>' +
            '</tr>';
    }).join('');

    if (offset === 0) {
        tbody.innerHTML = html;
    } else {
        tbody.insertAdjacentHTML('beforeend', html);
    }
}

/* ============================================
   LOAD MORE
   ============================================ */

function loadMore() {
    currentOffset += 100;
    loadTokens(currentSort, 100, currentOffset);
}

/* ============================================
   VIEW TOKEN (DETAIL VIEW)
   ============================================ */

async function viewToken(address) {
    if (!address) return;

    // Switch views
    const listView = document.getElementById('listView');
    const detailView = document.getElementById('detailView');
    if (listView) listView.style.display = 'none';
    if (detailView) detailView.classList.remove('hidden');

    // Clear previous chart
    destroyChart();

    // Reset panels to loading state
    resetDetailPanels();

    // Push state for browser back
    try {
        const basePath = window.location.pathname.startsWith('/dex') ? '/dex' : '';
        window.history.pushState({ address: address }, '', basePath + '/token/' + address);
    } catch (e) { /* ignore */ }

    try {
        const res = await fetch(API + '/token/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const token = data.token;
        if (!token) throw new Error('Token not found');

        currentToken = token;
        renderDetail(token);

        // Load supplementary data in parallel — default to LIVE 1s chart
        loadLiveChart(address);
        loadAI(address);
        loadQuantum(address);
        loadQuantumTrading(address);
        loadTrades(address);
        loadBonding(address);
        loadXBadge(address);
    } catch (err) {
        console.error('Failed to load token:', err);
        showToast('Failed to load token details', 'error');
    }
}

function resetDetailPanels() {
    const aiPanel = document.getElementById('aiPanel');
    if (aiPanel) {
        aiPanel.innerHTML = '<div class="ai-loading"><div class="ai-ring"></div><span>Querying the Collective...</span></div>';
    }
    const qPanel = document.getElementById('quantumPanel');
    if (qPanel) {
        qPanel.innerHTML = '<div class="q-loading"><div class="q-spinner"></div><span>Running simulations...</span></div>';
    }
    // Reset quantum trading panel
    var qtGated = document.getElementById('qtGated');
    var qtSignal = document.getElementById('qtSignal');
    if (qtGated) qtGated.classList.remove('hidden');
    if (qtSignal) qtSignal.classList.add('hidden');
    currentQuantumSignal = null;
    if (quantumWs) { try { quantumWs.close(); } catch {} quantumWs = null; }
}

/* ============================================
   RENDER DETAIL
   ============================================ */

function renderDetail(token) {
    if (!token) return;

    // Image
    const dImg = document.getElementById('dImg');
    if (dImg) {
        dImg.src = token.imageUrl || PLACEHOLDER;
        dImg.onerror = function () { this.src = PLACEHOLDER; };
    }

    // Symbol, name, address
    setText('dSym', token.symbol || '???');
    setText('dName', token.name || '');
    setText('dAddr', token.address || '');

    // DexScreener link
    const dDexLink = document.getElementById('dDexLink');
    if (dDexLink) {
        dDexLink.href = token.url || ('https://dexscreener.com/solana/' + (token.address || ''));
    }

    // Price
    setText('dPrice', formatPrice(token.price));

    // Main change (24h as default)
    const chg24 = token.priceChange ? token.priceChange.h24 : null;
    const dChg = document.getElementById('dChg');
    if (dChg) {
        dChg.textContent = formatPercent(chg24);
        dChg.className = 'price-chg ' + chgClass(chg24);
    }

    // Per-timeframe changes
    setChg('dChg5m', token.priceChange ? token.priceChange.m5 : null);
    setChg('dChg1h', token.priceChange ? token.priceChange.h1 : null);
    setChg('dChg6h', token.priceChange ? token.priceChange.h6 : null);
    setChg('dChg24h', token.priceChange ? token.priceChange.h24 : null);

    // Stats
    setText('dVol', formatNumber(token.volume ? token.volume.h24 : null));
    setText('dLiq', formatNumber(token.liquidity));
    setText('dFdv', formatNumber(token.fdv));
    setText('dMcap', formatNumber(token.marketCap));

    // Buy/Sell pressure (using h1 data)
    const txH1 = token.txns && token.txns.h1 ? token.txns.h1 : { buys: 0, sells: 0 };
    const totalTx = (txH1.buys || 0) + (txH1.sells || 0);
    const buyPct = totalTx > 0 ? ((txH1.buys || 0) / totalTx * 100) : 50;

    const pressureFill = document.getElementById('pressureFill');
    if (pressureFill) pressureFill.style.width = buyPct.toFixed(1) + '%';

    setText('dBuys', (txH1.buys || 0) + ' Buys');
    setText('dSells', (txH1.sells || 0) + ' Sells');

    // Transaction grid per timeframe
    setTxn('5m', token.txns ? token.txns.m5 : null);
    setTxn('1h', token.txns ? token.txns.h1 : null);
    setTxn('6h', token.txns ? token.txns.h6 : null);
    setTxn('24h', token.txns ? token.txns.h24 : null);

    // Links panel
    renderLinksPanel(token);

    // Pair info panel
    renderPairPanel(token);
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setChg(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = formatPercent(val);
    el.className = 'ps-v ' + chgClass(val);
}

function setTxn(tf, data) {
    const buys = data ? (data.buys || 0) : 0;
    const sells = data ? (data.sells || 0) : 0;
    setText('txB' + tf, buys.toLocaleString());
    setText('txS' + tf, sells.toLocaleString());
}

function renderLinksPanel(token) {
    const panel = document.getElementById('linksPanel');
    if (!panel) return;

    let html = '';

    if (token.websites && token.websites.length > 0) {
        token.websites.forEach(w => {
            const url = typeof w === 'string' ? w : (w.url || w.label || '');
            if (url) {
                html += '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener" class="link-item">Website</a>';
            }
        });
    }

    if (token.socials && token.socials.length > 0) {
        token.socials.forEach(s => {
            const url = typeof s === 'string' ? s : (s.url || '');
            const label = (s.type || s.platform || 'Link');
            if (url) {
                html += '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener" class="link-item">' + escapeHtml(label) + '</a>';
            }
        });
    }

    if (!html) {
        html = '<span class="no-data">No verified links</span>';
    }

    panel.innerHTML = html;
}

function renderPairPanel(token) {
    const panel = document.getElementById('pairPanel');
    if (!panel) return;

    let html = '';

    if (token.pairAddress) {
        html += '<div class="pair-row"><span class="pair-label">Pair</span><span class="pair-val mono">' + formatAddr(token.pairAddress) + '</span></div>';
    }
    if (token.dexId) {
        html += '<div class="pair-row"><span class="pair-label">DEX</span><span class="pair-val">' + escapeHtml(token.dexId) + '</span></div>';
    }
    if (token.chainId) {
        html += '<div class="pair-row"><span class="pair-label">Chain</span><span class="pair-val">' + escapeHtml(token.chainId) + '</span></div>';
    }
    if (token.pairCreatedAt) {
        html += '<div class="pair-row"><span class="pair-label">Created</span><span class="pair-val">' + timeAgo(token.pairCreatedAt) + '</span></div>';
    }
    if (token.priceNative) {
        html += '<div class="pair-row"><span class="pair-label">Native Price</span><span class="pair-val mono">' + token.priceNative + '</span></div>';
    }

    if (!html) {
        html = '<span class="no-data">--</span>';
    }

    panel.innerHTML = html;
}

/* ============================================
   CHART (Lightweight Charts)
   ============================================ */

function destroyChart() {
    if (liveInterval) { clearInterval(liveInterval); liveInterval = null; }
    // Unsubscribe from WebSocket price feed
    if (ws && ws.readyState === 1 && window.currentDetailAddress) {
        try { ws.send(JSON.stringify({ type: 'unsubscribe', token: window.currentDetailAddress })); } catch (e) {}
    }
    liveSeries = null;
    liveCandleSeries = null;
    liveVolumeSeries = null;
    liveDataPoints = [];
    liveCandles = [];
    currentCandle = null;
    liveLastPrice = null;
    livePriceDirection = 0;
    liveEmaPrice = null;
    pumpHistory = [];
    pumpOverlayActive = false;
    var overlay = document.getElementById('pumpOverlay');
    if (overlay) { overlay.className = 'pump-overlay'; }
    if (chartResizeObserver) {
        chartResizeObserver.disconnect();
        chartResizeObserver = null;
    }
    if (chartInstance) {
        try { chartInstance.remove(); } catch (e) { /* ignore */ }
        chartInstance = null;
    }
    const container = document.getElementById('chartContainer');
    if (container) container.innerHTML = '';
}

// Chart type toggle handler
function initChartTypeToggle() {
    var toggle = document.getElementById('chartTypeToggle');
    if (!toggle) return;
    toggle.querySelectorAll('.ct-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var newType = btn.getAttribute('data-type');
            if (newType === chartType) return;
            chartType = newType;
            toggle.querySelectorAll('.ct-btn').forEach(function(b) { b.classList.remove('active'); });
            btn.classList.add('active');
            // Reload current chart with new type
            if (window.currentDetailAddress) {
                var activeTf = document.querySelector('#tfTabs .tf.active');
                var tf = activeTf ? activeTf.getAttribute('data-tf') : '15m';
                if (tf === 'live') {
                    loadLiveChart(window.currentDetailAddress);
                } else {
                    loadChart(window.currentDetailAddress, tf);
                }
            }
        });
    });
}

// Pump velocity detection — returns 'none', 'mild', or 'extreme'
function detectPumpLevel(price) {
    var now = Date.now();
    pumpHistory.push({ price: price, ts: now });
    // Keep last 15 seconds of data
    pumpHistory = pumpHistory.filter(function(p) { return now - p.ts < 15000; });
    if (pumpHistory.length < 3) return 'none';
    var oldest = pumpHistory[0];
    var newest = pumpHistory[pumpHistory.length - 1];
    var pctChange = ((newest.price - oldest.price) / oldest.price) * 100;
    // Extreme pump: >5% in 15 seconds
    if (pctChange > 5) return 'extreme';
    // Mild pump: >2% in 15 seconds
    if (pctChange > 2) return 'mild';
    return 'none';
}

var lastPumpOverlayLevel = 'none';
function updatePumpOverlay(level) {
    if (level === lastPumpOverlayLevel) return; // skip if unchanged — avoids layout thrash
    lastPumpOverlayLevel = level;
    var overlay = document.getElementById('pumpOverlay');
    if (!overlay) return;
    if (level === 'extreme') {
        overlay.className = 'pump-overlay pumping';
        pumpOverlayActive = true;
    } else if (level === 'mild') {
        overlay.className = 'pump-overlay pumping-mild';
        pumpOverlayActive = true;
    } else {
        if (pumpOverlayActive) {
            overlay.className = 'pump-overlay';
            pumpOverlayActive = false;
        }
    }
}

// Fixed chart height — responsive via CSS media queries, not JS feedback loops
var CHART_HEIGHT = 450;
function getChartHeight() {
    if (window.innerWidth <= 768) return 320;
    if (window.innerWidth <= 1200) return 380;
    return CHART_HEIGHT;
}

async function loadChart(address, timeframe) {
    timeframe = timeframe || '15m';
    const container = document.getElementById('chartContainer');
    if (!container) return;
    window.currentDetailAddress = address;

    // Update active timeframe tab
    const tfBtns = document.querySelectorAll('#tfTabs .tf');
    tfBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tf') === timeframe);
    });

    destroyChart();

    var chartH = getChartHeight();
    container.style.height = '';
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666680;">Loading chart...</div>';

    try {
        const res = await fetch(API + '/chart/' + encodeURIComponent(address) + '?timeframe=' + timeframe);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const candles = data.candles || [];

        if (candles.length === 0) {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666680;">No chart data available</div>';
            return;
        }

        container.innerHTML = '';

        if (typeof LightweightCharts === 'undefined') {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666680;">Chart library not loaded</div>';
            return;
        }

        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: chartH,
            layout: {
                background: { type: 'solid', color: '#0a0a14' },
                textColor: '#8892a4'
            },
            grid: {
                vertLines: { color: 'rgba(255,255,255,0.04)' },
                horzLines: { color: 'rgba(255,255,255,0.04)' }
            },
            crosshair: {
                mode: 0,
                vertLine: { color: 'rgba(0,255,136,0.2)', width: 1, style: 2 },
                horzLine: { color: 'rgba(0,255,136,0.2)', width: 1, style: 2 }
            },
            rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
            timeScale: {
                borderColor: 'rgba(255,255,255,0.06)',
                timeVisible: true,
                secondsVisible: false
            }
        });

        chartInstance = chart;

        // Prepare candle data
        const candleData = candles
            .map(c => ({
                time: typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000),
                open: Number(c.open), high: Number(c.high),
                low: Number(c.low), close: Number(c.close),
                volume: Number(c.volume || 0)
            }))
            .sort((a, b) => a.time - b.time);

        const dedupedCandles = [];
        const seenTimes = new Set();
        for (const c of candleData) {
            if (!seenTimes.has(c.time)) {
                seenTimes.add(c.time);
                dedupedCandles.push(c);
            }
        }

        // Render based on chart type
        if (chartType === 'line') {
            // Thick animated line chart with gradient fill
            var lineSeries = chart.addAreaSeries({
                topColor: 'rgba(0, 255, 136, 0.25)',
                bottomColor: 'rgba(0, 255, 136, 0.0)',
                lineColor: '#00ff88',
                lineWidth: 3,
                crosshairMarkerVisible: true,
                crosshairMarkerRadius: 5,
                crosshairMarkerBorderColor: '#00ff88',
                crosshairMarkerBackgroundColor: 'rgba(0,255,136,0.3)',
            });
            lineSeries.setData(dedupedCandles.map(c => ({ time: c.time, value: c.close })));
        } else if (chartType === 'bar') {
            // Bar chart — each candle as a colored bar
            var barSeries = chart.addHistogramSeries({
                priceFormat: { type: 'price', minMove: 0.0000001, precision: 10 },
            });
            barSeries.setData(dedupedCandles.map(c => ({
                time: c.time,
                value: c.close,
                color: c.close >= c.open ? 'rgba(0,255,136,0.8)' : 'rgba(255,51,102,0.8)'
            })));
        } else {
            // Candlestick (default)
            var candleSeries = chart.addCandlestickSeries({
                upColor: '#00ff88',
                downColor: '#ff3366',
                borderUpColor: '#00ff88',
                borderDownColor: '#ff3366',
                wickUpColor: 'rgba(0,255,136,0.6)',
                wickDownColor: 'rgba(255,51,102,0.6)'
            });
            candleSeries.setData(dedupedCandles);
        }

        // Volume histogram (shown for all chart types)
        const volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume'
        });
        chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

        const dedupedVolume = [];
        const seenVolTimes = new Set();
        for (const c of dedupedCandles) {
            if (!seenVolTimes.has(c.time)) {
                seenVolTimes.add(c.time);
                dedupedVolume.push({
                    time: c.time,
                    value: c.volume,
                    color: c.close >= c.open ? 'rgba(0,255,136,0.25)' : 'rgba(255,51,102,0.25)'
                });
            }
        }
        volumeSeries.setData(dedupedVolume);

        chart.timeScale().fitContent();

        // Responsive resize — only update width, height is fixed
        chartResizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width } = entry.contentRect;
                if (width > 0) chart.applyOptions({ width: width });
            }
        });
        chartResizeObserver.observe(container);

    } catch (err) {
        console.error('Failed to load chart:', err);
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666680;">Chart unavailable</div>';
    }
}

/* ============================================
   AI COLLECTIVE ANALYSIS
   ============================================ */

async function loadAI(address) {
    const panel = document.getElementById('aiPanel');
    if (!panel) return;

    panel.innerHTML = '<div class="ai-loading"><div class="ai-ring"></div><span>Querying the Collective...</span></div>';

    try {
        const res = await fetch(API + '/ai/score/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        if (data.score === null || data.score === undefined) {
            panel.innerHTML = '<div class="ai-unavailable"><span class="no-data">Collective analysis not yet available for this token.</span></div>';
            return;
        }

        const score = Number(data.score);
        const scoreCls = score >= 70 ? 'high' : (score >= 40 ? 'mid' : 'low');
        const verdict = data.verdict || 'ANALYZING';

        let html = '<div class="ai-result">';
        html += '<div class="ai-score-row">';
        html += '<span class="ai-score-big ' + scoreCls + '">' + score + '</span>';
        html += '<span class="ai-verdict ' + scoreCls + '">' + escapeHtml(verdict) + '</span>';
        html += '</div>';

        // Detail rows
        html += '<div class="ai-details">';

        if (data.rugProbability !== null && data.rugProbability !== undefined) {
            const rugPct = (Number(data.rugProbability) * 100).toFixed(1);
            const rugCls = Number(data.rugProbability) < 0.3 ? 'pos' : (Number(data.rugProbability) < 0.6 ? 'warn' : 'neg');
            html += '<div class="ai-detail-row"><span class="ai-dl">Rug Probability</span><span class="ai-dv ' + rugCls + '">' + rugPct + '%</span></div>';
        }

        if (data.cabalScore !== null && data.cabalScore !== undefined) {
            html += '<div class="ai-detail-row"><span class="ai-dl">Cabal Score</span><span class="ai-dv">' + Number(data.cabalScore).toFixed(0) + '/100</span></div>';
        }

        if (data.bundleDetected !== null && data.bundleDetected !== undefined) {
            const bundleTxt = data.bundleDetected ? 'DETECTED' : 'None';
            const bundleCls = data.bundleDetected ? 'neg' : 'pos';
            html += '<div class="ai-detail-row"><span class="ai-dl">Bundle</span><span class="ai-dv ' + bundleCls + '">' + bundleTxt + '</span></div>';
        }

        if (data.swarmSentiment) {
            html += '<div class="ai-detail-row"><span class="ai-dl">Swarm Sentiment</span><span class="ai-dv">' + escapeHtml(data.swarmSentiment) + '</span></div>';
        }

        if (data.agents && data.agents.length > 0) {
            html += '<div class="ai-agents"><span class="ai-dl">Agents</span><div class="ai-agent-list">';
            data.agents.forEach(a => {
                const agentName = typeof a === 'string' ? a : (a.name || a.id || 'Agent');
                html += '<span class="ai-agent-tag">' + escapeHtml(agentName) + '</span>';
            });
            html += '</div></div>';
        }

        html += '</div></div>';
        panel.innerHTML = html;

    } catch (err) {
        console.error('Failed to load AI analysis:', err);
        panel.innerHTML = '<div class="ai-unavailable"><span class="no-data">Collective analysis unavailable.</span></div>';
    }
}

/* ============================================
   QUANTUM SIMULATION
   ============================================ */

async function loadQuantum(address) {
    const panel = document.getElementById('quantumPanel');
    if (!panel) return;

    panel.innerHTML = '<div class="q-loading"><div class="q-spinner"></div><span>Running simulations...</span></div>';

    try {
        const res = await fetch(API + '/quantum/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        if (!data.available) {
            panel.innerHTML = '<div class="q-unavailable"><span class="no-data">Quantum simulation not available for this token.</span></div>';
            return;
        }

        let html = '<div class="q-result">';

        if (data.rugProbability !== null && data.rugProbability !== undefined) {
            const rugPct = (Number(data.rugProbability) * 100).toFixed(1);
            const rugCls = Number(data.rugProbability) < 0.3 ? 'pos' : (Number(data.rugProbability) < 0.6 ? 'warn' : 'neg');
            html += '<div class="q-row"><span class="q-label">Rug Risk</span><span class="q-val q-big ' + rugCls + '">' + rugPct + '%</span></div>';
        }

        if (data.confidence !== null && data.confidence !== undefined) {
            html += '<div class="q-row"><span class="q-label">Confidence</span><span class="q-val">' + (Number(data.confidence) * 100).toFixed(0) + '%</span></div>';
        }

        if (data.simulations) {
            html += '<div class="q-row"><span class="q-label">Simulations</span><span class="q-val">' + Number(data.simulations).toLocaleString() + '</span></div>';
        }

        if (data.priceTargets) {
            html += '<div class="q-targets"><span class="q-label">Price Targets</span>';
            if (data.priceTargets.low !== undefined) {
                html += '<div class="q-target"><span>Bear</span><span class="neg">' + formatPrice(data.priceTargets.low) + '</span></div>';
            }
            if (data.priceTargets.mid !== undefined) {
                html += '<div class="q-target"><span>Base</span><span>' + formatPrice(data.priceTargets.mid) + '</span></div>';
            }
            if (data.priceTargets.high !== undefined) {
                html += '<div class="q-target"><span>Bull</span><span class="pos">' + formatPrice(data.priceTargets.high) + '</span></div>';
            }
            html += '</div>';
        }

        html += '</div>';
        panel.innerHTML = html;

    } catch (err) {
        console.error('Failed to load quantum simulation:', err);
        panel.innerHTML = '<div class="q-unavailable"><span class="no-data">Quantum simulation unavailable.</span></div>';
    }
}

/* ============================================
   QUANTUM TRADING INTELLIGENCE (premium, token-gated)
   ============================================ */

var quantumWs = null;          // WebSocket to /ws/quantum
var quantumAccess = false;     // whether user has FARNS access
var currentQuantumSignal = null;

async function checkQuantumAccess() {
    if (!walletAddress) { quantumAccess = false; return false; }
    try {
        const res = await fetch(API + '/quantum/check-access?wallet=' + encodeURIComponent(walletAddress));
        const data = await res.json();
        quantumAccess = data.hasAccess || false;
        return quantumAccess;
    } catch { quantumAccess = false; return false; }
}

async function loadQuantumTrading(address) {
    var panel = document.getElementById('quantumTradingPanel');
    var gated = document.getElementById('qtGated');
    var signal = document.getElementById('qtSignal');
    if (!panel || !gated || !signal) return;

    // Load public accuracy teaser regardless of access
    loadQuantumAccuracyTeaser();

    // Check access
    var hasAccess = await checkQuantumAccess();

    if (!hasAccess) {
        gated.classList.remove('hidden');
        signal.classList.add('hidden');
        return;
    }

    // User has FARNS — show signal
    gated.classList.add('hidden');
    signal.classList.remove('hidden');

    try {
        var res = await fetch(API + '/quantum/signal/' + encodeURIComponent(address) + '?wallet=' + encodeURIComponent(walletAddress));
        var data = await res.json();
        if (data.available) {
            renderQuantumSignal(data);
        } else {
            signal.innerHTML = '<div class="q-unavailable"><span class="no-data">Generating quantum signal...</span></div>';
        }
    } catch (err) {
        console.error('Quantum trading signal load failed:', err);
        signal.innerHTML = '<div class="q-unavailable"><span class="no-data">Quantum signal unavailable</span></div>';
    }

    // Connect quantum WebSocket for live updates
    connectQuantumWs(address);
}

function renderQuantumSignal(data) {
    var dirEl = document.getElementById('qtDirection');
    var confEl = document.getElementById('qtConfidence');
    var strEl = document.getElementById('qtStrength');
    var emaEl = document.getElementById('qtEma');
    var qEl = document.getElementById('qtQuantum');
    var collEl = document.getElementById('qtCollective');
    var methEl = document.getElementById('qtMethod');
    var reasonEl = document.getElementById('qtReasoning');

    if (!dirEl) return;
    currentQuantumSignal = data;

    // Direction
    var dir = data.direction || 'HOLD';
    dirEl.textContent = dir;
    dirEl.className = 'qt-direction ' + dir.toLowerCase();

    // Confidence
    var conf = Math.round((data.confidence || 0) * 100);
    confEl.textContent = conf + '%';

    // Strength bars (1-5)
    var strength = data.strength || 1;
    var barsHtml = '';
    for (var i = 1; i <= 5; i++) {
        var cls = 'qt-strength-bar';
        if (i <= strength) {
            cls += ' active';
            if (strength >= 4) cls += ' hot';
            else if (strength >= 3) cls += ' warn';
        }
        barsHtml += '<div class="' + cls + '"></div>';
    }
    strEl.innerHTML = barsHtml;

    // Details
    var cross = data.ema_crossover || '--';
    var mom = data.momentum_score !== undefined ? (data.momentum_score > 0 ? '+' : '') + data.momentum_score.toFixed(2) : '--';
    emaEl.innerHTML = '<span class="' + (cross === 'bullish' ? 'pos' : (cross === 'bearish' ? 'neg' : '')) + '">' + cross + '</span> (' + mom + ')';

    var bull = data.quantum_bull_prob !== undefined ? Math.round(data.quantum_bull_prob * 100) + '% bull' : '--';
    var qConf = data.quantum_confidence !== undefined ? Math.round(data.quantum_confidence * 100) + '%' : '';
    qEl.innerHTML = bull + (qConf ? ' <span class="qt-label">' + qConf + ' conf</span>' : '');

    var collDir = data.collective_direction || '--';
    var agents = (data.agents_consulted || []).join(', ');
    collEl.innerHTML = '<span class="' + (collDir === 'bullish' ? 'pos' : (collDir === 'bearish' ? 'neg' : '')) + '">' + collDir + '</span>' + (agents ? ' <span class="qt-label">(' + agents + ')</span>' : '');

    methEl.textContent = data.quantum_method || 'classical';

    // Reasoning
    if (data.reasoning) {
        reasonEl.textContent = data.reasoning;
        reasonEl.classList.remove('hidden');
    } else {
        reasonEl.classList.add('hidden');
    }
}

async function loadQuantumAccuracyTeaser() {
    var el = document.getElementById('qtAccuracyTeaser');
    if (!el) return;
    try {
        var res = await fetch(API + '/quantum/accuracy');
        var data = await res.json();
        if (data.win_rate > 0) {
            el.textContent = 'Signal accuracy: ' + Math.round(data.win_rate * 100) + '% | ' + data.resolved + ' resolved signals';
        } else if (data.total_signals > 0) {
            el.textContent = data.total_signals + ' signals generated — tracking accuracy...';
        }
    } catch {}
}

function connectQuantumWs(address) {
    if (quantumWs) { try { quantumWs.close(); } catch {} quantumWs = null; }
    if (!quantumAccess || !walletAddress) return;

    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var wsUrl = proto + '//' + location.host + '/ws/quantum?wallet=' + encodeURIComponent(walletAddress);
    try {
        quantumWs = new WebSocket(wsUrl);
        quantumWs.onopen = function() {
            if (address) quantumWs.send(JSON.stringify({ type: 'subscribe', token: address }));
        };
        quantumWs.onmessage = function(evt) {
            try {
                var msg = JSON.parse(evt.data);
                if (msg.type === 'quantum_signal' && msg.data) {
                    renderQuantumSignal(msg.data);
                }
            } catch {}
        };
        quantumWs.onclose = function() { quantumWs = null; };
        quantumWs.onerror = function() { quantumWs = null; };
    } catch {}
}

/* ============================================
   LIVE CHART (real-time 1-second candlestick + area)
   ============================================ */

var liveCandleInterval = 5;   // seconds per candle
var liveEmaAlpha = 0.3;       // EMA smoothing factor (0.3 = moderate smoothing)
var liveEmaPrice = null;      // current EMA value

function emaSmooth(price) {
    if (liveEmaPrice === null) { liveEmaPrice = price; return price; }
    liveEmaPrice = liveEmaAlpha * price + (1 - liveEmaAlpha) * liveEmaPrice;
    return liveEmaPrice;
}

// Called by WebSocket price pushes to update the live chart in real-time
function updateLiveChartFromPrice(price) {
    if (!liveSeries || !chartInstance || price <= 0) return;

    var now = Math.floor(Date.now() / 1000);
    var isLineMode = (chartType === 'line');
    var isBarMode = (chartType === 'bar');

    if (liveLastPrice !== null) {
        livePriceDirection = price > liveLastPrice ? 1 : (price < liveLastPrice ? -1 : livePriceDirection);
    }
    liveLastPrice = price;

    var pumpLevel = detectPumpLevel(price);
    updatePumpOverlay(pumpLevel);

    if (isLineMode) {
        var smoothed = emaSmooth(price);
        var lineTime = now;
        if (liveDataPoints.length > 0 && lineTime <= liveDataPoints[liveDataPoints.length - 1].time) {
            lineTime = liveDataPoints[liveDataPoints.length - 1].time + 1;
        }
        liveDataPoints.push({ time: lineTime, value: smoothed });
        liveSeries.update({ time: lineTime, value: smoothed });
    } else if (isBarMode) {
        var barTime = now;
        if (liveCandles.length > 0 && barTime <= liveCandles[liveCandles.length - 1].time) {
            barTime = liveCandles[liveCandles.length - 1].time + 1;
        }
        var barColor = livePriceDirection >= 0 ? 'rgba(0,255,136,0.7)' : 'rgba(255,51,102,0.7)';
        liveCandles.push({ time: barTime, value: price, color: barColor });
        if (liveCandleSeries) liveCandleSeries.update({ time: barTime, value: price, color: barColor });
        var smoothed2 = emaSmooth(price);
        if (liveDataPoints.length > 0 && barTime <= liveDataPoints[liveDataPoints.length - 1].time) {
            barTime = liveDataPoints[liveDataPoints.length - 1].time + 1;
        }
        liveDataPoints.push({ time: barTime, value: smoothed2 });
        liveSeries.update({ time: barTime, value: smoothed2 });
    } else {
        var candleTime = now - (now % liveCandleInterval);
        if (!currentCandle || currentCandle.time !== candleTime) {
            if (currentCandle && liveCandles.length > 0 && candleTime <= liveCandles[liveCandles.length - 1].time) {
                candleTime = liveCandles[liveCandles.length - 1].time + liveCandleInterval;
            }
            currentCandle = { time: candleTime, open: price, high: price, low: price, close: price, ticks: 1 };
            liveCandles.push(currentCandle);
        } else {
            currentCandle.high = Math.max(currentCandle.high, price);
            currentCandle.low = Math.min(currentCandle.low, price);
            currentCandle.close = price;
            currentCandle.ticks++;
        }
        if (liveCandleSeries) {
            liveCandleSeries.update({
                time: currentCandle.time, open: currentCandle.open,
                high: currentCandle.high, low: currentCandle.low, close: currentCandle.close,
            });
        }
        var smoothed3 = emaSmooth(price);
        var lineTime3 = now;
        if (liveDataPoints.length > 0 && lineTime3 <= liveDataPoints[liveDataPoints.length - 1].time) {
            lineTime3 = liveDataPoints[liveDataPoints.length - 1].time + 1;
        }
        liveDataPoints.push({ time: lineTime3, value: smoothed3 });
        liveSeries.update({ time: lineTime3, value: smoothed3 });
        if (liveVolumeSeries) {
            liveVolumeSeries.update({
                time: currentCandle.time, value: currentCandle.ticks,
                color: currentCandle.close >= currentCandle.open ? 'rgba(0,255,136,0.25)' : 'rgba(255,51,102,0.25)',
            });
        }
    }

    // Trim old data
    if (liveCandles.length > 2000) {
        liveCandles = liveCandles.slice(-1500);
        if (liveCandleSeries) liveCandleSeries.setData(liveCandles);
    }
    if (liveDataPoints.length > 5000) {
        liveDataPoints = liveDataPoints.slice(-4000);
        if (liveSeries) liveSeries.setData(liveDataPoints);
    }

    chartInstance.timeScale().scrollToRealTime();

    // Price flash
    var priceEl = document.getElementById('dPrice');
    if (priceEl) {
        priceEl.classList.remove('price-flash-up', 'price-flash-down');
        if (livePriceDirection === 1) priceEl.classList.add('price-flash-up');
        else if (livePriceDirection === -1) priceEl.classList.add('price-flash-down');
        setTimeout(function() { priceEl.classList.remove('price-flash-up', 'price-flash-down'); }, 600);
    }
}

async function loadLiveChart(address) {
    destroyChart();
    var container = document.getElementById('chartContainer');
    if (!container || typeof LightweightCharts === 'undefined') return;
    window.currentDetailAddress = address;

    var tfBtns = document.querySelectorAll('#tfTabs .tf');
    tfBtns.forEach(function(b) { b.classList.toggle('active', b.getAttribute('data-tf') === 'live'); });

    var chartH = getChartHeight();
    container.style.height = '';
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#00ff88;font-size:14px;gap:8px;"><div class="live-tf-dot" style="width:8px;height:8px;border-radius:50%;background:#00ff88;animation:pulse-dot 1s infinite"></div>Connecting 1-second feed...</div>';

    // Seed with 1m candle data
    var seedCandles = [];
    try {
        var seedRes = await fetch(API + '/chart/' + encodeURIComponent(address) + '?timeframe=1m');
        if (seedRes.ok) {
            var seedData = await seedRes.json();
            seedCandles = (seedData.candles || []).map(function(c) {
                return {
                    time: typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000),
                    open: Number(c.open), high: Number(c.high),
                    low: Number(c.low), close: Number(c.close),
                    volume: Number(c.volume || 0)
                };
            }).sort(function(a, b) { return a.time - b.time; });
        }
    } catch (e) { /* ignore */ }

    container.innerHTML = '';

    var chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: chartH,
        layout: { background: { type: 'solid', color: '#0a0a14' }, textColor: '#8892a4' },
        grid: { vertLines: { color: 'rgba(255,255,255,0.04)' }, horzLines: { color: 'rgba(255,255,255,0.04)' } },
        crosshair: {
            mode: 0,
            vertLine: { color: 'rgba(0,255,136,0.2)', width: 1, style: 2 },
            horzLine: { color: 'rgba(0,255,136,0.2)', width: 1, style: 2 }
        },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
        timeScale: {
            borderColor: 'rgba(255,255,255,0.06)',
            timeVisible: true,
            secondsVisible: true,
            rightOffset: 5,
            barSpacing: 6,
        },
    });
    chartInstance = chart;

    var priceOpts = { type: 'price', minMove: 0.0000001, precision: 10 };
    var isLineMode = (chartType === 'line');
    var isBarMode = (chartType === 'bar');

    // Main price series — depends on chart type
    if (isLineMode) {
        // Thick animated line with gradient fill
        liveSeries = chart.addAreaSeries({
            topColor: 'rgba(0, 255, 136, 0.2)',
            bottomColor: 'rgba(0, 255, 136, 0.0)',
            lineColor: '#00ff88',
            lineWidth: 3,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 5,
            crosshairMarkerBorderColor: '#00ff88',
            crosshairMarkerBackgroundColor: 'rgba(0,255,136,0.3)',
            priceFormat: priceOpts,
        });
        liveCandleSeries = null;
    } else if (isBarMode) {
        // Bar chart — each tick is a bar
        liveCandleSeries = chart.addHistogramSeries({
            priceFormat: priceOpts,
        });
        // Also add thin line overlay for continuity
        liveSeries = chart.addAreaSeries({
            topColor: 'rgba(0, 255, 136, 0.05)',
            bottomColor: 'rgba(0, 255, 136, 0.0)',
            lineColor: 'rgba(0, 255, 136, 0.3)',
            lineWidth: 1,
            crosshairMarkerVisible: false,
            priceFormat: priceOpts,
            priceScaleId: 'right',
            lastValueVisible: false,
        });
    } else {
        // Candlestick + area overlay (default)
        liveCandleSeries = chart.addCandlestickSeries({
            upColor: '#00ff88',
            downColor: '#ff3366',
            borderUpColor: '#00ff88',
            borderDownColor: '#ff3366',
            wickUpColor: 'rgba(0,255,136,0.5)',
            wickDownColor: 'rgba(255,51,102,0.5)',
            priceFormat: priceOpts,
        });
        liveSeries = chart.addAreaSeries({
            topColor: 'rgba(0, 255, 136, 0.08)',
            bottomColor: 'rgba(0, 255, 136, 0.0)',
            lineColor: 'rgba(0, 255, 136, 0.4)',
            lineWidth: 1,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 3,
            crosshairMarkerBackgroundColor: '#00ff88',
            priceFormat: priceOpts,
            priceScaleId: 'right',
            lastValueVisible: false,
        });
    }

    // Volume histogram
    liveVolumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });

    // Seed with historical data
    liveCandles = [];
    liveDataPoints = [];
    var seenTimes = {};
    for (var i = 0; i < seedCandles.length; i++) {
        var sc = seedCandles[i];
        if (!seenTimes[sc.time] && sc.close > 0) {
            seenTimes[sc.time] = true;
            liveCandles.push({ time: sc.time, open: sc.open, high: sc.high, low: sc.low, close: sc.close });
            liveDataPoints.push({ time: sc.time, value: sc.close });
        }
    }
    if (liveCandles.length > 0) {
        if (isLineMode) {
            liveSeries.setData(liveDataPoints);
        } else if (isBarMode) {
            liveCandleSeries.setData(liveCandles.map(function(c) {
                return { time: c.time, value: c.close, color: c.close >= c.open ? 'rgba(0,255,136,0.7)' : 'rgba(255,51,102,0.7)' };
            }));
            liveSeries.setData(liveDataPoints);
        } else {
            liveCandleSeries.setData(liveCandles);
            liveSeries.setData(liveDataPoints);
        }
        var volData = seedCandles.filter(function(c) { return !seenTimes['v' + c.time]; }).map(function(c) {
            seenTimes['v' + c.time] = true;
            return { time: c.time, value: c.volume, color: c.close >= c.open ? 'rgba(0,255,136,0.25)' : 'rgba(255,51,102,0.25)' };
        });
        if (volData.length > 0) liveVolumeSeries.setData(volData);
    }

    currentCandle = null;
    liveLastPrice = null;
    livePriceDirection = 0;
    liveEmaPrice = null;
    pumpHistory = [];

    var pollAddress = address;
    var tickCount = 0;
    var lastPumpLevel = 'none'; // track pump state to avoid expensive applyOptions every tick
    var lastRainbowUpdate = 0;  // throttle rainbow recolor to max 4fps

    // Subscribe via WebSocket for real-time price pushes (no HTTP polling needed)
    if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'subscribe', token: pollAddress }));
    }

    // Fallback HTTP poll at 3s (only fills gaps if WebSocket misses)
    liveInterval = setInterval(async function() {
        try {
            var res = await fetch(API + '/live/' + encodeURIComponent(pollAddress));
            if (!res.ok) return;
            var data = await res.json();
            if (!data.price || data.price <= 0) return;

            var price = data.price;
            var now = Math.floor(Date.now() / 1000);
            var candleTime = now - (now % liveCandleInterval);
            tickCount++;

            if (liveLastPrice !== null) {
                livePriceDirection = price > liveLastPrice ? 1 : (price < liveLastPrice ? -1 : livePriceDirection);
            }
            liveLastPrice = price;

            // Pump detection
            var pumpLevel = detectPumpLevel(price);
            updatePumpOverlay(pumpLevel);

            // Pick per-bar pump color (cheap — no applyOptions, no re-render)
            var pumpColor = null;
            if (pumpLevel === 'extreme') {
                var hue = (Date.now() / 8) % 360;
                pumpColor = 'hsl(' + hue + ', 100%, 60%)';
            }

            // Update main series based on chart type
            if (isLineMode) {
                var smoothed = emaSmooth(price);
                var lineTime = now;
                if (liveDataPoints.length > 0 && lineTime <= liveDataPoints[liveDataPoints.length - 1].time) {
                    lineTime = liveDataPoints[liveDataPoints.length - 1].time + 1;
                }
                liveDataPoints.push({ time: lineTime, value: smoothed });
                liveSeries.update({ time: lineTime, value: smoothed });
                // Line/area series doesn't support per-point colors — pump overlay handles the visual
            } else if (isBarMode) {
                var barTime = now;
                if (liveCandles.length > 0 && barTime <= liveCandles[liveCandles.length - 1].time) {
                    barTime = liveCandles[liveCandles.length - 1].time + 1;
                }
                // Per-bar color: rainbow during pump, normal green/red otherwise
                var barColor = pumpColor || (livePriceDirection >= 0 ? 'rgba(0,255,136,0.7)' : 'rgba(255,51,102,0.7)');
                liveCandles.push({ time: barTime, value: price, color: barColor });
                liveCandleSeries.update({ time: barTime, value: price, color: barColor });

                var smoothed2 = emaSmooth(price);
                if (liveDataPoints.length > 0 && barTime <= liveDataPoints[liveDataPoints.length - 1].time) {
                    barTime = liveDataPoints[liveDataPoints.length - 1].time + 1;
                }
                liveDataPoints.push({ time: barTime, value: smoothed2 });
                liveSeries.update({ time: barTime, value: smoothed2 });
            } else {
                // Candlestick mode (default)
                if (!currentCandle || currentCandle.time !== candleTime) {
                    if (currentCandle) {
                        if (liveCandles.length > 0 && candleTime <= liveCandles[liveCandles.length - 1].time) {
                            candleTime = liveCandles[liveCandles.length - 1].time + liveCandleInterval;
                        }
                    }
                    currentCandle = { time: candleTime, open: price, high: price, low: price, close: price, ticks: 1 };
                    liveCandles.push(currentCandle);
                } else {
                    currentCandle.high = Math.max(currentCandle.high, price);
                    currentCandle.low = Math.min(currentCandle.low, price);
                    currentCandle.close = price;
                    currentCandle.ticks++;
                }

                // Per-candle color: rainbow during pump, no applyOptions needed
                var candleUpdate = {
                    time: currentCandle.time,
                    open: currentCandle.open, high: currentCandle.high,
                    low: currentCandle.low, close: currentCandle.close,
                };
                if (pumpColor) {
                    candleUpdate.color = pumpColor;
                    candleUpdate.borderColor = pumpColor;
                    candleUpdate.wickColor = pumpColor;
                }
                liveCandleSeries.update(candleUpdate);

                var smoothed3 = emaSmooth(price);
                var lineTime3 = now;
                if (liveDataPoints.length > 0 && lineTime3 <= liveDataPoints[liveDataPoints.length - 1].time) {
                    lineTime3 = liveDataPoints[liveDataPoints.length - 1].time + 1;
                }
                liveDataPoints.push({ time: lineTime3, value: smoothed3 });
                liveSeries.update({ time: lineTime3, value: smoothed3 });

                liveVolumeSeries.update({
                    time: currentCandle.time,
                    value: currentCandle.ticks,
                    color: pumpColor || (currentCandle.close >= currentCandle.open ? 'rgba(0,255,136,0.25)' : 'rgba(255,51,102,0.25)'),
                });
            }

            // Trim old data
            if (liveCandles.length > 2000) {
                liveCandles = liveCandles.slice(-1500);
                if (liveCandleSeries) liveCandleSeries.setData(liveCandles);
            }
            if (liveDataPoints.length > 5000) {
                liveDataPoints = liveDataPoints.slice(-4000);
                if (liveSeries) liveSeries.setData(liveDataPoints);
            }

            chart.timeScale().scrollToRealTime();

            // Price display flash
            var priceEl = document.getElementById('dPrice');
            if (priceEl) {
                priceEl.textContent = formatPrice(price);
                priceEl.classList.remove('price-flash-up', 'price-flash-down');
                if (pumpLevel === 'extreme') {
                    priceEl.style.textShadow = '0 0 20px rgba(0,255,136,0.8), 0 0 40px rgba(0,255,136,0.4)';
                } else {
                    priceEl.style.textShadow = '';
                }
                if (livePriceDirection === 1) {
                    priceEl.classList.add('price-flash-up');
                } else if (livePriceDirection === -1) {
                    priceEl.classList.add('price-flash-down');
                }
                setTimeout(function() { priceEl.classList.remove('price-flash-up', 'price-flash-down'); }, 600);
            }

            if (data.priceChange && data.priceChange.m5 !== undefined) {
                setChg('dChg5m', data.priceChange.m5);
            }
        } catch (e) { /* ignore */ }
    }, 3000); // 3s HTTP fallback — primary updates come via WebSocket

    // Responsive resize — only update width, height is fixed
    chartResizeObserver = new ResizeObserver(function(entries) {
        for (var j = 0; j < entries.length; j++) {
            var cr = entries[j].contentRect;
            if (cr.width > 0) chart.applyOptions({ width: cr.width });
        }
    });
    chartResizeObserver.observe(container);
    chart.timeScale().fitContent();
}

/* ============================================
   LIVE TRADE FEED
   ============================================ */

async function loadTrades(address) {
    var panel = document.getElementById('tradesPanel');
    if (!panel) return;
    panel.innerHTML = '<div class="trades-loading">Loading trades...</div>';

    await fetchAndRenderTrades(address, panel);

    // Auto-refresh every 8 seconds
    if (tradesInterval) clearInterval(tradesInterval);
    tradesInterval = setInterval(function() {
        fetchAndRenderTrades(address, panel);
    }, 8000);
}

async function fetchAndRenderTrades(address, panel) {
    try {
        var res = await fetch(API + '/trades/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var data = await res.json();
        var trades = data.trades || [];

        if (trades.length === 0) {
            panel.innerHTML = '<div class="trades-empty">No recent trades</div>';
            setText('tradeCount', '--');
            return;
        }

        setText('tradeCount', trades.length + ' trades');

        var html = '<div class="trades-header">' +
            '<span class="th-type">Type</span>' +
            '<span class="th-size">Size</span>' +
            '<span class="th-tprice">Price</span>' +
            '<span class="th-wallet">Wallet</span>' +
            '<span class="th-time">Time</span>' +
        '</div>';

        html += trades.slice(0, 30).map(function(trade, idx) {
            var isBuy = trade.type === 'buy';
            var cls = isBuy ? 'trade-buy' : 'trade-sell';
            var icon = isBuy ? '&#9650;' : '&#9660;';
            var typeLabel = isBuy ? 'BUY' : 'SELL';
            var size = trade.volumeUsd > 0 ? formatNumber(trade.volumeUsd) : '--';
            var price = trade.priceUsd > 0 ? formatPrice(trade.priceUsd) : '--';
            var wallet = trade.maker ? (trade.maker.slice(0, 4) + '..' + trade.maker.slice(-4)) : '--';
            var time = trade.timestamp ? timeAgo(trade.timestamp) : '--';
            var delay = Math.min(idx * 40, 600);

            return '<div class="trade-row ' + cls + '" style="animation-delay:' + delay + 'ms">' +
                '<span class="trade-type">' + icon + ' ' + typeLabel + '</span>' +
                '<span class="trade-size mono">' + size + '</span>' +
                '<span class="trade-price mono">' + price + '</span>' +
                '<span class="trade-wallet mono">' + wallet + '</span>' +
                '<span class="trade-time">' + time + '</span>' +
            '</div>';
        }).join('');

        panel.innerHTML = html;
    } catch (e) {
        panel.innerHTML = '<div class="trades-empty">Trade feed unavailable</div>';
    }
}

function stopTrades() {
    if (tradesInterval) { clearInterval(tradesInterval); tradesInterval = null; }
}

/* ============================================
   BONDING CURVE METER
   ============================================ */

async function loadBonding(address) {
    var card = document.getElementById('bondingCard');
    var panel = document.getElementById('bondingPanel');
    if (!card || !panel) return;

    try {
        var res = await fetch(API + '/bonding/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var data = await res.json();

        if (!data.available || data.platform !== 'pump') {
            card.classList.add('hidden');
            return;
        }

        card.classList.remove('hidden');

        if (data.graduated) {
            panel.innerHTML = '<div class="bonding-graduated">' +
                '<div class="bonding-bar-wrap">' +
                    '<div class="bonding-bar"><div class="bonding-fill graduated" style="width:100%"></div></div>' +
                '</div>' +
                '<div class="bonding-status">' +
                    '<span class="bonding-badge graduated-badge">GRADUATED</span>' +
                    '<span class="bonding-dex">Trading on ' + escapeHtml(data.dexId || 'Raydium') + '</span>' +
                '</div>' +
                '<div class="bonding-stats">' +
                    '<span class="bonding-mcap">MCap: ' + formatNumber(data.marketCap) + '</span>' +
                    '<span class="bonding-liq">Liq: ' + formatNumber(data.liquidity) + '</span>' +
                '</div>' +
            '</div>';
        } else {
            var pct = data.progress || 0;
            var remaining = data.remainingUsd || 0;
            var nearGrad = pct > 80;
            panel.innerHTML = '<div class="bonding-active' + (nearGrad ? ' bonding-near' : '') + '">' +
                '<div class="bonding-bar-wrap">' +
                    '<div class="bonding-bar"><div class="bonding-fill' + (nearGrad ? ' bonding-pulse' : '') + '" style="width:' + pct + '%"></div></div>' +
                    '<div class="bonding-pct">' + pct.toFixed(1) + '%</div>' +
                '</div>' +
                '<div class="bonding-progress-info">' +
                    '<span>' + formatNumber(data.marketCap) + ' / ' + formatNumber(data.threshold) + '</span>' +
                    '<span class="bonding-remaining">' + formatNumber(remaining).replace('$', '$') + ' to graduation</span>' +
                '</div>' +
            '</div>';
        }
    } catch (e) {
        card.classList.add('hidden');
    }
}

/* ============================================
   SEARCH
   ============================================ */

const handleSearch = debounce(async function (query) {
    const drop = document.getElementById('searchDrop');
    if (!drop) return;

    if (!query || query.length < 2) {
        drop.classList.add('hidden');
        drop.innerHTML = '';
        return;
    }

    try {
        const res = await fetch(API + '/search?q=' + encodeURIComponent(query));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const tokens = data.tokens || [];

        if (tokens.length === 0) {
            drop.innerHTML = '<div class="search-item search-empty">No results found</div>';
            drop.classList.remove('hidden');
            return;
        }

        drop.innerHTML = tokens.slice(0, 10).map(t => {
            const sym = escapeHtml(t.symbol || '???');
            const name = escapeHtml(t.name || '');
            const addr = t.address || '';
            const imgSrc = t.imageUrl || PLACEHOLDER;
            const chg = t.priceChange ? t.priceChange.h24 : null;

            return '<div class="search-item" onclick="viewToken(\'' + addr + '\')">' +
                '<img src="' + imgSrc + '" alt="' + sym + '" onerror="this.src=\'' + PLACEHOLDER + '\'" class="search-img">' +
                '<div class="search-info">' +
                    '<span class="search-sym">' + sym + '</span>' +
                    '<span class="search-name">' + name + '</span>' +
                '</div>' +
                '<div class="search-right">' +
                    '<span class="search-price">' + formatPrice(t.price) + '</span>' +
                    '<span class="search-chg ' + chgClass(chg) + '">' + formatPercent(chg) + '</span>' +
                '</div>' +
            '</div>';
        }).join('');

        drop.classList.remove('hidden');
    } catch (err) {
        console.error('Search failed:', err);
        drop.innerHTML = '<div class="search-item search-empty">Search failed</div>';
        drop.classList.remove('hidden');
    }
}, 300);

/* ============================================
   NAVIGATION & VIEW MANAGEMENT
   ============================================ */

function goBack() {
    destroyChart();
    stopTrades();
    currentToken = null;

    const listView = document.getElementById('listView');
    const detailView = document.getElementById('detailView');
    if (listView) listView.style.display = '';
    if (detailView) detailView.classList.add('hidden');

    // Update URL
    try {
        const basePath = window.location.pathname.startsWith('/dex') ? '/dex' : '/';
        window.history.pushState({}, '', basePath);
    } catch (e) { /* ignore */ }
}

function goHome(event) {
    if (event) event.preventDefault();
    goBack();
    loadTokens('trending', 100, 0);
}

/* ============================================
   CLIPBOARD & BOOST
   ============================================ */

function copyAddr() {
    if (!currentToken || !currentToken.address) {
        showToast('No address to copy', 'error');
        return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(currentToken.address).then(() => {
            showToast('Address copied to clipboard', 'success');
        }).catch(() => {
            fallbackCopy(currentToken.address);
        });
    } else {
        fallbackCopy(currentToken.address);
    }
}

function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
        document.execCommand('copy');
        showToast('Address copied to clipboard', 'success');
    } catch (e) {
        showToast('Failed to copy address', 'error');
    }
    document.body.removeChild(ta);
}

/* ============================================
   WALLET CONNECTION (Phantom / Solflare)
   ============================================ */

function getWalletProvider() {
    if (window.phantom && window.phantom.solana && window.phantom.solana.isPhantom) return window.phantom.solana;
    if (window.solana && window.solana.isPhantom) return window.solana;
    if (window.solflare && window.solflare.isSolflare) return window.solflare;
    return null;
}

async function connectWallet() {
    var provider = getWalletProvider();
    if (!provider) {
        showToast('Install Phantom or Solflare wallet', 'error');
        window.open('https://phantom.app/', '_blank');
        return;
    }
    try {
        var resp = await provider.connect();
        walletProvider = provider;
        walletAddress = resp.publicKey.toString();
        updateWalletUI();
        showToast('Wallet connected: ' + walletAddress.slice(0, 4) + '..' + walletAddress.slice(-4), 'success');
        checkXConnection();
        provider.on('disconnect', function () {
            walletAddress = null;
            walletProvider = null;
            updateWalletUI();
        });
    } catch (err) {
        if (err.code !== 4001) {
            showToast('Failed to connect wallet', 'error');
        }
    }
}

function disconnectWallet() {
    if (walletProvider) {
        try { walletProvider.disconnect(); } catch (e) { /* ignore */ }
    }
    walletAddress = null;
    walletProvider = null;
    xConnected = false;
    xUsername = null;
    updateWalletUI();
    updateXUI();
    showToast('Wallet disconnected', 'info');
}

function updateWalletUI() {
    var btn = document.getElementById('walletBtn');
    if (!btn) return;
    if (walletAddress) {
        btn.textContent = walletAddress.slice(0, 4) + '..' + walletAddress.slice(-4);
        btn.classList.add('wallet-connected');
    } else {
        btn.textContent = 'Connect';
        btn.classList.remove('wallet-connected');
    }
    var boostBtn = document.getElementById('boostConfirmBtn');
    if (boostBtn) {
        boostBtn.textContent = walletAddress ? 'Confirm Boost' : 'Connect Wallet to Boost';
    }
}

async function getSolPrice() {
    try {
        var res = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd');
        var data = await res.json();
        solPrice = data.solana.usd;
    } catch (e) {
        solPrice = 200; // fallback
    }
    return solPrice;
}

/* ============================================
   CONNECT X (OAuth 2.0 read-only)
   ============================================ */

async function connectX() {
    if (xConnected) {
        // Disconnect
        if (walletAddress) {
            try {
                await fetch(API + '/x/disconnect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ wallet: walletAddress })
                });
            } catch (e) { /* ignore */ }
        }
        xConnected = false;
        xUsername = null;
        updateXUI();
        showToast('X account disconnected', 'info');
        return;
    }

    if (!walletAddress) {
        showToast('Connect your wallet first, then connect X', 'error');
        return;
    }

    try {
        var res = await fetch(API + '/x/auth?wallet=' + encodeURIComponent(walletAddress));
        var data = await res.json();
        if (data.authUrl) {
            window.open(data.authUrl, '_blank', 'width=600,height=700');
        } else {
            showToast('Could not start X authorization', 'error');
        }
    } catch (e) {
        showToast('Failed to connect X', 'error');
    }
}

function updateXUI() {
    var btn = document.getElementById('xConnectBtn');
    if (!btn) return;
    if (xConnected && xUsername) {
        btn.textContent = '@' + xUsername;
        btn.classList.add('x-connected');
    } else {
        btn.textContent = 'Connect X';
        btn.classList.remove('x-connected');
    }
}

async function checkXConnection() {
    if (!walletAddress) return;
    try {
        var res = await fetch(API + '/x/connection?wallet=' + encodeURIComponent(walletAddress));
        var data = await res.json();
        if (data.connected) {
            xConnected = true;
            xUsername = data.username;
            updateXUI();
        }
    } catch (e) { /* ignore */ }
}

function handleXCallbackParams() {
    var params = new URLSearchParams(window.location.search);
    var xUser = params.get('x_connected');
    var xError = params.get('x_error');

    if (xUser) {
        xConnected = true;
        xUsername = xUser;
        updateXUI();
        showToast('X connected: @' + xUser, 'success');
        // Clean URL
        try { window.history.replaceState({}, '', window.location.pathname); } catch (e) { /* ignore */ }
    }

    if (xError) {
        var msgs = {
            missing_params: 'X authorization was incomplete',
            invalid_state: 'X authorization expired, try again',
            token_exchange: 'X token exchange failed',
            profile_fetch: 'Could not fetch X profile',
            server_error: 'X connection error, try again',
        };
        showToast(msgs[xError] || 'X connection failed', 'error');
        try { window.history.replaceState({}, '', window.location.pathname); } catch (e) { /* ignore */ }
    }
}

async function loadXBadge(address) {
    var card = document.getElementById('xBadgeCard');
    var panel = document.getElementById('xBadgePanel');
    if (!card || !panel) return;

    card.classList.add('hidden');
    panel.innerHTML = '';

    try {
        var res = await fetch(API + '/x/badge/' + encodeURIComponent(address));
        var data = await res.json();
        var badges = data.badges || [];

        if (badges.length === 0) {
            card.classList.add('hidden');
            return;
        }

        card.classList.remove('hidden');

        var html = badges.map(function (b) {
            var roleLabel = b.role === 'deployer' ? 'Deployer' : 'Fee Recipient';
            var roleCls = b.role === 'deployer' ? 'xb-deployer' : 'xb-fee';
            var imgSrc = b.xProfileImage || '';
            return '<div class="xbadge-item">' +
                (imgSrc ? '<img src="' + escapeHtml(imgSrc) + '" class="xbadge-avatar" onerror="this.style.display=\'none\'">' : '') +
                '<div class="xbadge-info">' +
                    '<a href="https://x.com/' + escapeHtml(b.xUsername) + '" target="_blank" rel="noopener" class="xbadge-handle">@' + escapeHtml(b.xUsername) + '</a>' +
                    '<span class="xbadge-name">' + escapeHtml(b.xName || '') + '</span>' +
                '</div>' +
                '<div class="xbadge-meta">' +
                    '<span class="xbadge-role ' + roleCls + '">' + roleLabel + '</span>' +
                    '<span class="xbadge-wallet">' + escapeHtml(b.wallet) + '</span>' +
                '</div>' +
            '</div>';
        }).join('');

        panel.innerHTML = html;
    } catch (e) {
        card.classList.add('hidden');
    }
}

/* ============================================
   WELCOME POPUP (first-time visitors)
   ============================================ */

function showWelcome() {
    var modal = document.getElementById('welcomeModal');
    if (modal) modal.classList.remove('hidden');
}

function dismissWelcome() {
    var modal = document.getElementById('welcomeModal');
    if (modal) modal.classList.add('hidden');
    try { localStorage.setItem('dexai_welcomed', '1'); } catch (e) { /* ignore */ }
}

function checkFirstVisit() {
    try {
        if (!localStorage.getItem('dexai_welcomed')) {
            showWelcome();
        }
    } catch (e) {
        // localStorage unavailable, don't show
    }
}

/* ============================================
   BOOST SYSTEM (levels + on-chain transactions)
   ============================================ */

function openBoost() {
    if (!currentToken) { showToast('Select a token first', 'error'); return; }
    var modal = document.getElementById('boostModal');
    if (modal) modal.classList.remove('hidden');
    var symEl = document.getElementById('boostTokenSym');
    if (symEl) symEl.textContent = currentToken.symbol || '';
    var statusEl = document.getElementById('boostStatus');
    if (statusEl) statusEl.innerHTML = '';
    selectedBoostLevel = 1;
    // Reset level selection
    var levels = document.querySelectorAll('.boost-level');
    levels.forEach(function (l) { l.classList.remove('selected'); });
    var first = document.querySelector('.boost-level[data-level="1"]');
    if (first) first.classList.add('selected');
    // Clear requirement indicators
    var r2 = document.getElementById('lvl2Reqs');
    var r3 = document.getElementById('lvl3Reqs');
    if (r2) r2.innerHTML = '';
    if (r3) r3.innerHTML = '';
    updateWalletUI();
}

function closeBoost() {
    var modal = document.getElementById('boostModal');
    if (modal) modal.classList.add('hidden');
}

function selectBoostLevel(el) {
    if (!el) return;
    var level = parseInt(el.getAttribute('data-level')) || 1;
    selectedBoostLevel = level;
    var levels = document.querySelectorAll('.boost-level');
    levels.forEach(function (l) { l.classList.remove('selected'); });
    el.classList.add('selected');
    // If level 2+, check requirements
    if (level >= 2 && currentToken) {
        checkBoostEligibility(currentToken.address, level);
    }
}

function selectBoostType(el) {
    if (!el) return;
    var opts = document.querySelectorAll('.boost-opt');
    opts.forEach(function (o) { o.classList.remove('selected'); });
    el.classList.add('selected');
}

async function checkBoostEligibility(address, level) {
    var reqsId = level === 2 ? 'lvl2Reqs' : 'lvl3Reqs';
    var reqsEl = document.getElementById(reqsId);
    if (!reqsEl) return;
    reqsEl.innerHTML = '<span class="bl-req bl-req-loading">Checking requirements...</span>';
    try {
        var res = await fetch(API + '/boost/check-level', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address: address, level: level })
        });
        var data = await res.json();
        var html = '';
        if (data.checks) {
            for (var i = 0; i < data.checks.length; i++) {
                var check = data.checks[i];
                var cls = check.passed ? 'bl-req-pass' : 'bl-req-fail';
                var icon = check.passed ? '&#10003;' : '&#10007;';
                html += '<span class="bl-req ' + cls + '">' + icon + ' ' + escapeHtml(check.name) + '</span>';
            }
        }
        if (!data.eligible) {
            html += '<span class="bl-req bl-req-fail">Not eligible: ' + escapeHtml(data.reason || 'Requirements not met') + '</span>';
        }
        reqsEl.innerHTML = html;
    } catch (e) {
        reqsEl.innerHTML = '<span class="bl-req bl-req-fail">Could not check requirements</span>';
    }
}

function encodeU64LE(value) {
    var buf = new Uint8Array(8);
    var lo = value & 0xFFFFFFFF;
    var hi = Math.floor(value / 0x100000000) & 0xFFFFFFFF;
    buf[0] = lo & 0xFF; buf[1] = (lo >> 8) & 0xFF;
    buf[2] = (lo >> 16) & 0xFF; buf[3] = (lo >> 24) & 0xFF;
    buf[4] = hi & 0xFF; buf[5] = (hi >> 8) & 0xFF;
    buf[6] = (hi >> 16) & 0xFF; buf[7] = (hi >> 24) & 0xFF;
    return buf;
}

async function confirmBoost() {
    if (!walletAddress) {
        connectWallet();
        return;
    }
    if (!currentToken || !currentToken.address) {
        showToast('No token selected', 'error');
        return;
    }

    var level = selectedBoostLevel;
    var priceUsd = BOOST_PRICES[level] || 25;
    var selected = document.querySelector('.boost-opt.selected');
    var payType = selected ? (selected.getAttribute('data-type') || 'sol') : 'sol';
    var statusEl = document.getElementById('boostStatus');
    var btn = document.getElementById('boostConfirmBtn');
    btn.disabled = true;
    btn.textContent = 'Processing...';

    try {
        // For levels 2+, verify eligibility first
        if (level >= 2) {
            statusEl.innerHTML = '<span class="boost-checking">Verifying eligibility for Level ' + level + '...</span>';
            var eligRes = await fetch(API + '/boost/check-level', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ address: currentToken.address, level: level })
            });
            var eligData = await eligRes.json();
            if (!eligData.eligible) {
                statusEl.innerHTML = '<span class="boost-fail">' + escapeHtml(eligData.reason || 'Token does not meet Level ' + level + ' requirements') + '</span>';
                btn.disabled = false;
                btn.textContent = 'Confirm Boost';
                return;
            }
            statusEl.innerHTML = '<span class="boost-pass">Eligibility confirmed for Level ' + level + '</span>';
        }

        if (typeof solanaWeb3 === 'undefined') {
            statusEl.innerHTML = '<span class="boost-fail">Solana library not loaded. Refresh the page.</span>';
            btn.disabled = false;
            btn.textContent = 'Confirm Boost';
            return;
        }

        var Connection = solanaWeb3.Connection;
        var PublicKey = solanaWeb3.PublicKey;
        var Transaction = solanaWeb3.Transaction;
        var SystemProgram = solanaWeb3.SystemProgram;
        var TransactionInstruction = solanaWeb3.TransactionInstruction;
        var connection = new Connection('https://api.mainnet-beta.solana.com', 'confirmed');

        if (payType === 'sol') {
            // --- SOL TRANSFER ---
            if (!solPrice) await getSolPrice();
            var solAmount = priceUsd / solPrice;
            var lamports = Math.ceil(solAmount * 1e9);

            statusEl.innerHTML = '<span class="boost-checking">Sending ' + solAmount.toFixed(4) + ' SOL ($' + priceUsd + ')...</span>';

            var tx = new Transaction().add(
                SystemProgram.transfer({
                    fromPubkey: new PublicKey(walletAddress),
                    toPubkey: new PublicKey(ECOSYSTEM_WALLET),
                    lamports: lamports,
                })
            );
            tx.feePayer = new PublicKey(walletAddress);
            var bh = await connection.getLatestBlockhash();
            tx.recentBlockhash = bh.blockhash;

            statusEl.innerHTML = '<span class="boost-checking">Confirm in your wallet...</span>';
            var signed = await walletProvider.signAndSendTransaction(tx);

            statusEl.innerHTML = '<span class="boost-pass">Transaction sent! Confirming...</span>';

            // Report to server
            await fetch(API + '/boost/confirm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    address: currentToken.address,
                    txSignature: signed.signature,
                    paymentType: 'sol',
                    level: level,
                    wallet: walletAddress,
                    amountUsd: priceUsd,
                })
            });

            statusEl.innerHTML = '<span class="boost-pass">Level ' + level + ' boost confirmed! TX: ' + signed.signature.slice(0, 12) + '...</span>';
            showToast('Level ' + level + ' boost applied!', 'success');

        } else {
            // --- FARNS BURN ---
            statusEl.innerHTML = '<span class="boost-checking">Fetching FARNS price...</span>';

            var farnsPrice = 0;
            try {
                var fpRes = await fetch(API + '/token/' + FARNS_MINT);
                var fpData = await fpRes.json();
                farnsPrice = fpData.token ? fpData.token.price : 0;
            } catch (e) { /* ignore */ }
            if (farnsPrice <= 0) {
                try {
                    var dsRes = await fetch('https://api.dexscreener.com/latest/dex/tokens/' + FARNS_MINT);
                    var dsData = await dsRes.json();
                    if (dsData.pairs && dsData.pairs.length > 0) farnsPrice = parseFloat(dsData.pairs[0].priceUsd) || 0;
                } catch (e) { /* ignore */ }
            }
            if (farnsPrice <= 0) {
                statusEl.innerHTML = '<span class="boost-fail">Could not determine FARNS price. Try SOL payment instead.</span>';
                btn.disabled = false;
                btn.textContent = 'Confirm Boost';
                return;
            }

            // 3x power = you pay 1/3 the USD equivalent in FARNS
            var farnsNeeded = priceUsd / farnsPrice / 3;

            // Get FARNS decimals from mint account
            var mintPubkey = new PublicKey(FARNS_MINT);
            var ownerPubkey = new PublicKey(walletAddress);
            var decimals = 6; // default
            try {
                var mintAcct = await connection.getAccountInfo(mintPubkey);
                if (mintAcct && mintAcct.data) decimals = mintAcct.data[44];
            } catch (e) { /* use default */ }

            var farnsRaw = Math.ceil(farnsNeeded * Math.pow(10, decimals));

            // Derive user's FARNS ATA
            var TOKEN_PROGRAM = new PublicKey('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA');
            var ATA_PROGRAM = new PublicKey('ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL');
            var userAta = PublicKey.findProgramAddressSync(
                [ownerPubkey.toBuffer(), TOKEN_PROGRAM.toBuffer(), mintPubkey.toBuffer()],
                ATA_PROGRAM
            )[0];

            // SPL Token burn instruction (index 8): [8, amount_u64_le]
            var burnData = new Uint8Array(9);
            burnData[0] = 8;
            burnData.set(encodeU64LE(farnsRaw), 1);

            var burnIx = new TransactionInstruction({
                programId: TOKEN_PROGRAM,
                keys: [
                    { pubkey: userAta, isSigner: false, isWritable: true },
                    { pubkey: mintPubkey, isSigner: false, isWritable: true },
                    { pubkey: ownerPubkey, isSigner: true, isWritable: false },
                ],
                data: burnData,
            });

            statusEl.innerHTML = '<span class="boost-checking">Burning ' + farnsNeeded.toFixed(2) + ' FARNS (3x = $' + priceUsd + ' boost)...</span>';

            var tx2 = new Transaction().add(burnIx);
            tx2.feePayer = ownerPubkey;
            var bh2 = await connection.getLatestBlockhash();
            tx2.recentBlockhash = bh2.blockhash;

            statusEl.innerHTML = '<span class="boost-checking">Confirm FARNS burn in wallet...</span>';
            var signed2 = await walletProvider.signAndSendTransaction(tx2);

            statusEl.innerHTML = '<span class="boost-pass">FARNS burned! Confirming...</span>';

            await fetch(API + '/boost/confirm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    address: currentToken.address,
                    txSignature: signed2.signature,
                    paymentType: 'farns',
                    level: level,
                    wallet: walletAddress,
                    amountUsd: priceUsd,
                    farnsBurned: farnsNeeded,
                })
            });

            statusEl.innerHTML = '<span class="boost-pass">Level ' + level + ' boost confirmed! FARNS burned forever.</span>';
            showToast('Level ' + level + ' boost applied! FARNS burned!', 'success');
        }

    } catch (err) {
        console.error('Boost error:', err);
        if (err.code === 4001 || (err.message && err.message.indexOf('rejected') !== -1)) {
            statusEl.innerHTML = '<span class="boost-fail">Transaction cancelled by user</span>';
        } else {
            statusEl.innerHTML = '<span class="boost-fail">Error: ' + escapeHtml(err.message || 'Transaction failed') + '</span>';
        }
    }

    btn.disabled = false;
    btn.textContent = walletAddress ? 'Confirm Boost' : 'Connect Wallet to Boost';
}

/* ============================================
   WEBSOCKET
   ============================================ */

let wsRetries = 0;
const WS_MAX_RETRIES = 10;

function connectWebSocket() {
    if (wsRetries >= WS_MAX_RETRIES) {
        console.log('[DEXAI] WebSocket unavailable, using polling only');
        return;
    }
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPath = '/dex/ws';
        const url = protocol + '//' + window.location.host + wsPath;

        ws = new WebSocket(url);

        ws.onopen = function () {
            console.log('[DEXAI] WebSocket connected');
            wsRetries = 0;
            // Re-subscribe to live chart token if viewing one
            if (window.currentDetailAddress && liveSeries) {
                ws.send(JSON.stringify({ type: 'subscribe', token: window.currentDetailAddress }));
            }
        };

        ws.onmessage = function (event) {
            try {
                const msg = JSON.parse(event.data);
                handleWsMessage(msg);
            } catch (e) {
                // Ignore malformed messages
            }
        };

        ws.onclose = function () {
            ws = null;
            wsRetries++;
            if (wsRetries < WS_MAX_RETRIES) {
                const delay = Math.min(3000 * Math.pow(2, wsRetries), 30000);
                console.log('[DEXAI] WebSocket disconnected, retry ' + wsRetries + '/' + WS_MAX_RETRIES + ' in ' + (delay/1000) + 's');
                setTimeout(connectWebSocket, delay);
            } else {
                console.log('[DEXAI] WebSocket unavailable, using polling only');
            }
        };

        ws.onerror = function () {
            // onclose will fire after this
        };

    } catch (err) {
        wsRetries++;
        if (wsRetries < WS_MAX_RETRIES) {
            setTimeout(connectWebSocket, 5000);
        }
    }
}

function handleWsMessage(msg) {
    if (!msg || !msg.type) return;

    switch (msg.type) {
        case 'update':
            // If on list view, refresh data
            const listView = document.getElementById('listView');
            if (listView && listView.style.display !== 'none') {
                loadTokens(currentSort, 100, 0);
            }
            break;

        case 'price':
            // Real-time price update for current token — also feeds live chart
            if (currentToken && msg.token === currentToken.address && msg.price) {
                setText('dPrice', formatPrice(msg.price));
                if (msg.priceChange) {
                    const dChg = document.getElementById('dChg');
                    if (dChg) {
                        dChg.textContent = formatPercent(msg.priceChange);
                        dChg.className = 'price-chg ' + chgClass(msg.priceChange);
                    }
                }
                // Feed live chart series if active
                if (liveSeries && chartInstance) {
                    updateLiveChartFromPrice(msg.price);
                }
            }
            break;

        case 'ticker':
            if (msg.tokens) {
                buildTicker(msg.tokens);
            }
            break;

        default:
            break;
    }
}

/* ============================================
   INITIALIZATION
   ============================================ */

document.addEventListener('DOMContentLoaded', function () {
    // 0. Init chart type toggle
    initChartTypeToggle();

    // 1. Init particles
    initParticles();

    // 2. Setup nav button clicks
    const navBtns = document.querySelectorAll('#mainNav .nav-btn');
    navBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            const sort = btn.getAttribute('data-sort');
            if (sort) {
                // Close mobile menu after selection
                var nav = document.getElementById('mainNav');
                if (nav) nav.classList.remove('open');
                var menuBtn = document.getElementById('mobileMenuBtn');
                if (menuBtn) menuBtn.textContent = '\u2630';
                // Ensure we're showing list view
                goBack();
                loadTokens(sort, 100, 0);
            }
        });
    });

    // 3. Setup search
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', function () {
            handleSearch(searchInput.value.trim());
        });

        searchInput.addEventListener('focus', function () {
            if (searchInput.value.trim().length >= 2) {
                const drop = document.getElementById('searchDrop');
                if (drop && drop.innerHTML) drop.classList.remove('hidden');
            }
        });
    }

    // Close search dropdown on outside click
    document.addEventListener('click', function (e) {
        const searchBox = e.target.closest('.search-box');
        const drop = document.getElementById('searchDrop');
        if (!searchBox && drop) {
            drop.classList.add('hidden');
        }
    });

    // "/" key focuses search input
    document.addEventListener('keydown', function (e) {
        if (e.key === '/' && document.activeElement !== searchInput) {
            e.preventDefault();
            if (searchInput) searchInput.focus();
        }
        // Escape closes search dropdown
        if (e.key === 'Escape') {
            const drop = document.getElementById('searchDrop');
            if (drop) drop.classList.add('hidden');
            if (searchInput) searchInput.blur();
        }
    });

    // 4. Setup timeframe tab clicks
    const tfTabs = document.getElementById('tfTabs');
    if (tfTabs) {
        tfTabs.addEventListener('click', function (e) {
            const btn = e.target.closest('.tf');
            if (!btn) return;
            const tf = btn.getAttribute('data-tf');
            if (tf && currentToken) {
                if (tf === 'live') {
                    loadLiveChart(currentToken.address);
                } else {
                    loadChart(currentToken.address, tf);
                }
            }
        });
    }

    // 5. Load initial tokens
    loadTokens('trending', 100, 0);

    // 6. Connect WebSocket
    connectWebSocket();

    // 7. URL routing: check if path includes /token/
    const path = window.location.pathname;
    const tokenMatch = path.match(/\/token\/([A-Za-z0-9]+)/);
    if (tokenMatch && tokenMatch[1]) {
        viewToken(tokenMatch[1]);
    }

    // 8. Handle browser back/forward
    window.addEventListener('popstate', function (e) {
        if (e.state && e.state.address) {
            viewToken(e.state.address);
        } else {
            goBack();
        }
    });

    // 9. Auto-refresh list every 30 seconds (matches backend cache refresh)
    refreshInterval = setInterval(function () {
        const listView = document.getElementById('listView');
        if (listView && listView.style.display !== 'none') {
            loadTokens(currentSort, 100, 0);
        }
        updateMetaRefresh();
    }, 30000);

    // 10. Wallet connection
    var walletBtn = document.getElementById('walletBtn');
    if (walletBtn) {
        walletBtn.addEventListener('click', function () {
            if (walletAddress) {
                disconnectWallet();
            } else {
                connectWallet();
            }
        });
    }

    // 11. Auto-connect wallet if previously connected
    var provider = getWalletProvider();
    if (provider && provider.isConnected) {
        walletProvider = provider;
        try {
            walletAddress = provider.publicKey ? provider.publicKey.toString() : null;
            if (walletAddress) updateWalletUI();
        } catch (e) { /* ignore */ }
    }

    // 12. Welcome popup for first-time visitors
    checkFirstVisit();

    // 13. Pre-fetch SOL price
    getSolPrice();

    // 14. Handle X OAuth callback params
    handleXCallbackParams();

    // 15. Check X connection if wallet is already connected
    if (walletAddress) checkXConnection();
});
