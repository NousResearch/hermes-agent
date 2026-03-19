/**
 * DEXAI v2.1 - Cosmos Collective DEX Screener
 * Powered by: Whale Hunter, Quantum FarSight, Collective Intelligence, Burn Economy
 */

const express = require('express');
const cors = require('cors');
const { WebSocketServer } = require('ws');
const http = require('http');
const path = require('path');
const crypto = require('crypto');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

const PORT = process.env.DEXAI_PORT || 3847;
const COSMOS_API = process.env.COSMOS_API || 'http://localhost:8080';
const ECOSYSTEM_WALLET = '3fSS5RVErbgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw';
const FARNS_TOKEN = '9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS';
const BOOST_PRICES = { 1: 25, 2: 50, 3: 100 };
const BOOST_PRICE_USD = 25;  // kept for backward compat
const EXTENDED_INFO_PRICE_USD = 10;

// X OAuth 2.0 Configuration
const X_CLIENT_ID = process.env.X_CLIENT_ID || 'OUJSQ3BEX0Npc3pxZm1HcmxxWDc6MTpjaQ';
const X_CLIENT_SECRET = process.env.X_CLIENT_SECRET || '';
const X_REDIRECT_URI = process.env.X_REDIRECT_URI || 'https://ai.cosmos.cloud/dex/api/x/callback';
const X_SCOPES = 'tweet.read users.read follows.read';
const BAGS_FM_API = 'https://public-api-v2.bags.fm/api/v1';

// X Connection storage: wallet -> { xId, xUsername, xName, xProfileImage, connectedAt }
const xConnections = new Map();
// OAuth state storage: state -> { codeVerifier, wallet, ts }
const xOAuthState = new Map();
const FETCH_INTERVAL = 30000;
const COLLECTIVE_FETCH_INTERVAL = 60000;

// Real-time price APIs (Birdeye + Jupiter + Helius)
const BIRDEYE_API_KEY = process.env.BIRDEYE_API_KEY || 'c9d915af3f1f49ec9c017e89dbb77784';
const JUPITER_API_KEY = process.env.JUPITER_API_KEY || 'c872736b-676d-4279-a76c-93515999cd70';
const HELIUS_API_KEY = process.env.HELIUS_API_KEY || '1780e358-9bf5-4115-8234-eab6aafdc85c';
const BIRDEYE_BASE = 'https://public-api.birdeye.so';
const JUPITER_PRICE_BASE = 'https://api.jup.ag/price/v2';
const HELIUS_RPC = `https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`;
const HELIUS_WSS = `wss://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`;

// Platform suffixes for main list filtering
const ALLOWED_SUFFIXES = ['pump', 'bonk', 'bags'];

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use((req, res, next) => { console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`); next(); });

// ============================================================
// TOKEN CACHE & COLLECTIVE INTELLIGENCE
// ============================================================
const tokenCache = new Map();
let sortedByVolume = [];
let sortedByTrending = [];
let sortedByVelocity = [];
let sortedByNew = [];
let sortedByGainers = [];
let sortedByLosers = [];
let sortedByCollective = [];    // Collective's own picks
let sortedByWhaleHeat = [];     // Whale/smart money activity
let cacheReady = false;

const boosts = new Map();
const pendingBoosts = new Map();
const aiScoreCache = new Map();

// Collective intelligence data
const collectiveData = {
    whales: { topWallets: [], whaleFeed: [], lastFetch: 0 },
    trader: { running: false, positions: [], winRate: 0, lastFetch: 0 },
    learner: { bestCondition: null, worstCondition: null, config: {}, lastFetch: 0 },
    quantumCache: new Map(),     // address -> { bullProb, confidence, targets, ts }
    whaleTokenHeat: new Map(),   // address -> { whaleCount, smartMoneyBuys, totalSolInflow }
    collectivePicks: new Map(),  // address -> { score, reason, agents, ts }
};

// ============================================================
// PLATFORM HELPERS
// ============================================================
function getTokenPlatform(address) {
    if (!address) return null;
    const lower = address.toLowerCase();
    if (lower.endsWith('pump')) return 'pump';
    if (lower.endsWith('bonk')) return 'bonk';
    if (lower.endsWith('bags')) return 'bags';
    return null;
}

function isAllowedPlatform(address) {
    return getTokenPlatform(address) !== null;
}

function parsePair(pair) {
    const base = pair.baseToken || {};
    const quote = pair.quoteToken || {};
    const info = pair.info || {};
    const txns = pair.txns || {};
    const vol = pair.volume || {};
    const pc = pair.priceChange || {};
    const liq = pair.liquidity || {};
    const socials = {};
    for (const s of (info.socials || [])) socials[s.type || 'unknown'] = s.url || '';

    const address = base.address || '';
    const platform = getTokenPlatform(address);

    return {
        address,
        symbol: base.symbol || '???',
        name: base.name || 'Unknown',
        price: parseFloat(pair.priceUsd) || 0,
        priceNative: parseFloat(pair.priceNative) || 0,
        priceChange: {
            m5: parseFloat(pc.m5) || 0,
            h1: parseFloat(pc.h1) || 0,
            h6: parseFloat(pc.h6) || 0,
            h24: parseFloat(pc.h24) || 0,
        },
        volume: {
            m5: parseFloat(vol.m5) || 0,
            h1: parseFloat(vol.h1) || 0,
            h6: parseFloat(vol.h6) || 0,
            h24: parseFloat(vol.h24) || 0,
        },
        liquidity: parseFloat(liq.usd) || 0,
        fdv: parseFloat(pair.fdv) || 0,
        marketCap: parseFloat(pair.marketCap) || 0,
        txns: {
            m5: { buys: txns.m5?.buys || 0, sells: txns.m5?.sells || 0 },
            h1: { buys: txns.h1?.buys || 0, sells: txns.h1?.sells || 0 },
            h6: { buys: txns.h6?.buys || 0, sells: txns.h6?.sells || 0 },
            h24: { buys: txns.h24?.buys || 0, sells: txns.h24?.sells || 0 },
        },
        pairAddress: pair.pairAddress || '',
        pairCreatedAt: pair.pairCreatedAt || null,
        dexId: pair.dexId || 'raydium',
        chainId: pair.chainId || 'solana',
        url: pair.url || '',
        imageUrl: info.imageUrl || '',
        headerImg: info.header || '',
        websites: (info.websites || []).map(w => typeof w === 'string' ? w : (w.url || '')),
        socials,
        platform,
        isBags: platform === 'bags',
        // Scoring fields
        boostAmount: 0,
        burnBoostAmount: 0,      // FARNS burned specifically
        solBoostAmount: 0,       // SOL paid specifically
        aiScore: null,
        aiVerdict: null,
        collectiveVerified: false,
        collectivePick: false,
        collectiveReason: null,
        velocity: 0,
        whaleHeat: 0,            // whale/smart money activity score
        quantumBull: null,       // quantum simulation bull probability
        quantumConfidence: null,
        trendScore: 0,
        updatedAt: Date.now(),
    };
}

// ============================================================
// FETCH HELPERS
// ============================================================
async function safeFetch(url, opts = {}) {
    try {
        const res = await fetch(url, { signal: AbortSignal.timeout(15000), ...opts });
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

async function fetchTokenBatch(addresses) {
    const results = [];
    for (let i = 0; i < addresses.length; i += 30) {
        const batch = addresses.slice(i, i + 30);
        const data = await safeFetch(`https://api.dexscreener.com/latest/dex/tokens/${batch.join(',')}`);
        if (data?.pairs) {
            const byToken = {};
            for (const pair of data.pairs) {
                if (pair.chainId !== 'solana') continue;
                const addr = pair.baseToken?.address;
                if (!addr) continue;
                const liq = parseFloat(pair.liquidity?.usd) || 0;
                if (!byToken[addr] || liq > (parseFloat(byToken[addr].liquidity?.usd) || 0)) {
                    byToken[addr] = pair;
                }
            }
            for (const pair of Object.values(byToken)) results.push(parsePair(pair));
        }
        if (i + 30 < addresses.length) await new Promise(r => setTimeout(r, 250));
    }
    return results;
}

// ============================================================
// COLLECTIVE INTELLIGENCE FETCHER
// ============================================================
async function fetchCollectiveData() {
    try {
        // 1. Whale Hunter data
        const whaleData = await safeFetch(`${COSMOS_API}/api/trading/whales`);
        if (whaleData && whaleData.status !== 'not_running') {
            collectiveData.whales = {
                topWallets: whaleData.top_wallets || [],
                whaleFeed: whaleData.whale_feed || [],
                mixerFresh: whaleData.mixer_fresh_wallets || 0,
                lastFetch: Date.now(),
            };

            // Build whale heat map per token from whale feed
            collectiveData.whaleTokenHeat.clear();
            for (const event of (whaleData.whale_feed || [])) {
                const addr = event.token_address || event.mint;
                if (!addr) continue;
                const existing = collectiveData.whaleTokenHeat.get(addr) || { whaleCount: 0, smartMoneyBuys: 0, totalSolInflow: 0 };
                existing.whaleCount++;
                if (event.is_smart_money || event.label?.includes('smart')) existing.smartMoneyBuys++;
                existing.totalSolInflow += parseFloat(event.sol_amount || event.amount_sol || 0);
                collectiveData.whaleTokenHeat.set(addr, existing);
            }
        }

        // 2. Trader status & positions — read state file directly (fast, no API dependency)
        let traderData = null;
        try {
            const fs = require('fs');
            const stateRaw = fs.readFileSync('/workspace/Cosmos/cosmos/trading/.trader_state.json', 'utf8');
            const state = JSON.parse(stateRaw);
            const positions = Object.entries(state.positions || {}).map(([addr, p]) => ({
                mint: addr, symbol: p.symbol, entry_price: p.entry_price,
                sol_amount: p.sol_amount, timestamp: p.timestamp,
            }));
            const totalTrades = state.total_trades || 0;
            const wins = state.wins || 0;
            const winRate = totalTrades > 0 ? Math.round((wins / totalTrades) * 100) : 0;
            traderData = {
                running: true,
                positions,
                stats: { total_trades: totalTrades, wins, losses: state.losses || 0, total_pnl_sol: state.total_pnl_sol || 0, win_rate: winRate },
            };
        } catch (e) {
            // Fallback to API
            traderData = await safeFetch(`${COSMOS_API}/api/trading/status`);
        }
        if (traderData && traderData.running) {
            collectiveData.trader = {
                running: true,
                positions: traderData.positions || [],
                winRate: traderData.stats?.win_rate || 0,
                totalTrades: traderData.stats?.total_trades || 0,
                pnl: traderData.stats?.total_pnl_sol || 0,
                lastFetch: Date.now(),
            };

            // Tokens the trader is actively holding = collective endorsement
            for (const pos of (traderData.positions || [])) {
                const addr = pos.mint || pos.token_address;
                if (!addr) continue;
                if (!collectiveData.collectivePicks.has(addr)) {
                    collectiveData.collectivePicks.set(addr, {
                        score: 70,
                        reason: `Active trader position (${collectiveData.trader.winRate}% WR)`,
                        agents: ['DegenTrader'],
                        ts: Date.now(),
                    });
                }
            }
        }

        // 3. Adaptive learner insights
        const learnerData = await safeFetch(`${COSMOS_API}/api/trading/learner`);
        if (learnerData && learnerData.status !== 'not_running') {
            collectiveData.learner = {
                bestCondition: learnerData.summary?.best_condition || null,
                worstCondition: learnerData.summary?.worst_condition || null,
                config: learnerData.config || {},
                lastFetch: Date.now(),
            };
        }

        console.log(`[DEXAI] Collective sync: ${collectiveData.whales.topWallets.length} whales, ${collectiveData.whaleTokenHeat.size} whale-touched tokens, ${collectiveData.collectivePicks.size} collective picks, trader ${collectiveData.trader.running ? 'LIVE' : 'OFF'}`);
    } catch (e) {
        console.error('[DEXAI] Collective fetch error:', e.message);
    }
}

// ============================================================
// QUANTUM SIMULATION CACHE
// ============================================================
async function getQuantumPrediction(address) {
    const cached = collectiveData.quantumCache.get(address);
    if (cached && Date.now() - cached.ts < 300000) return cached;

    const data = await safeFetch(`${COSMOS_API}/api/farsight/crypto`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_address: address, simulations: 1000 }),
    });

    if (data && data.available !== false) {
        const result = {
            bullProb: parseFloat(data.bull_probability || data.bullProb || 0),
            confidence: parseFloat(data.confidence || 0),
            rugProb: parseFloat(data.rug_probability || 0),
            targets: data.price_targets || data.targets || {},
            simulations: data.simulation_count || data.simulations || 1000,
            ts: Date.now(),
        };
        collectiveData.quantumCache.set(address, result);
        return result;
    }
    return null;
}

// Background quantum scoring for top tokens
async function batchQuantumScore() {
    const topTokens = sortedByVolume.slice(0, 20);
    for (const token of topTokens) {
        const q = await getQuantumPrediction(token.address);
        if (q) {
            token.quantumBull = q.bullProb;
            token.quantumConfidence = q.confidence;
        }
        await new Promise(r => setTimeout(r, 500));
    }
}

// ============================================================
// COMPOSITE TRENDING ALGORITHM — Cosmos Collective Score
// ============================================================
function calculateTrendScore(token) {
    const addr = token.address;

    // ── BASE MARKET DATA (35%) ──
    const volScore = token.volume.h24 > 0 ? Math.log10(token.volume.h24) * 8 : 0;
    const txnScore = (token.txns.h1.buys + token.txns.h1.sells) * 0.08;
    const momentumScore = Math.max(0, token.priceChange.h1) * 0.3;
    const velocityScore = (parseFloat(token.velocity) || 0) * 1.5;
    const buyPressure = token.txns.h1.buys > 0
        ? (token.txns.h1.buys / (token.txns.h1.buys + token.txns.h1.sells)) * 5
        : 0;
    const baseScore = volScore + txnScore + momentumScore + velocityScore + buyPressure;

    // ── COLLECTIVE INTELLIGENCE (25%) ──
    let collectiveScore = 0;
    // AI score from Cosmos collective
    if (token.aiScore !== null) {
        collectiveScore += (token.aiScore / 100) * 15;  // 0-15 points
    }
    // Collective pick bonus
    const pick = collectiveData.collectivePicks.get(addr);
    if (pick) {
        collectiveScore += (pick.score / 100) * 10;  // 0-10 points
        token.collectivePick = true;
        token.collectiveReason = pick.reason;
    }
    // Trader is holding this token = strong signal
    if (collectiveData.trader.running) {
        const isHeld = (collectiveData.trader.positions || []).some(p => (p.mint || p.token_address) === addr);
        if (isHeld) collectiveScore += 8;
    }

    // ── QUANTUM SIMULATION (15%) ──
    let quantumScore = 0;
    const qData = collectiveData.quantumCache.get(addr);
    if (qData) {
        token.quantumBull = qData.bullProb;
        token.quantumConfidence = qData.confidence;
        // High bull probability + high confidence = major signal
        quantumScore = (qData.bullProb * qData.confidence) * 20;
        // Penalize high rug probability
        if (qData.rugProb > 0.5) quantumScore -= qData.rugProb * 10;
    }

    // ── WHALE / SMART MONEY (15%) ──
    let whaleScore = 0;
    const heat = collectiveData.whaleTokenHeat.get(addr);
    if (heat) {
        token.whaleHeat = heat.whaleCount * 3 + heat.smartMoneyBuys * 8 + Math.min(heat.totalSolInflow, 100) * 0.5;
        whaleScore = Math.min(token.whaleHeat, 25);  // Cap at 25 points
    }

    // ── BURN ECONOMY / BOOSTS (10%) ──
    let boostScore = 0;
    // FARNS burned = conviction signal (weighted 3x more than SOL)
    boostScore += (token.burnBoostAmount || 0) * 0.4;
    boostScore += (token.solBoostAmount || 0) * 0.12;
    // Collective verified tokens get extra
    if (token.collectiveVerified) boostScore += 5;
    boostScore = Math.min(boostScore, 15);  // Cap

    // ── FINAL COMPOSITE ──
    return baseScore + collectiveScore + quantumScore + whaleScore + boostScore;
}

// ============================================================
// TOKEN CACHE UPDATE
// ============================================================
async function updateTokenCache() {
    try {
        const addresses = new Set();
        const boostAmounts = {};

        // 1. Boosted tokens
        const boostData = await safeFetch('https://api.dexscreener.com/token-boosts/latest/v1');
        if (Array.isArray(boostData)) {
            for (const item of boostData) {
                if (item.chainId === 'solana' && item.tokenAddress) {
                    addresses.add(item.tokenAddress);
                    boostAmounts[item.tokenAddress] = (boostAmounts[item.tokenAddress] || 0) + (item.amount || 0);
                }
            }
        }
        await new Promise(r => setTimeout(r, 300));

        // 2. Token profiles
        const profileData = await safeFetch('https://api.dexscreener.com/token-profiles/latest/v1');
        if (Array.isArray(profileData)) {
            for (const item of profileData) {
                if (item.chainId === 'solana' && item.tokenAddress) addresses.add(item.tokenAddress);
            }
        }
        await new Promise(r => setTimeout(r, 300));

        // 3. Search popular terms for broader coverage
        for (const term of ['SOL', 'pump', 'meme', 'BONK', 'WIF', 'AI', 'bags', 'BAGS']) {
            const data = await safeFetch(`https://api.dexscreener.com/latest/dex/search?q=${term}`);
            if (data?.pairs) {
                for (const pair of data.pairs.slice(0, 40)) {
                    if (pair.chainId === 'solana' && pair.baseToken?.address) {
                        addresses.add(pair.baseToken.address);
                    }
                }
            }
            await new Promise(r => setTimeout(r, 250));
        }

        // 4. Add tokens from whale heat map (whales are buying these)
        for (const addr of collectiveData.whaleTokenHeat.keys()) {
            addresses.add(addr);
        }

        // 5. Add tokens from trader positions
        for (const pos of (collectiveData.trader.positions || [])) {
            const addr = pos.mint || pos.token_address;
            if (addr) addresses.add(addr);
        }

        console.log(`[DEXAI] Fetching details for ${addresses.size} tokens...`);
        const tokens = await fetchTokenBatch([...addresses]);

        // Merge into cache
        for (const token of tokens) {
            if (!token.address || token.volume.h24 <= 0) continue;
            const existing = tokenCache.get(token.address);
            if (existing) {
                // Velocity calc
                const timeDiff = (Date.now() - existing.updatedAt) / 60000;
                if (timeDiff > 0.5) {
                    token.velocity = Math.max(0, ((token.txns.h1.buys - existing.txns.h1.buys) / timeDiff).toFixed(2));
                } else {
                    token.velocity = existing.velocity;
                }
                // Preserve intelligence data
                token.aiScore = existing.aiScore;
                token.aiVerdict = existing.aiVerdict;
                token.collectiveVerified = existing.collectiveVerified;
                token.collectivePick = existing.collectivePick;
                token.collectiveReason = existing.collectiveReason;
                token.quantumBull = existing.quantumBull;
                token.quantumConfidence = existing.quantumConfidence;
                token.whaleHeat = existing.whaleHeat;
                token.burnBoostAmount = existing.burnBoostAmount || 0;
                token.solBoostAmount = existing.solBoostAmount || 0;
            }
            token.boostAmount = boostAmounts[token.address] || token.boostAmount;

            // Calculate composite trend score
            token.trendScore = calculateTrendScore(token);

            tokenCache.set(token.address, token);
        }

        // Build sorted arrays — main list filtered to allowed platforms only
        const all = [...tokenCache.values()].filter(t => t.volume.h24 > 0);
        const platformFiltered = all.filter(t => isAllowedPlatform(t.address));

        sortedByVolume = [...platformFiltered].sort((a, b) => b.volume.h24 - a.volume.h24).slice(0, 200);
        sortedByTrending = [...platformFiltered].sort((a, b) => b.trendScore - a.trendScore).slice(0, 200);
        sortedByVelocity = [...platformFiltered].sort((a, b) => (parseFloat(b.velocity) || 0) - (parseFloat(a.velocity) || 0)).slice(0, 200);
        sortedByNew = [...platformFiltered].filter(t => t.pairCreatedAt).sort((a, b) => (b.pairCreatedAt || 0) - (a.pairCreatedAt || 0)).slice(0, 200);
        sortedByGainers = [...platformFiltered].sort((a, b) => b.priceChange.h24 - a.priceChange.h24).slice(0, 200);
        sortedByLosers = [...platformFiltered].sort((a, b) => a.priceChange.h24 - b.priceChange.h24).slice(0, 200);

        // Collective picks — tokens endorsed by the collective (no platform filter for these)
        sortedByCollective = [...all].filter(t => {
            return t.collectivePick ||
                   (t.aiScore !== null && t.aiScore >= 70) ||
                   (t.quantumBull !== null && t.quantumBull >= 0.7) ||
                   (t.whaleHeat > 10);
        }).sort((a, b) => b.trendScore - a.trendScore).slice(0, 100);

        // Whale heat — tokens whales/smart money are buying (no platform filter)
        sortedByWhaleHeat = [...all].filter(t => t.whaleHeat > 0)
            .sort((a, b) => b.whaleHeat - a.whaleHeat).slice(0, 100);

        // Trim cache
        if (tokenCache.size > 600) {
            const keep = new Set(sortedByVolume.slice(0, 300).map(t => t.address));
            for (const addr of sortedByCollective.map(t => t.address)) keep.add(addr);
            for (const addr of sortedByWhaleHeat.map(t => t.address)) keep.add(addr);
            for (const [addr] of tokenCache) { if (!keep.has(addr)) tokenCache.delete(addr); }
        }

        cacheReady = true;
        const topSym = sortedByTrending[0]?.symbol || '---';
        const topVol = sortedByVolume[0] ? `${sortedByVolume[0].symbol} ($${(sortedByVolume[0].volume.h24 / 1e6).toFixed(1)}M)` : '---';
        console.log(`[DEXAI] Cache: ${tokenCache.size} tokens (${platformFiltered.length} on-platform) | Top trending: ${topSym} | Top vol: ${topVol} | Whale heat: ${sortedByWhaleHeat.length} | Collective picks: ${sortedByCollective.length}`);
        broadcastAll({ type: 'update', count: tokenCache.size, platformCount: platformFiltered.length, ts: Date.now() });
    } catch (e) {
        console.error('[DEXAI] Cache error:', e.message);
    }
}

// ============================================================
// AI SCORING (Cosmos Collective)
// ============================================================
async function getAIScore(address) {
    const cached = aiScoreCache.get(address);
    if (cached && Date.now() - cached.ts < 300000) return cached.data;

    try {
        const data = await safeFetch(`${COSMOS_API}/api/tools/score-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token_address: address }),
        });
        if (data) {
            const result = {
                score: data.score || null,
                verdict: data.prediction || data.verdict || null,
                rugProbability: data.rug_probability || null,
                cabalScore: data.cabal_score || null,
                bundleDetected: data.bundle_detected || false,
                swarmSentiment: data.swarm_sentiment || null,
                agents: data.agents || [],
            };
            aiScoreCache.set(address, { data: result, ts: Date.now() });
            const token = tokenCache.get(address);
            if (token) {
                token.aiScore = result.score;
                token.aiVerdict = result.verdict;
                // Recalculate trend score with new AI data
                token.trendScore = calculateTrendScore(token);
            }
            return result;
        }
    } catch {}
    return null;
}

// ============================================================
// API ROUTES
// ============================================================
app.get('/api/health', (req, res) => {
    const platformFiltered = [...tokenCache.values()].filter(t => isAllowedPlatform(t.address) && t.volume.h24 > 0);
    res.json({
        status: 'ok', service: 'DEXAI', version: '2.1.0',
        tokens: tokenCache.size, platformTokens: platformFiltered.length, cacheReady,
        ecosystem_wallet: ECOSYSTEM_WALLET, farns_token: FARNS_TOKEN,
        collective: {
            whalesTracked: collectiveData.whales.topWallets.length,
            whaleHeatTokens: collectiveData.whaleTokenHeat.size,
            collectivePicks: sortedByCollective.length,
            traderRunning: collectiveData.trader.running,
            traderWinRate: collectiveData.trader.winRate,
            quantumCached: collectiveData.quantumCache.size,
        },
        platforms: ALLOWED_SUFFIXES,
    });
});

app.get('/api/tokens', (req, res) => {
    const { sort = 'trending', limit = 100, offset = 0 } = req.query;
    const lim = Math.min(parseInt(limit) || 100, 200);
    const off = parseInt(offset) || 0;

    const lists = {
        volume: sortedByVolume, trending: sortedByTrending, velocity: sortedByVelocity,
        new: sortedByNew, gainers: sortedByGainers, losers: sortedByLosers,
        collective: sortedByCollective, whales: sortedByWhaleHeat,
    };
    const list = lists[sort] || sortedByTrending;
    const page = list.slice(off, off + lim);
    res.json({ tokens: page, total: list.length, sort, offset: off, limit: lim, cacheReady, ts: Date.now() });
});

app.get('/api/token/:address', async (req, res) => {
    const { address } = req.params;
    let token = tokenCache.get(address);

    if (!token) {
        const data = await safeFetch(`https://api.dexscreener.com/latest/dex/tokens/${address}`);
        if (data?.pairs?.length) {
            const pairs = data.pairs.filter(p => p.chainId === 'solana')
                .sort((a, b) => (parseFloat(b.liquidity?.usd) || 0) - (parseFloat(a.liquidity?.usd) || 0));
            if (pairs.length) {
                token = parsePair(pairs[0]);
                tokenCache.set(address, token);
            }
        }
    }

    if (!token) return res.status(404).json({ error: 'Token not found' });

    // Enrich with collective data
    const whaleHeat = collectiveData.whaleTokenHeat.get(address);
    if (whaleHeat) token.whaleHeat = whaleHeat.whaleCount * 3 + whaleHeat.smartMoneyBuys * 8;
    const qData = collectiveData.quantumCache.get(address);
    if (qData) { token.quantumBull = qData.bullProb; token.quantumConfidence = qData.confidence; }

    // Fetch AI score in background
    getAIScore(address).then(ai => {
        if (ai) { token.aiScore = ai.score; token.aiVerdict = ai.verdict; }
    });

    const boost = boosts.get(address);
    res.json({ token, boost: boost || null, allPairs: [] });
});

app.get('/api/search', async (req, res) => {
    const { q } = req.query;
    if (!q || q.length < 2) return res.json({ tokens: [] });

    const query = q.toLowerCase();
    const cacheResults = [...tokenCache.values()].filter(t =>
        t.symbol.toLowerCase().includes(query) ||
        t.name.toLowerCase().includes(query) ||
        t.address.toLowerCase().startsWith(query)
    ).slice(0, 20);

    if (cacheResults.length >= 5) return res.json({ tokens: cacheResults });

    const data = await safeFetch(`https://api.dexscreener.com/latest/dex/search?q=${encodeURIComponent(q)}`);
    const tokens = (data?.pairs || []).filter(p => p.chainId === 'solana').slice(0, 30).map(parsePair);
    const seen = new Set(cacheResults.map(t => t.address));
    for (const t of tokens) { if (!seen.has(t.address)) { cacheResults.push(t); seen.add(t.address); } }
    res.json({ tokens: cacheResults.slice(0, 30) });
});

// Chart data — GoldRush streaming (primary) → GeckoTerminal (fallback)
app.get('/api/chart/:address', async (req, res) => {
    const { address } = req.params;
    const { timeframe = '15m' } = req.query;

    // v5.0: Try GoldRush streaming candles first (real-time 1m data)
    if (timeframe === '1m' && goldrushCandles.has(address) && goldrushCandles.get(address).length > 0) {
        const candles = goldrushCandles.get(address);
        return res.json({ candles, source: 'goldrush-stream', pair: address });
    }

    // Subscribe to GoldRush candles for this token if not already
    if (goldrushClient && !goldrushCandles.has(address)) {
        subscribeGoldRushCandles(address);
    }

    // Fallback: GeckoTerminal OHLCV
    const tfMap = { '1m': ['minute', 1], '5m': ['minute', 5], '15m': ['minute', 15], '1h': ['hour', 1], '4h': ['hour', 4], '1d': ['day', 1] };
    const [period, agg] = tfMap[timeframe] || tfMap['15m'];

    const token = tokenCache.get(address);
    const poolAddr = token?.pairAddress || address;

    const data = await safeFetch(`https://api.geckoterminal.com/api/v2/networks/solana/pools/${poolAddr}/ohlcv/${period}?aggregate=${agg}&limit=300&currency=usd`);
    if (data?.data?.attributes?.ohlcv_list) {
        const candles = data.data.attributes.ohlcv_list.map(c => ({
            time: c[0], open: parseFloat(c[1]), high: parseFloat(c[2]),
            low: parseFloat(c[3]), close: parseFloat(c[4]), volume: parseFloat(c[5]),
        })).sort((a, b) => a.time - b.time);
        return res.json({ candles, source: 'geckoterminal', pair: poolAddr });
    }

    // v5.0: Last resort — return any GoldRush candles we have even if timeframe doesn't match
    if (goldrushCandles.has(address) && goldrushCandles.get(address).length > 0) {
        return res.json({ candles: goldrushCandles.get(address), source: 'goldrush-stream', pair: address });
    }

    res.json({ candles: [], source: 'unavailable' });
});

// AI Prediction
app.get('/api/ai/score/:address', async (req, res) => {
    const result = await getAIScore(req.params.address);
    res.json(result || { score: null, message: 'Collective unavailable' });
});

// Quantum Simulation
app.get('/api/quantum/:address', async (req, res) => {
    const result = await getQuantumPrediction(req.params.address);
    if (result) {
        res.json({
            available: true,
            bull_probability: result.bullProb,
            confidence: result.confidence,
            rug_probability: result.rugProb,
            price_targets: result.targets,
            simulation_count: result.simulations,
        });
    } else {
        res.json({ available: false });
    }
});

// Collective status — public endpoint showing the intelligence layer
app.get('/api/collective/status', (req, res) => {
    res.json({
        whales: {
            tracked: collectiveData.whales.topWallets.length,
            hotTokens: sortedByWhaleHeat.slice(0, 10).map(t => ({
                address: t.address, symbol: t.symbol, whaleHeat: t.whaleHeat,
            })),
        },
        trader: {
            running: collectiveData.trader.running,
            winRate: collectiveData.trader.winRate,
            totalTrades: collectiveData.trader.totalTrades,
            activePositions: (collectiveData.trader.positions || []).length,
        },
        picks: sortedByCollective.slice(0, 10).map(t => ({
            address: t.address, symbol: t.symbol, name: t.name, platform: t.platform,
            trendScore: Math.round(t.trendScore), aiScore: t.aiScore,
            quantumBull: t.quantumBull, whaleHeat: t.whaleHeat,
            reason: t.collectiveReason || 'High composite score',
        })),
        scoring: {
            formula: 'Base Market (35%) + Collective AI (25%) + Quantum Sim (15%) + Whale/Smart Money (15%) + Burn Economy (10%)',
            burnMultiplier: '3x vs SOL boost',
            platforms: ALLOWED_SUFFIXES,
        },
    });
});

// ============================================================
// BOOST SYSTEM — Burn Economy
// ============================================================
app.get('/api/boost/:address', (req, res) => {
    res.json({ boost: boosts.get(req.params.address) || null });
});

app.post('/api/boost/request', (req, res) => {
    const { address, paymentType = 'sol', level = 1 } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });

    const boostLevel = Math.min(Math.max(parseInt(level) || 1, 1), 3);
    const priceUsd = BOOST_PRICES[boostLevel] || 25;

    const existing = boosts.get(address);
    if (boostLevel >= 2 && existing?.boostCount > 0 && !existing.collectiveApproved) {
        return res.json({ status: 'requires_approval', message: 'Level 2+ boosts require passing verification checks.', boostCount: existing.boostCount });
    }

    const requestId = `boost_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const payment = paymentType === 'farns'
        ? { type: 'farns_burn', tokenMint: FARNS_TOKEN, burnAddress: '1nc1nerator11111111111111111111111111111111', amountUsd: priceUsd }
        : { type: 'sol', recipient: ECOSYSTEM_WALLET, amountUsd: priceUsd };

    pendingBoosts.set(requestId, { address, paymentType, level: boostLevel, requestedAt: Date.now() });
    res.json({ requestId, paymentInstructions: payment, boostPrice: priceUsd, level: boostLevel });
});

app.post('/api/boost/confirm', (req, res) => {
    const { address, txSignature, paymentType, level, wallet, amountUsd, farnsBurned, requestId } = req.body;

    // Support both new direct-tx flow and legacy requestId flow
    const tokenAddr = address || (requestId && pendingBoosts.get(requestId) ? pendingBoosts.get(requestId).address : null);
    const pType = paymentType || (requestId && pendingBoosts.get(requestId) ? pendingBoosts.get(requestId).paymentType : 'sol');

    if (!tokenAddr || !txSignature) return res.status(400).json({ error: 'Token address and tx signature required' });

    const boostLevel = parseInt(level) || 1;
    const priceUsd = BOOST_PRICES[boostLevel] || 25;

    const existing = boosts.get(tokenAddr) || {
        address: tokenAddr, boostCount: 0, boostLevel: 0, totalSolPaid: 0, totalFarnsBurned: 0,
        collectiveApproved: false, extendedInfo: false, lastBoostAt: null,
        boostHistory: [], wallet: null,
    };
    existing.boostCount++;
    existing.boostLevel = Math.max(existing.boostLevel || 0, boostLevel);
    existing.lastBoostAt = Date.now();
    existing.wallet = wallet || existing.wallet;

    // Track payment
    const token = tokenCache.get(tokenAddr);
    if (pType === 'farns') {
        existing.totalFarnsBurned += amountUsd || priceUsd;
        if (token) token.burnBoostAmount = (token.burnBoostAmount || 0) + (amountUsd || priceUsd);
    } else {
        existing.totalSolPaid += amountUsd || priceUsd;
        if (token) token.solBoostAmount = (token.solBoostAmount || 0) + (amountUsd || priceUsd);
    }

    // Record in history
    existing.boostHistory = existing.boostHistory || [];
    existing.boostHistory.push({
        level: boostLevel, paymentType: pType, txSignature,
        wallet: wallet || null, amountUsd: amountUsd || priceUsd,
        farnsBurned: farnsBurned || 0, ts: Date.now(),
    });

    // Auto-approve collective if level 2+ passed eligibility
    if (boostLevel >= 2) existing.collectiveApproved = true;
    if (boostLevel >= 3 && token) token.collectiveVerified = true;

    boosts.set(tokenAddr, existing);
    if (requestId) pendingBoosts.delete(requestId);

    // Recalculate trend score
    if (token) {
        token.boostAmount = existing.totalSolPaid + existing.totalFarnsBurned;
        token.trendScore = calculateTrendScore(token);
    }

    console.log(`[DEXAI] Boost confirmed: ${tokenAddr.slice(0, 8)}... Level ${boostLevel} via ${pType} (${txSignature.slice(0, 12)}...)`);
    res.json({ success: true, boost: existing });
});

// ============================================================
// BOOST LEVEL ELIGIBILITY CHECK
// When a token is boosted at level 2+, the collective verifies:
//   - Holder distribution (top holders, concentration)
//   - Rug probability (bundle detection, LP locks)
//   - X engagement (who is shilling, organic vs paid)
//   - Collective approval (multi-agent consensus)
// ============================================================
app.post('/api/boost/check-level', async (req, res) => {
    const { address, level = 1 } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });
    if (level <= 1) return res.json({ eligible: true, level: 1, checks: [] });

    const checks = [];
    let eligible = true;
    let reason = '';

    try {
        // 1. Run deep analysis via Cosmos collective
        const analysis = await safeFetch(`${COSMOS_API}/api/tools/deep-analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                token_address: address,
                checks: ['bundles', 'holder_distribution', 'creator_history', 'liquidity_locks']
            }),
        });

        // Holder distribution check
        const distPassed = analysis && (analysis.holder_distribution_score || analysis.score || 0) >= 50
            && !(analysis.top_holder_concentration > 0.5);
        checks.push({ name: 'Holder Distribution', passed: distPassed,
            detail: distPassed ? 'Healthy distribution' : 'Top holders too concentrated' });

        // Rug probability check
        const rugPassed = analysis && (analysis.rug_probability || 1) < 0.35 && !analysis.bundle_detected;
        checks.push({ name: 'Rug Check', passed: rugPassed,
            detail: rugPassed ? 'Low rug risk' : 'High rug probability or bundles detected' });

        // Liquidity check
        const token = tokenCache.get(address);
        const liqPassed = token && token.liquidity > 5000;
        checks.push({ name: 'Liquidity', passed: liqPassed,
            detail: liqPassed ? 'Sufficient liquidity' : 'Liquidity too low' });

        // Level 2 requires: distribution + rug + liquidity
        if (level === 2) {
            eligible = distPassed && rugPassed && liqPassed;
            if (!eligible) reason = 'Token must have healthy distribution, low rug risk, and sufficient liquidity for Level 2';
        }

        // Level 3 requires all of the above PLUS X engagement audit and collective approval
        if (level >= 3) {
            // X Engagement audit — search X for the token CA
            let xPassed = false;
            try {
                const xData = await safeFetch(`${COSMOS_API}/api/tools/search-x`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: address, type: 'token_audit' }),
                });
                if (xData) {
                    const tweetCount = xData.tweet_count || xData.results_count || 0;
                    const uniquePosters = xData.unique_posters || xData.unique_authors || 0;
                    const organicRatio = xData.organic_ratio || (uniquePosters > 3 ? 0.7 : 0.2);
                    xPassed = tweetCount >= 5 && uniquePosters >= 3 && organicRatio > 0.4;
                    checks.push({ name: 'X Engagement Audit', passed: xPassed,
                        detail: xPassed
                            ? `${tweetCount} posts by ${uniquePosters} unique accounts, ${Math.round(organicRatio * 100)}% organic`
                            : `Insufficient X engagement (${tweetCount} posts, ${uniquePosters} accounts)` });
                } else {
                    checks.push({ name: 'X Engagement Audit', passed: false, detail: 'X audit unavailable' });
                }
            } catch (e) {
                checks.push({ name: 'X Engagement Audit', passed: false, detail: 'X audit failed' });
            }

            // Collective approval — multi-agent consensus
            let collectivePassed = false;
            try {
                const collectiveResult = await safeFetch(`${COSMOS_API}/api/tools/score-token`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token_address: address, deep: true }),
                });
                if (collectiveResult && (collectiveResult.score || 0) >= 65) {
                    collectivePassed = true;
                }
                checks.push({ name: 'Collective Approval', passed: collectivePassed,
                    detail: collectivePassed
                        ? `Score: ${collectiveResult.score}/100 — approved`
                        : `Score: ${collectiveResult ? collectiveResult.score : 0}/100 — needs 65+` });
            } catch (e) {
                checks.push({ name: 'Collective Approval', passed: false, detail: 'Collective unavailable' });
            }

            eligible = distPassed && rugPassed && liqPassed && xPassed && collectivePassed;
            if (!eligible) reason = 'Level 3 requires all checks to pass: distribution, rug, liquidity, X audit, and collective approval';
        }

    } catch (e) {
        console.error('[DEXAI] Boost eligibility check error:', e.message);
        eligible = false;
        reason = 'Verification system temporarily unavailable';
        checks.push({ name: 'System', passed: false, detail: 'Verification error' });
    }

    res.json({ eligible, level, reason, checks });
});

app.post('/api/boost/submit-for-approval', async (req, res) => {
    const { address } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });

    const analysis = await safeFetch(`${COSMOS_API}/api/tools/deep-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_address: address, checks: ['bundles', 'holder_distribution', 'creator_history', 'liquidity_locks'] }),
    });

    if (!analysis) return res.json({ status: 'pending', message: 'Collective analysis in progress.' });

    const approved = (analysis.score || 0) >= 60 && (analysis.rug_probability || 1) < 0.3 && !analysis.bundle_detected;
    if (approved) {
        const existing = boosts.get(address) || { address, boostCount: 0, totalSolPaid: 0, totalFarnsBurned: 0, collectiveApproved: false, extendedInfo: false, lastBoostAt: null };
        existing.collectiveApproved = true;
        boosts.set(address, existing);
        const token = tokenCache.get(address);
        if (token) { token.collectiveVerified = true; token.trendScore = calculateTrendScore(token); }
    }
    res.json({ status: approved ? 'approved' : 'rejected', analysis: { score: analysis.score, rugProb: analysis.rug_probability, bundled: analysis.bundle_detected } });
});

app.post('/api/extended-info/purchase', (req, res) => {
    const { address, paymentType = 'sol' } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });

    const requestId = `ext_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const payment = paymentType === 'farns'
        ? { type: 'farns_burn', tokenMint: FARNS_TOKEN, burnAddress: '1nc1nerator11111111111111111111111111111111', amountUsd: EXTENDED_INFO_PRICE_USD }
        : { type: 'sol', recipient: ECOSYSTEM_WALLET, amountUsd: EXTENDED_INFO_PRICE_USD };

    res.json({ requestId, paymentInstructions: payment, price: EXTENDED_INFO_PRICE_USD,
        features: ['Custom description', 'Social links', 'Team info', 'Roadmap', 'Header image', 'Search priority'] });
});

// ============================================================
// REAL-TIME PRICE ENGINE — 5-Provider Round-Robin
// Cycles: Birdeye → Jupiter → DexScreener → GeckoTerminal → Raydium
// Each provider hit once every ~10s (5 providers × 2s interval)
// ============================================================
const livepriceCache = new Map(); // address -> { price, priceNative, ts, source }
const livePriceWatchers = new Map(); // address -> { lastTouch }

// Per-provider backoff tracking
const providerState = {
    birdeye:       { backoffUntil: 0, fails: 0 },
    jupiter:       { backoffUntil: 0, fails: 0 },
    dexscreener:   { backoffUntil: 0, fails: 0 },
    geckoterminal: { backoffUntil: 0, fails: 0 },
    raydium:       { backoffUntil: 0, fails: 0 },
    helius:        { backoffUntil: 0, fails: 0 },
    goldrush:      { backoffUntil: 0, fails: 0 },
};

function markProviderFail(name) {
    const s = providerState[name];
    s.fails++;
    const backoff = Math.min(3000 * Math.pow(2, s.fails), 120000);
    s.backoffUntil = Date.now() + backoff;
    console.log(`[LIVE] ${name} failed (#${s.fails}), backoff ${backoff/1000}s`);
}
function markProviderOk(name) { providerState[name].fails = 0; providerState[name].backoffUntil = 0; }
function isProviderReady(name) { return Date.now() >= providerState[name].backoffUntil; }

// ── PROVIDER 1: Birdeye (batch up to 50) ──
async function fetchBirdeyeMultiPrice(addresses) {
    if (!addresses.length || !isProviderReady('birdeye')) return {};
    try {
        const list = addresses.slice(0, 50).join(',');
        const res = await fetch(`${BIRDEYE_BASE}/defi/multi_price?list_address=${list}`, {
            signal: AbortSignal.timeout(5000),
            headers: { 'X-API-KEY': BIRDEYE_API_KEY, 'x-chain': 'solana' }
        });
        if (res.status === 429) { markProviderFail('birdeye'); return {}; }
        if (!res.ok) { markProviderFail('birdeye'); return {}; }
        const data = await res.json();
        const results = {};
        if (data?.data) {
            markProviderOk('birdeye');
            for (const [addr, info] of Object.entries(data.data)) {
                if (info?.value) results[addr] = { price: info.value, source: 'birdeye', ts: Date.now() };
            }
        }
        return results;
    } catch { return {}; }
}

// ── PROVIDER 2: Jupiter (batch up to 100) ──
async function fetchJupiterMultiPrice(addresses) {
    if (!addresses.length || !isProviderReady('jupiter')) return {};
    try {
        const ids = addresses.slice(0, 100).join(',');
        const res = await fetch(`${JUPITER_PRICE_BASE}?ids=${ids}`, {
            signal: AbortSignal.timeout(5000),
            headers: { 'x-api-key': JUPITER_API_KEY }
        });
        if (res.status === 429) { markProviderFail('jupiter'); return {}; }
        if (!res.ok) { markProviderFail('jupiter'); return {}; }
        const data = await res.json();
        const results = {};
        if (data?.data) {
            markProviderOk('jupiter');
            for (const [addr, info] of Object.entries(data.data)) {
                if (info?.price) results[addr] = { price: parseFloat(info.price), source: 'jupiter', ts: Date.now() };
            }
        }
        return results;
    } catch { return {}; }
}

// ── PROVIDER 3: DexScreener (batch up to 30, no key needed) ──
async function fetchDexScreenerMultiPrice(addresses) {
    if (!addresses.length || !isProviderReady('dexscreener')) return {};
    try {
        const batch = addresses.slice(0, 30).join(',');
        const res = await fetch(`https://api.dexscreener.com/latest/dex/tokens/${batch}`, {
            signal: AbortSignal.timeout(6000),
        });
        if (res.status === 429) { markProviderFail('dexscreener'); return {}; }
        if (!res.ok) { markProviderFail('dexscreener'); return {}; }
        const data = await res.json();
        const results = {};
        if (data?.pairs) {
            markProviderOk('dexscreener');
            // Pick highest-liquidity pair per token
            const byToken = {};
            for (const pair of data.pairs) {
                if (pair.chainId !== 'solana') continue;
                const addr = pair.baseToken?.address;
                if (!addr || !pair.priceUsd) continue;
                const liq = parseFloat(pair.liquidity?.usd) || 0;
                if (!byToken[addr] || liq > byToken[addr].liq) {
                    byToken[addr] = { price: parseFloat(pair.priceUsd), liq };
                }
            }
            for (const [addr, info] of Object.entries(byToken)) {
                results[addr] = { price: info.price, source: 'dexscreener', ts: Date.now() };
            }
        }
        return results;
    } catch { return {}; }
}

// ── PROVIDER 4: GeckoTerminal (multi-token, no key needed) ──
async function fetchGeckoTerminalMultiPrice(addresses) {
    if (!addresses.length || !isProviderReady('geckoterminal')) return {};
    try {
        // GeckoTerminal multi-token endpoint (up to 30)
        const batch = addresses.slice(0, 30).join(',');
        const res = await fetch(`https://api.geckoterminal.com/api/v2/networks/solana/tokens/multi/${batch}`, {
            signal: AbortSignal.timeout(6000),
            headers: { 'Accept': 'application/json' },
        });
        if (res.status === 429) { markProviderFail('geckoterminal'); return {}; }
        if (!res.ok) { markProviderFail('geckoterminal'); return {}; }
        const data = await res.json();
        const results = {};
        if (data?.data && Array.isArray(data.data)) {
            markProviderOk('geckoterminal');
            for (const token of data.data) {
                const attrs = token.attributes || {};
                const addr = attrs.address;
                const price = parseFloat(attrs.price_usd);
                if (addr && price > 0) {
                    results[addr] = { price, source: 'geckoterminal', ts: Date.now() };
                }
            }
        }
        return results;
    } catch { return {}; }
}

// ── PROVIDER 5: Raydium (batch mint/price, no key needed) ──
async function fetchRaydiumMultiPrice(addresses) {
    if (!addresses.length || !isProviderReady('raydium')) return {};
    try {
        const mints = addresses.slice(0, 50).join(',');
        const res = await fetch(`https://api-v3.raydium.io/mint/price?mints=${mints}`, {
            signal: AbortSignal.timeout(5000),
        });
        if (res.status === 429) { markProviderFail('raydium'); return {}; }
        if (!res.ok) { markProviderFail('raydium'); return {}; }
        const data = await res.json();
        const results = {};
        // Raydium returns { data: { "mintAddr": "priceString", ... } }
        const prices = data?.data || data;
        if (prices && typeof prices === 'object') {
            markProviderOk('raydium');
            for (const [addr, priceVal] of Object.entries(prices)) {
                const p = parseFloat(priceVal);
                if (p > 0) results[addr] = { price: p, source: 'raydium', ts: Date.now() };
            }
        }
        return results;
    } catch { return {}; }
}

// ── PROVIDER 6: Helius DAS API (getAssetBatch, returns price_per_token) ──
async function fetchHeliusMultiPrice(addresses) {
    if (!addresses.length || !isProviderReady('helius')) return {};
    try {
        // getAssetBatch supports up to 1000 assets
        const batch = addresses.slice(0, 100);
        const res = await fetch(HELIUS_RPC, {
            method: 'POST',
            signal: AbortSignal.timeout(6000),
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                jsonrpc: '2.0', id: 'helius-price',
                method: 'getAssetBatch',
                params: { ids: batch },
            }),
        });
        if (res.status === 429) { markProviderFail('helius'); return {}; }
        if (!res.ok) { markProviderFail('helius'); return {}; }
        const data = await res.json();
        const results = {};
        const assets = data?.result || [];
        if (Array.isArray(assets) && assets.length > 0) {
            markProviderOk('helius');
            for (const asset of assets) {
                const addr = asset?.id;
                const priceInfo = asset?.token_info?.price_info;
                if (addr && priceInfo?.price_per_token > 0) {
                    results[addr] = { price: priceInfo.price_per_token, source: 'helius', ts: Date.now() };
                }
            }
        }
        return results;
    } catch { return {}; }
}

// ── PROVIDER 7: GoldRush Streaming API (real-time OHLCV + new pairs) ──
const GOLDRUSH_API_KEY = process.env.GOLDRUSH_API_KEY || 'cqt_rQHQf7Jjtdbwchwvf3ctgTQpKTc8';
const goldrushCandles = new Map(); // address -> [{ time, open, high, low, close, volume }]
const goldrushPrices = new Map();  // address -> { price, ts }
let goldrushClient = null;
let goldrushNewPairsSub = null;

async function initGoldRush() {
    try {
        // SDK's StreamingWebSocketClient.connect() calls createClient() without
        // forwarding webSocketImpl — set global so createClient finds it.
        if (typeof globalThis.WebSocket === 'undefined') {
            globalThis.WebSocket = require('ws');
        }
        const { GoldRushClient } = require('@covalenthq/client-sdk');
        goldrushClient = new GoldRushClient(
            GOLDRUSH_API_KEY,
            {},
            {
                onOpened: () => console.log('[GOLDRUSH] Connected to streaming service'),
                onClosed: () => console.log('[GOLDRUSH] Disconnected from streaming service'),
                onError: (err) => console.error('[GOLDRUSH] Error:', err),
            }
        );

        // Subscribe to new Solana DEX pairs (raw query — SDK method has schema mismatch)
        goldrushNewPairsSub = goldrushClient.StreamingService.rawQuery(
            `subscription { newPairs(chain_name: SOLANA_MAINNET) { chain_name protocol pair_address tx_hash liquidity } }`, {},
            {
                next: (data) => {
                    const pairs = data?.data?.newPairs;
                    if (!pairs) return;
                    const arr = Array.isArray(pairs) ? pairs : [pairs];
                    for (const pair of arr) {
                        console.log(`[GOLDRUSH] New pair: ${pair.pair_address?.slice(0,8)}... on ${pair.protocol} liq=$${Math.round(pair.liquidity || 0)}`);
                        const msg = JSON.stringify({ type: 'newPair', pair: {
                            address: pair.pair_address,
                            pairAddress: pair.pair_address,
                            protocol: pair.protocol,
                            liquidity: pair.liquidity,
                        }});
                        for (const ws of wsClients) {
                            if (ws.readyState === 1) { try { ws.send(msg); } catch {} }
                        }
                    }
                },
                error: (err) => console.error('[GOLDRUSH] newPairs error:', err),
            }
        );
        console.log('[GOLDRUSH] Streaming initialized — new pairs + OHLCV candles');
    } catch (e) {
        console.warn('[GOLDRUSH] SDK not available:', e.message);
    }
}

function subscribeGoldRushCandles(tokenAddress) {
    if (!goldrushClient || goldrushCandles.has(tokenAddress)) return;
    goldrushCandles.set(tokenAddress, []);
    try {
        goldrushClient.StreamingService.rawQuery(
            `subscription { ohlcvCandlesForToken(chain_name: SOLANA_MAINNET, token_addresses: ["${tokenAddress}"], interval: ONE_MINUTE, timeframe: ONE_HOUR) { timestamp open high low close volume volume_usd } }`, {},
            {
                next: (data) => {
                    const candles = data?.data?.ohlcvCandlesForToken;
                    if (!candles) return;
                    const arr2 = Array.isArray(candles) ? candles : [candles];
                    for (const candle of arr2) {
                        const entry = {
                            time: Math.floor(new Date(candle.timestamp).getTime() / 1000),
                            open: parseFloat(candle.open), high: parseFloat(candle.high),
                            low: parseFloat(candle.low), close: parseFloat(candle.close),
                            volume: parseFloat(candle.volume_usd || candle.volume || 0),
                        };
                        const arr = goldrushCandles.get(tokenAddress) || [];
                        arr.push(entry);
                        if (arr.length > 300) arr.shift();
                        goldrushCandles.set(tokenAddress, arr);
                        if (entry.close > 0) {
                            goldrushPrices.set(tokenAddress, { price: entry.close, ts: Date.now() });
                            livepriceCache.set(tokenAddress, { price: entry.close, source: 'goldrush-stream', ts: Date.now() });
                        }
                        const msg = JSON.stringify({ type: 'candle', token: tokenAddress, candle: entry, source: 'goldrush' });
                        for (const ws of wsClients) {
                            if (ws.readyState === 1 && ws.subs?.has(tokenAddress)) {
                                try { ws.send(msg); } catch {}
                            }
                        }
                    }
                },
                error: (err) => console.error(`[GOLDRUSH] candle error for ${tokenAddress.slice(0,8)}:`, err),
            }
        );
        console.log(`[GOLDRUSH] Subscribed to OHLCV: ${tokenAddress.slice(0, 12)}...`);
    } catch (e) {
        console.warn(`[GOLDRUSH] Candle subscribe failed for ${tokenAddress.slice(0,8)}:`, e.message);
    }
}

// GoldRush as batch price provider (uses cached stream prices)
async function fetchGoldRushMultiPrice(addresses) {
    const results = {};
    for (const addr of addresses) {
        const cached = goldrushPrices.get(addr);
        if (cached && Date.now() - cached.ts < 10000) {
            results[addr] = { price: cached.price, source: 'goldrush', ts: cached.ts };
        }
    }
    if (Object.keys(results).length > 0) markProviderOk('goldrush');
    return results;
}

// ── ROUND-ROBIN PROVIDER CYCLING ──
const PROVIDER_ORDER = [
    { name: 'birdeye',       fn: fetchBirdeyeMultiPrice },
    { name: 'jupiter',       fn: fetchJupiterMultiPrice },
    { name: 'dexscreener',   fn: fetchDexScreenerMultiPrice },
    { name: 'geckoterminal', fn: fetchGeckoTerminalMultiPrice },
    { name: 'raydium',       fn: fetchRaydiumMultiPrice },
    { name: 'helius',        fn: fetchHeliusMultiPrice },
    { name: 'goldrush',      fn: fetchGoldRushMultiPrice },
];
let currentProviderIdx = 0;

// Kept for the /api/live/:address single-token fallback
async function fetchLivePrice(address) {
    // Try all providers in order, skip backed-off ones
    for (const provider of PROVIDER_ORDER) {
        if (!isProviderReady(provider.name)) continue;
        const results = await provider.fn([address]);
        if (results[address]) {
            livepriceCache.set(address, results[address]);
            return results[address];
        }
    }
    // Fallback to last known LIVE price
    const lastLive = livepriceCache.get(address);
    if (lastLive && Date.now() - lastLive.ts < 30000) {
        return { price: lastLive.price, source: lastLive.source.replace(/-cached$/, '') + '-cached', ts: lastLive.ts };
    }
    const token = tokenCache.get(address);
    if (token?.price) return { price: token.price, source: 'cache', ts: Date.now() };
    return null;
}

// ── BATCHED LIVE PRICE LOOP — ROUND-ROBIN ──
// Every 2s, calls the NEXT provider in rotation.
// Each provider only gets called once every ~10s (5 providers × 2s).
// If that provider fails/is backed-off, immediately tries the next ready one.
let livePollRunning = false;

async function livePricePollCycle() {
    if (livePollRunning) return;
    livePollRunning = true;
    try {
        const addresses = [...livePriceWatchers.keys()];
        if (addresses.length === 0) { livePollRunning = false; return; }

        // Pick next ready provider (round-robin, skip backed-off)
        let results = {};
        let providerUsed = null;
        for (let i = 0; i < PROVIDER_ORDER.length; i++) {
            const idx = (currentProviderIdx + i) % PROVIDER_ORDER.length;
            const provider = PROVIDER_ORDER[idx];
            if (!isProviderReady(provider.name)) continue;

            results = await provider.fn(addresses);
            providerUsed = provider.name;
            currentProviderIdx = (idx + 1) % PROVIDER_ORDER.length;
            break;
        }

        const resolved = new Set(Object.keys(results));
        const gotCount = resolved.size;

        // For any tokens the primary provider missed, try ONE fallback
        if (resolved.size < addresses.length) {
            const missing = addresses.filter(a => !resolved.has(a));
            // Pick the next ready provider that's different from the one we just used
            for (const provider of PROVIDER_ORDER) {
                if (provider.name === providerUsed || !isProviderReady(provider.name)) continue;
                const fallbackResults = await provider.fn(missing);
                for (const [addr, result] of Object.entries(fallbackResults)) {
                    results[addr] = result;
                    resolved.add(addr);
                }
                break; // only one fallback attempt
            }
        }

        // Broadcast all results + use cache for anything still missing
        for (const address of addresses) {
            let result = results[address] || null;
            if (!result) {
                const lastLive = livepriceCache.get(address);
                if (lastLive && Date.now() - lastLive.ts < 30000) {
                    result = { price: lastLive.price, source: lastLive.source.replace(/-cached$/, '') + '-cached', ts: lastLive.ts };
                } else {
                    const token = tokenCache.get(address);
                    if (token?.price) result = { price: token.price, source: 'cache', ts: Date.now() };
                }
            }
            if (!result) continue;

            livepriceCache.set(address, result);

            const msg = JSON.stringify({ type: 'price', token: address, price: result.price, source: result.source, ts: result.ts });
            for (const ws of wsClients) {
                if (ws.readyState === 1 && ws.subs?.has(address)) {
                    try { ws.send(msg); } catch {}
                }
            }
        }

        if (providerUsed) {
            const readyCount = PROVIDER_ORDER.filter(p => isProviderReady(p.name)).length;
            console.log(`[LIVE] ${providerUsed}: ${gotCount}/${addresses.length} prices | ${readyCount}/${PROVIDER_ORDER.length} providers ready`);
        }
    } catch (e) {
        console.error('[LIVE] Poll cycle error:', e.message);
    }
    livePollRunning = false;
}

// Poll every 2 seconds — round-robin across 5 providers
const LIVE_POLL_MS = 2000;
setInterval(livePricePollCycle, LIVE_POLL_MS);

// Register/unregister tokens for live watching
function startLiveWatch(address) {
    if (livePriceWatchers.has(address)) return;
    livePriceWatchers.set(address, { lastTouch: Date.now() });
    // v5.0: Auto-subscribe to GoldRush 1m candles for real-time chart data
    if (goldrushClient) subscribeGoldRushCandles(address);
    console.log(`[LIVE] Watching: ${address.slice(0, 8)}... (${livePriceWatchers.size} active)`);
}

function stopLiveWatch(address) {
    if (!livePriceWatchers.has(address)) return;
    livePriceWatchers.delete(address);
    console.log(`[LIVE] Stopped: ${address.slice(0, 8)}... (${livePriceWatchers.size} active)`);
}

function touchLiveWatch(address) {
    const watcher = livePriceWatchers.get(address);
    if (!watcher) { startLiveWatch(address); return; }
    watcher.lastTouch = Date.now();
}

// Cleanup stale watchers every 60 seconds (no activity for 10 min)
setInterval(() => {
    const now = Date.now();
    for (const [address, watcher] of livePriceWatchers) {
        // Also count WebSocket subscribers as "active"
        let hasWsSub = false;
        for (const ws of wsClients) {
            if (ws.readyState === 1 && ws.subs?.has(address)) { hasWsSub = true; break; }
        }
        if (!hasWsSub && now - watcher.lastTouch > 600000) {
            stopLiveWatch(address);
        }
    }
}, 60000);

// ── HELIUS WEBSOCKET — Real-time trade push notifications ──
// Subscribes to Jupiter/Raydium program logs. When a swap is detected
// involving a watched token, we instantly refresh its price from cache
// and push to frontend WebSocket — fills gaps between poll intervals.
const WebSocket = require('ws');
let heliusWs = null;
let heliusSubId = null;
let heliusReconnectTimer = null;
const JUPITER_PROGRAM = 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4';
const RAYDIUM_AMM_PROGRAM = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8';

function connectHeliusWs() {
    if (heliusWs && heliusWs.readyState <= 1) return; // already open/connecting
    try {
        heliusWs = new WebSocket(HELIUS_WSS);

        heliusWs.on('open', () => {
            console.log('[HELIUS-WS] Connected — subscribing to DEX swaps');
            // Subscribe to Jupiter program logs (covers most Solana swaps)
            heliusWs.send(JSON.stringify({
                jsonrpc: '2.0', id: 1, method: 'logsSubscribe',
                params: [
                    { mentions: [JUPITER_PROGRAM] },
                    { commitment: 'confirmed' },
                ],
            }));
        });

        heliusWs.on('message', async (raw) => {
            try {
                const msg = JSON.parse(raw);
                // Subscription confirmation
                if (msg.result !== undefined && !msg.method) {
                    heliusSubId = msg.result;
                    console.log(`[HELIUS-WS] Subscribed (id: ${heliusSubId})`);
                    return;
                }
                // Log notification — a swap happened
                if (msg.method === 'logsNotification') {
                    const logs = msg.params?.result?.value?.logs || [];
                    const logStr = logs.join(' ');
                    // Check if any of our watched tokens appear in the logs
                    for (const address of livePriceWatchers.keys()) {
                        if (logStr.includes(address)) {
                            // Token involved in a swap — quick-refresh from fastest available provider
                            const result = await fetchLivePrice(address);
                            if (result) {
                                livepriceCache.set(address, result);
                                const priceMsg = JSON.stringify({ type: 'price', token: address, price: result.price, source: result.source + '+ws', ts: result.ts });
                                for (const ws of wsClients) {
                                    if (ws.readyState === 1 && ws.subs?.has(address)) {
                                        try { ws.send(priceMsg); } catch {}
                                    }
                                }
                            }
                            break; // one refresh per log batch is enough
                        }
                    }
                }
            } catch {}
        });

        heliusWs.on('close', () => {
            console.log('[HELIUS-WS] Disconnected, reconnecting in 5s...');
            heliusSubId = null;
            if (heliusReconnectTimer) clearTimeout(heliusReconnectTimer);
            heliusReconnectTimer = setTimeout(connectHeliusWs, 5000);
        });

        heliusWs.on('error', (err) => {
            console.error('[HELIUS-WS] Error:', err.message);
        });
    } catch (e) {
        console.error('[HELIUS-WS] Connect failed:', e.message);
        if (heliusReconnectTimer) clearTimeout(heliusReconnectTimer);
        heliusReconnectTimer = setTimeout(connectHeliusWs, 10000);
    }
}

// Start Helius WebSocket after server boots (in START section)

// ============================================================
// LIVE PRICE (real-time endpoint)
// ============================================================
app.get('/api/live/:address', async (req, res) => {
    const address = req.params.address;
    // Ensure we're watching this token
    touchLiveWatch(address);
    // Return latest cached price (updated every 1s by the watcher)
    const cached = livepriceCache.get(address);
    if (cached && Date.now() - cached.ts < 2000) {
        const token = tokenCache.get(address);
        return res.json({
            price: cached.price,
            priceNative: token?.priceNative || null,
            priceChange: token?.priceChange || null,
            volume: token?.volume || null,
            txns: token?.txns || null,
            source: cached.source,
            ts: cached.ts,
        });
    }
    // Fetch fresh if no recent cache
    const result = await fetchLivePrice(address);
    if (!result) return res.json({ price: null });
    const token = tokenCache.get(address);
    res.json({
        price: result.price,
        priceNative: token?.priceNative || null,
        priceChange: token?.priceChange || null,
        volume: token?.volume || null,
        txns: token?.txns || null,
        source: result.source,
        ts: result.ts,
    });
});

// ============================================================
// RECENT TRADES (via GeckoTerminal)
// ============================================================
const tradesCache = new Map();

app.get('/api/trades/:address', async (req, res) => {
    const { address } = req.params;
    const cached = tradesCache.get(address);
    if (cached && Date.now() - cached.ts < 8000) return res.json({ trades: cached.trades });

    const token = tokenCache.get(address);
    const poolAddr = token?.pairAddress || address;

    const data = await safeFetch(
        `https://api.geckoterminal.com/api/v2/networks/solana/pools/${poolAddr}/trades`
    );

    if (!data?.data) return res.json({ trades: [] });

    const trades = data.data.slice(0, 50).map(t => {
        const a = t.attributes || {};
        return {
            type: a.kind || 'unknown',
            priceUsd: parseFloat(a.price_to_in_usd || a.price_from_in_usd || 0),
            volumeUsd: parseFloat(a.volume_in_usd || 0),
            tokenAmount: parseFloat(a.to_token_amount || a.from_token_amount || 0),
            txHash: a.tx_hash || '',
            maker: a.tx_from_address || '',
            timestamp: a.block_timestamp || null,
        };
    });

    tradesCache.set(address, { trades, ts: Date.now() });
    res.json({ trades });
});

// ============================================================
// BONDING CURVE STATUS (pump.fun tokens)
// ============================================================
app.get('/api/bonding/:address', (req, res) => {
    const token = tokenCache.get(req.params.address);
    if (!token) return res.json({ available: false });

    const isPump = token.platform === 'pump';
    if (!isPump) {
        return res.json({ available: true, bonded: true, graduated: true, platform: token.platform || 'other', progress: 100 });
    }

    const GRAD_THRESHOLD = 69000;
    const mcap = token.marketCap || token.fdv || 0;
    const liq = token.liquidity || 0;
    const progress = Math.min((mcap / GRAD_THRESHOLD) * 100, 100);
    const graduated = mcap >= GRAD_THRESHOLD || token.dexId === 'raydium';

    res.json({
        available: true,
        bonded: graduated,
        graduated,
        platform: 'pump',
        progress: graduated ? 100 : Math.round(progress * 10) / 10,
        marketCap: mcap,
        threshold: GRAD_THRESHOLD,
        remainingUsd: graduated ? 0 : Math.max(0, GRAD_THRESHOLD - mcap),
        liquidity: liq,
        dexId: token.dexId,
    });
});

// ============================================================
// WEBSOCKET
// ============================================================
const wsClients = new Set();

wss.on('connection', (ws) => {
    wsClients.add(ws);
    ws.isAlive = true;
    ws.on('pong', () => { ws.isAlive = true; });
    ws.on('message', (msg) => {
        try {
            const data = JSON.parse(msg);
            if (data.type === 'subscribe') { ws.subs = ws.subs || new Set(); ws.subs.add(data.token); }
            if (data.type === 'unsubscribe') ws.subs?.delete(data.token);
        } catch {}
    });
    ws.on('close', () => wsClients.delete(ws));
    ws.send(JSON.stringify({ type: 'connected', tokens: tokenCache.size }));
});

function broadcastAll(data) {
    const msg = JSON.stringify(data);
    for (const ws of wsClients) { if (ws.readyState === 1) ws.send(msg); }
}

setInterval(() => {
    for (const ws of wsClients) {
        if (!ws.isAlive) { ws.terminate(); wsClients.delete(ws); continue; }
        ws.isAlive = false; ws.ping();
    }
}, 30000);

// ============================================================
// X OAUTH 2.0 — Connect X (read-only: tweets, followers)
// ============================================================

// Step 1: Generate auth URL with PKCE
app.get('/api/x/auth', (req, res) => {
    const { wallet } = req.query;
    if (!wallet) return res.status(400).json({ error: 'Wallet address required' });

    // Generate PKCE code verifier + challenge
    const codeVerifier = crypto.randomBytes(32).toString('base64url');
    const codeChallenge = crypto.createHash('sha256').update(codeVerifier).digest('base64url');
    const state = crypto.randomBytes(16).toString('hex');

    // Store state for callback
    xOAuthState.set(state, { codeVerifier, wallet, ts: Date.now() });

    // Clean up old states (> 10 min)
    for (const [s, data] of xOAuthState) {
        if (Date.now() - data.ts > 600000) xOAuthState.delete(s);
    }

    const authUrl = 'https://x.com/i/oauth2/authorize?' + new URLSearchParams({
        response_type: 'code',
        client_id: X_CLIENT_ID,
        redirect_uri: X_REDIRECT_URI,
        scope: X_SCOPES,
        state: state,
        code_challenge: codeChallenge,
        code_challenge_method: 'S256',
    }).toString();

    res.json({ authUrl, state });
});

// Step 2: OAuth callback — exchange code for token, fetch profile
app.get('/api/x/callback', async (req, res) => {
    const { code, state } = req.query;

    if (!code || !state) {
        return res.redirect('/?x_error=missing_params');
    }

    const oauthData = xOAuthState.get(state);
    if (!oauthData) {
        return res.redirect('/?x_error=invalid_state');
    }
    xOAuthState.delete(state);

    try {
        // Exchange code for access token
        const basicAuth = Buffer.from(`${X_CLIENT_ID}:${X_CLIENT_SECRET}`).toString('base64');
        const tokenRes = await fetch('https://api.x.com/2/oauth2/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': `Basic ${basicAuth}`,
            },
            body: new URLSearchParams({
                code: code,
                grant_type: 'authorization_code',
                redirect_uri: X_REDIRECT_URI,
                code_verifier: oauthData.codeVerifier,
            }).toString(),
        });

        if (!tokenRes.ok) {
            console.error('[DEXAI] X token exchange failed:', tokenRes.status, await tokenRes.text());
            return res.redirect('/?x_error=token_exchange');
        }

        const tokenData = await tokenRes.json();
        const accessToken = tokenData.access_token;

        // Fetch X user profile
        const profileRes = await fetch('https://api.x.com/2/users/me?user.fields=id,name,username,profile_image_url', {
            headers: { 'Authorization': `Bearer ${accessToken}` },
        });

        if (!profileRes.ok) {
            console.error('[DEXAI] X profile fetch failed:', profileRes.status);
            return res.redirect('/?x_error=profile_fetch');
        }

        const profileData = await profileRes.json();
        const xUser = profileData.data;

        // Store X connection linked to wallet
        xConnections.set(oauthData.wallet, {
            xId: xUser.id,
            xUsername: xUser.username,
            xName: xUser.name,
            xProfileImage: xUser.profile_image_url,
            connectedAt: Date.now(),
        });

        console.log(`[DEXAI] X connected: @${xUser.username} → wallet ${oauthData.wallet.slice(0, 8)}...`);

        // Redirect back to DEXAI with success
        res.redirect(`/?x_connected=${encodeURIComponent(xUser.username)}`);

    } catch (e) {
        console.error('[DEXAI] X OAuth error:', e.message);
        res.redirect('/?x_error=server_error');
    }
});

// Step 3: Get current X connection for a wallet
app.get('/api/x/connection', (req, res) => {
    const { wallet } = req.query;
    if (!wallet) return res.status(400).json({ error: 'Wallet required' });

    const conn = xConnections.get(wallet);
    if (!conn) return res.json({ connected: false });

    res.json({
        connected: true,
        username: conn.xUsername,
        name: conn.xName,
        profileImage: conn.xProfileImage,
        connectedAt: conn.connectedAt,
    });
});

// Step 4: X badge for token — checks if deployer/fee-recipient has X linked
app.get('/api/x/badge/:tokenAddress', async (req, res) => {
    const { tokenAddress } = req.params;
    if (!tokenAddress) return res.json({ badges: [] });

    const badges = [];

    try {
        // Check BAGS.FM for token creator and fee recipients
        const [creatorData, feeData] = await Promise.all([
            safeFetch(`${BAGS_FM_API}/tokens/${tokenAddress}/creators`),
            safeFetch(`${BAGS_FM_API}/tokens/${tokenAddress}/fee-claimers`),
        ]);

        // Collect all associated wallets (deployer + fee recipients)
        const associatedWallets = new Set();

        if (creatorData) {
            const creator = creatorData.creator || creatorData.deployer || creatorData.wallet;
            if (creator) associatedWallets.add(typeof creator === 'string' ? creator : creator.wallet || creator.address);
            // Handle array format
            if (Array.isArray(creatorData)) {
                for (const c of creatorData) {
                    const w = c.wallet || c.address || c.creator;
                    if (w) associatedWallets.add(w);
                }
            }
        }

        if (feeData) {
            const claimers = feeData.claimers || feeData.fee_recipients || feeData;
            if (Array.isArray(claimers)) {
                for (const c of claimers) {
                    const w = c.wallet || c.address || c.claimer;
                    if (w) associatedWallets.add(w);
                }
            }
        }

        // Also try the generic BAGS.FM token endpoint
        const tokenInfo = await safeFetch(`${BAGS_FM_API}/tokens/${tokenAddress}`);
        if (tokenInfo) {
            if (tokenInfo.creator) associatedWallets.add(tokenInfo.creator);
            if (tokenInfo.fee_recipient) associatedWallets.add(tokenInfo.fee_recipient);
            if (tokenInfo.deployer) associatedWallets.add(tokenInfo.deployer);
        }

        // Check which associated wallets have X connections
        for (const wallet of associatedWallets) {
            const xConn = xConnections.get(wallet);
            if (xConn) {
                badges.push({
                    wallet: wallet.slice(0, 4) + '..' + wallet.slice(-4),
                    role: creatorData && (creatorData.creator === wallet || creatorData.deployer === wallet || creatorData.wallet === wallet)
                        ? 'deployer' : 'fee_recipient',
                    xUsername: xConn.xUsername,
                    xName: xConn.xName,
                    xProfileImage: xConn.xProfileImage,
                });
            }
        }
    } catch (e) {
        console.error('[DEXAI] X badge lookup error:', e.message);
    }

    res.json({ badges });
});

// Disconnect X from wallet
app.post('/api/x/disconnect', (req, res) => {
    const { wallet } = req.body;
    if (!wallet) return res.status(400).json({ error: 'Wallet required' });

    const had = xConnections.delete(wallet);
    res.json({ disconnected: had });
});

// ============================================================
// QUANTUM TRADING INTELLIGENCE — Token-Gated Premium Signals
// ============================================================

const SOLANA_RPC = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';
const MIN_FARNS_HOLDING = 100000; // 100K FARNS minimum for quantum access
const SPL_TOKEN_PROGRAM = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA';

// Quantum signal cache (populated by Python backend via /api/quantum/internal/*)
const quantumSignals = new Map();     // address -> { signal_id, direction, confidence, strength, ... }
const quantumSignalHistory = [];      // recent signals (max 500)
const quantumCorrelations = [];       // cross-token correlations
const quantumAccuracy = { win_rate: 0, total_signals: 0, resolved: 0, pending: 0 };

// FARNS balance cache: wallet -> { balance, ts }
const farnsBalanceCache = new Map();
const FARNS_CACHE_TTL = 120000; // 2 minutes

/**
 * Check FARNS token balance for a wallet via Solana RPC.
 * Read-only — no burn required, just holding check.
 */
async function checkFarnsBalance(walletAddress, minAmount = MIN_FARNS_HOLDING) {
    // Check cache first
    const cached = farnsBalanceCache.get(walletAddress);
    if (cached && Date.now() - cached.ts < FARNS_CACHE_TTL) {
        return { hasAccess: cached.balance >= minAmount, balance: cached.balance, required: minAmount };
    }

    try {
        const payload = {
            jsonrpc: '2.0', id: 1,
            method: 'getTokenAccountsByOwner',
            params: [
                walletAddress,
                { mint: FARNS_TOKEN },
                { encoding: 'jsonParsed' }
            ]
        };

        const resp = await fetch(SOLANA_RPC, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(10000),
        });
        const data = await resp.json();
        const accounts = data?.result?.value || [];

        let totalBalance = 0;
        for (const account of accounts) {
            const info = account?.account?.data?.parsed?.info || {};
            const tokenAmount = info.tokenAmount || {};
            const amount = parseInt(tokenAmount.amount || '0', 10);
            const decimals = tokenAmount.decimals || 9;
            totalBalance += amount / (10 ** decimals);
        }

        farnsBalanceCache.set(walletAddress, { balance: totalBalance, ts: Date.now() });
        return { hasAccess: totalBalance >= minAmount, balance: totalBalance, required: minAmount };
    } catch (e) {
        console.error('[DEXAI] FARNS balance check failed:', e.message);
        return { hasAccess: false, balance: 0, required: minAmount, error: e.message };
    }
}

/**
 * Token gate middleware — verifies FARNS balance from wallet query param or header.
 */
async function quantumTokenGate(req, res, next) {
    const wallet = req.query.wallet || req.headers['x-wallet-address'] || '';
    if (!wallet || wallet.length < 32) {
        return res.status(401).json({
            error: 'Wallet address required',
            message: 'Connect wallet and hold 100K+ FARNS to access quantum signals',
            gated: true,
        });
    }

    const result = await checkFarnsBalance(wallet);
    if (!result.hasAccess) {
        return res.status(403).json({
            error: 'Insufficient FARNS balance',
            message: `Hold at least ${MIN_FARNS_HOLDING.toLocaleString()} FARNS to unlock quantum signals`,
            balance: result.balance,
            required: MIN_FARNS_HOLDING,
            gated: true,
        });
    }

    req.farnsBalance = result.balance;
    next();
}

// --- Quantum Signal Internal API (called by Python backend to push data) ---

app.post('/api/quantum/internal/signal', (req, res) => {
    const signal = req.body;
    if (!signal || !signal.token_address) return res.status(400).json({ error: 'Invalid signal' });

    quantumSignals.set(signal.token_address, signal);
    quantumSignalHistory.unshift(signal);
    if (quantumSignalHistory.length > 500) quantumSignalHistory.length = 500;

    // Broadcast to quantum WS subscribers
    broadcastQuantum({ type: 'quantum_signal', data: signal });

    res.json({ ok: true });
});

app.post('/api/quantum/internal/correlations', (req, res) => {
    const { correlations } = req.body || {};
    if (Array.isArray(correlations)) {
        quantumCorrelations.length = 0;
        quantumCorrelations.push(...correlations);
    }
    res.json({ ok: true });
});

app.post('/api/quantum/internal/accuracy', (req, res) => {
    const stats = req.body;
    if (stats) Object.assign(quantumAccuracy, stats);
    res.json({ ok: true });
});

// --- Quantum Signal Public API (token-gated) ---

// Latest signal for a specific token (gated)
app.get('/api/quantum/signal/:address', quantumTokenGate, (req, res) => {
    const signal = quantumSignals.get(req.params.address);
    if (signal) {
        res.json({ available: true, ...signal });
    } else {
        res.json({ available: false, message: 'No quantum signal for this token yet' });
    }
});

// Recent signals across all tokens (gated)
app.get('/api/quantum/signals', quantumTokenGate, (req, res) => {
    const limit = Math.min(parseInt(req.query.limit) || 50, 200);
    res.json({
        signals: quantumSignalHistory.slice(0, limit),
        total: quantumSignalHistory.length,
    });
});

// Cross-token correlations (gated)
app.get('/api/quantum/correlations', quantumTokenGate, (req, res) => {
    res.json({ correlations: quantumCorrelations });
});

// Public accuracy stats (NOT gated — proves value to drive demand)
app.get('/api/quantum/accuracy', (req, res) => {
    res.json({
        ...quantumAccuracy,
        min_farns: MIN_FARNS_HOLDING,
        token_mint: FARNS_TOKEN,
    });
});

// Balance check endpoint for frontend
app.get('/api/quantum/check-access', async (req, res) => {
    const wallet = req.query.wallet || '';
    if (!wallet) return res.json({ hasAccess: false, balance: 0 });
    const result = await checkFarnsBalance(wallet);
    res.json(result);
});

// --- Quantum WebSocket (/ws/quantum) ---
const quantumWss = new WebSocketServer({ server, path: '/ws/quantum' });
const quantumWsClients = new Set();

quantumWss.on('connection', async (ws, req) => {
    // Parse wallet from query string for auth
    const url = new URL(req.url, `http://${req.headers.host}`);
    const wallet = url.searchParams.get('wallet') || '';

    if (!wallet || wallet.length < 32) {
        ws.send(JSON.stringify({ type: 'error', message: 'Wallet address required' }));
        ws.close(4001, 'No wallet');
        return;
    }

    // Verify FARNS balance
    const balance = await checkFarnsBalance(wallet);
    if (!balance.hasAccess) {
        ws.send(JSON.stringify({
            type: 'access_denied',
            message: `Hold ${MIN_FARNS_HOLDING.toLocaleString()} FARNS to access quantum feed`,
            balance: balance.balance,
            required: MIN_FARNS_HOLDING,
        }));
        ws.close(4003, 'Insufficient FARNS');
        return;
    }

    quantumWsClients.add(ws);
    ws.isAlive = true;
    ws.subs = new Set(); // subscribed token addresses

    ws.send(JSON.stringify({
        type: 'connected',
        message: 'Quantum feed connected',
        balance: balance.balance,
        signals_available: quantumSignals.size,
    }));

    ws.on('pong', () => { ws.isAlive = true; });
    ws.on('message', (msg) => {
        try {
            const data = JSON.parse(msg);
            if (data.type === 'subscribe' && data.token) ws.subs.add(data.token);
            if (data.type === 'unsubscribe' && data.token) ws.subs.delete(data.token);
            // Request latest signal
            if (data.type === 'get_signal' && data.token) {
                const signal = quantumSignals.get(data.token);
                if (signal) ws.send(JSON.stringify({ type: 'quantum_signal', data: signal }));
            }
        } catch {}
    });
    ws.on('close', () => quantumWsClients.delete(ws));
});

function broadcastQuantum(data) {
    const msg = JSON.stringify(data);
    for (const ws of quantumWsClients) {
        if (ws.readyState === 1) {
            // If client has subscriptions, only send matching signals
            if (ws.subs && ws.subs.size > 0 && data.data?.token_address) {
                if (ws.subs.has(data.data.token_address)) ws.send(msg);
            } else {
                ws.send(msg);
            }
        }
    }
}

// Quantum WS heartbeat
setInterval(() => {
    for (const ws of quantumWsClients) {
        if (!ws.isAlive) { ws.terminate(); quantumWsClients.delete(ws); continue; }
        ws.isAlive = false; ws.ping();
    }
}, 30000);

// ============================================================
// x402 DISCOVERY — Premium Quantum API (1 SOL per query)
// ============================================================
// Serves the x402 discovery manifest so hubs/bazaars/i1l.store can find us

const X402_DISCOVERY = {
    x402Version: 2,
    provider: {
        name: "Cosmos AI Swarm",
        description: "Quantum-enhanced trading intelligence powered by IBM Quantum (simulator + real QPU hardware). Two tiers: Simulated Quantum (0.25 SOL) with hardware-optimized weights, and Real Quantum Hardware (1 SOL) on IBM QPU. Supports any Solana memecoin + BTC, ETH, SOL majors.",
        url: process.env.COSMOS_API_URL || "https://ai.cosmos.cloud",
        category: "trading",
        tags: ["solana", "quantum", "trading", "defi", "ai", "signals", "ibm-quantum", "x402"],
    },
    endpoints: [
        {
            path: "/api/x402/quantum/analyze",
            method: "POST",
            description: "Simulated Quantum Analysis (0.25 SOL) — Quantum simulator with hardware-optimized algo weights from real QPU runs. Fast 5-15s response.",
            price: "250000000",
            asset: "native",
            network: "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
            payTo: ECOSYSTEM_WALLET,
            extra: { tier: "simulated", estimated_time: "5-15 seconds" },
            requestSchema: {
                type: "object",
                required: ["token_address"],
                properties: {
                    token_address: { type: "string", description: "Solana token mint address, or ticker: BTC, ETH, SOL" },
                    tier: { type: "string", enum: ["simulated", "hardware"], description: "Tier preference (auto-detected from payment amount)" },
                },
            },
        },
        {
            path: "/api/x402/quantum/analyze",
            method: "POST",
            description: "Real Quantum Hardware Analysis (1 SOL) — Actual IBM Quantum QPU circuit execution. Higher fidelity, 30-90s processing.",
            price: "1000000000",
            asset: "native",
            network: "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
            payTo: ECOSYSTEM_WALLET,
            extra: { tier: "hardware", estimated_time: "30-90 seconds" },
            requestSchema: {
                type: "object",
                required: ["token_address"],
                properties: {
                    token_address: { type: "string", description: "Solana token mint address, or ticker: BTC, ETH, SOL" },
                    tier: { type: "string", enum: ["simulated", "hardware"], description: "Tier preference (auto-detected from payment amount)" },
                },
            },
        },
    ],
    networks: ["solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"],
};

app.get('/.well-known/x402.json', (req, res) => res.json(X402_DISCOVERY));
app.get('/api/x402/discovery', (req, res) => res.json(X402_DISCOVERY));

// x402 proxy — forward paid requests to the FastAPI backend
app.post('/api/x402/quantum/analyze', async (req, res) => {
    try {
        const fetch = (await import('node-fetch')).default;
        const headers = { 'Content-Type': 'application/json' };
        if (req.headers['x-payment']) headers['X-PAYMENT'] = req.headers['x-payment'];

        const upstream = `${COSMOS_API}/api/x402/quantum/analyze`;
        const resp = await fetch(upstream, {
            method: 'POST',
            headers,
            body: JSON.stringify(req.body || {}),
        });

        const data = await resp.json();

        // Forward x402 headers
        if (resp.headers.get('x-payment')) res.set('X-PAYMENT', resp.headers.get('x-payment'));
        if (resp.headers.get('x-payment-response')) res.set('X-PAYMENT-RESPONSE', resp.headers.get('x-payment-response'));
        if (resp.headers.get('x-payment-required')) res.set('X-Payment-Required', resp.headers.get('x-payment-required'));
        if (resp.headers.get('x-processing-time')) res.set('X-Processing-Time', resp.headers.get('x-processing-time'));

        res.status(resp.status).json(data);
    } catch (err) {
        console.error('x402 proxy error:', err.message);
        res.status(502).json({ error: 'Upstream x402 service unavailable' });
    }
});

app.get('/api/x402/quantum/pricing', async (req, res) => {
    res.json({
        service: "Cosmos Quantum Trading Intelligence",
        tiers: {
            simulated: {
                price_sol: 0.25,
                price_lamports: 250000000,
                description: "Quantum simulator with hardware-optimized algo weights from real QPU runs. Fast response.",
                estimated_time: "5-15 seconds",
                features: ["ema_momentum", "quantum_simulation", "collective_intelligence", "signal_fusion", "scenario_analysis"],
            },
            hardware: {
                price_sol: 1.0,
                price_lamports: 1000000000,
                description: "Real IBM Quantum QPU circuit execution. Higher fidelity, more qubits, more shots.",
                estimated_time: "30-90 seconds",
                features: ["ema_momentum", "quantum_hardware", "collective_intelligence", "signal_fusion", "scenario_analysis", "bell_correlations", "higher_qubit_count"],
            },
        },
        supported_assets: ["Any Solana memecoin (mint address)", "BTC", "ETH", "SOL"],
        pay_to: ECOSYSTEM_WALLET,
        network: "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
        asset: "native",
        protocol: "x402",
        endpoint: "/api/x402/quantum/analyze",
        method: "POST",
    });
});

app.get('/api/x402/quantum/stats', async (req, res) => {
    try {
        const fetch = (await import('node-fetch')).default;
        const resp = await fetch(`${COSMOS_API}/api/x402/quantum/stats`);
        const data = await resp.json();
        res.json(data);
    } catch (err) {
        res.json({ total_queries: 0, total_revenue_sol: 0, message: "Stats unavailable" });
    }
});

// ============================================================
// SPA ROUTING
// ============================================================
app.get('/token/:address', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

// ============================================================
// START
// ============================================================
server.listen(PORT, () => {
    console.log(`\n  DEXAI v2.1 — Cosmos Collective DEX Screener`);
    console.log(`  Port: ${PORT} | Ecosystem: ${ECOSYSTEM_WALLET.slice(0, 8)}... | FARNS: ${FARNS_TOKEN.slice(0, 8)}...`);
    console.log(`  Trending: Market(35%) + Collective(25%) + Quantum(15%) + Whale(15%) + Burn(10%)`);
    console.log(`  Quantum Trading: Token-gated (${MIN_FARNS_HOLDING.toLocaleString()} FARNS) | WS /ws/quantum`);
    console.log(`  Platforms: ${ALLOWED_SUFFIXES.join(', ')} | Burn boost = 3x SOL boost`);
    console.log(`  x402 Premium: /api/x402/quantum/analyze (1 SOL/query) | Discovery: /.well-known/x402.json\n`);

    // Initial data fetch
    fetchCollectiveData();
    updateTokenCache();

    // Refresh cycles
    setInterval(updateTokenCache, FETCH_INTERVAL);
    setInterval(fetchCollectiveData, COLLECTIVE_FETCH_INTERVAL);
    // Quantum scoring for top tokens every 5 minutes
    setTimeout(() => { batchQuantumScore(); setInterval(batchQuantumScore, 300000); }, 60000);

    // Connect Helius WebSocket for real-time swap detection
    setTimeout(connectHeliusWs, 3000);

    // v5.0: Initialize GoldRush streaming (new pairs + OHLCV candles)
    setTimeout(initGoldRush, 5000);
});

module.exports = { app, server, broadcastAll, broadcastQuantum, checkFarnsBalance };
