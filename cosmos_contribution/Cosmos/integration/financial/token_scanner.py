"""
Token Scanner for Swarm Chat.

Detects contract addresses dropped in chat and provides DEEP token analysis:
- DexScreener market data
- Bankr trading info
- X/Twitter social search via Grok
- Collective analysis for risk assessment

"Drop any CA in chat - Cosmos scans it for you."
"""

import re
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)

# Regex patterns for contract addresses
SOLANA_CA_PATTERN = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')
EVM_CA_PATTERN = re.compile(r'\b0x[a-fA-F0-9]{40}\b')

# Common non-token addresses to ignore
IGNORE_ADDRESSES = {
    'So11111111111111111111111111111111111111112',  # Wrapped SOL
    '11111111111111111111111111111111',  # System program
}


class TokenScanner:
    """
    Scans chat messages for contract addresses and provides DEEP token analysis.

    Features:
    - Market data from DexScreener
    - Trading info from Bankr
    - Social sentiment from X via Grok search
    - Risk assessment and recommendations
    """

    def __init__(self):
        self._dex_screener = None
        self._bankr_client = None
        self._grok_provider = None
        self._last_scan = {}  # Track recently scanned CAs to avoid spam

    @property
    def dex_screener(self):
        if self._dex_screener is None:
            try:
                from Cosmos.integration.financial.dexscreener import dex_screener
                self._dex_screener = dex_screener
            except ImportError:
                logger.warning("DexScreener not available")
        return self._dex_screener

    @property
    def bankr_client(self):
        if self._bankr_client is None:
            try:
                from Cosmos.integration.bankr import get_bankr_client
                self._bankr_client = get_bankr_client()
            except ImportError:
                pass
        return self._bankr_client

    @property
    def grok_provider(self):
        """Get Grok for X/Twitter search."""
        if self._grok_provider is None:
            try:
                from Cosmos.integration.external.grok import get_grok_provider
                self._grok_provider = get_grok_provider()
            except ImportError:
                logger.debug("Grok provider not available for X search")
        return self._grok_provider

    def detect_ca(self, message: str) -> Optional[Dict[str, str]]:
        """
        Detect a contract address in a message.

        Returns:
            Dict with 'address' and 'chain' if found, None otherwise
        """
        # Check for EVM address first (more specific pattern)
        evm_match = EVM_CA_PATTERN.search(message)
        if evm_match:
            addr = evm_match.group()
            return {"address": addr, "chain": "ethereum"}  # Could be any EVM chain

        # Check for Solana address
        sol_match = SOLANA_CA_PATTERN.search(message)
        if sol_match:
            addr = sol_match.group()
            # Skip known non-token addresses
            if addr in IGNORE_ADDRESSES:
                return None
            # Basic validation - Solana addresses are base58
            if len(addr) >= 32 and len(addr) <= 44:
                return {"address": addr, "chain": "solana"}

        return None

    def _should_scan(self, address: str) -> bool:
        """Check if we should scan this address (rate limiting)."""
        now = datetime.now().timestamp()
        last = self._last_scan.get(address, 0)
        # Don't scan same address more than once per 5 minutes
        if now - last < 300:
            return False
        self._last_scan[address] = now
        return True

    async def scan_token(self, address: str, chain: str = None) -> Optional[Dict[str, Any]]:
        """
        Scan a token and return detailed information.

        Args:
            address: The contract address
            chain: Optional chain hint

        Returns:
            Token information dict or None
        """
        if not self._should_scan(address):
            logger.debug(f"Skipping recently scanned address: {address}")
            return None

        try:
            # Try DexScreener first
            if self.dex_screener:
                pairs = await self.dex_screener.get_token_pairs(chain or "solana", address)
                if pairs:
                    return self._format_dex_data(pairs, address)

            # Fallback to Bankr if available
            if self.bankr_client:
                try:
                    result = await self.bankr_client.execute(
                        f"Get token info for {address}"
                    )
                    if result:
                        return self._format_bankr_data(result, address)
                except Exception as e:
                    logger.debug(f"Bankr lookup failed: {e}")

            return None

        except Exception as e:
            logger.error(f"Token scan failed for {address}: {e}")
            return None

    def _format_dex_data(self, pairs: List[Dict], address: str) -> Dict[str, Any]:
        """Format DexScreener pair data into token info."""
        if not pairs:
            return None

        # Get the most liquid pair
        pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))

        base_token = pair.get('baseToken', {})
        quote_token = pair.get('quoteToken', {})

        # Determine which is our token
        if base_token.get('address', '').lower() == address.lower():
            token = base_token
        else:
            token = quote_token

        price_usd = pair.get('priceUsd', '0')
        price_native = pair.get('priceNative', '0')

        # Price changes
        price_change = pair.get('priceChange', {})
        change_5m = price_change.get('m5', 0)
        change_1h = price_change.get('h1', 0)
        change_6h = price_change.get('h6', 0)
        change_24h = price_change.get('h24', 0)

        # Volume
        volume = pair.get('volume', {})
        vol_24h = volume.get('h24', 0)

        # Liquidity
        liquidity = pair.get('liquidity', {})
        liq_usd = liquidity.get('usd', 0)

        # Market cap (FDV)
        fdv = pair.get('fdv', 0)
        market_cap = pair.get('marketCap', fdv)

        # Transactions
        txns = pair.get('txns', {})
        txns_24h = txns.get('h24', {})
        buys_24h = txns_24h.get('buys', 0)
        sells_24h = txns_24h.get('sells', 0)

        return {
            "address": address,
            "name": token.get('name', 'Unknown'),
            "symbol": token.get('symbol', '???'),
            "chain": pair.get('chainId', 'unknown'),
            "dex": pair.get('dexId', 'unknown'),
            "pair_address": pair.get('pairAddress', ''),
            "price_usd": float(price_usd) if price_usd else 0,
            "price_native": float(price_native) if price_native else 0,
            "market_cap": float(market_cap) if market_cap else 0,
            "fdv": float(fdv) if fdv else 0,
            "liquidity_usd": float(liq_usd) if liq_usd else 0,
            "volume_24h": float(vol_24h) if vol_24h else 0,
            "change_5m": float(change_5m) if change_5m else 0,
            "change_1h": float(change_1h) if change_1h else 0,
            "change_6h": float(change_6h) if change_6h else 0,
            "change_24h": float(change_24h) if change_24h else 0,
            "buys_24h": buys_24h,
            "sells_24h": sells_24h,
            "url": pair.get('url', f"https://dexscreener.com/{pair.get('chainId', 'solana')}/{address}"),
            "info": pair.get('info', {}),
            "created_at": pair.get('pairCreatedAt'),
        }

    def _format_bankr_data(self, result: Dict, address: str) -> Dict[str, Any]:
        """Format Bankr API response into token info."""
        return {
            "address": address,
            "name": result.get('name', 'Unknown'),
            "symbol": result.get('symbol', '???'),
            "chain": result.get('chain', 'unknown'),
            "price_usd": result.get('price', 0),
            "market_cap": result.get('marketCap', 0),
            "raw": result,
        }

    async def search_x_mentions(self, symbol: str, address: str) -> Optional[Dict[str, Any]]:
        """
        Search X/Twitter for mentions of this token via Grok.

        Returns social sentiment and key mentions.
        """
        if not self.grok_provider:
            return None

        try:
            # Search for token mentions on X
            search_query = f"${symbol} OR {address[:12]} crypto token"
            result = await self.grok_provider.search(search_query)

            if not result:
                return None

            # Parse the sentiment from Grok's response
            sentiment = "neutral"
            if any(word in result.lower() for word in ["bullish", "moon", "pump", "buy", "gem"]):
                sentiment = "bullish"
            elif any(word in result.lower() for word in ["bearish", "dump", "sell", "rug", "scam"]):
                sentiment = "bearish"

            return {
                "sentiment": sentiment,
                "mentions": result[:500] if result else "",
                "source": "x_via_grok"
            }

        except Exception as e:
            logger.debug(f"X search failed for {symbol}: {e}")
            return None

    async def get_deep_analysis(self, token_info: Dict[str, Any]) -> str:
        """
        Generate a deep analysis of the token using collective deliberation.
        """
        try:
            from Cosmos.core.collective.session_manager import get_session_manager

            symbol = token_info.get('symbol', 'Unknown')
            address = token_info.get('address', '')
            price = token_info.get('price_usd', 0)
            mcap = token_info.get('market_cap', 0)
            liq = token_info.get('liquidity_usd', 0)
            vol = token_info.get('volume_24h', 0)
            c24h = token_info.get('change_24h', 0)

            analysis_prompt = f"""Analyze this token for a user who just dropped the CA in chat:

Token: ${symbol}
Address: {address}
Price: ${price}
Market Cap: ${mcap}
Liquidity: ${liq}
24h Volume: ${vol}
24h Change: {c24h}%

Provide a thorough analysis covering:
1. Is this token tradeable? (enough liquidity?)
2. Any red flags? (low liq, high volatility, suspicious patterns?)
3. What type of token is this likely? (meme, utility, scam?)
4. Risk assessment and recommendation

Be thorough and helpful. Depth of analysis is valuable for trading decisions.
"""
            session_manager = get_session_manager()
            result = await session_manager.deliberate_in_session(
                session_type="quick_response",
                prompt=analysis_prompt,
                context={"token": symbol, "address": address}
            )

            return result.final_response

        except Exception as e:
            logger.debug(f"Deep analysis failed: {e}")
            return None

    async def format_response_async(self, token_info: Dict[str, Any]) -> str:
        """
        Format token info into a chat-friendly response with deep analysis.
        """
        if not token_info:
            return None

        name = token_info.get('name', 'Unknown')
        symbol = token_info.get('symbol', '???')
        chain = token_info.get('chain', 'unknown').upper()
        address = token_info.get('address', '')
        price = token_info.get('price_usd', 0)
        mcap = token_info.get('market_cap', 0)
        fdv = token_info.get('fdv', 0)
        liq = token_info.get('liquidity_usd', 0)
        vol = token_info.get('volume_24h', 0)

        # Price changes with emoji indicators
        def change_emoji(val):
            if val > 10:
                return "🚀"
            elif val > 0:
                return "📈"
            elif val < -10:
                return "💀"
            elif val < 0:
                return "📉"
            return "➡️"

        c5m = token_info.get('change_5m', 0)
        c1h = token_info.get('change_1h', 0)
        c6h = token_info.get('change_6h', 0)
        c24h = token_info.get('change_24h', 0)

        buys = token_info.get('buys_24h', 0)
        sells = token_info.get('sells_24h', 0)
        buy_sell_ratio = buys / sells if sells > 0 else float('inf') if buys > 0 else 0

        # Format large numbers
        def fmt_num(n):
            if n >= 1_000_000_000:
                return f"${n/1_000_000_000:.2f}B"
            elif n >= 1_000_000:
                return f"${n/1_000_000:.2f}M"
            elif n >= 1_000:
                return f"${n/1_000:.2f}K"
            else:
                return f"${n:.2f}"

        def fmt_price(p):
            if p < 0.00001:
                return f"${p:.10f}"
            elif p < 0.01:
                return f"${p:.6f}"
            elif p < 1:
                return f"${p:.4f}"
            else:
                return f"${p:.2f}"

        # Risk assessment
        risk_factors = []
        if liq < 10000:
            risk_factors.append("LOW LIQUIDITY")
        if mcap < 50000:
            risk_factors.append("MICRO CAP")
        if buy_sell_ratio < 0.5:
            risk_factors.append("MORE SELLS THAN BUYS")
        if abs(c24h) > 50:
            risk_factors.append("HIGH VOLATILITY")

        risk_warning = ""
        if risk_factors:
            risk_warning = f"\n⚠️ **Risks:** {', '.join(risk_factors)}"

        dex = token_info.get('dex', '').upper()
        url = token_info.get('url', '')

        # Get X mentions (non-blocking)
        x_section = ""
        try:
            x_data = await self.search_x_mentions(symbol, address)
            if x_data:
                sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(x_data['sentiment'], "⚪")
                x_section = f"\n\n**X/Twitter Sentiment:** {sentiment_emoji} {x_data['sentiment'].upper()}"
                if x_data.get('mentions'):
                    x_section += f"\n_{x_data['mentions'][:200]}..._"
        except Exception as e:
            logger.debug(f"X search failed: {e}")

        # Get deep analysis from collective (non-blocking)
        analysis_section = ""
        try:
            deep_analysis = await self.get_deep_analysis(token_info)
            if deep_analysis:
                analysis_section = f"\n\n🧠 **Collective Analysis:**\n{deep_analysis[:400]}"
        except Exception as e:
            logger.debug(f"Deep analysis failed: {e}")

        response = f"""🔍 **TOKEN SCAN: {symbol}** ({chain})

**{name}** (${symbol})

💰 **Price:** {fmt_price(price)}
📊 **Market Cap:** {fmt_num(mcap)}
💎 **FDV:** {fmt_num(fdv)}
💧 **Liquidity:** {fmt_num(liq)}
📈 **24h Volume:** {fmt_num(vol)}

**Price Action:**
{change_emoji(c5m)} 5m: {c5m:+.1f}% | {change_emoji(c1h)} 1h: {c1h:+.1f}%
{change_emoji(c6h)} 6h: {c6h:+.1f}% | {change_emoji(c24h)} 24h: {c24h:+.1f}%

**24h Activity:** {buys} buys / {sells} sells (ratio: {buy_sell_ratio:.2f})
**DEX:** {dex}{risk_warning}{x_section}{analysis_section}

🔗 {url}"""

        return response

    def format_response(self, token_info: Dict[str, Any]) -> str:
        """
        Format token info into a chat-friendly response (sync version).
        """
        if not token_info:
            return None

        name = token_info.get('name', 'Unknown')
        symbol = token_info.get('symbol', '???')
        chain = token_info.get('chain', 'unknown').upper()
        price = token_info.get('price_usd', 0)
        mcap = token_info.get('market_cap', 0)
        fdv = token_info.get('fdv', 0)
        liq = token_info.get('liquidity_usd', 0)
        vol = token_info.get('volume_24h', 0)

        # Price changes with emoji indicators
        def change_emoji(val):
            if val > 10:
                return "🚀"
            elif val > 0:
                return "📈"
            elif val < -10:
                return "💀"
            elif val < 0:
                return "📉"
            return "➡️"

        c5m = token_info.get('change_5m', 0)
        c1h = token_info.get('change_1h', 0)
        c6h = token_info.get('change_6h', 0)
        c24h = token_info.get('change_24h', 0)

        buys = token_info.get('buys_24h', 0)
        sells = token_info.get('sells_24h', 0)
        buy_sell_ratio = buys / sells if sells > 0 else float('inf') if buys > 0 else 0

        # Format large numbers
        def fmt_num(n):
            if n >= 1_000_000_000:
                return f"${n/1_000_000_000:.2f}B"
            elif n >= 1_000_000:
                return f"${n/1_000_000:.2f}M"
            elif n >= 1_000:
                return f"${n/1_000:.2f}K"
            else:
                return f"${n:.2f}"

        def fmt_price(p):
            if p < 0.00001:
                return f"${p:.10f}"
            elif p < 0.01:
                return f"${p:.6f}"
            elif p < 1:
                return f"${p:.4f}"
            else:
                return f"${p:.2f}"

        # Risk assessment
        risk_factors = []
        if liq < 10000:
            risk_factors.append("LOW LIQUIDITY")
        if mcap < 50000:
            risk_factors.append("MICRO CAP")
        if buy_sell_ratio < 0.5:
            risk_factors.append("MORE SELLS THAN BUYS")
        if abs(c24h) > 50:
            risk_factors.append("HIGH VOLATILITY")

        risk_warning = ""
        if risk_factors:
            risk_warning = f"\n⚠️ **Risks:** {', '.join(risk_factors)}"

        dex = token_info.get('dex', '').upper()
        url = token_info.get('url', '')

        response = f"""🔍 **TOKEN SCAN: {symbol}** ({chain})

**{name}** (${symbol})

💰 **Price:** {fmt_price(price)}
📊 **Market Cap:** {fmt_num(mcap)}
💎 **FDV:** {fmt_num(fdv)}
💧 **Liquidity:** {fmt_num(liq)}
📈 **24h Volume:** {fmt_num(vol)}

**Price Action:**
{change_emoji(c5m)} 5m: {c5m:+.1f}% | {change_emoji(c1h)} 1h: {c1h:+.1f}%
{change_emoji(c6h)} 6h: {c6h:+.1f}% | {change_emoji(c24h)} 24h: {c24h:+.1f}%

**24h Activity:** {buys} buys / {sells} sells (ratio: {buy_sell_ratio:.2f})
**DEX:** {dex}{risk_warning}

🔗 {url}"""

        return response

    def _detect_token_type(self, token_info: Dict[str, Any]) -> str:
        """
        Detect the type of token based on available data.

        Returns: meme, utility, defi, nft, unknown
        """
        name = token_info.get('name', '').lower()
        symbol = token_info.get('symbol', '').lower()
        mcap = token_info.get('market_cap', 0)
        liq = token_info.get('liquidity_usd', 0)

        # Check for meme indicators
        meme_keywords = ['doge', 'shib', 'pepe', 'wojak', 'moon', 'elon', 'inu', 'floki',
                         'baby', 'safe', 'chad', 'meme', 'frog', 'cat', 'dog', 'bonk']
        if any(kw in name or kw in symbol for kw in meme_keywords):
            return "meme"

        # Check for DeFi indicators
        defi_keywords = ['swap', 'finance', 'yield', 'stake', 'lend', 'dao', 'governance',
                         'vault', 'pool', 'liquidity', 'protocol']
        if any(kw in name for kw in defi_keywords):
            return "defi"

        # Check for NFT/gaming indicators
        nft_keywords = ['nft', 'meta', 'verse', 'game', 'play', 'earn', 'land', 'world']
        if any(kw in name for kw in nft_keywords):
            return "nft/gaming"

        # Check for AI/tech indicators
        ai_keywords = ['ai', 'gpt', 'neural', 'learn', 'agent', 'bot', 'smart', 'quantum']
        if any(kw in name or kw in symbol for kw in ai_keywords):
            return "ai/tech"

        # Based on market cap and liquidity patterns
        if mcap > 10_000_000 and liq > 100_000:
            return "established"
        elif mcap < 100_000:
            return "micro-cap"

        return "unknown"

    async def process_message(self, content: str) -> Optional[str]:
        """
        Process a chat message and return a DEEP token analysis if CA detected.

        Features:
        - DexScreener market data
        - Token type detection
        - X/Twitter sentiment search via Grok
        - Collective analysis for recommendations

        Args:
            content: The chat message content

        Returns:
            Formatted response string or None
        """
        # Detect CA in message
        ca_info = self.detect_ca(content)
        if not ca_info:
            return None

        address = ca_info['address']
        chain = ca_info['chain']

        logger.info(f"Detected CA in chat: {address} ({chain})")

        # Scan the token
        token_info = await self.scan_token(address, chain)
        if not token_info:
            return f"🔍 Could not find token info for `{address[:8]}...{address[-6:]}`"

        # Detect token type
        token_type = self._detect_token_type(token_info)
        token_info['token_type'] = token_type

        # Format with deep analysis (async version with X search and collective)
        response = await self.format_response_async(token_info)

        # Add token type badge
        type_badges = {
            "meme": "🐸 MEME TOKEN",
            "defi": "🔄 DeFi TOKEN",
            "nft/gaming": "🎮 NFT/GAMING",
            "ai/tech": "🤖 AI/TECH",
            "established": "✅ ESTABLISHED",
            "micro-cap": "🔬 MICRO-CAP",
            "unknown": "❓ UNKNOWN TYPE"
        }
        badge = type_badges.get(token_type, "❓ UNKNOWN TYPE")

        # Insert badge after the header
        if response:
            lines = response.split('\n')
            if len(lines) > 0:
                lines[0] = f"{lines[0]} | {badge}"
                response = '\n'.join(lines)

        return response


# Global instance
token_scanner = TokenScanner()


async def scan_message_for_token(content: str) -> Optional[str]:
    """
    Convenience function to scan a message for tokens.

    Args:
        content: Chat message content

    Returns:
        Token analysis response or None
    """
    return await token_scanner.process_message(content)
