"""
Entity Extractor for Natural Language Commands.

Extracts structured entities (files, URLs, amounts, etc.) from text.
"""

import re
import logging
from typing import   Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Extracted entities from text."""
    urls: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    amounts: list[dict] = field(default_factory=list)  # {value, currency}
    tokens: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)  # @mentions
    dates: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    raw_text: str = ""


class EntityExtractor:
    """
    Extract structured entities from natural language text.

    Handles:
    - URLs and file paths
    - Monetary amounts
    - Crypto tokens and chains
    - Email addresses
    - Social mentions
    - Dates and numbers
    """

    # Patterns
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    )

    FILE_PATH_PATTERN = re.compile(
        r'(?:[A-Za-z]:)?[\\/][\w\-. \\\/]+\.\w+|'
        r'\.\/[\w\-. \/]+|'
        r'[\w\-]+\.(?:py|js|ts|json|yaml|yml|md|txt|csv|html|css)'
    )

    AMOUNT_PATTERN = re.compile(
        r'\$(\d+(?:,\d{3})*(?:\.\d{1,2})?)|'
        r'(\d+(?:,\d{3})*(?:\.\d{1,8})?)\s*(USD|USDC|ETH|BTC|SOL|BNKR|tokens?)?',
        re.IGNORECASE
    )

    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )

    MENTION_PATTERN = re.compile(r'@(\w+)')

    # Known tokens and chains
    KNOWN_TOKENS = {
        "eth", "ethereum", "btc", "bitcoin", "sol", "solana",
        "usdc", "usdt", "bnkr", "matic", "link", "uni",
        "aave", "comp", "mkr", "snx", "yfi", "crv",
        "doge", "shib", "pepe", "bonk", "wif"
    }

    KNOWN_CHAINS = {
        "base", "ethereum", "mainnet", "solana", "polygon",
        "arbitrum", "optimism", "avalanche", "bsc", "fantom"
    }

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from text.

        Args:
            text: Input text to extract from

        Returns:
            ExtractedEntities with all found entities
        """
        entities = ExtractedEntities(raw_text=text)

        # Extract URLs
        entities.urls = self._extract_urls(text)

        # Extract file paths
        entities.file_paths = self._extract_file_paths(text)

        # Extract amounts
        entities.amounts = self._extract_amounts(text)

        # Extract emails
        entities.emails = self.EMAIL_PATTERN.findall(text)

        # Extract mentions
        entities.mentions = self.MENTION_PATTERN.findall(text)

        # Extract tokens and chains
        entities.tokens, entities.chains = self._extract_crypto(text)

        # Extract raw numbers
        entities.numbers = self._extract_numbers(text)

        return entities

    def _extract_urls(self, text: str) -> list[str]:
        """Extract and validate URLs."""
        urls = []
        for match in self.URL_PATTERN.finditer(text):
            url = match.group()
            # Add protocol if missing
            if url.startswith('www.'):
                url = 'https://' + url
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    urls.append(url)
            except Exception:
                pass
        return urls

    def _extract_file_paths(self, text: str) -> list[str]:
        """Extract file paths."""
        paths = []
        for match in self.FILE_PATH_PATTERN.finditer(text):
            path = match.group()
            # Skip URLs that might match
            if not path.startswith(('http', 'www')):
                paths.append(path)
        return paths

    def _extract_amounts(self, text: str) -> list[dict]:
        """Extract monetary amounts with currency."""
        amounts = []

        for match in self.AMOUNT_PATTERN.finditer(text):
            groups = match.groups()

            if groups[0]:  # $X format
                amounts.append({
                    "value": float(groups[0].replace(',', '')),
                    "currency": "USD",
                    "raw": match.group()
                })
            elif groups[1]:  # X CURRENCY format
                value = float(groups[1].replace(',', ''))
                currency = (groups[2] or "").upper() or None
                amounts.append({
                    "value": value,
                    "currency": currency,
                    "raw": match.group()
                })

        return amounts

    def _extract_crypto(self, text: str) -> tuple:
        """Extract crypto tokens and chain names."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        tokens = [t for t in self.KNOWN_TOKENS if t in words]
        chains = [c for c in self.KNOWN_CHAINS if c in words]

        return tokens, chains

    def _extract_numbers(self, text: str) -> list[float]:
        """Extract standalone numbers."""
        numbers = []
        # Find numbers not part of amounts
        for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            try:
                num = float(match.group(1))
                numbers.append(num)
            except ValueError:
                pass
        return numbers

    def get_primary_token(self, entities: ExtractedEntities) -> Optional[str]:
        """Get the most likely primary token from entities."""
        if entities.tokens:
            return entities.tokens[0].upper()
        return None

    def get_primary_chain(self, entities: ExtractedEntities) -> Optional[str]:
        """Get the most likely primary chain from entities."""
        if entities.chains:
            return entities.chains[0].lower()
        return None

    def get_primary_amount(self, entities: ExtractedEntities) -> Optional[dict]:
        """Get the primary amount from entities."""
        if entities.amounts:
            return entities.amounts[0]
        return None
