"""Parser registry — site-specific + generic fallback.

Provides :func:`get_parser` (URL → parser) and :func:`list_supported_sites`.
"""

import logging
from typing import List, Optional

from .base import BaseSiteParser, ProductData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARSERS: List[BaseSiteParser] = []
_PARSERS_LOADED: bool = False


def _ensure_loaded() -> None:
    """Lazy-load all parsers on first access."""
    global _PARSERS_LOADED
    if _PARSERS_LOADED:
        return
    _PARSERS_LOADED = True

    from .amazon_global import AmazonGlobalParser
    from .ebay_global import EbayGlobalParser
    from .bestbuy import BestBuyParser
    from .newegg import NeweggParser
    from .walmart import WalmartParser
    from .price_comparison import IdealoParser, PriceSpyParser, CamelParser
    from .generic import GenericParser

    # Order matters: specific parsers first, GenericParser last
    _PARSERS.extend([
        AmazonGlobalParser(),
        EbayGlobalParser(),
        BestBuyParser(),
        NeweggParser(),
        WalmartParser(),
        IdealoParser(),
        PriceSpyParser(),
        CamelParser(),
        GenericParser(),  # always last — catches everything
    ])

    logger.debug("Loaded %d parsers", len(_PARSERS))


def register_parser(parser: BaseSiteParser) -> None:
    """Register a custom parser (inserted before GenericParser)."""
    _ensure_loaded()
    # Insert before the last element (GenericParser)
    _PARSERS.insert(len(_PARSERS) - 1, parser)


def get_parser(url: str) -> Optional[BaseSiteParser]:
    """Return the best parser for a given URL, or ``None``."""
    _ensure_loaded()
    for parser in _PARSERS:
        try:
            if parser.can_handle(url):
                return parser
        except Exception:
            continue
    return None


def list_supported_sites() -> List[dict]:
    """Return a list of supported sites with their domains."""
    _ensure_loaded()
    sites = []
    for parser in _PARSERS:
        sites.append({
            "site": parser.get_site_name(),
            "domains": parser.get_domains(),
        })
    return sites
