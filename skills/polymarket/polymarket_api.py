#!/usr/bin/env python3
"""
Polymarket API Client

Access Polymarket prediction market data.
"""

import requests
from typing import Optional, List


class PolymarketAPI:
    """Polymarket API 客户端"""
    
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Hermes-Agent-Polymarket-Skill/1.0"
        })
    
    def get_markets(self, active: bool = True, limit: int = 50) -> List[dict]:
        """获取市场列表"""
        params = {
            "active": str(active).lower(),
            "limit": limit,
            "order": "volume",
            "sort": "desc"
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/events",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # API 返回列表
            if isinstance(data, list):
                events = data
            else:
                events = data.get('_events', [])
            
            markets = []
            for event in events:
                for market in event.get('markets', []):
                    market_data = {
                        "slug": market.get('slug', ''),
                        "question": market.get('question', ''),
                        "yes_bid": float(market.get('yesBid', 0)) / 100,
                        "yes_ask": float(market.get('yesAsk', 0)) / 100,
                        "no_bid": float(market.get('noBid', 0)) / 100,
                        "no_ask": float(market.get('noAsk', 0)) / 100,
                        "last_price": float(market.get('lastPrice', 0)) / 100,
                        "volume": float(market.get('volume', 0)),
                        "volume_24h": float(market.get('volume24h', 0)),
                        "liquidity": float(market.get('liquidity', 0)),
                        "open_interest": float(market.get('openInterest', 0)),
                        "category": event.get('category', ''),
                        "tags": event.get('tags', []),
                        "end_date": event.get('endDate'),
                        "url": f"https://polymarket.com/event/{market.get('slug', '')}"
                    }
                    markets.append(market_data)
            
            return markets[:limit]
            
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []
    
    def search_markets(self, query: str, limit: int = 20) -> List[dict]:
        """搜索市场"""
        markets = self.get_markets(limit=200)
        
        query_lower = query.lower()
        filtered = [
            m for m in markets 
            if query_lower in m['question'].lower() or 
               query_lower in m['slug'].lower()
        ]
        
        return filtered[:limit]
    
    def get_market(self, slug: str) -> Optional[dict]:
        """获取单个市场详情"""
        markets = self.get_markets(limit=200)
        
        for market in markets:
            if market['slug'] == slug:
                return market
        
        return None


# Skill 入口函数
def get_markets(limit: int = 20, category: str = None) -> List[dict]:
    """获取活跃市场"""
    api = PolymarketAPI()
    markets = api.get_markets(limit=limit)
    
    if category:
        markets = [m for m in markets if m.get('category', '').lower() == category.lower()]
    
    return markets


def search_markets(query: str, limit: int = 20) -> List[dict]:
    """搜索市场"""
    api = PolymarketAPI()
    return api.search_markets(query, limit=limit)


def get_market(slug: str) -> dict:
    """获取市场详情"""
    api = PolymarketAPI()
    market = api.get_market(slug)
    return market if market else {}
