"""
Cosmos Tool Router
======================

Routes tool calls to the appropriate provider.
Integrates MCP tools with built-in capabilities.
"""

import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger

from .mcp_manager import get_mcp_manager


class ToolRouter:
    """
    Routes tool calls to appropriate handlers.

    Priority:
    1. Built-in tools (web search, browser via API)
    2. MCP tools (1ly, filesystem, etc.)
    3. Skill-provided tools
    """

    def __init__(self):
        self.mcp = get_mcp_manager()
        self.builtin_tools: Dict[str, callable] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in tool handlers."""
        self.builtin_tools.update({
            # Web search tools
            "web_search": self._web_search,
            "web_search_grok": self._web_search_grok,
            "web_search_gemini": self._web_search_gemini,

            # Browser tools (if headless available)
            "browser_fetch": self._browser_fetch,

            # Feedback tools
            "collect_feedback": self._collect_feedback,
            "get_suggestions": self._get_suggestions,

            # Summary tools
            "get_collective_summary": self._get_collective_summary,
        })

    async def route(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route a tool call to the appropriate handler."""
        # Check built-in tools first
        if tool_name in self.builtin_tools:
            try:
                return await self.builtin_tools[tool_name](params)
            except Exception as e:
                logger.error(f"Built-in tool {tool_name} failed: {e}")
                return {"error": str(e)}

        # Check MCP tools
        if tool_name in self.mcp.tools:
            return await self.mcp.call_tool(tool_name, params)

        # Check if it's a 1ly tool (auto-load server)
        if tool_name.startswith("1ly_"):
            if "1ly" not in self.mcp.servers:
                # Add 1ly server config
                self.mcp.add_server(
                    "1ly",
                    "npx",
                    ["@1ly/mcp-server"],
                    {}
                )
            await self.mcp.start_server("1ly")
            return await self.mcp.call_tool(tool_name, params)

        return {"error": f"Unknown tool: {tool_name}"}

    def list_available_tools(self) -> List[Dict[str, str]]:
        """List all available tools with descriptions."""
        tools = []

        # Built-in tools
        for name in self.builtin_tools:
            tools.append({
                "name": name,
                "source": "builtin",
                "description": self._get_builtin_description(name)
            })

        # MCP tools
        for name, tool in self.mcp.tools.items():
            tools.append({
                "name": name,
                "source": f"mcp:{tool.server_name}",
                "description": tool.description
            })

        return tools

    def _get_builtin_description(self, name: str) -> str:
        """Get description for a built-in tool."""
        descriptions = {
            "web_search": "Search the web for information",
            "web_search_grok": "Search via Grok (xAI) - includes X/Twitter",
            "web_search_gemini": "Search via Gemini (Google) - grounded search",
            "browser_fetch": "Fetch and parse a web page",
            "collect_feedback": "Collect user feedback on a response",
            "get_suggestions": "Get improvement suggestions",
            "get_collective_summary": "Get summary of collective deliberations",
        }
        return descriptions.get(name, "")

    # =========================================================================
    # Built-in Tool Implementations
    # =========================================================================

    async def _web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Web search using best available provider."""
        query = params.get("query", "")
        if not query:
            return {"error": "Query required"}

        # Try Grok first (has X integration)
        result = await self._web_search_grok(params)
        if "error" not in result:
            return result

        # Fallback to Gemini
        result = await self._web_search_gemini(params)
        if "error" not in result:
            return result

        # Fallback to DuckDuckGo
        return await self._web_search_ddg(params)

    async def _web_search_grok(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search via Grok API."""
        try:
            from Cosmos.integration.external.grok import get_grok_provider

            grok = get_grok_provider()
            if grok and grok.api_key:
                query = params.get("query", "")
                result = await grok.chat(
                    f"Search the web and provide current information about: {query}",
                    max_tokens=2000
                )
                return {"results": result.get("content", ""), "source": "grok"}
        except Exception as e:
            logger.debug(f"Grok search failed: {e}")

        return {"error": "Grok search unavailable"}

    async def _web_search_gemini(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search via Gemini API with grounding."""
        try:
            from Cosmos.integration.external.gemini import get_gemini_provider

            gemini = get_gemini_provider()
            if gemini:
                query = params.get("query", "")
                result = await gemini.chat(
                    f"Search and provide current information about: {query}",
                    max_tokens=2000
                )
                return {"results": result.get("content", ""), "source": "gemini"}
        except Exception as e:
            logger.debug(f"Gemini search failed: {e}")

        return {"error": "Gemini search unavailable"}

    async def _web_search_ddg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search via DuckDuckGo (no API key needed)."""
        try:
            import httpx

            query = params.get("query", "")
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json"},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "results": data.get("AbstractText", "") or data.get("RelatedTopics", []),
                        "source": "duckduckgo"
                    }
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")

        return {"error": "DuckDuckGo search failed"}

    async def _browser_fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch and parse a web page."""
        try:
            import httpx
            from bs4 import BeautifulSoup

            url = params.get("url", "")
            if not url:
                return {"error": "URL required"}

            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=15.0, follow_redirects=True)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')

                    # Extract main content
                    for script in soup(["script", "style"]):
                        script.decompose()

                    text = soup.get_text(separator="\n", strip=True)

                    return {
                        "url": str(resp.url),
                        "title": soup.title.string if soup.title else "",
                        "content": text[:10000],  # Limit content size
                        "status": resp.status_code
                    }
        except Exception as e:
            logger.error(f"Browser fetch failed: {e}")

        return {"error": f"Failed to fetch {params.get('url', '')}"}

    async def _collect_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect user feedback."""
        try:
            from Cosmos.core.collective.evolution import get_evolution_engine

            feedback = params.get("feedback", "")
            context = params.get("context", {})

            if not feedback:
                return {"error": "Feedback content required"}

            evolution = get_evolution_engine()
            if evolution:
                evolution.record_feedback(feedback, context)
                return {"status": "recorded", "message": "Feedback integrated"}

        except Exception as e:
            logger.error(f"Feedback collection failed: {e}")

        return {"error": "Failed to collect feedback"}

    async def _get_suggestions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get improvement suggestions."""
        try:
            from Cosmos.core.collective.evolution import get_evolution_engine

            evolution = get_evolution_engine()
            if evolution:
                suggestions = evolution.get_improvement_suggestions()
                return {"suggestions": suggestions}

        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")

        return {"suggestions": []}

    async def _get_collective_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of collective deliberations."""
        try:
            from Cosmos.core.collective.deliberation import get_deliberation_stats

            stats = await get_deliberation_stats()
            return {
                "total_deliberations": stats.get("total", 0),
                "average_participation": stats.get("avg_participation", 0),
                "consensus_rate": stats.get("consensus_rate", 0),
                "latest": stats.get("latest", None)
            }

        except Exception as e:
            logger.debug(f"Failed to get collective summary: {e}")

        return {
            "total_deliberations": 0,
            "average_participation": 0,
            "consensus_rate": 0,
            "latest": None
        }


# Global tool router instance
_tool_router: Optional[ToolRouter] = None


def get_tool_router() -> ToolRouter:
    """Get or create the global tool router."""
    global _tool_router
    if _tool_router is None:
        _tool_router = ToolRouter()
    return _tool_router


async def route_tool_call(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to route a tool call."""
    return await get_tool_router().route(tool_name, params)
