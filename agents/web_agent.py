"""agents/web_agent.py
WebAgent: performs live web searches and returns structured research findings.
Used in the self-evolving swarm to gather context and validate architecture decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("agents.web_agent")


@dataclass
class AgentResult:
    success: bool
    output: str
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WebAgent:
    """Autonomous web search agent for research and validation."""

    name = "web_agent"
    SYSTEM_PROMPT = """You are a research specialist in the Kairos self-evolving swarm.
You receive a task or question and perform targeted web searches to gather context.
You output a structured research summary with:
1. Key findings relevant to the task
2. Best practices and patterns discovered
3. Potential tools, libraries, or approaches
4. Links to useful resources
5. Summary of findings in actionable format

Keep research concise and focused on the immediate need."""

    def __init__(
        self,
        tools: Any = None,
        memory: Any = None,
        llm_call: Optional[Callable[[str, str], str]] = None,
    ):
        self.tools = tools
        self.memory = memory
        self.llm_call = llm_call

    def run(self, task: str, context: str = "") -> AgentResult:
        """
        Perform web research for a given task.

        Args:
            task: The research query or task
            context: Optional architectural context

        Returns:
            AgentResult with research findings
        """
        logger.info(f"WEB_AGENT researching: {task[:80]}...")

        try:
            # Use DuckDuckGo search if available
            search_results = self._search(task)
            research_summary = self._synthesize_research(task, search_results, context)

            return AgentResult(
                success=True,
                output=research_summary,
                metadata={
                    "task": task,
                    "search_count": len(search_results),
                    "agent": "web_agent",
                },
            )
        except Exception as e:
            logger.error(f"Web research failed: {e}")
            return AgentResult(
                success=False,
                output=f"Web research failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def _search(self, query: str) -> list[dict[str, Any]]:
        """Perform web search using DuckDuckGo."""
        results = []
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=5))
                results = [
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "link": r.get("link", ""),
                    }
                    for r in search_results
                ]
            logger.info(f"Found {len(results)} results for: {query}")
        except ImportError:
            logger.warning("DuckDuckGo search not available, using fallback")
        except Exception as e:
            logger.warning(f"Search error: {e}")

        return results

    def _synthesize_research(
        self, task: str, search_results: list[dict[str, Any]], context: str = ""
    ) -> str:
        """Synthesize search results into actionable research summary."""
        if not search_results:
            return "No web search results found. Using existing context only."

        prompt = f"""Task: {task}

Context: {context}

Search Results:
{self._format_results(search_results)}

Synthesize these findings into a concise research summary with:
1. Key Findings (3-5 bullets)
2. Best Practices
3. Recommended Approaches
4. Resources (links)

Format as structured text."""

        if self.llm_call:
            try:
                summary = self.llm_call(self.SYSTEM_PROMPT, prompt)
                return summary
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}, using raw results")

        # Fallback: raw summary
        return self._format_results(search_results)

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results as readable text."""
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r.get('title', 'No title')}")
            formatted.append(f"   {r.get('body', 'No description')[:200]}...")
            if r.get("link"):
                formatted.append(f"   Link: {r['link']}")
        return "\n".join(formatted)
