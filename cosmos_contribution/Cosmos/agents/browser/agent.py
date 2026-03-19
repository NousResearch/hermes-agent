"""
Cosmos Browser Agent.

Autonomous browser agent for web tasks using browser-use.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Check for browser-use availability
try:
    from browser_use import Agent, Browser, BrowserConfig
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    Agent = None
    Browser = None


@dataclass
class BrowserResult:
    """Result of a browser task."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    url: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CosmosBrowserAgent:
    """
    Autonomous browser agent for web tasks.

    Uses browser-use library with Playwright backend.

    Capabilities:
    - Navigate to URLs
    - Fill forms
    - Click buttons
    - Extract data
    - Take screenshots
    - Execute multi-step tasks
    """

    def __init__(self, llm=None, headless: bool = True):
        """
        Initialize browser agent.

        Args:
            llm: LLM to use for decision making
            headless: Whether to run browser in headless mode
        """
        if not BROWSER_USE_AVAILABLE:
            logger.warning("browser-use not available - install with: pip install browser-use")

        self.llm = llm
        self.headless = headless
        self._browser = None
        self._agent = None

    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if not BROWSER_USE_AVAILABLE:
            raise ImportError("browser-use is required: pip install browser-use")

        if self._browser is None:
            from .stealth import StealthBrowser
            self._browser = await StealthBrowser.create(headless=self.headless)

    async def execute_task(self, task: str) -> BrowserResult:
        """
        Execute a web browsing task.

        Args:
            task: Natural language description of the task

        Returns:
            BrowserResult with extracted data and status
        """
        start_time = datetime.now()

        try:
            await self._ensure_browser()

            # Create browser-use agent
            if BROWSER_USE_AVAILABLE:
                from browser_use import Agent

                agent = Agent(
                    task=task,
                    llm=self.llm or self._get_default_llm(),
                    browser=self._browser,
                )

                result = await agent.run()

                return BrowserResult(
                    success=result.is_done() if hasattr(result, 'is_done') else True,
                    data=result.extracted_content() if hasattr(result, 'extracted_content') else {},
                    screenshots=result.screenshots if hasattr(result, 'screenshots') else [],
                    actions_taken=result.history if hasattr(result, 'history') else [],
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
            else:
                # Fallback to basic Playwright
                return await self._execute_with_playwright(task)

        except Exception as e:
            logger.error(f"Browser task failed: {e}")
            return BrowserResult(
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    async def _execute_with_playwright(self, task: str) -> BrowserResult:
        """Fallback execution using raw Playwright."""
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page()

                actions = []

                # Parse basic commands from task
                task_lower = task.lower()

                if "go to" in task_lower or "navigate" in task_lower:
                    # Extract URL
                    import re
                    url_match = re.search(r'(https?://\S+)', task)
                    if url_match:
                        url = url_match.group(1)
                        await page.goto(url)
                        actions.append(f"Navigated to {url}")

                # Take screenshot
                screenshot = await page.screenshot()

                await browser.close()

                return BrowserResult(
                    success=True,
                    url=page.url if hasattr(page, 'url') else "",
                    actions_taken=actions,
                    data={"note": "Executed with basic Playwright fallback"},
                )

        except Exception as e:
            return BrowserResult(
                success=False,
                error=f"Playwright fallback failed: {e}",
            )

    async def navigate_and_extract(
        self,
        url: str,
        extraction_prompt: str
    ) -> Dict[str, Any]:
        """
        Navigate to URL and extract specific data.

        Args:
            url: Target URL
            extraction_prompt: What data to extract

        Returns:
            Extracted data
        """
        task = f"Go to {url} and {extraction_prompt}"
        result = await self.execute_task(task)
        return result.data

    async def fill_form(
        self,
        url: str,
        form_data: Dict[str, str],
        submit: bool = False
    ) -> BrowserResult:
        """
        Fill a form on a webpage.

        Args:
            url: URL of the form page
            form_data: Dict of field names/labels to values
            submit: Whether to submit the form

        Returns:
            BrowserResult
        """
        fields = ", ".join(f"{k}: {v}" for k, v in form_data.items())
        task = f"Go to {url}, fill in the form with {fields}"
        if submit:
            task += ", and submit the form"

        return await self.execute_task(task)

    async def take_screenshot(self, url: str) -> bytes:
        """Take a screenshot of a URL."""
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url)
                screenshot = await page.screenshot()
                await browser.close()
                return screenshot

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return b""

    def _get_default_llm(self):
        """Get default LLM for browser-use."""
        # Try to use local LLM first
        try:
            from Cosmos.core.local_llm import get_local_llm
            return get_local_llm()
        except Exception:
            pass

        # Fall back to OpenAI if available
        try:
            from langchain_openai import ChatOpenAI
            import os
            if os.environ.get("OPENAI_API_KEY"):
                return ChatOpenAI(model="gpt-4")
        except Exception:
            pass

        return None

    async def close(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
