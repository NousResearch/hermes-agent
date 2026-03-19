"""
cosmos Web Agent - Intelligent Web Browsing

Novel Approaches:
1. Semantic Page Understanding - Extract meaning, not just text
2. Action Planning - Multi-step web interactions
3. Smart Navigation - Learn site patterns
4. Content Synthesis - Combine info across pages
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
from urllib.parse import urljoin, urlparse
import json

from loguru import logger


class ActionType(Enum):
    """Types of web actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SCROLL = "scroll"
    EXTRACT = "extract"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    BACK = "back"
    FORWARD = "forward"


@dataclass
class PageElement:
    """An interactive element on a page."""
    selector: str
    tag: str
    text: str = ""
    attributes: dict = field(default_factory=dict)
    is_visible: bool = True
    is_interactive: bool = False
    element_type: str = ""  # "link", "button", "input", "form", etc.


@dataclass
class PageState:
    """Current state of a web page."""
    url: str
    title: str = ""
    content: str = ""
    html: str = ""

    # Extracted elements
    links: list[PageElement] = field(default_factory=list)
    buttons: list[PageElement] = field(default_factory=list)
    inputs: list[PageElement] = field(default_factory=list)
    forms: list[PageElement] = field(default_factory=list)

    # Metadata
    loaded_at: datetime = field(default_factory=datetime.now)
    load_time_ms: float = 0.0

    # Semantic understanding
    page_type: str = ""  # "article", "form", "list", "search", "login", etc.
    main_content: str = ""
    structured_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "page_type": self.page_type,
            "link_count": len(self.links),
            "button_count": len(self.buttons),
            "input_count": len(self.inputs),
        }


@dataclass
class WebAction:
    """A web action to perform."""
    action_type: ActionType
    target: Optional[str] = None  # URL or selector
    value: Optional[str] = None   # Input value
    wait_after_ms: int = 500

    def to_dict(self) -> dict:
        return {
            "action": self.action_type.value,
            "target": self.target,
            "value": self.value,
        }


@dataclass
class ActionResult:
    """Result of a web action."""
    success: bool
    action: WebAction
    page_state: Optional[PageState] = None
    error: Optional[str] = None
    screenshot_path: Optional[str] = None
    extracted_data: Optional[Any] = None


@dataclass
class BrowsingSession:
    """A web browsing session."""
    id: str
    goal: str
    actions: list[ActionResult] = field(default_factory=list)
    visited_urls: list[str] = field(default_factory=list)

    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Collected data
    extracted_content: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)


class WebAgent:
    """
    Intelligent web browsing agent.

    Features:
    - Navigate and interact with web pages
    - Extract and synthesize information
    - Fill forms and complete multi-step tasks
    - Learn navigation patterns
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        headless: bool = True,
        timeout_ms: int = 30000,
    ):
        self.llm_fn = llm_fn
        self.headless = headless
        self.timeout_ms = timeout_ms

        self._browser = None
        self._page = None
        self._initialized = False

        self.sessions: dict[str, BrowsingSession] = {}
        self._session_counter = 0

        # Site patterns learned
        self.site_patterns: dict[str, dict] = {}

        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize browser (Playwright)."""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            self._page = await self._browser.new_page()
            self._initialized = True

            logger.info("Web agent initialized with Playwright")

        except ImportError:
            logger.warning("Playwright not installed. Install with: pip install playwright && playwright install")
            # Fallback to requests-based browsing
            self._initialized = True

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
        if hasattr(self, '_playwright'):
            await self._playwright.stop()
        self._initialized = False

    async def browse(
        self,
        goal: str,
        start_url: Optional[str] = None,
        max_actions: int = 10,
    ) -> BrowsingSession:
        """
        Browse the web to achieve a goal.

        Args:
            goal: What to find or accomplish
            start_url: Starting URL (or will search)
            max_actions: Maximum actions to take

        Returns:
            BrowsingSession with results
        """
        await self.initialize()

        async with self._lock:
            self._session_counter += 1
            session_id = f"session_{self._session_counter}"

        session = BrowsingSession(id=session_id, goal=goal)
        self.sessions[session_id] = session

        logger.info(f"Starting browse session {session_id}: {goal}")

        # Determine starting point
        if not start_url:
            start_url = await self._generate_search_url(goal)

        # Navigate to start
        result = await self._execute_action(WebAction(
            action_type=ActionType.NAVIGATE,
            target=start_url,
        ))
        session.actions.append(result)

        if not result.success:
            session.ended_at = datetime.now()
            return session

        session.visited_urls.append(start_url)

        # Iterative browsing
        for _ in range(max_actions - 1):
            # Analyze current page
            page_state = result.page_state
            if not page_state:
                break

            # Decide next action
            next_action = await self._plan_next_action(session, page_state)

            if next_action is None:
                # Goal achieved or no more actions
                break

            # Execute action
            result = await self._execute_action(next_action)
            session.actions.append(result)

            if result.success and result.page_state:
                if result.page_state.url not in session.visited_urls:
                    session.visited_urls.append(result.page_state.url)

                # Extract relevant content
                content = await self._extract_relevant_content(
                    result.page_state, goal
                )
                if content:
                    session.extracted_content.append(content)

        session.ended_at = datetime.now()

        # Synthesize findings
        session.findings = await self._synthesize_findings(session)

        logger.info(f"Session {session_id} complete: {len(session.findings)} findings")
        return session

    async def _generate_search_url(self, goal: str) -> str:
        """Generate a search URL for the goal."""
        # Extract search terms
        if self.llm_fn:
            prompt = f"Extract 3-5 search keywords from this goal: {goal}\nReturn just the keywords separated by spaces."
            try:
                if asyncio.iscoroutinefunction(self.llm_fn):
                    keywords = await self.llm_fn(prompt)
                else:
                    keywords = self.llm_fn(prompt)
                keywords = keywords.strip()
            except Exception as e:
                logger.debug(f"Keyword extraction failed, using raw goal: {e}")
                keywords = goal
        else:
            keywords = goal

        # Use DuckDuckGo for privacy
        from urllib.parse import quote
        return f"https://duckduckgo.com/?q={quote(keywords)}"

    async def _execute_action(self, action: WebAction) -> ActionResult:
        """Execute a web action."""
        try:
            page_state = None

            if self._page:
                # Playwright-based execution
                if action.action_type == ActionType.NAVIGATE:
                    await self._page.goto(action.target, timeout=self.timeout_ms)
                    await asyncio.sleep(action.wait_after_ms / 1000)
                    page_state = await self._get_page_state()

                elif action.action_type == ActionType.CLICK:
                    await self._page.click(action.target, timeout=self.timeout_ms)
                    await asyncio.sleep(action.wait_after_ms / 1000)
                    page_state = await self._get_page_state()

                elif action.action_type == ActionType.FILL:
                    await self._page.fill(action.target, action.value or "")
                    await asyncio.sleep(action.wait_after_ms / 1000)
                    page_state = await self._get_page_state()

                elif action.action_type == ActionType.SCROLL:
                    await self._page.evaluate("window.scrollBy(0, 500)")
                    await asyncio.sleep(action.wait_after_ms / 1000)
                    page_state = await self._get_page_state()

                elif action.action_type == ActionType.EXTRACT:
                    content = await self._page.inner_text(action.target)
                    page_state = await self._get_page_state()
                    return ActionResult(
                        success=True,
                        action=action,
                        page_state=page_state,
                        extracted_data=content,
                    )

                elif action.action_type == ActionType.BACK:
                    await self._page.go_back()
                    await asyncio.sleep(action.wait_after_ms / 1000)
                    page_state = await self._get_page_state()

            else:
                # Requests-based fallback
                if action.action_type == ActionType.NAVIGATE:
                    page_state = await self._fetch_page(action.target)

            return ActionResult(
                success=True,
                action=action,
                page_state=page_state,
            )

        except Exception as e:
            logger.error(f"Action failed: {e}")
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
            )

    async def _get_page_state(self) -> PageState:
        """Extract current page state."""
        url = self._page.url
        title = await self._page.title()
        content = await self._page.inner_text("body")

        state = PageState(
            url=url,
            title=title,
            content=content[:10000],  # Limit size
        )

        # Extract interactive elements
        try:
            # Links
            links = await self._page.query_selector_all("a[href]")
            for link in links[:50]:  # Limit
                href = await link.get_attribute("href")
                text = await link.inner_text()
                if href and text.strip():
                    state.links.append(PageElement(
                        selector=f"a[href='{href}']",
                        tag="a",
                        text=text.strip()[:100],
                        attributes={"href": href},
                        element_type="link",
                    ))

            # Buttons
            buttons = await self._page.query_selector_all("button, input[type='submit']")
            for btn in buttons[:20]:
                text = await btn.inner_text() or await btn.get_attribute("value") or ""
                state.buttons.append(PageElement(
                    selector="button",
                    tag="button",
                    text=text.strip()[:50],
                    element_type="button",
                ))

            # Inputs
            inputs = await self._page.query_selector_all("input:not([type='hidden']), textarea")
            for inp in inputs[:20]:
                input_type = await inp.get_attribute("type") or "text"
                name = await inp.get_attribute("name") or ""
                placeholder = await inp.get_attribute("placeholder") or ""
                state.inputs.append(PageElement(
                    selector=f"input[name='{name}']" if name else "input",
                    tag="input",
                    text=placeholder,
                    attributes={"type": input_type, "name": name},
                    element_type="input",
                ))

        except Exception as e:
            logger.debug(f"Element extraction error: {e}")

        # Classify page type
        state.page_type = self._classify_page(state)

        return state

    async def _fetch_page(self, url: str) -> PageState:
        """Fetch page using requests (fallback)."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator=' ', strip=True)

            state = PageState(
                url=url,
                title=soup.title.string if soup.title else "",
                content=content[:10000],
                html=html[:50000],
            )

            # Extract links
            for a in soup.find_all('a', href=True)[:50]:
                state.links.append(PageElement(
                    selector=f"a[href='{a['href']}']",
                    tag="a",
                    text=a.get_text(strip=True)[:100],
                    attributes={"href": urljoin(url, a['href'])},
                    element_type="link",
                ))

            state.page_type = self._classify_page(state)
            return state

        except Exception as e:
            logger.error(f"Page fetch failed: {e}")
            return PageState(url=url)

    def _classify_page(self, state: PageState) -> str:
        """Classify the type of page."""
        url = state.url.lower()
        content = state.content.lower()
        title = state.title.lower()

        # URL patterns
        if "/login" in url or "/signin" in url:
            return "login"
        if "/search" in url or "q=" in url:
            return "search"
        if "/article" in url or "/blog" in url or "/post" in url:
            return "article"
        if "/product" in url or "/item" in url:
            return "product"
        if "/cart" in url or "/checkout" in url:
            return "checkout"

        # Content patterns
        if len(state.inputs) > 3:
            return "form"
        if len(state.links) > 20 and len(state.content) < 2000:
            return "list"
        if "login" in title or "sign in" in title:
            return "login"
        if len(content) > 3000 and len(state.links) < 20:
            return "article"

        return "general"

    async def _plan_next_action(
        self,
        session: BrowsingSession,
        page_state: PageState,
    ) -> Optional[WebAction]:
        """Plan the next action based on goal and current state."""
        if self.llm_fn:
            return await self._llm_plan_action(session, page_state)
        else:
            return self._heuristic_next_action(session, page_state)

    async def _llm_plan_action(
        self,
        session: BrowsingSession,
        page_state: PageState,
    ) -> Optional[WebAction]:
        """Use LLM to plan next action."""
        # Summarize available actions
        links_summary = "\n".join([
            f"- Link: {l.text[:50]} -> {l.attributes.get('href', '')[:50]}"
            for l in page_state.links[:10]
        ])

        buttons_summary = "\n".join([
            f"- Button: {b.text[:30]}"
            for b in page_state.buttons[:5]
        ])

        visited = ", ".join(session.visited_urls[-5:])

        prompt = f"""You are browsing the web to: {session.goal}

Current page:
- URL: {page_state.url}
- Title: {page_state.title}
- Type: {page_state.page_type}

Available links:
{links_summary}

Available buttons:
{buttons_summary}

Recently visited: {visited}

What should we do next? Return JSON:
{{"action": "navigate/click/extract/done", "target": "url or selector", "reason": "why"}}

Return "done" if goal is achieved or no progress possible."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            data = json.loads(self._extract_json(response))

            action_str = data.get("action", "done")

            if action_str == "done":
                return None

            action_type = {
                "navigate": ActionType.NAVIGATE,
                "click": ActionType.CLICK,
                "extract": ActionType.EXTRACT,
                "scroll": ActionType.SCROLL,
            }.get(action_str, ActionType.NAVIGATE)

            return WebAction(
                action_type=action_type,
                target=data.get("target"),
            )

        except Exception as e:
            logger.error(f"Action planning failed: {e}")
            return None

    def _heuristic_next_action(
        self,
        session: BrowsingSession,
        page_state: PageState,
    ) -> Optional[WebAction]:
        """Simple heuristic for next action."""
        goal_words = session.goal.lower().split()

        # Look for relevant links
        for link in page_state.links:
            link_text = link.text.lower()
            href = link.attributes.get("href", "")

            # Skip already visited
            if href in session.visited_urls:
                continue

            # Check relevance
            if any(word in link_text for word in goal_words if len(word) > 3):
                return WebAction(
                    action_type=ActionType.NAVIGATE,
                    target=href,
                )

        # No more relevant links
        return None

    async def _extract_relevant_content(
        self,
        page_state: PageState,
        goal: str,
    ) -> Optional[str]:
        """Extract content relevant to the goal."""
        if self.llm_fn:
            prompt = f"""Extract information relevant to: {goal}

Page content:
{page_state.content[:3000]}

Return just the relevant facts and information. If nothing relevant, return "NONE"."""

            try:
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                if "NONE" not in response.upper():
                    return response.strip()

            except Exception as e:
                logger.error(f"Content extraction failed: {e}")

        # Fallback: return page summary
        if len(page_state.content) > 200:
            return f"From {page_state.title}: {page_state.content[:500]}..."

        return None

    async def _synthesize_findings(
        self,
        session: BrowsingSession,
    ) -> list[dict]:
        """Synthesize collected content into findings."""
        if not session.extracted_content:
            return []

        if self.llm_fn:
            combined = "\n---\n".join(session.extracted_content[:10])

            prompt = f"""Synthesize these pieces of information into clear findings.
Goal: {session.goal}

Collected information:
{combined}

Return JSON array of findings:
[{{"finding": "...", "confidence": 0.0-1.0, "source_count": N}}]"""

            try:
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                return json.loads(self._extract_json(response))

            except Exception as e:
                logger.error(f"Synthesis failed: {e}")

        # Fallback: return raw content as findings
        return [
            {"finding": content, "confidence": 0.5, "source_count": 1}
            for content in session.extracted_content[:5]
        ]

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        # Find array
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return text[start:end]

        # Find object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return text[start:end]

        return '[]'

    async def fill_form(
        self,
        url: str,
        form_data: dict[str, str],
        submit: bool = True,
    ) -> ActionResult:
        """Fill a form with provided data."""
        await self.initialize()

        # Navigate to form page
        result = await self._execute_action(WebAction(
            action_type=ActionType.NAVIGATE,
            target=url,
        ))

        if not result.success:
            return result

        # Fill each field
        for field_name, value in form_data.items():
            selector = f"input[name='{field_name}'], textarea[name='{field_name}']"
            result = await self._execute_action(WebAction(
                action_type=ActionType.FILL,
                target=selector,
                value=value,
            ))

            if not result.success:
                return result

        # Submit if requested
        if submit:
            result = await self._execute_action(WebAction(
                action_type=ActionType.CLICK,
                target="button[type='submit'], input[type='submit']",
            ))

        return result

    async def extract_structured_data(
        self,
        url: str,
        schema: dict,
    ) -> dict:
        """Extract structured data from a page according to schema."""
        await self.initialize()

        # Navigate
        result = await self._execute_action(WebAction(
            action_type=ActionType.NAVIGATE,
            target=url,
        ))

        if not result.success or not result.page_state:
            return {}

        extracted = {}

        if self.llm_fn:
            prompt = f"""Extract data from this page according to the schema.

Schema:
{json.dumps(schema, indent=2)}

Page content:
{result.page_state.content[:4000]}

Return JSON matching the schema."""

            try:
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                extracted = json.loads(self._extract_json(response))

            except Exception as e:
                logger.error(f"Structured extraction failed: {e}")

        return extracted

    def get_stats(self) -> dict:
        """Get web agent statistics."""
        return {
            "initialized": self._initialized,
            "total_sessions": len(self.sessions),
            "total_pages_visited": sum(
                len(s.visited_urls) for s in self.sessions.values()
            ),
            "total_actions": sum(
                len(s.actions) for s in self.sessions.values()
            ),
        }
