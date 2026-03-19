"""
Browser Controller for Direct Playwright Access.
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class BrowserController:
    """
    Low-level browser controller for direct Playwright access.

    For cases where the AI agent isn't needed and you want
    direct control over browser actions.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._context = None
        self._page = None

    async def start(self):
        """Start the browser."""
        try:
            from playwright.async_api import async_playwright
            from .stealth import StealthBrowser

            self._browser = await StealthBrowser.create(headless=self.headless)
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()

        except ImportError:
            raise ImportError("playwright is required: pip install playwright")

    async def stop(self):
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._context = None
            self._page = None

    async def goto(self, url: str, wait_until: str = "networkidle"):
        """Navigate to a URL."""
        if not self._page:
            await self.start()
        await self._page.goto(url, wait_until=wait_until)
        return self._page.url

    async def click(self, selector: str):
        """Click an element."""
        if self._page:
            await self._page.click(selector)

    async def fill(self, selector: str, value: str):
        """Fill a form field."""
        if self._page:
            await self._page.fill(selector, value)

    async def type(self, selector: str, text: str, delay: int = 50):
        """Type text with delay."""
        if self._page:
            await self._page.type(selector, text, delay=delay)

    async def press(self, key: str):
        """Press a key."""
        if self._page:
            await self._page.keyboard.press(key)

    async def screenshot(self, full_page: bool = False) -> bytes:
        """Take a screenshot."""
        if self._page:
            return await self._page.screenshot(full_page=full_page)
        return b""

    async def get_text(self, selector: str) -> str:
        """Get text content of an element."""
        if self._page:
            element = await self._page.query_selector(selector)
            if element:
                return await element.text_content() or ""
        return ""

    async def get_html(self) -> str:
        """Get page HTML."""
        if self._page:
            return await self._page.content()
        return ""

    async def evaluate(self, script: str) -> Any:
        """Evaluate JavaScript."""
        if self._page:
            return await self._page.evaluate(script)
        return None

    async def wait_for(self, selector: str, timeout: int = 30000):
        """Wait for an element."""
        if self._page:
            await self._page.wait_for_selector(selector, timeout=timeout)

    async def extract_table(self, selector: str) -> List[Dict]:
        """Extract data from an HTML table."""
        script = f'''
        (() => {{
            const table = document.querySelector('{selector}');
            if (!table) return [];
            const rows = Array.from(table.querySelectorAll('tr'));
            const headers = Array.from(rows[0]?.querySelectorAll('th, td') || [])
                .map(cell => cell.textContent?.trim() || '');
            return rows.slice(1).map(row => {{
                const cells = Array.from(row.querySelectorAll('td'));
                const obj = {{}};
                cells.forEach((cell, i) => {{
                    obj[headers[i] || `col_${{i}}`] = cell.textContent?.trim() || '';
                }});
                return obj;
            }});
        }})()
        '''
        return await self.evaluate(script) or []
