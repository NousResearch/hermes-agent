"""
Stealth Browser Configuration.

Bot detection countermeasures for Playwright.
"""

import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Stealth JavaScript to inject
STEALTH_SCRIPTS = '''
// Hide webdriver property
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined,
    configurable: true
});

// Add chrome runtime
window.chrome = {
    runtime: {},
    loadTimes: function() {},
    csi: function() {},
    app: {}
};

// Override plugins
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
    configurable: true
});

// Override languages
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
    configurable: true
});

// Override permissions
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
);
'''


class StealthBrowser:
    """
    Browser with bot detection countermeasures.

    Features:
    - User agent rotation
    - WebDriver property hiding
    - Chrome runtime spoofing
    - Plugin/language spoofing
    """

    STEALTH_ARGS = [
        "--disable-blink-features=AutomationControlled",
        "--disable-features=IsolateOrigins,site-per-process",
        "--disable-site-isolation-trials",
        "--disable-web-security",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-accelerated-2d-canvas",
        "--no-first-run",
        "--no-zygote",
        "--disable-gpu",
    ]

    @classmethod
    async def create(cls, headless: bool = True):
        """Create a stealth browser instance."""
        try:
            from playwright.async_api import async_playwright

            playwright = await async_playwright().start()

            browser = await playwright.chromium.launch(
                headless=headless,
                args=cls.STEALTH_ARGS,
            )

            return cls(browser, playwright)

        except ImportError:
            raise ImportError("playwright is required: pip install playwright && playwright install")

    def __init__(self, browser, playwright):
        self.browser = browser
        self.playwright = playwright

    async def new_context(self, **kwargs):
        """Create a new browser context with stealth settings."""
        user_agent = random.choice(USER_AGENTS)

        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=user_agent,
            locale="en-US",
            timezone_id="America/New_York",
            **kwargs
        )

        # Inject stealth scripts
        await context.add_init_script(STEALTH_SCRIPTS)

        return context

    async def new_page(self):
        """Create a new page with stealth settings."""
        context = await self.new_context()
        page = await context.new_page()
        return page

    async def close(self):
        """Close the browser."""
        await self.browser.close()
        await self.playwright.stop()

    @staticmethod
    def get_random_user_agent() -> str:
        """Get a random user agent."""
        return random.choice(USER_AGENTS)
