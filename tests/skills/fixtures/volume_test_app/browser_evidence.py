"""Optional technical UI verification. Not a Hermes agent/browser_exec run."""

import time

from data import FIELDS, reference


class BrowserEvidence:
    def __init__(self, base, executable):
        self.base, self.executable = base, executable
        self.errors = []
        self.screenshots = []

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self.manager = sync_playwright().start()
        self.browser = self.manager.chromium.launch(executable_path=self.executable, headless=True, args=["--no-sandbox"])
        self.context = self.browser.new_context(viewport={"width": 1100, "height": 800})
        self.page = self.context.new_page()
        self.page.set_default_timeout(3000)
        self.page.set_default_navigation_timeout(3000)
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("console", lambda message: self.errors.append(message.text) if message.type == "error" else None)
        self.page.on("response", lambda response: self.errors.append(f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)
        return self

    def capture(self, run, volume, rows, output, budget, save):
        from playwright.sync_api import TimeoutError as BrowserTimeout

        self.page.goto(f"{self.base}/?run={run}")
        self.page.wait_for_function("window.catalogState?.complete", timeout=3000)
        samples = []
        cases = [("listing", {}), ("search", {"q": "Café"}), ("category", {"category": "0"}), ("price-sort", {"sort": "price"}), ("last-page", {})]
        for name, args in cases:
            expected = reference(rows, **args)
            page_number = max(1, (len(expected) + 49) // 50) if name == "last-page" else 1
            visible = expected[(page_number - 1) * 50:page_number * 50]
            for phase in ["first-request", "warmup"] + ["warm"] * 10:
                budget.check()
                self.page.locator("#search").fill(args.get("q", ""))
                self.page.locator("#category").select_option(args.get("category", ""))
                self.page.locator("#sort").select_option(args.get("sort", "name"))
                self.page.locator("#page").fill(str(page_number))
                previous = self.page.evaluate("window.catalogState.request")
                start = time.perf_counter()
                sample = {"volume": volume, "operation": name, "phase": phase, "unit": "ms", "error": None, "timeout": False, "correct": False}
                try:
                    self.page.locator("#go").click()
                    self.page.wait_for_function("""expected => {
                      const state=window.catalogState;
                      const rows=[...document.querySelectorAll('#rows tr')].map(tr=>[...tr.cells].map(td=>td.textContent));
                      return state?.complete && state.request>expected.previous && state.total===expected.total &&
                        state.page===expected.page && JSON.stringify(rows)===JSON.stringify(expected.rows) &&
                        document.querySelector('#status').textContent.startsWith(expected.total+' products');
                    }""", arg={"previous": previous, "total": len(expected), "page": page_number,
                               "rows": [[str(row[field]) for field in FIELDS] for row in visible]}, timeout=3000)
                    sample["correct"] = True
                except BrowserTimeout as error:
                    sample["timeout"] = True
                    sample["error"] = str(error)
                sample["duration_ms"] = (time.perf_counter() - start) * 1000
                samples.append(sample)
                save(samples)
            if name == "listing":
                image = output / f"ui-{volume}.png"
                self.page.screenshot(path=str(image))
                self.screenshots.append(image.name)
        return samples

    def __exit__(self, *_args):
        self.context.close()
        self.browser.close()
        self.manager.stop()
