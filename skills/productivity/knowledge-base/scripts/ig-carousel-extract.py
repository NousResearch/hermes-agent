"""Extract first 2 slides + alt text from an Instagram carousel post using Playwright.

Instagram only loads the first 2 carousel images on initial page load.
Slides 3+ require carousel navigation that is blocked by Instagram's anti-bot
in all headless browser modes (tested: headless=old, headless=new, touch events,
mobile viewport, arrow keys, mouse clicks, network interception — all fail).

The alt attribute on Instagram carousel images contains the FULL text content
of each slide — no vision_analyze needed. This is the preferred extraction method.

Usage:
    python ig-carousel-extract.py URL [OUTPUT_DIR]

Requires: playwright (pip install playwright), npx playwright install chromium
Cookies: /tmp/ig_cookies.txt (Netscape format, must contain sessionid)
"""
import asyncio, json, sys, os, subprocess
from playwright.async_api import async_playwright


def load_cookies_netscape(path: str) -> list[dict]:
    cookies = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                cookies.append({
                    "name": parts[5],
                    "value": parts[6],
                    "domain": parts[0].lstrip(".") if parts[0].startswith(".") else parts[0],
                    "path": parts[2],
                    "secure": parts[3] == "TRUE",
                })
    return cookies


async def extract(post_url: str, out_dir: str = "/tmp/ig_slides") -> list[dict]:
    cookies = load_cookies_netscape("/tmp/ig_cookies.txt")
    os.makedirs(out_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(viewport={"width": 1080, "height": 1920})
        await context.add_cookies(cookies)
        page = await context.new_page()

        await page.goto(post_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(8000)

        # Check for login wall
        content = await page.content()
        if "login" in content.lower() and len(content) < 80000:
            print("ERROR: Login wall — cookies may be expired", file=sys.stderr)
            await browser.close()
            return []

        # Extract all carousel images (first 2 slides only)
        result = await page.evaluate("""() => {
            const images = [];
            const seen = new Set();
            document.querySelectorAll('img').forEach(img => {
                const src = img.src;
                if (src && src.includes('cdninstagram') && !src.includes('s150x150') && !src.includes('profile')) {
                    if (!seen.has(src)) {
                        seen.add(src);
                        images.push({
                            src,
                            alt: img.alt || '',
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                        });
                    }
                }
            });
            return images;
        }""")

        await browser.close()

    # Download each image
    slides = []
    for i, img in enumerate(result):
        ext = ".webp" if ".webp" in img["src"] else ".jpg"
        fname = f"slide_{i:02d}{ext}"
        fpath = os.path.join(out_dir, fname)

        subprocess.run([
            "curl", "-sLo", fpath,
            "-H", "Referer: https://www.instagram.com/",
            "-H", "User-Agent: Mozilla/5.0",
            img["src"],
        ], timeout=30)

        size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        if size < 1000:
            print(f"  WARNING: slide {i} download too small ({size} bytes) — CDN URL may have expired", file=sys.stderr)

        slides.append({
            "index": i,
            "file": fpath,
            "alt": img["alt"],
            "size_bytes": size,
        })

    print(f"Extracted {len(slides)} slides to {out_dir}", file=sys.stderr)
    return slides


def main():
    if len(sys.argv) < 2:
        print("Usage: python ig-carousel-extract.py URL [OUTPUT_DIR]", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ig_slides"
    slides = asyncio.run(extract(url, out_dir))
    print(json.dumps(slides, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
