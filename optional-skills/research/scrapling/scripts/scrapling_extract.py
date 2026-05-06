#!/usr/bin/env python3
"""Selector-driven Scrapling extraction runner for the optional pilot runtime.

The script emits a JSON receipt for both success and failure. It imports
Scrapling lazily so --help and schema tests work without installing the optional
runtime into Hermes' main environment.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Callable, Sequence

VALID_MODES = ("static", "dynamic", "stealth")
VALID_SELECTOR_TYPES = ("css", "xpath", "text", "regex")


def make_receipt(
    *,
    mode: str,
    url: str,
    selector: str,
    selector_type: str,
    content: str,
    elapsed_ms: int,
    fallback_reason: str,
    errors: list[dict] | None = None,
) -> dict:
    return {
        "backend": "scrapling",
        "mode": mode,
        "url": url,
        "selector": selector,
        "selector_type": selector_type,
        "content": content,
        "elapsed_ms": int(elapsed_ms),
        "fallback_reason": fallback_reason,
        "errors": errors or [],
    }


def normalize_content(value, max_chars: int) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        text = "\n".join(str(item) for item in value if item is not None)
    else:
        text = str(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def select_from_page(page, *, selector: str, selector_type: str):
    if selector_type == "css":
        selected = page.css(selector)
        if hasattr(selected, "getall"):
            return selected.getall()
        if hasattr(selected, "get"):
            return selected.get()
        return selected
    if selector_type == "xpath":
        selected = page.xpath(selector)
        if hasattr(selected, "getall"):
            return selected.getall()
        if hasattr(selected, "get"):
            return selected.get()
        return selected
    if selector_type == "text":
        if hasattr(page, "find_by_text"):
            return page.find_by_text(selector)
        return str(page)
    if selector_type == "regex":
        source = str(page)
        return re.findall(selector, source)
    raise ValueError(f"unsupported selector_type: {selector_type}")


def default_fetcher(
    *,
    url: str,
    selector: str,
    selector_type: str,
    mode: str,
    timeout: int,
    wait_selector: str | None,
    network_idle: bool,
    max_chars: int,
) -> str:
    if mode == "static":
        from scrapling.fetchers import Fetcher

        page = Fetcher.get(url, timeout=timeout)
    elif mode == "dynamic":
        from scrapling.fetchers import DynamicFetcher

        kwargs = {"headless": True}
        if wait_selector:
            kwargs["wait_selector"] = (wait_selector, "visible")
        if network_idle:
            kwargs["network_idle"] = True
        page = DynamicFetcher.fetch(url, **kwargs)
    elif mode == "stealth":
        from scrapling.fetchers import StealthyFetcher

        page = StealthyFetcher.fetch(url, headless=True)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    return normalize_content(select_from_page(page, selector=selector, selector_type=selector_type), max_chars)


def run_extract(
    *,
    url: str,
    selector: str,
    selector_type: str,
    mode: str,
    timeout: int,
    wait_selector: str | None,
    network_idle: bool,
    max_chars: int,
    fallback_reason: str,
    fetcher: Callable[..., str] | None = None,
) -> dict:
    started = time.monotonic()
    fetcher = fetcher or default_fetcher
    try:
        content = fetcher(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            timeout=timeout,
            wait_selector=wait_selector,
            network_idle=network_idle,
            max_chars=max_chars,
        )
        return make_receipt(
            mode=mode,
            url=url,
            selector=selector,
            selector_type=selector_type,
            content=normalize_content(content, max_chars),
            elapsed_ms=int((time.monotonic() - started) * 1000),
            fallback_reason=fallback_reason,
            errors=[],
        )
    except Exception as exc:  # deliberate: CLI must return JSON, not a traceback wall
        return make_receipt(
            mode=mode,
            url=url,
            selector=selector,
            selector_type=selector_type,
            content="",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            fallback_reason=fallback_reason,
            errors=[{"type": exc.__class__.__name__, "message": str(exc)}],
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a narrow Scrapling selector extraction and emit a JSON receipt.")
    parser.add_argument("--url", required=True, help="Public URL to fetch.")
    parser.add_argument("--selector", required=True, help="CSS, XPath, text, or regex selector to extract.")
    parser.add_argument("--selector-type", choices=VALID_SELECTOR_TYPES, default="css", help="Selector strategy, default: css.")
    parser.add_argument("--mode", choices=VALID_MODES, default="static", help="Fetcher mode: static, dynamic, or stealth.")
    parser.add_argument("--timeout", type=int, default=20, help="Static fetch timeout in seconds where supported.")
    parser.add_argument("--wait-selector", default=None, help="Dynamic mode selector to wait for before extraction.")
    parser.add_argument("--network-idle", action="store_true", help="Dynamic mode: wait for network idle when supported.")
    parser.add_argument("--max-chars", type=int, default=50000, help="Maximum content characters to emit; <=0 disables truncation.")
    parser.add_argument("--fallback-reason", default="selector_required", help="Auditable reason for using Scrapling fallback.")
    return parser


def main(argv: Sequence[str] | None = None, *, fetcher: Callable[..., str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    receipt = run_extract(
        url=args.url,
        selector=args.selector,
        selector_type=args.selector_type,
        mode=args.mode,
        timeout=args.timeout,
        wait_selector=args.wait_selector,
        network_idle=args.network_idle,
        max_chars=args.max_chars,
        fallback_reason=args.fallback_reason,
        fetcher=fetcher,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt["errors"]:
        if receipt["errors"][0].get("type") == "ModuleNotFoundError":
            return 2
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
