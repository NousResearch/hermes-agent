#!/usr/bin/env python3
"""
Probe a Hermes OpenAI-compatible custom endpoint.

This is designed to catch the common failure mode where `/models` works
but inference endpoints return 5xx (upstream/proxy issues).

Examples:
  ./.venv/bin/python scripts/probe_custom_endpoint.py --from-hermes-config
  python scripts/probe_custom_endpoint.py --base-url http://localhost:8000/v1 --api-key sk-... --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ProbeResult:
    label: str
    url: str
    status_code: Optional[int]
    content_type: str
    body_preview: str
    error: str = ""


def _candidate_base_urls(base_url: str) -> list[str]:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return []
    candidates = [base]
    if not base.endswith("/v1"):
        candidates.append(f"{base}/v1")
    return candidates


def _looks_like_html(result: ProbeResult) -> bool:
    content_type = (result.content_type or "").lower()
    if "text/html" in content_type:
        return True
    body = (result.body_preview or "").lstrip().lower()
    return body.startswith("<!doctype html") or body.startswith("<html")


def _mask_secret(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        return ""
    if len(s) <= 10:
        return s[:2] + "..." + s[-2:]
    return s[:4] + "..." + s[-4:]


def _preview(text: str, limit: int = 800) -> str:
    t = (text or "").replace("\r\n", "\n")
    return t if len(t) <= limit else (t[:limit] + "...(truncated)")


def _request(method: str, url: str, headers: dict[str, str], payload: Optional[dict[str, Any]], timeout: float) -> ProbeResult:
    import httpx

    try:
        if method == "GET":
            resp = httpx.get(url, headers=headers, timeout=timeout)
        else:
            resp = httpx.request(
                method,
                url,
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        return ProbeResult(
            label=f"{method} {url.rsplit('/', 1)[-1]}",
            url=url,
            status_code=resp.status_code,
            content_type=str(resp.headers.get("content-type", "")),
            body_preview=_preview(resp.text),
        )
    except Exception as e:
        return ProbeResult(
            label=f"{method} {url.rsplit('/', 1)[-1]}",
            url=url,
            status_code=None,
            content_type="",
            body_preview="",
            error=repr(e),
        )


def _get_hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return str(get_hermes_home())
    except Exception:
        return os.getenv("HERMES_HOME", "").strip() or str((__import__("pathlib").Path.home() / ".hermes"))


def _load_hermes_env_key(hermes_home: str, env_key: str) -> str:
    from pathlib import Path

    # 1) If already present in environment (doctor loads it), use it.
    v = os.getenv(env_key, "").strip()
    if v:
        return v

    # 2) Parse ~/.hermes/.env directly.
    env_path = Path(hermes_home) / ".env"
    if not env_path.exists():
        return ""
    try:
        from dotenv import dotenv_values  # type: ignore

        vals = dotenv_values(env_path)
        return str(vals.get(env_key, "") or "").strip()
    except Exception:
        return ""


def _load_from_hermes_config(env_key: str = "OPENAI_API_KEY") -> tuple[str, str, str]:
    from pathlib import Path

    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing PyYAML. Run with Hermes' venv: ./.venv/bin/python ...") from e

    hermes_home = _get_hermes_home()
    cfg_path = Path(hermes_home) / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    base_url = str(model_cfg.get("base_url", "") or "").strip()
    api_key = str(model_cfg.get("api_key", "") or "").strip() or _load_hermes_env_key(hermes_home, env_key)
    model = str(model_cfg.get("default", "") or "").strip()
    if not base_url:
        raise RuntimeError(f"No model.base_url in {cfg_path}")
    return base_url, api_key, model


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from-hermes-config", action="store_true", help="Load base_url/api_key/model from ~/.hermes/config.yaml")
    p.add_argument("--base-url", default="", help="Base URL ending in /v1 (e.g. https://api.example.com/v1)")
    p.add_argument("--api-key", default="", help="API key (optional)")
    p.add_argument("--model", default="", help="Model name (optional)")
    p.add_argument("--env-key", default="OPENAI_API_KEY", help="Env var name to read from ~/.hermes/.env when api_key isn't in config.yaml")
    p.add_argument("--timeout", type=float, default=15.0, help="Timeout seconds")
    args = p.parse_args(argv)

    if args.from_hermes_config:
        base_url, api_key, model = _load_from_hermes_config(env_key=args.env_key)
    else:
        base_url, api_key, model = args.base_url.strip(), args.api_key.strip(), args.model.strip()

    base_url = base_url.rstrip("/")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print("Hermes custom endpoint probe")
    print(f"- base_url: {base_url}")
    print(f"- api_key:  {_mask_secret(api_key) if api_key else '(none)'}")
    print(f"- model:    {model or '(none)'}")
    print()

    results: list[ProbeResult] = []
    candidate_bases = _candidate_base_urls(base_url)
    inference_base = base_url
    selected_models_url = ""
    for idx, candidate in enumerate(candidate_bases):
        r_models = _request("GET", f"{candidate}/models", headers, None, timeout=args.timeout)
        results.append(r_models)
        if r_models.status_code == 200 and not _looks_like_html(r_models):
            inference_base = candidate
            selected_models_url = r_models.url
            break
        if idx == 0:
            inference_base = candidate

    if model:
        chat_payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        results.append(_request("POST", f"{inference_base}/chat/completions", headers, chat_payload, timeout=args.timeout))
        resp_payload = {"model": model, "input": "ping", "max_output_tokens": 1}
        results.append(_request("POST", f"{inference_base}/responses", headers, resp_payload, timeout=args.timeout))

    bad = False
    for r in results:
        status = str(r.status_code) if r.status_code is not None else "ERROR"
        print(f"== {r.label} ==")
        print(f"url:    {r.url}")
        print(f"status: {status}")
        if r.error:
            print(f"error:  {r.error}")
            bad = True
        if r.content_type:
            print(f"type:   {r.content_type}")
        if r.body_preview:
            print(r.body_preview)
        print()

        is_models_probe = r.url.endswith("/models")
        html_result = _looks_like_html(r)
        if r.status_code is not None and r.status_code >= 400:
            bad = True
        elif html_result and (not is_models_probe or r.url == selected_models_url or not selected_models_url):
            bad = True

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
