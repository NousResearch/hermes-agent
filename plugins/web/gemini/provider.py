"""Gemini web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
official Google GenAI SDK (``google-genai``) which is lazy-loaded via
:func:`tools.lazy_deps.ensure`.

Config keys this provider responds to::

    web:
      search_backend: "gemini"     # explicit per-capability
      extract_backend: "gemini"    # explicit per-capability
      backend: "gemini"            # shared fallback for both
      gemini_model: "gemini-3.1-flash-lite" # Model used for grounding

Env var::

    GEMINI_API_KEY=...    # or GOOGLE_API_KEY
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _get_gemini_client() -> Any:
    """Lazy-import and cache a Gemini SDK client.

    Raises ``ValueError`` when ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` is unset.
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_gemini_client", None)
    if cached is not None:
        return cached

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Get your API key at https://aistudio.google.com/app/apikey"
        )

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("google.genai", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise ImportError(str(exc))

    from google import genai

    client = genai.Client(api_key=api_key)
    _wt._gemini_client = client
    return client


def _reset_client_for_tests() -> None:
    """Drop the cached Gemini client so tests can re-instantiate cleanly."""
    import tools.web_tools as _wt

    _wt._gemini_client = None


class GeminiWebSearchProvider(WebSearchProvider):
    """Gemini search + extract provider using Google Search Grounding."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Gemini Google Search"

    def is_available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def _get_model(self) -> str:
        from hermes_cli.config import load_config
        cfg = load_config().get("web", {})
        return cfg.get("gemini_model", "gemini-3.1-flash-lite")

    def _resolve_url(self, url: str) -> str:
        """Resolve vertexaisearch URLs using a HEAD request."""
        if not url.startswith("https://vertexaisearch.cloud.google.com/"):
            return url
        try:
            import httpx
            # Do a non-following HEAD request to extract the location
            with httpx.Client(follow_redirects=False, timeout=5.0) as client:
                res = client.head(url)
                if res.is_redirect and "location" in res.headers:
                    return res.headers["location"]
        except Exception as e:
            logger.debug(f"Failed to resolve vertex url {url}: {e}")
        return url

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Gemini search via Google Search Grounding."""
        try:
            from tools.interrupt import is_interrupted
            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            model = self._get_model()
            logger.info("Gemini search: '%s' (model=%s)", query, model)
            
            client = _get_gemini_client()
            from google.genai import types

            response = client.models.generate_content(
                model=model,
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0,
                )
            )

            web_results = []
            
            # Google Search grounding metadata is typically in response.candidates[0].grounding_metadata.search_entry_point
            # or in the chunks. We'll extract URLs from grounding chunks if available.
            # But we also just return the summarized answer as the main description.
            
            # Provide the generated answer as the first result.
            if response.text:
                web_results.append({
                    "url": "https://google.com/search?q=" + query.replace(" ", "+"),
                    "title": "Gemini Grounded Answer",
                    "description": response.text,
                    "position": 1,
                })
                
            # Try to extract supporting links
            try:
                candidate = response.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    if hasattr(candidate.grounding_metadata, "grounding_chunks"):
                        for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks):
                            if hasattr(chunk, "web") and chunk.web:
                                web_results.append({
                                    "url": self._resolve_url(getattr(chunk.web, "uri", getattr(chunk.web, "url", ""))),
                                    "title": getattr(chunk.web, "title", ""),
                                    "description": "Supporting source",
                                    "position": len(web_results) + 1,
                                })
            except Exception as e:
                logger.debug(f"Failed to extract grounding chunks: {e}")

            # Fallback if no results at all
            if not web_results:
                 web_results.append({
                    "url": "",
                    "title": "No Results",
                    "description": "Gemini did not return a response.",
                    "position": 1,
                })

            # Trim to limit
            web_results = web_results[:limit]
            
            return {"success": True, "data": {"web": web_results}}
            
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {"success": False, "error": f"Google GenAI SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini search error: %s", exc)
            return {"success": False, "error": f"Gemini search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Gemini.
        
        Gemini can fetch and summarize URLs passed directly to it with Google Search Grounding.
        """
        try:
            from tools.interrupt import is_interrupted
            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

            client = _get_gemini_client()
            from google.genai import types
            model = self._get_model()

            results: List[Dict[str, Any]] = []
            for url in urls:
                logger.info("Gemini extract: %s", url)
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=f"Extract and summarize the full content from this URL: {url}",
                        config=types.GenerateContentConfig(
                            tools=[{"google_search": {}}],
                            temperature=0.0,
                        )
                    )
                    content = response.text or ""
                    results.append({
                        "url": url,
                        "title": f"Gemini Extraction: {url}",
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url},
                    })
                except Exception as inner_exc:
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": str(inner_exc)
                    })
                    
            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [{"url": u, "title": "", "content": "", "error": f"Google GenAI SDK not installed: {exc}"} for u in urls]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini extract error: %s", exc)
            return [{"url": u, "title": "", "content": "", "error": f"Gemini extract failed: {exc}"} for u in urls]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Gemini Google Search Grounding",
            "badge": "free tier available",
            "tag": "Google Search grounded generation using Gemini API.",
            "env_vars": [
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Gemini API key",
                    "url": "https://aistudio.google.com/app/apikey",
                },
            ],
        }

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    provider = GeminiWebSearchProvider()

    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        urls = sys.argv[2:]
        print(f"Extracting {urls}...")
        results = provider.extract(urls)
        for r in results:
            print(f"\n--- {r.get('title', r.get('url'))} ---")
            print(r.get('content', r.get('error')))
    else:
        query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "latest news on artificial intelligence"
        print(f"Searching for: {query}")
        result = provider.search(query)
        import json
        print(json.dumps(result, indent=2))
