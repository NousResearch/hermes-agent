"""Free native Google model discovery, including every page.

Google determines listed IDs; models.dev supplies text/tool capabilities.
Unknown capabilities remain usable by explicit model ID, not advertised as
verified agent models. Failures never return a partial/authoritative catalog.
"""
import json
import time
from urllib.parse import urlencode
from urllib.request import Request

from agent.models_dev import fetch_models_dev, get_model_info
from hermes_cli.urllib_security import open_credentialed_url


def fetch_models(api_key: str | None, *, timeout: float = 8.0) -> list[str] | None:
    if not api_key:
        return None
    if not fetch_models_dev():
        return None
    deadline = time.monotonic() + timeout
    result = []
    tokens = set()
    token = ""
    try:
        for _ in range(20):
            query = {"pageSize": 1000, **({"pageToken": token} if token else {})}
            request = Request("https://generativelanguage.googleapis.com/v1beta/models?" + urlencode(query),
                              headers={"x-goog-api-key": api_key, "Accept": "application/json"})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            with open_credentialed_url(request, timeout=remaining) as response:
                data = json.load(response)
            if not isinstance(data, dict) or not isinstance(data.get("models"), list):
                return None
            for entry in data["models"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    return None
                model = entry["name"].removeprefix("models/")
                # Dedicated Computer Use routes require Google's built-in tool;
                # Hermes' generic function tools cannot invoke them. The native
                # list and models.dev's tool_call flag do not encode this prerequisite.
                if "-computer-use-" in model:
                    continue
                info = get_model_info("gemini", model, allow_network=False)
                if ("generateContent" in (entry.get("supportedGenerationMethods") or [])
                        and info and info.tool_call and info.output_modalities == ("text",)
                        and info.status != "deprecated"):
                    result.append(model)
            token = data.get("nextPageToken")
            if not token:
                return list(dict.fromkeys(result))
            if not isinstance(token, str) or token in tokens:
                return None
            tokens.add(token)
    except Exception:
        return None  # no credential-bearing exceptions in logs
    return None
