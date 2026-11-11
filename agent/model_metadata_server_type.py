"""Local server-type verdicts on disk: the fingerprint legs and their day-long revalidated cache
(sibling of ``model_metadata``)."""

from __future__ import annotations

from typing import Optional

# A positive server-type verdict ("ollama"/"llamacpp"/"vllm"/"lm-studio") describes which software
# answers a URL, which only changes on a server swap under the same base_url. The 5-minute probe TTL
# turned every >=5-min idle gap (the normal first-message-of-the-session rhythm) into a re-run of the
# serial fingerprint waterfall: a 404 burst per cold window, up to each leg's 2 s connect timeout
# off-LAN. Verdicts persist for a day, but a disk hit is REVALIDATED with the one leg that certified
# it, so a swap self-heals on next use at one request per cold window, not five. Runtime-window
# values (ollama_num_ctx) keep the short TTL: a restart with a new --max-model-len must show fast.
SERVER_TYPE_DISK_TTL_SECONDS = 86400.0


def server_type_probe_legs(server_url: str, lmstudio_url: str):
    """Ordered waterfall legs: (name, paths tried until one answers 200, body check).

    Most specific first. LM Studio answers /api/tags with {"error": ...} and 200, so Ollama's
    body must carry "models"; older llama.cpp builds have no /v1 prefix."""
    return (
        ("lm-studio", (f"{lmstudio_url}/api/v1/models",), lambda r: True),
        ("ollama", (f"{server_url}/api/tags",), lambda r: "models" in r.json()),
        ("llamacpp", (f"{server_url}/v1/props", f"{server_url}/props"), lambda r: "default_generation_settings" in r.text),
        ("vllm", (f"{server_url}/version",), lambda r: "version" in r.json()),
    )


def _revalidate(name, server_url, lmstudio_url, base_url, api_key) -> str:
    """"match", "mismatch" (another surface answers the certified leg) or "unreachable" (a blip:
    the durable verdict survives it, and the host is NOT marked blackholed, so the next call
    gets its own chance)."""
    import httpx

    from agent import model_metadata_http
    from agent.model_metadata import _auth_headers

    leg = next((leg for leg in server_type_probe_legs(server_url, lmstudio_url) if leg[0] == name), None)
    if leg is None:
        return "mismatch"
    _, urls, check = leg
    try:
        with httpx.Client(timeout=2.0, headers=_auth_headers(api_key), verify=model_metadata_http.resolve_verify(base_url)) as client:
            for url in urls:
                r = client.get(url)
                if r.status_code == 200:
                    break  # not 200: try the leg's next path (llama.cpp /props)
            else:
                return "mismatch"
    except httpx.HTTPError:
        return "unreachable"
    try:
        return "match" if check(r) else "mismatch"
    except ValueError:  # a 200 that is not the certified JSON body: another server answers
        return "mismatch"


def revalidated_disk_server_type(server_url, lmstudio_url, base_url, api_key) -> Optional[str]:
    """The disk verdict for *server_url* if it still holds, else None (the caller runs the full
    waterfall, which rewrites the entry). A confirmed match buys the verdict another day."""
    from agent.model_metadata import _local_probe_disk_get, _local_probe_disk_put

    disk_hit = _local_probe_disk_get("server_type", server_url, SERVER_TYPE_DISK_TTL_SECONDS)
    if not isinstance(disk_hit, str):
        return None
    verdict = _revalidate(disk_hit, server_url, lmstudio_url, base_url, api_key)
    if verdict == "mismatch":
        return None
    if verdict == "match":
        _local_probe_disk_put("server_type", server_url, disk_hit)
    return disk_hit
