"""Inline HTML sidecars share canonical transcript persistence, never a second artifact store."""
import html as html_escape
import json
import re

ARTIFACT_ID = re.compile(r"html_[0-9a-f]{64}\Z")
CSP = ("default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; "
       "img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; "
       "object-src 'none'; base-uri 'none'; form-action 'none'")
RENDER_POLICY = {"sandbox": "", "referrer_policy": "no-referrer", "scripts": False,
                 "network_resources": False, "min_height": 120, "default_height": 320, "max_height": 800}


def artifact_from_result(name: str, result) -> dict | None:
    """Recognize only the publishing tool's versioned envelope, not arbitrary model text."""
    if name != "publish_html":
        return None
    try:
        data = json.loads(result) if isinstance(result, str) else result
        artifact = data.get("artifact") if isinstance(data, dict) else None
        if (not isinstance(artifact, dict) or artifact.get("version") != 1
                or artifact.get("format") != "html" or not ARTIFACT_ID.fullmatch(str(artifact.get("id", "")))
                or not all(isinstance(artifact.get(k), str) for k in ("title", "html", "fallback"))):
            return None
        if len(artifact["html"].encode("utf-8")) > 65536:
            return None
        return dict(artifact)
    except (ValueError, TypeError, UnicodeError):
        return None


def artifact_summary(artifact: dict) -> dict:
    """History/events carry metadata only; authenticated retrieval carries content."""
    return {**{key: artifact[key] for key in ("version", "id", "title", "format", "fallback")},
            "byte_length": len(artifact["html"].encode("utf-8"))}


def project_artifact_metadata(metadata: dict) -> dict:
    artifact = artifact_from_result("publish_html", {"artifact": metadata.get("inline_artifact")})
    if artifact is None:
        return {key: value for key, value in metadata.items() if key != "inline_artifact"}
    return {**metadata, "inline_artifact": artifact_summary(artifact)}


def artifact_response(artifact: dict, *, session_id: str, row_id: int, tool_call_id: str) -> dict:
    """JSON only. Clients must use sandboxed srcdoc, never inject HTML into the app DOM."""
    prefix = '<meta http-equiv="Content-Security-Policy" content="' + html_escape.escape(CSP, quote=False) + '">'
    return {"artifact": artifact_summary(artifact), "html": artifact["html"],
            "render_html": prefix + artifact["html"], "render_policy": dict(RENDER_POLICY),
            "association": {"session_id": session_id, "row_id": row_id, "tool_call_id": tool_call_id}}
