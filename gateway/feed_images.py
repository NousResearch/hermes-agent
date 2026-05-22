"""Fanhearts feed-image API integration helpers.

This module is intentionally small and dependency-light so the Discord gateway can
forward uploaded images without blocking the main agent flow, and a cron/script
worker can later process queued jobs. The Fanhearts API endpoints are being built
in parallel, so all request/response parsing is tolerant of the expected field
name variants.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://dev-api.fanhearts.com"
DEFAULT_LIMIT = 1
DEFAULT_TIMEOUT_SECONDS = 60.0

_IMAGE_ID_KEYS = ("id", "feed_image_id", "feedImageId")
_IMAGE_URL_KEYS = ("image_url", "imageUrl", "input_image_url", "inputImageUrl", "url")
_PROMPT_KEYS = ("prompt", "user_prompt", "userPrompt", "transform_prompt", "transformPrompt")


def env_enabled() -> bool:
    """Return whether Discord -> Fanhearts forwarding is enabled."""

    return os.getenv("FANHEARTS_FEED_IMAGES_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_api_base_url() -> str:
    return os.getenv("FANHEARTS_FEED_IMAGES_API_BASE_URL", DEFAULT_API_BASE_URL).strip() or DEFAULT_API_BASE_URL


def env_jwt() -> str:
    return os.getenv("FANHEARTS_FEED_IMAGES_JWT", "").strip()


def _headers(jwt: str) -> dict[str, str]:
    if not jwt:
        raise ValueError("FANHEARTS_FEED_IMAGES_JWT is required")
    return {"Authorization": f"Bearer {jwt}"}


def _endpoint(api_base_url: str, path: str) -> str:
    base = api_base_url.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _first_present(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_jobs(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("feed_images", "feedImages", "jobs", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


async def post_discord_feed_image(
    *,
    image_path: str,
    message_text: str,
    source: Mapping[str, Any],
    api_base_url: str | None = None,
    jwt: str | None = None,
) -> dict[str, Any]:
    """POST a Discord image attachment to Fanhearts /feed_images.

    The endpoint is not implemented yet, so this sends a conservative multipart
    body with the image, prompt text, source label, and JSON metadata. The API can
    ignore unknown fields while preserving all Discord routing data.
    """

    api_base_url = api_base_url or env_api_base_url()
    jwt = jwt if jwt is not None else env_jwt()
    image = Path(image_path).expanduser()
    if not image.exists():
        raise FileNotFoundError(str(image))

    metadata = {
        "platform": source.get("platform") or "discord",
        "discord_channel_id": source.get("chat_id") or source.get("channel_id"),
        "discord_thread_id": source.get("thread_id"),
        "discord_message_id": source.get("message_id"),
        "discord_user_id": source.get("user_id"),
        "discord_user_name": source.get("user_name"),
    }
    content_type = mimetypes.guess_type(str(image))[0] or "application/octet-stream"
    data = {
        "source": "discord",
        "prompt": message_text or "",
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }

    with image.open("rb") as fh:
        files = {"image": (image.name, fh, content_type)}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            response = await client.post(
                _endpoint(api_base_url, "/feed_images"),
                headers=_headers(jwt or ""),
                data=data,
                files=files,
            )
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"ok": True, "status_code": response.status_code, "text": response.text}


async def _download_image(client: httpx.AsyncClient, url: str, workdir: Path) -> Path:
    response = await client.get(url)
    response.raise_for_status()
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        suffix = ".png"
    path = workdir / f"input{suffix}"
    path.write_bytes(response.content)
    return path


FEED_IMAGE_TRANSFORM_PROMPT = """첨부한 사진을 기반으로 자연스러운 합성 이미지를 만들어줘.

현재 사진은 풍경을 배경으로, 한 손에 인물 사진을 들고 있는 장면이다.
손에 들린 사진 속 인물을 실제 풍경 속에 자연스럽게 등장한 것처럼 합성해줘.

가장 중요한 조건은 다음과 같다.

사진 속 인물의 얼굴 생김새, 표정, 시선, 포즈, 체형, 신체 비율은 최대한 그대로 유지해줘.
인물이 다른 사람처럼 보이면 안 된다. 얼굴 구조, 눈·코·입의 인상, 분위기, 표정의 뉘앙스를 원본 인물과 동일하게 유지해줘.

다만 인물이 풍경 속에 실제로 서 있는 것처럼 보이도록,
의상은 배경의 계절감과 장소 분위기에 맞는 자연스러운 캐주얼 스타일로 바꿔줘.
옷은 과하게 화려하지 않게, 실제 여행 사진이나 일상 스냅처럼 자연스럽게 연출해줘.

머리스타일이나 메이크업, 분장이 원본에서 너무 과하거나 배경과 어울리지 않는다면,
인물의 정체성과 얼굴 인상은 유지한 채 캐주얼한 의상과 어울리도록 자연스럽게 정리해줘.
단, 헤어스타일을 완전히 다른 사람처럼 바꾸지는 말고, 원본의 분위기를 살린 자연스러운 변화만 적용해줘.

합성된 인물은 배경의 조명, 그림자, 원근감, 색감, 해상도와 자연스럽게 어울려야 한다.
사진을 들고 있는 손이나 원래의 종이 사진 느낌은 제거하고, 인물이 실제 풍경 안에 존재하는 것처럼 만들어줘.

전체 결과물은 인위적인 AI 합성 느낌이 아니라, 실제 카메라로 촬영한 자연스러운 여행/라이프스타일 사진처럼 보여야 한다.

네거티브 프롬프트

다른 사람처럼 변형된 얼굴, 얼굴 인상 변화, 표정 변화, 포즈 변화, 과한 성형 느낌, 과한 메이크업, 과한 헤어스타일, 비현실적인 의상, 판타지 의상, 과도한 보정, 플라스틱 피부, AI 합성 티, 흐릿한 얼굴, 왜곡된 손, 이상한 손가락, 부자연스러운 그림자, 배경과 맞지 않는 조명, 원근감 오류, 종이 사진이 남아 있음, 손에 사진을 들고 있는 상태 유지, 인물이 배경에 떠 보임"""


def build_transform_prompt(image_path: Path, user_prompt: str) -> str:
    """Build the GPT Image 2 prompt for a queued feed image."""

    return FEED_IMAGE_TRANSFORM_PROMPT


CODEX_IMAGE_CHAT_MODEL = "gpt-5.4"
CODEX_IMAGE_MODEL = "gpt-image-2"
CODEX_IMAGE_PROVIDER_MODEL_LABEL = "codex-oauth/gpt-5.4-image_generation"
CODEX_IMAGE_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_IMAGE_INSTRUCTIONS = (
    "You are an image editing assistant operating through Codex OAuth. "
    "Use the supplied source image and produce the best possible completed image for the prompt."
)


def _codex_image_generation_payload(image_path: Path, prompt: str) -> dict[str, Any]:
    """Build a Codex OAuth Responses image-generation request payload.

    This intentionally avoids Hermes' built-in image_generate/FAL path. Codex
    OAuth exposes image generation through the Responses API `image_generation`
    tool, with the source image supplied as an input image for edits/variations.
    """

    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    size = os.getenv("FANHEARTS_FEED_IMAGES_CODEX_SIZE", "1024x1024").strip() or "1024x1024"
    quality = os.getenv("FANHEARTS_FEED_IMAGES_CODEX_QUALITY", "medium").strip() or "medium"
    output_format = os.getenv("FANHEARTS_FEED_IMAGES_CODEX_FORMAT", "png").strip() or "png"
    background = os.getenv("FANHEARTS_FEED_IMAGES_CODEX_BACKGROUND", "auto").strip() or "auto"

    return {
        "model": os.getenv("FANHEARTS_FEED_IMAGES_CODEX_MODEL", CODEX_IMAGE_CHAT_MODEL).strip() or CODEX_IMAGE_CHAT_MODEL,
        "store": False,
        "instructions": CODEX_IMAGE_INSTRUCTIONS,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded_image}",
                    },
                ],
            }
        ],
        "tools": [
            {
                "type": "image_generation",
                "model": os.getenv("FANHEARTS_FEED_IMAGES_CODEX_IMAGE_MODEL", CODEX_IMAGE_MODEL).strip() or CODEX_IMAGE_MODEL,
                "size": size,
                "quality": quality,
                "output_format": output_format,
                "background": background,
            }
        ],
    }


def _build_codex_oauth_client() -> Any | None:
    """Return an OpenAI client authenticated with Hermes' Codex OAuth token."""

    try:
        import openai
        from agent.auxiliary_client import _codex_cloudflare_headers, _read_codex_access_token

        token = _read_codex_access_token()
        if not isinstance(token, str) or not token.strip():
            return None
        base_url = os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or CODEX_IMAGE_BASE_URL
        return openai.OpenAI(
            api_key=token.strip(),
            base_url=base_url,
            default_headers=_codex_cloudflare_headers(token.strip()),
        )
    except Exception as exc:
        logger.debug("Could not build Codex OAuth image client: %s", exc)
        return None


def _get_event_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _collect_codex_oauth_image(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Stream a Codex Responses image_generation call and return image metadata."""

    image_item = None
    response_id = None
    usage = None
    with client.responses.stream(**payload) as stream:
        for event in stream:
            event_type = _get_event_attr(event, "type", "")
            if event_type == "response.output_item.done":
                item = _get_event_attr(event, "item")
                if _get_event_attr(item, "type") == "image_generation_call" and _get_event_attr(item, "result"):
                    image_item = item
            elif event_type == "response.image_generation_call.partial_image":
                partial = _get_event_attr(event, "partial_image_b64")
                if isinstance(partial, str) and partial:
                    image_item = {"type": "image_generation_call", "result": partial, "status": "partial"}
        final = stream.get_final_response()

    response_id = _get_event_attr(final, "id")
    usage = _get_event_attr(final, "usage")
    for item in _get_event_attr(final, "output", []) or []:
        if _get_event_attr(item, "type") == "image_generation_call" and _get_event_attr(item, "result"):
            image_item = item

    if not image_item:
        raise RuntimeError("Codex response contained no image_generation_call result")
    return {"image_item": image_item, "response_id": response_id, "usage": usage}


def generate_with_codex_oauth_image(image_path: Path, prompt: str, output_dir: Path) -> tuple[Path | None, str | None]:
    """Generate/edit a feed image through Codex OAuth, not FAL.

    Returns ``(local_output_path, remote_image_url)``. Codex returns image bytes
    directly, so the remote URL is always ``None``.
    """

    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    client = _build_codex_oauth_client()
    if client is None:
        raise RuntimeError("No Codex/ChatGPT OAuth credentials available. Run `hermes auth codex` to sign in.")

    payload = _codex_image_generation_payload(image_path, prompt)
    parsed = _collect_codex_oauth_image(client, payload)
    image_item = parsed["image_item"]
    image_b64 = _get_event_attr(image_item, "result")
    if not isinstance(image_b64, str) or not image_b64:
        raise RuntimeError("Codex image_generation_call did not include image bytes")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "completed_image.png"
    output_path.write_bytes(base64.b64decode(image_b64))

    metadata = {
        "ok": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(output_path),
        "model": payload["model"],
        "image_model": payload["tools"][0].get("model"),
        "tool": payload["tools"][0],
        "prompt": prompt,
        "response_id": parsed.get("response_id"),
        "usage": _jsonable(parsed.get("usage")),
        "revised_prompt": _get_event_attr(image_item, "revised_prompt"),
        "status": _get_event_attr(image_item, "status"),
        "image_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, None


async def _mark_failed(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    jwt: str,
    feed_image_id: str,
    error: str,
) -> None:
    try:
        response = await client.put(
            _endpoint(api_base_url, f"/feed_images/{feed_image_id}"),
            headers=_headers(jwt),
            data={"status": "failed", "error": error[:2000]},
        )
        response.raise_for_status()
    except Exception:
        logger.exception("Failed to mark feed image %s as failed", feed_image_id)


async def process_queued_feed_images(
    *,
    api_base_url: str | None = None,
    jwt: str | None = None,
    limit: int = DEFAULT_LIMIT,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """Process queued Fanhearts feed-image jobs one by one."""

    api_base_url = api_base_url or env_api_base_url()
    jwt = jwt if jwt is not None else env_jwt()
    if not jwt:
        return {"processed": 0, "completed": 0, "failed": 0, "skipped": 0, "error": "missing FANHEARTS_FEED_IMAGES_JWT"}

    root = Path(workdir) if workdir is not None else Path(tempfile.gettempdir()) / "hermes-feed-images"
    root.mkdir(parents=True, exist_ok=True)
    summary = {"processed": 0, "completed": 0, "failed": 0, "skipped": 0}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.get(
            _endpoint(api_base_url, "/feed_images/status=queued"),
            headers=_headers(jwt),
            params={"limit": int(limit)},
        )
        response.raise_for_status()
        jobs = _extract_jobs(response.json())[: int(limit)]

        for job in jobs:
            feed_image_id = str(_first_present(job, _IMAGE_ID_KEYS) or "")
            image_url = str(_first_present(job, _IMAGE_URL_KEYS) or "")
            user_prompt = str(_first_present(job, _PROMPT_KEYS) or "")
            if not feed_image_id or not image_url:
                summary["skipped"] += 1
                continue

            summary["processed"] += 1
            job_dir = root / feed_image_id
            job_dir.mkdir(parents=True, exist_ok=True)
            try:
                claim = await client.post(
                    _endpoint(api_base_url, f"/feed_images/{feed_image_id}/claim"),
                    headers=_headers(jwt),
                )
                if claim.status_code in {409, 423}:
                    summary["skipped"] += 1
                    continue
                claim.raise_for_status()

                input_path = await _download_image(client, image_url, job_dir)
                transform_prompt = build_transform_prompt(input_path, user_prompt)
                output_path, output_url = generate_with_codex_oauth_image(input_path, transform_prompt, job_dir)

                data = {
                    "status": "completed",
                    "transform_prompt": transform_prompt,
                    "output_image_url": output_url or "",
                    "model": "codex-oauth/gpt-5.4-image_generation",
                }
                files = None
                file_handle = None
                try:
                    if output_path and output_path.exists():
                        file_handle = output_path.open("rb")
                        files = {"completed_image": (output_path.name, file_handle, "image/png")}
                    update = await client.put(
                        _endpoint(api_base_url, f"/feed_images/{feed_image_id}"),
                        headers=_headers(jwt),
                        data=data,
                        files=files,
                    )
                    update.raise_for_status()
                finally:
                    if file_handle:
                        file_handle.close()
                summary["completed"] += 1
            except Exception as exc:
                summary["failed"] += 1
                await _mark_failed(
                    client,
                    api_base_url=api_base_url,
                    jwt=jwt,
                    feed_image_id=feed_image_id,
                    error=str(exc),
                )

    return summary


def process_queued_feed_images_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(process_queued_feed_images(**kwargs))
