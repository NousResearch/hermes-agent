"""Behavior tests for the image-serve base-url contextvar + URL rewriting.

The API server binds ``set_image_serve_base_url`` around each agent run so
absolute local image paths come back as fetchable /images/ URLs. These
tests pin the rewrite contract: absolute paths only, URLs untouched,
per-context isolation.
"""

from pathlib import Path

import pytest

from agent.image_gen_provider import (
    _IMAGE_SERVE_BASE_URL,
    _maybe_rewrite_image_url,
    reset_image_serve_base_url,
    set_image_serve_base_url,
    success_response,
)

BASE = "http://127.0.0.1:8199"


@pytest.fixture(autouse=True)
def _fresh_context():
    token = _IMAGE_SERVE_BASE_URL.set(None)
    yield
    _IMAGE_SERVE_BASE_URL.reset(token)


def test_no_binding_leaves_path_untouched():
    assert _maybe_rewrite_image_url("/tmp/x.png") == "/tmp/x.png"


def test_absolute_host_path_is_rewritten(tmp_path):
    set_image_serve_base_url(BASE)
    p = str(Path(tmp_path) / "image_20260823_ab12cd34.png")
    out = _maybe_rewrite_image_url(p)
    assert out.startswith(BASE + "/images/")
    assert out.endswith("image_20260823_ab12cd34.png")


def test_relative_path_passes_through():
    set_image_serve_base_url(BASE)
    assert _maybe_rewrite_image_url("cache/images/x.png") == "cache/images/x.png"


def test_http_urls_pass_through():
    set_image_serve_base_url(BASE)
    url = "https://cdn.example.com/img.png"
    assert _maybe_rewrite_image_url(url) == url


@pytest.mark.linux_only
def test_posix_absolute_literal_rewritten():
    set_image_serve_base_url(BASE)
    assert (
        _maybe_rewrite_image_url("/home/u/.hermes/cache/images/a.png")
        == f"{BASE}/images/a.png"
    )


@pytest.mark.windows_only
def test_windows_drive_path_rewritten():
    set_image_serve_base_url(BASE)
    out = _maybe_rewrite_image_url(r"C:\Users\u\.hermes\cache\images\a.png")
    assert out == f"{BASE}/images/a.png"


def test_reset_restores_previous_state():
    assert _IMAGE_SERVE_BASE_URL.get() is None
    token = set_image_serve_base_url(BASE)
    reset_image_serve_base_url(token)
    assert _IMAGE_SERVE_BASE_URL.get() is None


def test_success_response_rewrites_image_field():
    set_image_serve_base_url(BASE)
    p = str(Path.cwd().resolve() / "b64_provider.png")
    payload = success_response(
        image=p,
        model="m",
        prompt="p",
        aspect_ratio="1:1",
        provider="openai",
    )
    assert payload["image"].startswith(BASE + "/images/")


def test_success_response_without_binding_keeps_raw_value():
    raw = "https://cdn.example.com/x.png"
    payload = success_response(
        image=raw,
        model="m",
        prompt="p",
        aspect_ratio="1:1",
        provider="xai",
    )
    assert payload["image"] == raw


def test_non_string_image_passes_through():
    set_image_serve_base_url(BASE)
    assert _maybe_rewrite_image_url(None) is None
