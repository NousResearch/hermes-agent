#!/usr/bin/env python3
"""Repro: Weixin iLink getUploadUrl returns ret:-2 without context_token.

Bug class: #64704 / #70776 / #70792 — outbound media (images, files,
voice) fails with ret: -2 (parameter validation error) because the
iLink getUploadUrl endpoint requires context_token for upload
authorization.

On main: FAILS (context_token missing from payload). With the fix:
PASSES (context_token included when available).
"""
import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gateway.platforms.weixin import _get_upload_url  # noqa: E402

sig = inspect.signature(_get_upload_url)
params = list(sig.parameters.keys())

if "context_token" not in params:
    print(f"FAIL: _get_upload_url has no context_token param: {params}")
    sys.exit(1)

# Verify the payload builder threads it through
src = inspect.getsource(_get_upload_url)
if 'payload["context_token"] = context_token' not in src:
    print("FAIL: context_token not threaded into payload")
    sys.exit(1)

print(f"PASS: _get_upload_url accepts context_token (params: {params})")
print("PASS: payload includes context_token when provided")
sys.exit(0)
