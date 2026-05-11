#!/usr/bin/env python3
"""
run_workflow.py — Inject parameters into a ComfyUI workflow, submit it, monitor
execution, and download outputs.

Improvements over v1:
  - Cloud-aware URL routing (handles /api prefix and /history_v2 / /experiment/models renames)
  - API key from CLI flag OR $COMFY_CLOUD_API_KEY env var
  - WebSocket progress monitoring (--ws), with HTTP polling fallback
  - Streaming download (no whole-file buffering — handles GB-size video outputs)
  - Path-traversal-safe output writes
  - Subfolder-aware download paths (no silent overwrites)
  - Retry with exponential backoff on transient errors
  - Status-error correctly classified before "completed: true"
  - Image upload helper (--input-image NAME=PATH)
  - Auto-randomize seed when value is -1 or omitted on a randomize-seed flag
  - Auto-extends timeout heuristically for video workflows
  - Editor-format detection with helpful error
  - Doesn't pollute extra_data.api_key_comfy_org with the cloud auth key
    unless --partner-key is provided (correct semantic per cloud docs)

Usage:
    # Local server
    python3 run_workflow.py --workflow workflow_api.json \
        --args '{"prompt": "a cat", "seed": 42}' \
        --output-dir ./outputs

    # Cloud server (API key from env var)
    export COMFY_CLOUD_API_KEY="comfyui-xxxxxxx"
    python3 run_workflow.py --workflow workflow_api.json \
        --args '{"prompt": "a cat"}' \
        --host https://cloud.comfy.org \
        --output-dir ./outputs

    # With image input (auto-uploads, then references)
    python3 run_workflow.py --workflow img2img.json \
        --input-image image=./photo.png \
        --args '{"prompt": "make it cyberpunk"}'

    # WebSocket real-time progress
    python3 run_workflow.py --workflow flux_dev.json \
        --args '{"prompt": "..."}' \
        --ws

Stdlib-only by default (Python 3.10+). Will use `requests`/`websocket-client`
if installed for nicer behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

# Local import — _common.py sits next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DEFAULT_LOCAL_HOST, ENV_API_KEY,
    coerce_seed, emit_json, http_get, http_post, http_request,
    is_cloud_host, is_link, log, looks_like_video_workflow,
    media_type_from_filename, new_client_id, resolve_api_key, resolve_url,
    safe_path_join, unwrap_workflow,
)


# =============================================================================
# Runner
# =============================================================================

class WorkflowRunError(Exception):
    """Raised when a workflow run fails (validation, execution, timeout)."""

    def __init__(self, status: str, message: str, **details: Any):
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        d = {"status": self.status, "error": self.message}
        d.update(self.details)
        return d


class ComfyRunner:
    def __init__(
        self,
        host: str = DEFAULT_LOCAL_HOST,
        api_key: str | None = None,
        client_id: str | None = None,
        partner_key: str | None = None,
    ):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.partner_key = partner_key
        self.is_cloud = is_cloud_host(self.host)
        self.client_id = client_id or new_client_id()

    @property
    def headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _url(self, path: str) -> str:
        return resolve_url(self.host, path, is_cloud=self.is_cloud)

    # ---------- server health ----------
    def check_server(self) -> tuple[bool, dict | None]:
        try:
            r = http_get(self._url("/system_stats"), headers=self.headers, retries=2)
            if r.status == 200:
                try:
                    return True, r.json()
                except Exception:
                    return True, None
            return False, {"http_status": r.status, "body": r.text()[:500]}
        except Exception as e:
            return False, {"error": str(e)}

    # ---------- upload ----------
    def upload_image(self, path: Path, *, image_type: str = "input", overwrite: bool = True,
                     endpoint: str = "/upload/image", extra_form: dict | None = None) -> dict:
        """Upload an image file via multipart. Returns server-side ref dict."""
        if not path.exists():
            raise FileNotFoundError(f"input image not found: {path}")
        # Stream the file via a handle to avoid OOM on huge inputs (16MP+ photos).
        with path.open("rb") as fh:
            files = {"image": (path.name, fh)}
            form = {"type": image_type}
            if overwrite:
                form["overwrite"] = "true"
            if extra_form:
                form.update({k: str(v) for k, v in extra_form.items()})
            r = http_request(
                "POST", self._url(endpoint),
                headers=self.headers, files=files, form=form,
                timeout=300, retries=2,
            )
        if r.status != 200:
            raise WorkflowRunError(
                "upload_failed",
                f"Upload of {path.name} failed: HTTP {r.status}",
                body=r.text()[:500],
            )
        try:
            return r.json()
        except Exception:
            return {"name": path.name}

    def upload_mask(self, path: Path, original_ref: dict) -> dict:
        """Upload an inpaint mask, linked to a previously uploaded source image.

        `original_ref` should be the dict returned by `upload_image()` for the
        source image (or `{"filename": ..., "subfolder": ..., "type": "input"}`).
        """
        return self.upload_image(
            path,
            endpoint="/upload/mask",
            extra_form={
                "subfolder": "clipspace",
                "original_ref": json.dumps(original_ref),
            },
        )

    # ---------- submit ----------
    def submit(self, workflow: dict) -> dict:
        payload: dict[str, Any] = {"prompt": workflow, "client_id": self.client_id}
        if self.partner_key:
            payload["extra_data"] = {"api_key_comfy_org": self.partner_key}

        r = http_post(self._url("/prompt"), headers=self.headers, json_body=payload, timeout=120)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text()[:500]}
        if r.status != 200:
            return {"_http_error": r.status, "body": body}
        return body

    # ---------- HTTP polling ----------
    def poll_status(self, prompt_id: str, *, timeout: float = 300.0,
                    initial_interval: float = 1.5, max_interval: float = 8.0) -> dict:
        start = time.time()
        interval = initial_interval

        while time.time() - start < timeout:
            if self.is_cloud:
                r = http_get(
                    self._url(f"/job/{prompt_id}/status"),
                    headers=self.headers, retries=2, timeout=30,
                )
                if r.status == 200:
                    try:
                        data = r.json()
                    except Exception:
                        data = {}
                    s = data.get("status")
                    if s == "completed":
                        return {"status": "success", "data": data}
                    if s in ("failed",):
                        return {"status": "error", "data": data}
                    if s == "cancelled":
                        return {"status": "cancelled", "data": data}
                    # pending / in_progress → continue
                elif r.status == 404:
                    # Cloud sometimes 404s briefly between submit and dispatcher pickup
                    pass
                else:
                    # transient error — retry loop covers it
                    pass
            else:
                # Local: /history/{id} grows once execution completes
                r = http_get(
                    self._url(f"/history/{prompt_id}"),
                    headers=self.headers, retries=2, timeout=30,
                )
                if r.status == 200:
                    try:
                        data = r.json() or {}
                    except Exception:
                        data = {}
                    entry = data.get(prompt_id)
                    if isinstance(entry, dict):
                        st = entry.get("status") or {}
                        # IMPORTANT: check error first — `completed: true` can coexist with errors
                        status_str = st.get("status_str")
                        if status_str == "error":
                            return {"status": "error", "data": entry}
                        if st.get("completed", False):
                            return {"status": "success", "outputs": entry.get("outputs", {})}
                # not in history yet → continue polling

            time.sleep(interval)
            interval = min(max_interval, interval * 1.4)

        return {"status": "timeout", "elapsed": time.time() - start}

    # ---------- WebSocket monitoring ----------
    def monitor_ws(self, prompt_id: str, *, timeout: float = 300.0,
                   on_progress: Any = None) -> dict:
        """Connect to /ws and listen until execution_success / execution_error.

        Falls back to HTTP polling if `websocket-client` is not installed.
        Returns same shape as poll_status.
        """
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError:
            log("websocket-client not installed; falling back to HTTP polling")
            return self.poll_status(prompt_id, timeout=timeout)

        # Build WS URL. Preserve any base-path components the user gave us
        # (e.g. http://example.com/comfyui → ws://example.com/comfyui/ws).
        parsed = urlparse(self.host)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        base_path = parsed.path.rstrip("/")
        ws_url = f"{scheme}://{netloc}{base_path}/ws?clientId={self.client_id}"
        if self.is_cloud and self.api_key:
            ws_url += f"&token={self.api_key}"

        outputs: dict[str, Any] = {}
        error_payload: dict[str, Any] | None = None
        success = False
        seen_executed = False

        ws = websocket.create_connection(ws_url, timeout=timeout)
        try:
            ws.settimeout(timeout)
            deadline = time.time() + timeout
            while time.time() < deadline:
                msg = ws.recv()
                if isinstance(msg, bytes):
                    # Binary preview frame — ignore for now; ws_monitor.py prints them
                    continue
                try:
                    payload = json.loads(msg)
                except Exception:
                    continue
                mtype = payload.get("type", "")
                mdata = payload.get("data", {}) or {}

                # Filter to our job (cloud broadcasts; local filters via client_id)
                pid = mdata.get("prompt_id")
                if pid is not None and pid != prompt_id:
                    continue

                if mtype == "progress":
                    if callable(on_progress):
                        on_progress({
                            "type": "progress",
                            "value": mdata.get("value"),
                            "max": mdata.get("max"),
                            "node": mdata.get("node"),
                        })
                elif mtype == "progress_state":
                    if callable(on_progress):
                        on_progress({"type": "progress_state", "nodes": mdata.get("nodes", {})})
                elif mtype == "executing":
                    node = mdata.get("node")
                    if callable(on_progress):
                        on_progress({"type": "executing", "node": node})
                    # When `node` is None on a local server, that signals end-of-run
                    if node is None and not self.is_cloud and seen_executed:
                        success = True
                        break
                elif mtype == "executed":
                    seen_executed = True
                    nid = mdata.get("node")
                    out = mdata.get("output") or {}
                    if nid:
                        outputs[nid] = out
                elif mtype == "notification":
                    if callable(on_progress):
                        on_progress({"type": "notification", "message": mdata.get("value", "")})
                elif mtype == "execution_success":
                    success = True
                    break
                elif mtype == "execution_error":
                    error_payload = mdata
                    break
                elif mtype == "execution_interrupted":
                    error_payload = {"interrupted": True, **mdata}
                    break
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if error_payload is not None:
            return {"status": "error", "data": error_payload}
        if success:
            return {"status": "success", "outputs": outputs}
        return {"status": "timeout", "elapsed": timeout}

    # ---------- outputs ----------
    def get_outputs(self, prompt_id: str) -> dict:
        if self.is_cloud:
            # Try /jobs/{id} first (returns full job with outputs); fall back to /history_v2
            r = http_get(self._url(f"/jobs/{prompt_id}"), headers=self.headers, retries=2)
            if r.status == 200:
                try:
                    return (r.json() or {}).get("outputs", {}) or {}
                except Exception:
                    pass
            # Fallback
            r = http_get(self._url(f"/history/{prompt_id}"), headers=self.headers, retries=2)
            if r.status == 200:
                try:
                    body = r.json() or {}
                except Exception:
                    body = {}
                if isinstance(body, dict) and prompt_id in body:
                    return body[prompt_id].get("outputs", {}) or {}
                if isinstance(body, dict) and "outputs" in body:
                    return body["outputs"] or {}
            return {}
        # Local
        r = http_get(self._url(f"/history/{prompt_id}"), headers=self.headers, retries=2)
        if r.status != 200:
            return {}
        try:
            body = r.json() or {}
        except Exception:
            return {}
        entry = body.get(prompt_id) or {}
        return entry.get("outputs", {}) or {}

    def download_output(
        self, *, filename: str, subfolder: str, file_type: str,
        output_dir: Path, preserve_subfolder: bool = True, overwrite: bool = False,
    ) -> Path:
        """Stream a single output to disk. Path-traversal-safe."""
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        url = self._url("/view") + "?" + urlencode(params)

        # Compute target path safely. If preserve_subfolder, include subfolder in the
        # local path; otherwise put the file in output_dir flat.
        target_parts: list[str] = []
        if preserve_subfolder and subfolder:
            target_parts.extend(p for p in subfolder.split("/") if p and p not in (".", ".."))
        target_parts.append(filename)
        out_path = safe_path_join(output_dir, *target_parts)

        if out_path.exists() and not overwrite:
            stem, suffix = out_path.stem, out_path.suffix
            i = 1
            while True:
                candidate = out_path.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    out_path = candidate
                    break
                i += 1

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download. Two-step for cloud: get the 302, then fetch signed URL
        # so we don't accidentally send X-API-Key to the storage backend.
        # The HTTP transport already strips X-API-Key on cross-host redirect
        # via _strip_api_key_on_redirect, so a single follow_redirects=True call
        # is safe AND simpler.
        r = http_request(
            "GET", url, headers=self.headers,
            timeout=600, retries=3, follow_redirects=True,
            stream=True, sink=out_path,
        )
        if r.status != 200:
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            raise WorkflowRunError(
                "download_failed",
                f"Download of {filename} failed: HTTP {r.status}",
                url=url,
            )
        return out_path

    # ---------- queue / cancel ----------
    def cancel(self, prompt_id: str | None = None) -> bool:
        if prompt_id:
            r = http_post(
                self._url("/queue"), headers=self.headers,
                json_body={"delete": [prompt_id]}, retries=1,
            )
            return r.status == 200
        # Interrupt currently running
        r = http_post(self._url("/interrupt"), headers=self.headers, retries=1)
        return r.status == 200


# =============================================================================
# Schema / parameter injection
# =============================================================================

def _inline_schema(workflow: dict) -> dict:
    """Generate schema using the sibling extract_schema module."""
    from extract_schema import extract_schema  # noqa: WPS433
    return extract_schema(workflow)


def load_schema(schema_path: str | None, workflow: dict) -> dict:
    if schema_path:
        with open(schema_path) as f:
            return json.load(f)
    return _inline_schema(workflow)


def inject_params(
    workflow: dict, schema: dict, args: dict,
    *, randomize_seed_if_unset: bool = False,
) -> tuple[dict, list[str]]:
    """Inject user args into the workflow. Returns (new_workflow, warnings)."""
    wf = copy.deepcopy(workflow)
    params = schema.get("parameters", {}) or {}
    warnings: list[str] = []

    # Auto-randomize seed when it's -1 in args, or when randomize_seed_if_unset
    # and user didn't pass a seed.
    if "seed" in params:
        if "seed" in args and args["seed"] in (None, -1, "-1"):
            args = dict(args)
            args["seed"] = coerce_seed(args["seed"])
            warnings.append(f"seed=-1 expanded to {args['seed']}")
        elif randomize_seed_if_unset and "seed" not in args:
            args = dict(args)
            args["seed"] = coerce_seed(None)
            warnings.append(f"seed auto-randomized to {args['seed']}")

    for name, value in args.items():
        if name not in params:
            warnings.append(f"unknown parameter '{name}' (not in schema), skipping")
            continue
        m = params[name]
        nid, field = m["node_id"], m["field"]
        node = wf.get(nid)
        if not isinstance(node, dict) or "inputs" not in node:
            warnings.append(f"node '{nid}' for parameter '{name}' missing in workflow")
            continue
        # Refuse to overwrite a link with a literal — would silently break wiring
        cur = node["inputs"].get(field)
        if is_link(cur):
            warnings.append(
                f"parameter '{name}' targets {nid}.{field} which is currently a link; "
                f"refusing to overwrite (set the schema to point at the source node instead)"
            )
            continue
        node["inputs"][field] = value

    return wf, warnings


# =============================================================================
# Output download helper
# =============================================================================

def download_outputs(
    runner: ComfyRunner, outputs: dict, output_dir: Path,
    *, preserve_subfolder: bool = True, overwrite: bool = False,
) -> list[dict]:
    """Walk the outputs dict and download every file. Cloud uses `video` (singular);
    local uses `videos` (plural). We accept both."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []

    OUTPUT_KEYS = ("images", "gifs", "videos", "video", "audio", "files", "models", "3d")

    for node_id, node_output in (outputs or {}).items():
        if not isinstance(node_output, dict):
            continue
        for key in OUTPUT_KEYS:
            entries = node_output.get(key)
            if not entries:
                continue
            if not isinstance(entries, list):
                entries = [entries]
            for fi in entries:
                if not isinstance(fi, dict):
                    continue
                filename = fi.get("filename") or ""
                if not filename:
                    continue
                subfolder = fi.get("subfolder") or ""
                file_type = fi.get("type") or "output"
                try:
                    out_path = runner.download_output(
                        filename=filename, subfolder=subfolder, file_type=file_type,
                        output_dir=output_dir, preserve_subfolder=preserve_subfolder,
                        overwrite=overwrite,
                    )
                    downloaded.append({
                        "file": str(out_path),
                        "node_id": node_id,
                        "type": media_type_from_filename(filename),
                        "filename": filename,
                        "subfolder": subfolder,
                        "source_type": file_type,
                    })
                except Exception as e:
                    log(f"WARN: failed to download {filename}: {e}")
    return downloaded


# =============================================================================
# CLI
# =============================================================================

def parse_input_image_arg(spec: str) -> tuple[str, Path]:
    """Parse `name=path` (or `path` alone, defaulting to name='image')."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path).expanduser()
    return "image", Path(spec).expanduser()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run a ComfyUI workflow with parameter injection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workflow", required=True, help="Path to workflow API JSON file")
    p.add_argument("--args", default="{}",
                   help="JSON parameters to inject (or `@/path/to/args.json`)")
    p.add_argument("--schema", help="Path to schema JSON (auto-generated if omitted)")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST, help="ComfyUI server URL")
    p.add_argument("--api-key",
                   help=f"API key for cloud (or set ${ENV_API_KEY} env var)")
    p.add_argument("--partner-key",
                   help="Partner-node API key (extra_data.api_key_comfy_org). "
                        "Required for Flux Pro / Ideogram / etc. Defaults to --api-key if not set.")
    p.add_argument("--output-dir", default="./outputs", help="Directory to save outputs")
    p.add_argument("--timeout", type=int, default=0,
                   help="Max seconds to wait (0=auto: 300 / 900 for video workflows)")
    p.add_argument("--input-image", action="append", default=[],
                   help="Upload local image before running. Format: `name=path` or `path`. "
                        "The `name` becomes the value injected into the matching schema parameter.")
    p.add_argument("--randomize-seed", action="store_true",
                   help="If schema has a 'seed' parameter and --args didn't set one, randomize it")
    p.add_argument("--ws", action="store_true",
                   help="Use WebSocket for real-time progress (requires `websocket-client`)")
    p.add_argument("--no-download", action="store_true", help="Skip downloading outputs")
    p.add_argument("--flat-output", action="store_true",
                   help="Don't preserve server-side subfolder structure when saving outputs")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing files instead of appending _1, _2, ...")
    p.add_argument("--submit-only", action="store_true",
                   help="Submit and return prompt_id without waiting")
    p.add_argument("--client-id", help="Override generated client_id (UUID)")
    p.add_argument("--use-partner-key-as-auth", action="store_true",
                   help="(Compat) Use --partner-key value as cloud X-API-Key. Don't use unless you know why.")

    args = p.parse_args(argv)

    # ---- Load workflow ----
    wf_path = Path(args.workflow).expanduser()
    if not wf_path.exists():
        emit_json({"error": f"Workflow file not found: {args.workflow}"})
        return 1
    try:
        with wf_path.open() as f:
            workflow_raw = json.load(f)
        workflow = unwrap_workflow(workflow_raw)
    except ValueError as e:
        emit_json({"error": str(e)})
        return 1
    except json.JSONDecodeError as e:
        emit_json({"error": f"Invalid JSON in workflow file: {e}"})
        return 1

    # ---- Parse user args ----
    args_str = args.args
    if args_str.startswith("@"):
        try:
            args_str = Path(args_str[1:]).read_text()
        except OSError as e:
            emit_json({"error": f"Cannot read args file: {e}"})
            return 1
    try:
        user_args = json.loads(args_str) if args_str.strip() else {}
    except json.JSONDecodeError as e:
        emit_json({"error": f"Invalid --args JSON: {e}"})
        return 1
    if not isinstance(user_args, dict):
        emit_json({"error": "--args must be a JSON object"})
        return 1

    # ---- Resolve API key ----
    api_key = resolve_api_key(args.api_key)
    partner_key = args.partner_key or None
    if args.use_partner_key_as_auth and not api_key and partner_key:
        api_key = partner_key

    # ---- Connect ----
    runner = ComfyRunner(
        host=args.host, api_key=api_key, partner_key=partner_key,
        client_id=args.client_id,
    )

    # Server reachability
    ok, info = runner.check_server()
    if not ok:
        emit_json({
            "error": f"Cannot reach server at {args.host}",
            "details": info,
            "hint": (
                "Check `comfy launch --background` is running for local, "
                f"or set ${ENV_API_KEY} for cloud."
            ),
        })
        return 1

    # ---- Upload input images ----
    upload_warnings: list[str] = []
    for spec in args.input_image:
        try:
            param_name, path = parse_input_image_arg(spec)
        except Exception as e:
            emit_json({"error": f"Bad --input-image spec '{spec}': {e}"})
            return 1
        try:
            ref = runner.upload_image(path)
        except Exception as e:
            emit_json({"error": f"Upload failed for {path}: {e}"})
            return 1
        # Register as a user arg so inject_params consumes it through the schema
        uploaded_name = ref.get("name") or path.name
        if param_name not in user_args:
            user_args[param_name] = uploaded_name

    # ---- Inject params ----
    schema = load_schema(args.schema, workflow)
    workflow, inj_warnings = inject_params(
        workflow, schema, user_args, randomize_seed_if_unset=args.randomize_seed,
    )
    warnings = upload_warnings + inj_warnings
    for w in warnings:
        log(f"WARN: {w}")

    # ---- Submit ----
    submit_resp = runner.submit(workflow)
    if "_http_error" in submit_resp:
        emit_json({
            "error": "Submission HTTP error",
            "http_status": submit_resp["_http_error"],
            "body": submit_resp.get("body"),
        })
        return 1

    if isinstance(submit_resp.get("error"), dict):
        emit_json({
            "error": "Workflow validation failed",
            "details": submit_resp["error"],
            "node_errors": submit_resp.get("node_errors"),
        })
        return 1

    prompt_id = submit_resp.get("prompt_id")
    if not prompt_id:
        emit_json({"error": "No prompt_id in submit response", "response": submit_resp})
        return 1

    node_errors = submit_resp.get("node_errors") or {}
    if node_errors:
        emit_json({"error": "Workflow validation failed", "node_errors": node_errors})
        return 1

    if args.submit_only:
        emit_json({"status": "submitted", "prompt_id": prompt_id, "warnings": warnings})
        return 0

    # ---- Wait ----
    timeout = args.timeout
    if timeout <= 0:
        timeout = 900 if looks_like_video_workflow(workflow) else 300

    log(f"Submitted: prompt_id={prompt_id}, waiting (timeout={timeout}s)…")

    def _on_progress(evt: dict) -> None:
        t = evt.get("type")
        if t == "progress":
            log(f"  step {evt.get('value')}/{evt.get('max')} on node {evt.get('node')}")
        elif t == "executing":
            node = evt.get("node")
            if node:
                log(f"  executing node {node}")

    try:
        if args.ws:
            wait_result = runner.monitor_ws(prompt_id, timeout=timeout, on_progress=_on_progress)
        else:
            wait_result = runner.poll_status(prompt_id, timeout=timeout)
    except KeyboardInterrupt:
        log(f"Interrupted — cancelling job {prompt_id} on server…")
        try:
            runner.cancel(prompt_id)
        except Exception as e:
            log(f"  (cancel request failed: {e})")
        emit_json({
            "status": "interrupted",
            "prompt_id": prompt_id,
            "note": "Ctrl+C received; sent cancellation to server.",
        })
        return 130

    if wait_result["status"] == "timeout":
        emit_json({
            "status": "timeout",
            "prompt_id": prompt_id,
            "elapsed": wait_result.get("elapsed"),
            "hint": "Re-run with larger --timeout, or use --submit-only and check later.",
        })
        return 1
    if wait_result["status"] == "error":
        emit_json({"status": "error", "prompt_id": prompt_id, "details": wait_result.get("data")})
        return 1
    if wait_result["status"] == "cancelled":
        emit_json({"status": "cancelled", "prompt_id": prompt_id})
        return 1

    # ---- Outputs ----
    outputs = wait_result.get("outputs")
    if not outputs:
        outputs = runner.get_outputs(prompt_id)

    if args.no_download:
        emit_json({
            "status": "success", "prompt_id": prompt_id,
            "outputs": outputs, "warnings": warnings,
        })
        return 0

    downloaded = download_outputs(
        runner, outputs, Path(args.output_dir).expanduser(),
        preserve_subfolder=not args.flat_output, overwrite=args.overwrite,
    )

    emit_json({
        "status": "success",
        "prompt_id": prompt_id,
        "outputs": downloaded,
        "warnings": warnings,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
