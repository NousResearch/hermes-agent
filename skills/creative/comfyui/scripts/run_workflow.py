#!/usr/bin/env python3
"""
run_workflow.py — Inject parameters into a ComfyUI workflow, submit it, monitor execution,
and download outputs.

Usage:
    # Local server
    python3 run_workflow.py --workflow workflow_api.json \
        --args '{"prompt": "a cat", "seed": 42}' \
        --output-dir ./outputs

    # Cloud server
    python3 run_workflow.py --workflow workflow_api.json \
        --args '{"prompt": "a cat"}' \
        --host https://cloud.comfy.org \
        --api-key comfyui-xxxxxxx \
        --output-dir ./outputs

    # With schema file (pre-extracted)
    python3 run_workflow.py --workflow workflow_api.json \
        --schema schema.json \
        --args '{"prompt": "a cat"}' \
        --output-dir ./outputs

Requires: Python 3.10+, requests (or urllib as fallback)
"""

import json
import sys
import time
import uuid
import copy
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlencode, urlparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error


def http_get(url: str, headers: dict = None, follow_redirects: bool = True) -> tuple:
    """GET request, returns (status_code, body_bytes, response_headers)."""
    if HAS_REQUESTS:
        r = requests.get(url, headers=headers or {}, allow_redirects=follow_redirects, timeout=30)
        return r.status_code, r.content, dict(r.headers)
    else:
        req = urllib.request.Request(url, headers=headers or {})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)


def http_post(url: str, data: dict, headers: dict = None) -> tuple:
    """POST JSON request, returns (status_code, response_dict)."""
    payload = json.dumps(data).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if HAS_REQUESTS:
        r = requests.post(url, json=data, headers=hdrs, timeout=30)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}
    else:
        req = urllib.request.Request(url, data=payload, headers=hdrs, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


class ComfyRunner:
    def __init__(self, host: str = "http://127.0.0.1:8188", api_key: str = None):
        self.host = host.rstrip("/")
        self.api_key = api_key
        parsed_host = urlparse(self.host).hostname or ""
        self.is_cloud = parsed_host.lower() == "cloud.comfy.org" or api_key is not None
        self.client_id = str(uuid.uuid4())

    @property
    def headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def api_url(self, path: str) -> str:
        """Build URL. Cloud uses /api prefix for some endpoints."""
        if self.is_cloud and not path.startswith("/api"):
            # Cloud endpoints: /api/prompt, /api/view, /api/job, /api/queue
            return f"{self.host}/api{path}"
        return f"{self.host}{path}"

    def check_server(self) -> bool:
        """Check if server is reachable."""
        try:
            url = self.api_url("/system_stats") if not self.is_cloud else f"{self.host}/api/system_stats"
            status, _, _ = http_get(url, self.headers)
            return status == 200
        except Exception:
            return False

    def submit(self, workflow: dict) -> dict:
        """Submit workflow for execution. Returns {prompt_id, node_errors}."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        if self.api_key and self.is_cloud:
            payload.setdefault("extra_data", {})["api_key_comfy_org"] = self.api_key
        url = self.api_url("/prompt")
        status, resp = http_post(url, payload, self.headers)
        if status != 200:
            return {"error": f"HTTP {status}", "details": resp}
        return resp

    def poll_status(self, prompt_id: str, timeout: int = 120) -> dict:
        """Poll until job completes. Returns final status dict."""
        start = time.time()
        poll_interval = 2.0

        while time.time() - start < timeout:
            if self.is_cloud:
                # Cloud has a dedicated status endpoint
                url = f"{self.host}/api/job/{prompt_id}/status"
                status, body, _ = http_get(url, self.headers)
                if status == 200:
                    data = json.loads(body) if isinstance(body, bytes) else body
                    job_status = data.get("status", "unknown")
                    if job_status == "completed":
                        return {"status": "success", "data": data}
                    elif job_status == "failed":
                        return {"status": "error", "data": data}
                    elif job_status == "cancelled":
                        return {"status": "cancelled", "data": data}
                    # still running, continue polling
            else:
                # Local: check /history/{prompt_id}
                url = f"{self.host}/history/{prompt_id}"
                status, body, _ = http_get(url, self.headers)
                if status == 200:
                    data = json.loads(body) if isinstance(body, bytes) else body
                    if prompt_id in data:
                        entry = data[prompt_id]
                        if entry.get("status", {}).get("completed", False):
                            return {"status": "success", "outputs": entry.get("outputs", {})}
                        if entry.get("status", {}).get("status_str") == "error":
                            return {"status": "error", "data": entry}

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, 10.0)

        return {"status": "timeout", "elapsed": time.time() - start}

    def get_outputs(self, prompt_id: str) -> dict:
        """Get output file info from history."""
        if self.is_cloud:
            url = f"{self.host}/api/job/{prompt_id}/status"
        else:
            url = f"{self.host}/history/{prompt_id}"
        status, body, _ = http_get(url, self.headers)
        if status != 200:
            return {}
        data = json.loads(body) if isinstance(body, bytes) else body
        if self.is_cloud:
            return data.get("outputs", {})
        if prompt_id in data:
            return data[prompt_id].get("outputs", {})
        return {}

    def download_output(self, filename: str, subfolder: str, file_type: str, output_dir: Path) -> Path:
        """Download a single output file."""
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
        url = self.api_url(f"/view?{params}")
        status, body, _ = http_get(url, self.headers, follow_redirects=True)
        if status != 200:
            raise RuntimeError(f"Failed to download {filename}: HTTP {status}")
        out_path = output_dir / filename
        out_path.write_bytes(body)
        return out_path


def load_schema(schema_path: str = None, workflow: dict = None) -> dict:
    """Load or generate parameter schema."""
    if schema_path:
        with open(schema_path) as f:
            return json.load(f)
    # Inline extraction (same logic as extract_schema.py but simplified)
    if workflow is None:
        return {"parameters": {}}
    # Import from sibling script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from extract_schema import extract_schema
    return extract_schema(workflow)


def inject_params(workflow: dict, schema: dict, args: dict) -> dict:
    """Inject user parameters into workflow based on schema mapping."""
    wf = copy.deepcopy(workflow)
    params = schema.get("parameters", {})

    for param_name, value in args.items():
        if param_name not in params:
            print(f"Warning: unknown parameter '{param_name}', skipping", file=sys.stderr)
            continue
        mapping = params[param_name]
        node_id = mapping["node_id"]
        field = mapping["field"]
        if node_id in wf and "inputs" in wf[node_id]:
            wf[node_id]["inputs"][field] = value
        else:
            print(f"Warning: node {node_id} not found in workflow", file=sys.stderr)

    return wf


def main():
    parser = argparse.ArgumentParser(description="Run a ComfyUI workflow with parameter injection")
    parser.add_argument("--workflow", required=True, help="Path to workflow API JSON file")
    parser.add_argument("--args", default="{}", help="JSON parameters to inject")
    parser.add_argument("--schema", help="Path to schema JSON (from extract_schema.py). Auto-generated if omitted.")
    parser.add_argument("--host", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--api-key", help="API key for cloud (X-API-Key)")
    parser.add_argument("--output-dir", default="./outputs", help="Directory to save outputs")
    parser.add_argument("--timeout", type=int, default=120, help="Max seconds to wait for completion")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading outputs")
    parser.add_argument("--submit-only", action="store_true", help="Submit and return prompt_id without waiting")
    args = parser.parse_args()

    # Load workflow
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(json.dumps({"error": f"Workflow file not found: {args.workflow}"}))
        sys.exit(1)
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Validate format
    if "nodes" in workflow and "links" in workflow:
        print(json.dumps({"error": "Workflow is in editor format, not API format. Re-export with 'Save (API Format)'."}))
        sys.exit(1)

    # Parse user args
    try:
        user_args = json.loads(args.args)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid --args JSON: {e}"}))
        sys.exit(1)

    # Load/generate schema and inject params
    schema = load_schema(args.schema, workflow)
    if user_args:
        workflow = inject_params(workflow, schema, user_args)

    # Connect to server
    runner = ComfyRunner(host=args.host, api_key=args.api_key)

    # Check server
    if not runner.check_server():
        print(json.dumps({"error": f"Cannot reach server at {args.host}. Is ComfyUI running?"}))
        sys.exit(1)

    # Submit
    result = runner.submit(workflow)
    if "error" in result:
        print(json.dumps({"error": "Submission failed", "details": result}))
        sys.exit(1)

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(json.dumps({"error": "No prompt_id in response", "response": result}))
        sys.exit(1)

    # Check for node errors
    node_errors = result.get("node_errors", {})
    if node_errors:
        print(json.dumps({"error": "Workflow validation failed", "node_errors": node_errors}))
        sys.exit(1)

    if args.submit_only:
        print(json.dumps({"status": "submitted", "prompt_id": prompt_id}))
        sys.exit(0)

    # Poll for completion
    print(f"Submitted: {prompt_id}. Waiting...", file=sys.stderr)
    poll_result = runner.poll_status(prompt_id, timeout=args.timeout)

    if poll_result["status"] == "timeout":
        print(json.dumps({"status": "timeout", "prompt_id": prompt_id, "elapsed": poll_result["elapsed"]}))
        sys.exit(1)
    elif poll_result["status"] == "error":
        print(json.dumps({"status": "error", "prompt_id": prompt_id, "details": poll_result.get("data")}))
        sys.exit(1)
    elif poll_result["status"] == "cancelled":
        print(json.dumps({"status": "cancelled", "prompt_id": prompt_id}))
        sys.exit(1)

    # Download outputs
    outputs = poll_result.get("outputs") or runner.get_outputs(prompt_id)
    if args.no_download:
        print(json.dumps({"status": "success", "prompt_id": prompt_id, "outputs": outputs}))
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for node_id, node_output in outputs.items():
        # ComfyUI puts images/videos under "images" key (even for video)
        for key in ("images", "gifs", "videos", "audio"):
            if key not in node_output:
                continue
            for file_info in node_output[key]:
                filename = file_info.get("filename", "")
                subfolder = file_info.get("subfolder", "")
                file_type = file_info.get("type", "output")
                if not filename:
                    continue
                try:
                    out_path = runner.download_output(filename, subfolder, file_type, output_dir)
                    # Detect media type from extension
                    ext = Path(filename).suffix.lower()
                    if ext in (".mp4", ".webm", ".avi", ".mov", ".gif"):
                        media_type = "video"
                    elif ext in (".wav", ".mp3", ".flac", ".ogg"):
                        media_type = "audio"
                    else:
                        media_type = "image"
                    downloaded.append({
                        "file": str(out_path),
                        "node_id": node_id,
                        "type": media_type,
                        "filename": filename,
                    })
                except Exception as e:
                    print(f"Warning: failed to download {filename}: {e}", file=sys.stderr)

    print(json.dumps({
        "status": "success",
        "prompt_id": prompt_id,
        "outputs": downloaded,
    }, indent=2))


if __name__ == "__main__":
    main()
