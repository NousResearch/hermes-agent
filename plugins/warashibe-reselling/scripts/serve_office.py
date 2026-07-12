"""Serve the local MOA office over HTTP for browser/WebView imports."""
from __future__ import annotations
import argparse
import functools
import http.server
import pathlib
import urllib.request
import json
import os


def local_status() -> bytes:
    """Return redacted local MoA status when the gateway route is unavailable."""
    try:
        import yaml
        cfg = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes")) / "config.yaml"
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        preset_name = data.get("moa", {}).get("active_preset") or data.get("moa", {}).get("default_preset") or "hakuapulse-orchestrator"
        preset = data.get("moa", {}).get("presets", {}).get(preset_name, {})
        profiles_dir = cfg.parent / "profiles"
        profile_ids = ["sedori-secretary", "sedori-researcher", "sedori-buyer", "sedori-lister", "sedori-shipper", "sedori-ledger"]
        agents = []
        for profile_id in profile_ids:
            profile_cfg = profiles_dir / profile_id / "config.yaml"
            if profile_cfg.exists():
                profile_data = yaml.safe_load(profile_cfg.read_text(encoding="utf-8")) or {}
                agents.append({"id": profile_id, "provider": profile_data.get("model", {}).get("provider"), "model": profile_data.get("model", {}).get("default"), "load": 0.35})
        return json.dumps({"status": "local-config", "active_preset": preset_name, "aggregator": preset.get("aggregator", {}), "reference_models": preset.get("reference_models", []), "agents": agents}, ensure_ascii=False).encode("utf-8")
    except Exception as exc:
        return json.dumps({"status": "demo", "error": type(exc).__name__, "active_preset": "hakuapulse-orchestrator", "agents": []}).encode("utf-8")


class OfficeHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
    }

    def do_GET(self):
        if self.path == "/api/gateway-status":
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/moa/status", timeout=3) as response:
                    payload = response.read()
            except Exception:
                payload = local_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--directory", default=str(pathlib.Path(__file__).resolve().parents[1] / "web"))
    args = parser.parse_args()
    handler = functools.partial(OfficeHandler, directory=args.directory)
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"MOA office serving at http://{args.host}:{args.port}/office.html", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
