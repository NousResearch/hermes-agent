"""Ported from clawdbot-main/scripts/tools — run via: py -3 scripts/openclaw_ports/<name>

VOICEVOX ENGINE HTTP verification (version + optional synthesis).
Caption: English log lines for tooling; user-facing messages may be Japanese.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc,no-redef]
        if iterable is None:
            return range(kwargs.get("total", 0))
        return iterable


def setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def probe_version_only(endpoint: str, timeout: float = 5.0) -> bool:
    try:
        url = f"{endpoint.rstrip('/')}/version"
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except OSError:
        return False


def verify_voicevox(endpoint: str = "http://localhost:50021", speaker_id: int = 2) -> bool:
    try:
        url_version = f"{endpoint.rstrip('/')}/version"
        logging.info("Checking version at %s", url_version)
        v_resp = requests.get(url_version, timeout=5)
        if v_resp.status_code != 200:
            logging.error("VOICEVOX Version check failed: %d", v_resp.status_code)
            return False
        logging.info("VOICEVOX Version: %s", v_resp.text)

        text = "はくあ、顕現中。システムオールグリーン。"
        logging.info("Testing synthesis buffer generation...")
        url_query = f"{endpoint.rstrip('/')}/audio_query"
        params = {"text": text, "speaker": speaker_id}
        q_resp = requests.post(url_query, params=params, timeout=10)
        if q_resp.status_code != 200:
            logging.error("Audio query failed: %d", q_resp.status_code)
            return False
        query_data = q_resp.json()

        url_synth = f"{endpoint.rstrip('/')}/synthesis"
        s_resp = requests.post(
            url_synth,
            params={"speaker": speaker_id},
            data=json.dumps(query_data),
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        if s_resp.status_code != 200:
            logging.error("Synthesis failed: %d", s_resp.status_code)
            return False

        audio_data = s_resp.content
        if audio_data[:4] == b"RIFF":
            logging.info("VOICEVOX synthesis OK (RIFF). Audio substrate reactive.")
            return True
        logging.error("Synthesis result missing valid RIFF header.")
        return False
    except Exception as e:
        logging.error("VOICEVOX detection failure: %s", e)
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Verify VOICEVOX ENGINE HTTP API.")
    p.add_argument("--endpoint", default=os.getenv("VOICEVOX_ENDPOINT", "http://127.0.0.1:50021"))
    p.add_argument("--speaker", type=int, default=int(os.getenv("VOICEVOX_SPEAKER_ID", "2")))
    p.add_argument("--probe-only", action="store_true", help="GET /version only (fast).")
    p.add_argument("--quiet", action="store_true", help="Less log noise.")
    p.add_argument("--wait-seconds", type=int, default=0, help="Retry until success or timeout (uses tqdm).")
    args = p.parse_args()

    setup_logging(args.quiet)
    ep = args.endpoint.rstrip("/")

    if args.wait_seconds > 0:
        for _ in tqdm(range(args.wait_seconds), desc="VOICEVOX wait", unit="s"):
            if args.probe_only:
                ok = probe_version_only(ep)
            else:
                ok = verify_voicevox(ep, args.speaker)
            if ok:
                return 0
            time.sleep(1)
        return 1

    if args.probe_only:
        return 0 if probe_version_only(ep) else 1

    return 0 if verify_voicevox(ep, args.speaker) else 1


if __name__ == "__main__":
    sys.exit(main())
