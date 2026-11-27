#!/usr/bin/env python3
"""yt_proof.py — End-to-end YouTube video playback and frame capture proof in native Chromium on Termux.

Credits: @pjy010218
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import sys
import time

CDP = ("127.0.0.1", 9222)


def newtab(url: str) -> dict:
    c = http.client.HTTPConnection(*CDP, timeout=10)
    c.request("PUT", "/json/new?" + url)
    res = c.getresponse().read().decode("utf-8", errors="replace")
    c.close()
    return json.loads(res)


def drive(tab: dict):
    import websocket
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=45)
    n = [0]

    def cmd(method: str, **params):
        n[0] += 1
        ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == n[0]:
                return r
    return cmd


def main() -> None:
    try:
        import websocket
    except ImportError:
        print("Error: websocket-client required. Run: pip install websocket-client", file=sys.stderr)
        sys.exit(1)

    print("[1] Opening YouTube watch page...")
    tab = newtab("https://www.youtube.com/watch?v=jNQXAC9IVRw")
    cmd = drive(tab)
    time.sleep(12)

    r = cmd("Runtime.evaluate", expression="""
    JSON.stringify({
      title: document.title.slice(0, 80),
      channel: (document.querySelector('ytd-channel-name a') || {}).innerText || 'n/a',
      playerAlive: !!document.querySelector('#movie_player'),
      videoEl: !!document.querySelector('video'),
      t: (document.querySelector('video') || {}).currentTime || 0,
      dur: (document.querySelector('video') || {}).duration || 0,
      paused: (document.querySelector('video') || {paused: '?'})['paused']
    })
    """, returnByValue=True)
    state = json.loads(r.get("result", {}).get("result", {}).get("value", "{}"))
    for k, v in state.items():
        print(f"    {k}: {v}")

    print("[2] Capturing decoded frame from the <video> element...")
    r = cmd("Runtime.evaluate", expression="""
    (function(){
      const v = document.querySelector('video');
      if (!v || !v.videoWidth) return JSON.stringify({err: 'no decodable video'});
      const cv = document.createElement('canvas');
      cv.width = 640; cv.height = 360;
      cv.getContext('2d').drawImage(v, 0, 0, 640, 360);
      return JSON.stringify({frame: cv.toDataURL('image/jpeg', 0.75).split(',')[1], t: v.currentTime});
    })()
    """, returnByValue=True)
    val = json.loads(r.get("result", {}).get("result", {}).get("value", "{}"))
    if "frame" not in val:
        print("    Frame capture failed:", val)
    else:
        out_frame = os.path.join(os.path.expanduser("~"), "proof_yt_frame.jpg")
        with open(out_frame, "wb") as f:
            f.write(base64.b64decode(val["frame"]))
        print(f"    Frame captured at t={val.get('t', 0):.1f}s -> {out_frame}")

    print("[3] Full-page screenshot...")
    r = cmd("Page.captureScreenshot", format="png")
    data = r.get("result", {}).get("data")
    if data:
        out_page = os.path.join(os.path.expanduser("~"), "proof_yt_page.png")
        with open(out_page, "wb") as f:
            f.write(base64.b64decode(data))
        print(f"    Saved screenshot -> {out_page}")
    else:
        print("    Page screenshot failed:", str(r)[:150])

    print("Verification completed.")


if __name__ == "__main__":
    main()
