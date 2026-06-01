import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "skills" / "creative" / "p5js" / "scripts" / "export-frames.js"


def test_p5js_exporter_does_not_use_unsafe_chromium_file_flags():
    source = EXPORT_SCRIPT.read_text()

    assert "--disable-web-security" not in source
    assert "--allow-file-access-from-files" not in source
    assert "--no-sandbox" not in source
    assert "--disable-setuid-sandbox" not in source
    assert "file://" not in source


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for exporter smoke test")
def test_p5js_exporter_uses_localhost_and_blocks_outbound_requests(tmp_path):
    node_modules = tmp_path / "node_modules" / "puppeteer"
    node_modules.mkdir(parents=True)
    output_dir = tmp_path / "frames"
    sketch_dir = tmp_path / "sketch"
    sketch_dir.mkdir()
    (sketch_dir / "sketch.html").write_text(
        "<!doctype html><canvas></canvas><script>window._p5Ready = true; function redraw(){}</script>"
    )

    stub_log = tmp_path / "puppeteer.log"
    (node_modules / "index.js").write_text(
        textwrap.dedent(
            """
            const fs = require('fs');
            function log(event) { fs.appendFileSync(process.env.PUPPETEER_STUB_LOG, JSON.stringify(event) + '\\n'); }
            module.exports = {
              launch: async (opts) => {
                log({ event: 'launch', args: opts.args });
                let requestHandler;
                const fakeRequest = (url) => ({
                  url: () => url,
                  continue: () => log({ event: 'request-continued', url }),
                  abort: (reason) => log({ event: 'request-aborted', reason, url }),
                });
                return {
                  newPage: async () => ({
                    setRequestInterception: async (enabled) => log({ event: 'interception', enabled }),
                    on: (name, handler) => { requestHandler = handler; log({ event: 'handler', name }); },
                    setViewport: async (viewport) => log({ event: 'viewport', viewport }),
                    goto: async (url, opts) => {
                      log({ event: 'goto', url, opts });
                      requestHandler(fakeRequest(url));
                      requestHandler(fakeRequest('https://attacker.example/leak'));
                    },
                    waitForSelector: async (selector, opts) => log({ event: 'waitForSelector', selector, opts }),
                    waitForFunction: async (expr, opts) => log({ event: 'waitForFunction', expr, opts }),
                    evaluate: async () => log({ event: 'evaluate' }),
                    $: async (selector) => ({
                      screenshot: async ({ path }) => {
                        fs.writeFileSync(path, 'png');
                        log({ event: 'screenshot', selector, path });
                      },
                    }),
                  }),
                  close: async () => log({ event: 'close' }),
                };
              },
            };
            """
        )
    )

    result = subprocess.run(
        [
            "node",
            str(EXPORT_SCRIPT),
            str(sketch_dir / "sketch.html"),
            "--output",
            str(output_dir),
            "--frames",
            "1",
        ],
        check=True,
        env={
            **os.environ,
            "NODE_PATH": str(tmp_path / "node_modules"),
            "PUPPETEER_STUB_LOG": str(stub_log),
        },
        text=True,
        capture_output=True,
    )

    events = [json.loads(line) for line in stub_log.read_text().splitlines()]
    launch = next(event for event in events if event["event"] == "launch")
    goto = next(event for event in events if event["event"] == "goto")

    assert result.returncode == 0
    assert launch["args"] == ["--disable-gpu", "--disable-dev-shm-usage"]
    assert goto["url"].startswith("http://127.0.0.1:")
    assert goto["url"].endswith("/sketch.html")
    assert {"event": "interception", "enabled": True} in events
    assert {"event": "handler", "name": "request"} in events
    assert {"event": "request-continued", "url": goto["url"]} in events
    assert {
        "event": "request-aborted",
        "reason": "blockedbyclient",
        "url": "https://attacker.example/leak",
    } in events
    assert (output_dir / "frame-0000.png").exists()
