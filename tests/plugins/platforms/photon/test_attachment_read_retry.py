from __future__ import annotations

import json
import subprocess
from pathlib import Path


_HELPER = Path("plugins/platforms/photon/sidecar/attachment-read.mjs").resolve()


def _run_node(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_attachment_read_retries_with_linear_backoff() -> None:
    payload = _run_node(
        f"""
        import {{ readBinaryContentWithRetry }} from {json.dumps(_HELPER.as_uri())};
        let calls = 0;
        const delays = [];
        const logs = [];
        const value = await readBinaryContentWithRetry(
          {{
            async read() {{
              calls += 1;
              if (calls < 3) throw new Error(`reset-${{calls}}`);
              return Buffer.from("ok");
            }}
          }},
          {{
            label: "attachment photo.heic",
            attempts: 3,
            retryMs: 25,
            sleep: async (delay) => delays.push(delay),
            log: (message) => logs.push(message),
          }}
        );
        console.log(JSON.stringify({{
          calls,
          delays,
          logs,
          value: value.toString("utf8"),
        }}));
        """
    )

    assert payload["calls"] == 3
    assert payload["delays"] == [25, 50]
    assert len(payload["logs"]) == 2
    assert payload["value"] == "ok"


def test_attachment_read_raises_after_attempt_budget() -> None:
    payload = _run_node(
        f"""
        import {{ readBinaryContentWithRetry }} from {json.dumps(_HELPER.as_uri())};
        let calls = 0;
        let errorMessage = null;
        try {{
          await readBinaryContentWithRetry(
            {{ async read() {{ calls += 1; throw new Error("connection reset"); }} }},
            {{ attempts: 2, retryMs: 0, log: () => undefined }}
          );
        }} catch (error) {{
          errorMessage = error.message;
        }}
        console.log(JSON.stringify({{ calls, errorMessage }}));
        """
    )

    assert payload == {"calls": 2, "errorMessage": "connection reset"}
