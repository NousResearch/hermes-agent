"""冒烟测试: 启动一个本地 mock OpenAI 兼容服务, 验证 probe 全流程。

用法:
    cd /root/hermes-agent && python -m llm_api_probe.tests.smoke_test

会启动 http://127.0.0.1:18080/v1 的 mock 服务, 然后跑 connectivity + speed 两个模块。
"""
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# 当作 llm_api_probe.tests.smoke_test 跑, probes 在同级可相对导入
from pathlib import Path
from llm_api_probe.probes.config import load_config
from llm_api_probe.probes.probe_connectivity import run as run_conn
from llm_api_probe.probes.probe_speed import run as run_speed
from llm_api_probe.probes.report import render_console_table, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== Mock 服务 ====================

class MockHandler(BaseHTTPRequestHandler):
    """极简 OpenAI 兼容 mock: 总是成功, 假装输出 50 tokens。"""

    def log_message(self, *args, **kwargs):
        pass  # 静默

    def _read_body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n).decode() or "{}")

    def _respond_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._respond_json({"data": [{"id": "mock-1"}, {"id": "mock-2"}]})
        else:
            self._respond_json({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_body()
        # 故意延迟 100ms 模拟推理
        time.sleep(0.1)
        model = body.get("model", "mock-1")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # 假装输出 50 token
            for i in range(50):
                chunk = {
                    "id": f"cmpl-{i}",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            # usage chunk
            usage_chunk = {
                "id": "cmpl-final",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 50, "total_tokens": 70},
            }
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._respond_json({
                "id": "cmpl-1",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "我是 mock 模型, 用了 50 tokens。"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 50, "total_tokens": 70},
            })


def start_mock_server(port=18080):
    s = HTTPServer(("127.0.0.1", port), MockHandler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    return s


# ==================== 测试 ====================

def main():
    print("[smoke] 启动 mock server...")
    srv = start_mock_server(18080)
    time.sleep(0.3)

    # 写测试配置
    test_cfg = os.path.join(ROOT, "llm_api_probe", "reports", "smoke-config.yaml")
    os.makedirs(os.path.dirname(test_cfg), exist_ok=True)
    with open(test_cfg, "w") as f:
        f.write(f"""providers:
  - name: mock-provider
    label: 本地 mock
    category: self_hosted
    base_url: http://127.0.0.1:18080/v1
    api_key: test
    models: [mock-1]
""")

    from llm_api_probe.probes.config import load_config
    from llm_api_probe.probes.probe_connectivity import run as run_conn
    from llm_api_probe.probes.probe_speed import run as run_speed

    providers = load_config(test_cfg)
    assert len(providers) == 1, providers

    p = providers[0]

    print("[smoke] 跑 connectivity...")
    r = run_conn(p, "mock-1", verbose=False)
    assert r.ok, f"connectivity failed: {r.error}"
    assert r.metrics, "no metrics"
    print(f"  ✓ {len(r.metrics)} metrics, hello={r.raw.get('hello_response', '')[:40]}")

    print("[smoke] 跑 speed (1 round, concurrency=1)...")
    r = run_speed(p, "mock-1", rounds=1, concurrency=1, verbose=False)
    assert r.ok, f"speed failed: {r.error}"
    metric_names = [m.name for m in r.metrics]
    assert "ttft_median_ms" in metric_names, metric_names
    assert "output_tps_median" in metric_names, metric_names
    print(f"  ✓ ttft={dict(zip(metric_names, [m.value for m in r.metrics])).get('ttft_median_ms')}ms")

    # 跑 report
    print("[smoke] 渲染控制台报告...")
    from llm_api_probe.probes.report import render_console_table, write_json
    results = [run_conn(p, "mock-1", verbose=False), run_speed(p, "mock-1", rounds=1, concurrency=1, verbose=False)]
    table = render_console_table(providers, results)
    assert "mock-provider" in table, table
    print("  ✓ 控制台报告渲染成功")

    out_json = os.path.join(ROOT, "llm_api_probe", "reports", "smoke-report.json")
    write_json(providers, results, Path(out_json))
    assert os.path.exists(out_json)
    print(f"  ✓ JSON 报告: {out_json}")

    print("\n[smoke] ✓ 全部通过")
    srv.shutdown()


if __name__ == "__main__":
    main()