"""测试模型 API 连通性 — Qwen3-235B-A22B-w8a8

用法:
    python scripts/test_model_api.py
    python scripts/test_model_api.py --prompt "解释量子纠缠"
"""
import http.client
import json
import ssl
import sys

API_HOST = "ai-pool.evebattery.com"
API_PATH = "/v1/chat/completions"
API_KEY = "sk-dooFBpzVWgrvf32YLPFfq5r63dEYHELlUjMT84KrEH5wG0zN"
MODEL = "Qwen3-235B-A22B-w8a8"
DEFAULT_PROMPT = "你好，请用一句话介绍自己。"


def _get_prompt():
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_PROMPT


def test_chat_completions(prompt):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    })
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY,
    }

    print(f"[测试] POST https://{API_HOST}{API_PATH}")
    print(f"[测试] 模型: {MODEL}")
    print(f"[测试] 提示: {prompt}")
    print()

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_ciphers("DEFAULT:@SECLEVEL=1")

    try:
        conn = http.client.HTTPSConnection(API_HOST, timeout=60, context=ctx)
        conn.request("POST", API_PATH, payload, headers)
        resp = conn.getresponse()
        status = resp.status
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
    except Exception as e:
        print(f"[失败] 请求异常: {e}")
        sys.exit(1)

    if status != 200:
        print(f"[失败] HTTP {status}")
        print(f"[响应] {json.dumps(body, ensure_ascii=False, indent=2)}")
        sys.exit(1)

    print(f"[成功] HTTP {status}")

    try:
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        print(f"[回复] {content.strip()}")
        print()
        print(f"[用量] prompt_tokens={usage.get('prompt_tokens', '?')}, "
              f"completion_tokens={usage.get('completion_tokens', '?')}, "
              f"total_tokens={usage.get('total_tokens', '?')}")
    except (KeyError, IndexError) as e:
        print(f"[警告] 响应格式异常: {e}")
        print(f"[原始响应] {json.dumps(body, ensure_ascii=False, indent=2)}")
        sys.exit(1)

    print()
    print("模型 API 验证通过")


if __name__ == "__main__":
    test_chat_completions(_get_prompt())
