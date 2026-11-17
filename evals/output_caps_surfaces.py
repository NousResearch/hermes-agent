"""Full local execution controls; never sends a request to an inference vendor."""
import json
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path


def exercise_surfaces(agent, captures, url, home, config):
    results = {}
    start = len(captures)
    response = agent.run_conversation("Return the fixture response.")
    assert response["final_response"] == "LOCAL_CAPTURE_ONLY", response
    results["full-agent"] = captures[start:]

    from tools.delegate_tool import _build_child_agent, _run_single_child
    child = _build_child_agent(0, "Return fixture response", None, [], "fixture", 2, 1, agent)
    start = len(captures)
    response = _run_single_child(0, "Return fixture response", child, agent)
    assert response.get("status") == "completed", response
    results["full-child"] = captures[start:]

    slot = {"provider": "fixture-local", "model": "fixture", "max_tokens": 23}
    config["moa"] = {"default_preset": "fixture", "presets": {"fixture": {
        "reference_models": [slot, slot], "aggregator": slot,
        "max_tokens": 29, "reference_max_tokens": 31,
    }}}
    config["auxiliary"] = {"compression": {"provider": "fixture-local", "model": "fixture", "reasoning_effort": "none", "max_output_tokens": 47}}
    config["_setup_done"] = True
    (home / "config.yaml").write_text(json.dumps(config))
    from agent.moa_loop import MoAClient
    start = len(captures)
    response = MoAClient("fixture").chat.completions.create(model="fixture", messages=[{"role": "user", "content": "Return fixture"}])
    assert response.choices[0].message.content == "LOCAL_CAPTURE_ONLY"
    results["full-moa"] = captures[start:]
    assert len(results["full-moa"]) == 3, results["full-moa"]

    from agent.auxiliary_client import call_llm
    start = len(captures)
    response = call_llm(task="compression", messages=[{"role": "user", "content": "Return fixture"}])
    assert response.choices[0].message.content == "LOCAL_CAPTURE_ONLY"
    results["full-compression-call"] = captures[start:]

    from agent.curator import _run_llm_review
    start = len(captures)
    response = _run_llm_review("Return the fixture response without tools.")
    assert response["final"] == "LOCAL_CAPTURE_ONLY", response
    results["full-curator-fork"] = captures[start:]

    from agent.background_review import build_cache_parity_fork
    start = len(captures)
    review, _, _ = build_cache_parity_fork(agent, {}, max_iterations=2)
    try:
        response = review.run_conversation("Return fixture response without tools.")
        assert response["final_response"] == "LOCAL_CAPTURE_ONLY", response
    finally:
        review.close()
    results["full-review-fork"] = captures[start:]

    agent.max_tokens = 43
    start = len(captures)
    review, _, _ = build_cache_parity_fork(agent, {}, max_iterations=2)
    try:
        response = review.run_conversation("Return fixture response without tools.")
        assert response["final_response"] == "LOCAL_CAPTURE_ONLY", response
    finally:
        review.close()
        agent.max_tokens = None
    results["internal-review-budget"] = captures[start:]
    assert results["internal-review-budget"][0]["body"]["max_tokens"] == 43

    import boto3
    from agent.transports.bedrock import BedrockTransport
    native = boto3.client("bedrock-runtime", region_name="us-east-1", endpoint_url=url,
                          aws_access_key_id="fixture", aws_secret_access_key="fixture")
    kwargs = BedrockTransport().build_kwargs("amazon.nova-pro-v1:0", [{"role": "user", "content": "fixture"}])
    kwargs.pop("__bedrock_converse__", None)
    kwargs.pop("__bedrock_region__", None)
    start = len(captures)
    native.converse(**kwargs)
    results["bedrock-sdk-local-not-aws"] = captures[start:]

    master, slave = pty.openpty()
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("API_KEY", "TOKEN", "SECRET"))}
    env.update(HOME=str(home), HERMES_HOME=str(home), HERMES_MAX_TOKENS="13", TERM="xterm", NO_COLOR="1")
    start = len(captures)
    proc = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "chat", "--provider", "fixture-local", "-m", "fixture", "-q", "Return fixture"], stdin=slave, stdout=slave, stderr=slave, cwd=os.getcwd(), env=env)
    os.close(slave)
    output = bytearray()
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            elif proc.poll() is not None:
                break
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=10)
    finally:
        os.close(master)
    (home / "cli-pty.txt").write_bytes(output)
    assert proc.returncode == 0 and b"LOCAL_CAPTURE_ONLY" in output, output.decode(errors="replace")
    results["cli-pty"] = captures[start:]
    assert results["cli-pty"]
    if os.environ.get("VERIFY_OUTPUT_CAP_REMOVAL"):
        for label, requests in results.items():
            if label == "internal-review-budget":
                continue
            assert requests, label
            for request in requests:
                assert "max_tokens" not in request["body"] and "max_completion_tokens" not in request["body"], (label, request)
                assert "maxTokens" not in request["body"].get("inferenceConfig", {}), (label, request)
    return results
