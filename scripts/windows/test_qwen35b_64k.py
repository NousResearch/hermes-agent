import time
import json
import urllib.request
import psutil
import subprocess
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Qwen35B_64K_Bench")

def run_benchmark():
    url = "http://127.0.0.1:8080/v1/chat/completions"
    models_url = "http://127.0.0.1:8080/v1/models"
    
    # Check server availability
    for attempt in range(40):
        try:
            req_m = urllib.request.Request(models_url)
            with urllib.request.urlopen(req_m, timeout=3) as resp:
                if resp.status == 200:
                    logger.info("llama-server is ready and responding at 8080.")
                    break
        except Exception:
            time.sleep(3)
    else:
        logger.error("llama-server failed to become ready in time.")
        sys.exit(1)

    # Prompt testing
    prompt = "Explain the principles of quantum context, superposition, and entanglement in 3 concise paragraphs."
    payload = {
        "model": "Qwen3.6-35B-A3B-Uncensored-IQ3_M",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    logger.info("Executing 64K context benchmark request...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            res = json.loads(response.read().decode("utf-8"))
        t1 = time.time()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)

    usage = res.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_time = t1 - t0
    gen_speed = completion_tokens / total_time if total_time > 0 else 0

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        vram_used, vram_total, gpu_util = smi.split(",")
    except Exception:
        vram_used, vram_total, gpu_util = "N/A", "N/A", "N/A"

    ram_used = psutil.virtual_memory().used / (1024**3)
    ram_total = psutil.virtual_memory().total / (1024**3)

    logger.info("=== BENCHMARK EVALUATION RESULTS (64K Context / TurboQuant turbo3) ===")
    logger.info("Model: Qwen3.6-35B-A3B-Uncensored-IQ3_M")
    logger.info("Context Length: 65536 (64K)")
    logger.info("KV Cache Type: turbo3 (TurboQuant KV compression)")
    logger.info(f"Prompt Tokens: {prompt_tokens}")
    logger.info(f"Completion Tokens: {completion_tokens}")
    logger.info(f"Total Response Time: {total_time:.2f} s")
    logger.info(f"Generation Speed: {gen_speed:.2f} tokens/sec")
    logger.info(f"GPU VRAM Used: {vram_used.strip()} MB / {vram_total.strip()} MB")
    logger.info(f"GPU Utilization: {gpu_util.strip()}%")
    logger.info(f"System RAM Used: {ram_used:.2f} GB / {ram_total:.2f} GB")
    logger.info("=== SAMPLE RESPONSE ===")
    logger.info(res["choices"][0]["message"]["content"][:200] + "...")

if __name__ == "__main__":
    run_benchmark()
