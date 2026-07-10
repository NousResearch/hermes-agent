"""
gpu_agent.py — fully-offline sovereign GPU agent for Victus (RTX 3050 6GB).

Brain : ollama qwen2.5-coder:3b @ http://localhost:11434  (weights on disk, NO network)
Hands : CUDA tools dispatched on the local host
        - probe_gpu    : live nvidia-smi telemetry
        - compile_kernel: nvcc-compile a matmul kernel
        - run_kernel   : execute the compiled binary, capture host-side timing
Loop  : think -> tool -> observe -> answer  (multi-turn, offline)

No external calls except the local ollama socket. Verifiable via run() / pytest.
"""
from __future__ import annotations
import json, os, re, subprocess, tempfile, time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:3b"
SYS = (
    "You are a GPU-driven sovereign agent on Victus (NVIDIA RTX 3050 6GB Laptop GPU, "
    "CUDA 13.3, driver 592.27). You control the local machine via tool calls. "
    "To call a tool, output EXACTLY ONE line: TOOL: <name> [arg]. "
    "Available tools: probe_gpu, compile_kernel, run_kernel. "
    "After a TOOL_RESULT, continue reasoning. When finished, output FINAL: <answer>. Be terse."
)


class GPUAgent:
    def __init__(self, model: str = MODEL, ollama_url: str = OLLAMA_URL, max_steps: int = 6):
        self.model = model
        self.ollama_url = ollama_url
        self.max_steps = max_steps
        self.history: list[dict] = [{"role": "system", "content": SYS}]
        self.log: list[str] = []

    # ---- brain ----
    def _think(self) -> str:
        payload = {"model": self.model, "prompt": self._prompt(), "stream": False}
        try:
            out = subprocess.run(
                ["curl", "-s", "-m", "180", "-d", json.dumps(payload), self.ollama_url],
                capture_output=True, text=True, timeout=200,
            )
            data = json.loads(out.stdout)
            return data.get("response", "").strip()
        except Exception as e:  # network/parse failure -> deterministic fallback
            return f"FINAL: brain error ({e})"

    def _prompt(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)

    # ---- hands ----
    def probe_gpu(self) -> dict:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        p = [x.strip() for x in out.stdout.split(",")]
        used, total = float(p[1]), float(p[2])
        return {"name": p[0], "memory_used_mib": used, "memory_total_mib": total,
                "utilization_gpu": float(p[3]), "temperature_gpu": float(p[4])}

    _KERNEL_SRC = """
#include <stdio.h>
__global__ void matmul(float* a, float* b, float* c, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n*n) { float s=0; for(int k=0;k<n;k++) s+=a[i/n*n+k]*b[k*n+i%n]; c[i]=s; }
}
int main(){
  const int n = 512;
  size_t bytes = n*n*sizeof(float);
  float *a,*b,*c; cudaMalloc(&a,bytes); cudaMalloc(&b,bytes); cudaMalloc(&c,bytes);
  float *ha=(float*)malloc(bytes); for(int i=0;i<n*n;i++) ha[i]=(float)(i%7);
  cudaMemcpy(a,ha,bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(b,ha,bytes,cudaMemcpyHostToDevice);
  matmul<<<(n*n+255)/256,256>>>(a,b,c,n);
  cudaDeviceSynchronize();
  float *hc=(float*)malloc(bytes); cudaMemcpy(hc,c,bytes,cudaMemcpyDeviceToHost);
  printf("OK matmul n=%d c[0]=%.3f\\n", n, hc[0]);
  free(ha); free(hc); cudaFree(a); cudaFree(b); cudaFree(c);
  return 0;
}
"""

    _VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

    @classmethod
    def _msvc_cmd(cls, cmd: list[str]) -> str:
        """Return a single shell command string that sources MSVC (cl.exe/link.exe on PATH)
        then runs cmd. BuildTools is not on PATH by default, so nvcc needs vcvars first."""
        inner = " ".join(cmd)
        if not Path(cls._VCVARS).exists():
            return inner
        return f'call "{cls._VCVARS}" >nul && {inner}'

    def compile_kernel(self, name: str = "matmul") -> dict:
        cu = Path(tempfile.gettempdir()) / f"{name}.cu"
        cu.write_text(self._KERNEL_SRC)
        exe = Path(tempfile.gettempdir()) / f"{name}.exe"
        r = subprocess.run(f'cmd /c "{self._msvc_cmd(["nvcc", str(cu), "-o", str(exe)])}"', capture_output=True, text=True, timeout=120, shell=True)
        return {"kernel": name, "ok": r.returncode == 0, "output": (r.stderr or r.stdout)[:300], "bin": str(exe)}

    def run_kernel(self, name: str = "matmul", n: int = 512) -> dict:
        exe = Path(tempfile.gettempdir()) / f"{name}.exe"
        if not exe.exists():
            c = self.compile_kernel(name)
            if not c["ok"]:
                return {"ok": False, "reason": "compile failed", "detail": c["output"]}
        t0 = time.perf_counter()
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        dt = round((time.perf_counter() - t0) * 1000, 2)
        return {"ok": r.returncode == 0, "host_ms": dt, "n": n, "stderr": r.stderr[:200]}

    TOOLS = {
        "probe_gpu": lambda self, a: self.probe_gpu(),
        "compile_kernel": lambda self, a: self.compile_kernel(),
        "run_kernel": lambda self, a: self.run_kernel(),
    }

    # ---- loop ----
    def run(self, task: str) -> str:
        self.history.append({"role": "user", "content": task})
        self.log.append(f"agent: task -> {task}")
        for step in range(self.max_steps):
            out = self._think()
            self.history.append({"role": "assistant", "content": out})
            self.log.append(f"think[{step}]: {out[:140]}")
            m = re.search(r"TOOL:\s*(\w+)(?:\s+(\S+))?", out)
            if m and m.group(1) in self.TOOLS:
                fn = self.TOOLS[m.group(1)]
                res = fn(self, m.group(2))
                self.log.append(f"tool {m.group(1)} -> {res}")
                self.history.append({"role": "user", "content": f"TOOL_RESULT: {json.dumps(res)[:400]}"})
                continue
            fm = re.search(r"FINAL:\s*([\s\S]+)", out)
            if fm:
                ans = fm.group(1).strip()
                self.log.append(f"agent ▶ {ans}")
                return ans
        self.log.append("agent: step budget reached")
        return "[incomplete: step budget]"


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "Probe the GPU, then compile and run a CUDA matmul kernel, and report what happened."
    agent = GPUAgent()
    result = agent.run(task)
    print("\n=== GPU AGENT RESULT ===")
    print(result)
    print("\n=== TRACE ===")
    print("\n".join(agent.log))
