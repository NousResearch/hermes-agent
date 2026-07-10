"""
gpu_agent.py — fully-offline sovereign GPU agent on Victus (RTX 3050 6GB).

A concrete instance of environments.host_loop.LocalHostLoop:
  brain  : ollama qwen2.5-coder:3b @ :11434 (disk weights, ZERO network)
  hands  : CUDA tools on the host
           - probe_gpu     : live nvidia-smi telemetry
           - compile_kernel: nvcc (via MSVC vcvars64.bat) compile a matmul kernel
           - run_kernel    : execute the compiled .exe on the GPU, host-side timing
  manifest: a run record proving probe/compile/run happened offline.

Run:  python gpu_agent.py "probe the GPU then compile and run a CUDA matmul"
"""
from __future__ import annotations
import json, subprocess, tempfile, time
from pathlib import Path

from environments.host_loop import LocalHostLoop

_VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

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


class GPUAgent(LocalHostLoop):
    def __init__(self, model: str = "qwen2.5-coder:3b", max_steps: int = 6):
        self.model = model
        super().__init__(
            planner=self._ollama_planner,
            tools={
                "probe_gpu": self.probe_gpu,
                "compile_kernel": self.compile_kernel,
                "run_kernel": self.run_kernel,
            },
            max_steps=max_steps,
            name="gpu-agent",
        )

    # ---- brain (ollama, offline) ----
    def _ollama_planner(self, prompt: str) -> str:
        try:
            out = subprocess.run(
                ["curl", "-s", "-m", "180", "-d",
                 json.dumps({"model": self.model, "prompt": prompt, "stream": False}),
                 "http://localhost:11434/api/generate"],
                capture_output=True, text=True, timeout=200,
            )
            return json.loads(out.stdout).get("response", "").strip()
        except Exception as e:
            return f"FINAL: brain error ({e})"

    # ---- hands ----
    def probe_gpu(self, arg=None) -> dict:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        p = [x.strip() for x in out.stdout.split(",")]
        used, total = float(p[1]), float(p[2])
        return {"name": p[0], "memory_used_mib": used, "memory_total_mib": total,
                "utilization_gpu": float(p[3]), "temperature_gpu": float(p[4])}

    def _msvc_cmd(self, cmd: list[str]) -> str:
        inner = " ".join(cmd)
        if not Path(_VCVARS).exists():
            return inner
        return f'call "{_VCVARS}" >nul && {inner}'

    def compile_kernel(self, arg=None) -> dict:
        name = arg or "matmul"
        cu = Path(tempfile.gettempdir()) / f"{name}.cu"
        cu.write_text(_KERNEL_SRC)
        exe = Path(tempfile.gettempdir()) / f"{name}.exe"
        r = subprocess.run(f'cmd /c "{self._msvc_cmd(["nvcc", str(cu), "-o", str(exe)])}"',
                           capture_output=True, text=True, timeout=120, shell=True)
        return {"kernel": name, "ok": r.returncode == 0,
                "output": (r.stderr or r.stdout)[:300], "bin": str(exe)}

    def run_kernel(self, arg=None) -> dict:
        name = arg or "matmul"
        exe = Path(tempfile.gettempdir()) / f"{name}.exe"
        if not exe.exists():
            c = self.compile_kernel(name)
            if not c["ok"]:
                return {"ok": False, "reason": "compile failed", "detail": c["output"]}
        t0 = time.perf_counter()
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        dt = round((time.perf_counter() - t0) * 1000, 2)
        return {"ok": r.returncode == 0, "host_ms": dt, "n": 512, "stderr": r.stderr[:200]}


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "Probe the GPU, then compile and run a CUDA matmul kernel, and report what happened."
    agent = GPUAgent()
    result = agent.run(task)
    print("\n=== GPU AGENT RESULT ===")
    print(result["answer"])
    print(f"\n[manifest: {agent.manifest_path}]")
