"""
wasm_agent.py — fully-offline sovereign Rust/WASM agent on Victus (x86_64-pc-windows-msvc).

A concrete instance of environments.host_loop.LocalHostLoop:
  brain  : ollama qwen2.5-coder:3b @ :11434 (disk weights, ZERO network)
  hands  : Rust/WASM tools on the host
           - rust_build_wasm : cargo build --target wasm32-unknown-unknown
           - rust_run_wasm   : execute via wasmtime if present; else honest capability report
  manifest: a run record proving build/run happened offline.

The loop is platform-portable: any x86_64-pc-windows + rustup + wasm32 target
runs this unchanged (paths are resolved via rustup, not hardcoded to Victus).

Run:  python wasm_agent.py "build the wasm_hello crate to wasm32"
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path

from environments.host_loop import LocalHostLoop
from environments.rust_wasm_tools import build_wasm, run_wasm


class WasmAgent(LocalHostLoop):
    def __init__(self, crate_dir: str, model: str = "qwen2.5-coder:3b", max_steps: int = 6):
        self.crate_dir = crate_dir
        self.model = model
        self.SYSTEM = (
            "You are a sovereign local Rust/WASM agent on x86_64-pc-windows-msvc. "
            "You control the host via tool calls. Emit ONE tool per turn as "
            "`TOOL: <name> [arg]`, or `FINAL: <answer>` when done. "
            "Available tools:\n"
            "  rust_build_wasm [crate_dir]  build a Rust crate to wasm32-unknown-unknown\n"
            "  rust_run_wasm   [wasm_path]  run a .wasm (needs wasmtime) or report capability\n"
            f"The crate to build is at: {crate_dir}\n"
            "Be terse. Do not invent tools."
        )
        super().__init__(
            planner=self._ollama_planner,
            tools={
                "rust_build_wasm": self._hand_build,
                "rust_run_wasm": self._hand_run,
            },
            max_steps=max_steps,
            name="wasm-agent",
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

    # ---- hands (Rust/WASM) ----
    def _hand_build(self, arg=None) -> dict:
        target = arg or self.crate_dir
        return build_wasm(target, use_pack=False)

    def _hand_run(self, arg=None) -> dict:
        if arg and Path(arg).exists():
            return run_wasm(arg)
        built = build_wasm(self.crate_dir, use_pack=False)
        if not built.get("ok"):
            return built
        return run_wasm(built["wasm"])


if __name__ == "__main__":
    import sys
    crate = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path(__file__).resolve().parent.parent / "environments" / "testdata" / "wasm_hello")
    task = sys.argv[2] if len(sys.argv) > 2 else "Build this Rust crate to wasm32-unknown-unknown and report the artifact."
    agent = WasmAgent(crate)
    res = agent.run(task)
    print("\n=== WASM AGENT RESULT ===")
    print(res["answer"])
    print(f"\n[manifest: {agent.manifest_path}]")
