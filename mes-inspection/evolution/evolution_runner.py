"""进化执行器 — 封装 hermes-agent-self-evolution 的 GEPA 调用。

支持自定义模型（Qwen3-235B-A22B-w8a8 等私有部署），
通过 DSPy 的 OpenAI 兼容接口对接。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


MES_SKILLS = [
    "mes-nginx-check",
    "mes-jvm-check",
    "mes-rabbitmq-check",
    "mes-oracle-check",
    "mes-elk-check",
    "mes-skywalking-check",
]


class EvolutionRunner:
    """进化执行器。"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.iterations = self.config.get("iterations", 10)
        self.eval_source = self.config.get("eval_source", "synthetic")
        self.optimizer_model = self.config.get("optimizer_model", "openai/gpt-4.1")
        self.eval_model = self.config.get("eval_model", "openai/gpt-4.1-mini")
        self.hermes_repo = self.config.get("hermes_repo", os.getenv("HERMES_AGENT_REPO", ""))
        self.skills_dir = self.config.get("skills_dir", "")
        self.base_url = self.config.get("base_url", "")
        self.api_key_env = self.config.get("api_key_env", "OPENAI_API_KEY")

    def _find_evolution_lib(self) -> Optional[Path]:
        vendor_path = Path(__file__).parent.parent / "vendor" / "hermes-agent-self-evolution"
        if vendor_path.exists() and (vendor_path / "evolution" / "__init__.py").exists():
            return vendor_path
        return None

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        api_key = os.getenv(self.api_key_env, "")
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if self.base_url:
            base = self.base_url.rstrip("/")
            env["OPENAI_API_BASE"] = base if base.endswith("/v1") else base + "/v1"
        if self.hermes_repo:
            env["HERMES_AGENT_REPO"] = self.hermes_repo
        return env

    def evolve_skill(self, skill_name: str, dry_run: bool = False, timeout: int = 1800) -> Dict[str, Any]:
        lib_path = self._find_evolution_lib()
        if not lib_path:
            return self._format_result(skill_name, -1, "", "hermes-agent-self-evolution 未安装")

        cmd = [
            sys.executable, "-m", "evolution.skills.evolve_skill",
            "--skill", skill_name,
            "--iterations", str(self.iterations),
            "--eval-source", self.eval_source,
            "--optimizer-model", self.optimizer_model,
            "--eval-model", self.eval_model,
        ]
        if self.hermes_repo:
            cmd.extend(["--hermes-repo", self.hermes_repo])
        if dry_run:
            cmd.append("--dry-run")

        env = self._build_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(lib_path))
            return self._format_result(skill_name, result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            return self._format_result(skill_name, -1, "", f"进化超时（{timeout}秒）")
        except Exception as e:
            return self._format_result(skill_name, -1, "", str(e))

    def evolve_all(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        return [self.evolve_skill(name, dry_run=dry_run) for name in MES_SKILLS]

    def list_evolvable_skills(self) -> List[str]:
        return MES_SKILLS.copy()

    def _format_result(self, skill_name: str, returncode: int, stdout: str, stderr: str) -> Dict[str, Any]:
        return {"success": returncode == 0, "skill_name": skill_name, "stdout": stdout[-2000:] if stdout else "", "stderr": stderr[-1000:] if stderr else "", "returncode": returncode}

    def format_report(self, results: List[Dict[str, Any]]) -> str:
        lines = ["🧬 MES 技能进化报告", "━" * 20, ""]
        for r in results:
            icon = "✅" if r["success"] else "❌"
            lines.append(f"{icon} {r['skill_name']}")
            if r["success"] and r.get("stdout"):
                for line in r["stdout"].split("\n"):
                    if "improvement" in line.lower() or "score" in line.lower():
                        lines.append(f"   {line.strip()}")
            elif not r["success"] and r.get("stderr"):
                lines.append(f"   错误: {r['stderr'][:100]}")
            lines.append("")
        success_count = sum(1 for r in results if r["success"])
        lines.append(f"总计: {success_count}/{len(results)} 成功")
        return "\n".join(lines)
